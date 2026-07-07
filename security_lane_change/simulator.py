import csv
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.models import (
    CURRENT_LANE,
    DecisionSolveRecord,
    TARGET_LANE,
    SolveRecord,
    SimulationResult,
    VehicleState,
)
from security_lane_change.planner import SecurityLaneChangePlanner
from security_lane_change.safety import front_safety_margin, minimum_safe_gap
from security_lane_change.scenario import AttackedLaneChangeScenario


class RecedingHorizonSimulator:
    """Execute only the first step of each plan, then replan after dt."""

    def __init__(
        self,
        cfg: PlannerConfig,
        scenario: AttackedLaneChangeScenario,
        planner: SecurityLaneChangePlanner,
    ) -> None:
        self.cfg = cfg
        self.scenario = scenario
        self.planner = planner
        self.decision_solve_records: list[DecisionSolveRecord] = []

    def run(self) -> SimulationResult:
        steps = self.cfg.simulation_steps
        dt = self.cfg.dt
        collision_threshold = 1.0
        if hasattr(self.scenario, "reset_dynamic_trajectories"):
            self.scenario.reset_dynamic_trajectories()
        if hasattr(self.planner, "configure_for_scenario"):
            self.planner.configure_for_scenario(self.scenario)
        attacked_vehicle = getattr(self.scenario, "attacked_vehicle", "LT")
        prediction_label = getattr(self.scenario, "prediction_label", "attacked prediction at t=0")
        full_prediction_packet = self.scenario.build_prediction_packet(0, steps)
        first_attacked_prediction = (
            full_prediction_packet.lc if attacked_vehicle == "LC" else full_prediction_packet.lt
        )

        ego_x = np.full(steps + 1, np.nan)
        ego_v = np.full(steps + 1, np.nan)
        ego_a = np.full(steps + 1, np.nan)
        ego_lane = np.full(steps + 1, -1, dtype=int)
        front_gap = np.full(steps + 1, np.nan)
        required_front_gap = np.full(steps + 1, np.nan)
        front_safety = np.full(steps + 1, np.nan)
        ttc = np.full(steps + 1, np.inf)
        ttc_front_vehicle = [""] * (steps + 1)
        lc_x = self.scenario.true_lc.x[: steps + 1]
        lc_v = self.scenario.true_lc.v[: steps + 1]
        lt_x = self.scenario.true_lt.x[: steps + 1]
        lt_v = self.scenario.true_lt.v[: steps + 1]

        ego = self.scenario.ego_initial
        ego_x[0] = ego.x
        ego_v[0] = ego.v
        ego_a[0] = ego.a
        ego_lane[0] = CURRENT_LANE
        previous_accel = ego.a
        current_lane = CURRENT_LANE
        merge_time: Optional[float] = None
        collision_time: Optional[float] = None
        collision_index: Optional[int] = None
        collision_front_vehicle: Optional[str] = None

        self._update_front_metrics(
            step=0,
            ego_x=ego_x,
            ego_v=ego_v,
            ego_lane=ego_lane,
            lc_x=lc_x,
            lc_v=lc_v,
            lt_x=lt_x,
            lt_v=lt_v,
            front_gap=front_gap,
            required_front_gap=required_front_gap,
            front_safety=front_safety,
            ttc=ttc,
            ttc_front_vehicle=ttc_front_vehicle,
            collision_threshold=collision_threshold,
        )
        collision_vehicle = self._collision_vehicle_at_step(
            step=0,
            ego_x=ego_x,
            ego_lane=ego_lane,
            front_gap=front_gap,
            ttc_front_vehicle=ttc_front_vehicle,
            collision_threshold=collision_threshold,
        )
        if collision_vehicle is not None:
            collision_time = 0.0
            collision_index = 0
            collision_front_vehicle = collision_vehicle

        step = 0
        while step < steps and collision_time is None:
            if ego.v <= 1e-9:
                next_index = step + 1
                next_ego = VehicleState(x=ego.x, v=0.0, a=0.0)
                ego_x[next_index] = ego.x
                ego_v[next_index] = 0.0
                ego_a[step] = 0.0
                ego_lane[next_index] = current_lane
                if hasattr(self.scenario, "update_ft_after_ego_step"):
                    self.scenario.update_ft_after_ego_step(step, next_ego, current_lane)
                ego = next_ego
                previous_accel = 0.0
                step = next_index
                self._update_front_metrics(
                    step=step,
                    ego_x=ego_x,
                    ego_v=ego_v,
                    ego_lane=ego_lane,
                    lc_x=lc_x,
                    lc_v=lc_v,
                    lt_x=lt_x,
                    lt_v=lt_v,
                    front_gap=front_gap,
                    required_front_gap=required_front_gap,
                    front_safety=front_safety,
                    ttc=ttc,
                    ttc_front_vehicle=ttc_front_vehicle,
                    collision_threshold=collision_threshold,
                )
                collision_vehicle = self._collision_vehicle_at_step(
                    step=step,
                    ego_x=ego_x,
                    ego_lane=ego_lane,
                    front_gap=front_gap,
                    ttc_front_vehicle=ttc_front_vehicle,
                    collision_threshold=collision_threshold,
                )
                if collision_vehicle is not None:
                    collision_time = step * dt
                    collision_index = step
                    collision_front_vehicle = collision_vehicle
                continue

            if hasattr(self.planner, "build_planning_packet"):
                packet = self.planner.build_planning_packet(self.scenario, step)
            else:
                packet = self.scenario.build_prediction_packet(step, self.cfg.horizon_steps)
            solve_records_before = len(getattr(self.planner, "solve_records", []))
            planning_start = perf_counter()
            plan = self.planner.plan(
                ego,
                current_lane,
                previous_accel,
                packet,
                simulation_step=step,
                simulation_time=step * dt,
            )
            total_planning_time = perf_counter() - planning_start
            new_solve_records = getattr(self.planner, "solve_records", [])[solve_records_before:]
            self._record_decision_solve_time(
                step=step,
                simulation_time=step * dt,
                plan=plan,
                solve_records=new_solve_records,
                total_planning_time=total_planning_time,
            )

            execution_steps = min(self._execution_steps(plan), steps - step)
            for local_step in range(execution_steps):
                accel, next_lane, next_x, next_v = self._planned_next_state(
                    plan=plan,
                    local_step=local_step,
                    ego=ego,
                    current_lane=current_lane,
                    previous_accel=previous_accel,
                    step=step,
                )
                next_index = step + 1
                next_ego = VehicleState(x=next_x, v=next_v, a=accel)

                ego_x[next_index] = next_x
                ego_v[next_index] = next_v
                ego_a[step] = accel
                ego_lane[next_index] = next_lane
                if hasattr(self.scenario, "update_ft_after_ego_step"):
                    self.scenario.update_ft_after_ego_step(step, next_ego, next_lane)
                if hasattr(self.scenario, "update_lt_after_ego_step"):
                    self.scenario.update_lt_after_ego_step(step, next_ego)

                if current_lane == CURRENT_LANE and next_lane == TARGET_LANE and merge_time is None:
                    merge_time = next_index * dt

                current_lane = next_lane
                ego = next_ego
                previous_accel = accel
                step = next_index

                self._update_front_metrics(
                    step=step,
                    ego_x=ego_x,
                    ego_v=ego_v,
                    ego_lane=ego_lane,
                    lc_x=lc_x,
                    lc_v=lc_v,
                    lt_x=lt_x,
                    lt_v=lt_v,
                    front_gap=front_gap,
                    required_front_gap=required_front_gap,
                    front_safety=front_safety,
                    ttc=ttc,
                    ttc_front_vehicle=ttc_front_vehicle,
                    collision_threshold=collision_threshold,
                )
                collision_vehicle = self._collision_vehicle_at_step(
                    step=step,
                    ego_x=ego_x,
                    ego_lane=ego_lane,
                    front_gap=front_gap,
                    ttc_front_vehicle=ttc_front_vehicle,
                    collision_threshold=collision_threshold,
                )
                if collision_vehicle is not None:
                    collision_time = step * dt
                    collision_index = step
                    collision_front_vehicle = collision_vehicle
                    break

        if collision_index is not None and hasattr(self.scenario, "propagate_after_collision"):
            self.scenario.propagate_after_collision(
                collision_index=collision_index,
                collision_front_vehicle=collision_front_vehicle,
                ego_x=ego_x,
                ego_v=ego_v,
                ego_a=ego_a,
                ego_lane=ego_lane,
            )
            for post_step in range(collision_index + 1, steps + 1):
                self._update_front_metrics(
                    step=post_step,
                    ego_x=ego_x,
                    ego_v=ego_v,
                    ego_lane=ego_lane,
                    lc_x=lc_x,
                    lc_v=lc_v,
                    lt_x=lt_x,
                    lt_v=lt_v,
                    front_gap=front_gap,
                    required_front_gap=required_front_gap,
                    front_safety=front_safety,
                    ttc=ttc,
                    ttc_front_vehicle=ttc_front_vehicle,
                    collision_threshold=collision_threshold,
                )

        valid_indices = np.where(np.isfinite(ego_x))[0]
        if len(valid_indices) >= 2:
            ego_a[valid_indices[-1]] = ego_a[valid_indices[-2]]
        time = np.arange(steps + 1, dtype=float) * dt
        ft_x = self.scenario.true_ft.x[: steps + 1]
        ft_v = self.scenario.true_ft.v[: steps + 1]

        return SimulationResult(
            time=time,
            ego_x=ego_x,
            ego_v=ego_v,
            ego_a=ego_a,
            ego_lane=ego_lane,
            lc_x=lc_x,
            lc_v=lc_v,
            lt_x=lt_x,
            lt_v=lt_v,
            ft_x=ft_x,
            ft_v=ft_v,
            front_gap=front_gap,
            required_front_gap=required_front_gap,
            front_safety_margin=front_safety,
            ttc=ttc,
            ttc_front_vehicle=ttc_front_vehicle,
            merge_time=merge_time,
            collision_time=collision_time,
            collision_index=collision_index,
            collision_front_vehicle=collision_front_vehicle,
            collision_threshold=collision_threshold,
            first_attacked_prediction=first_attacked_prediction,
            attacked_vehicle=attacked_vehicle,
            prediction_label=prediction_label,
        )

    def _execution_steps(self, plan) -> int:
        configured_steps = int(getattr(self.planner, "execution_steps", 1))
        if getattr(self.planner, "open_loop_only_when_attacked", False):
            if not bool(getattr(self.scenario, "has_network_attack", True)):
                configured_steps = 1
        if not plan.feasible or len(plan.a) == 0:
            return 1
        return max(1, min(configured_steps, len(plan.a)))

    def _planned_next_state(
        self,
        plan,
        local_step: int,
        ego: VehicleState,
        current_lane: int,
        previous_accel: float,
        step: int,
    ) -> tuple[float, int, float, float]:
        if plan.feasible and local_step < len(plan.a):
            accel = float(plan.a[local_step])
            lane_idx = min(local_step + 1, len(plan.lane) - 1)
            next_lane = int(plan.lane[lane_idx]) if lane_idx >= 0 else current_lane
            state_idx = local_step + 1
            if state_idx < len(plan.x) and state_idx < len(plan.v):
                next_x = float(plan.x[state_idx])
                next_v = float(plan.v[state_idx])
                guarded = self._unmerged_lc_follow_guard(
                    current_lane=current_lane,
                    next_lane=next_lane,
                    next_v=next_v,
                    ego=ego,
                    previous_accel=previous_accel,
                    step=step,
                )
                if guarded is not None:
                    accel, next_x, next_v = guarded
                    next_lane = CURRENT_LANE
                return float(accel), int(next_lane), float(next_x), float(next_v)
        else:
            if getattr(self.planner, "infeasible_accel_policy", "emergency_brake") == "keep_speed":
                accel = 0.0
            else:
                accel = self.cfg.limits.a_min
            next_lane = current_lane

        if ego.v + accel * self.cfg.dt < self.cfg.limits.v_min:
            accel = (self.cfg.limits.v_min - ego.v) / self.cfg.dt
        next_v = max(self.cfg.limits.v_min, min(self.cfg.limits.v_max, ego.v + accel * self.cfg.dt))
        next_x = ego.x + ego.v * self.cfg.dt + 0.5 * accel * self.cfg.dt * self.cfg.dt
        guarded = self._unmerged_lc_follow_guard(
            current_lane=current_lane,
            next_lane=next_lane,
            next_v=next_v,
            ego=ego,
            previous_accel=previous_accel,
            step=step,
        )
        if guarded is not None:
            accel, next_x, next_v = guarded
            next_lane = CURRENT_LANE
        return float(accel), int(next_lane), float(next_x), float(next_v)

    def _unmerged_lc_follow_guard(
        self,
        current_lane: int,
        next_lane: int,
        next_v: float,
        ego: VehicleState,
        previous_accel: float,
        step: int,
    ) -> Optional[tuple[float, float, float]]:
        if not bool(getattr(self.planner, "prevent_unmerged_stop_on_safe_lc", False)):
            return None
        if current_lane != CURRENT_LANE or next_lane != CURRENT_LANE:
            return None

        speed_threshold = float(getattr(self.planner, "unmerged_stop_guard_speed_threshold_mps", 0.25))
        if float(next_v) > speed_threshold:
            return None

        if step + 1 >= len(self.scenario.true_lc.x):
            return None
        return self._current_lane_lc_following_step(ego, previous_accel, step)

    def _current_lane_lc_following_step(
        self,
        ego: VehicleState,
        previous_accel: float,
        step: int,
    ) -> Optional[tuple[float, float, float]]:
        dt = self.cfg.dt
        limits = self.cfg.limits
        jerk_step = limits.j_max * dt
        lower_accel = max(limits.a_min, previous_accel - jerk_step)
        upper_accel = min(limits.a_max, previous_accel + jerk_step)
        desired_accel = (self.cfg.desired_speed - ego.v) / dt
        desired_accel = float(np.clip(desired_accel, lower_accel, upper_accel))

        next_leader_idx = min(step + 1, len(self.scenario.true_lc.x) - 1)
        leader_x = float(self.scenario.true_lc.x[next_leader_idx])
        leader_v = float(self.scenario.true_lc.v[next_leader_idx])
        if leader_v <= float(getattr(self.planner, "unmerged_stop_guard_leader_speed_threshold_mps", 0.1)):
            return None

        candidate_accels = np.linspace(desired_accel, lower_accel, 81)
        for accel in candidate_accels:
            accel = self._clip_accel_for_speed(float(accel), ego.v)
            next_v = self._next_speed(ego.v, accel)
            next_x = ego.x + ego.v * dt + 0.5 * accel * dt * dt
            safe_gap = minimum_safe_gap(leader_v, next_v, self.cfg)
            if leader_x - next_x >= safe_gap - 1e-6:
                return float(accel), float(next_x), float(next_v)
        return None

    def _clip_accel_for_speed(self, accel: float, speed: float) -> float:
        if speed + accel * self.cfg.dt < self.cfg.limits.v_min:
            return (self.cfg.limits.v_min - speed) / self.cfg.dt
        if speed + accel * self.cfg.dt > self.cfg.limits.v_max:
            return (self.cfg.limits.v_max - speed) / self.cfg.dt
        return float(accel)

    def _next_speed(self, speed: float, accel: float) -> float:
        return max(self.cfg.limits.v_min, min(self.cfg.limits.v_max, speed + accel * self.cfg.dt))

    def _update_front_metrics(
        self,
        step: int,
        ego_x: np.ndarray,
        ego_v: np.ndarray,
        ego_lane: np.ndarray,
        lc_x: np.ndarray,
        lc_v: np.ndarray,
        lt_x: np.ndarray,
        lt_v: np.ndarray,
        front_gap: np.ndarray,
        required_front_gap: np.ndarray,
        front_safety: np.ndarray,
        ttc: np.ndarray,
        ttc_front_vehicle: list[str],
        collision_threshold: float,
    ) -> None:
        if not np.isfinite(ego_x[step]) or not np.isfinite(ego_v[step]):
            return
        if int(ego_lane[step]) == TARGET_LANE:
            leader_x = lt_x[step]
            leader_v = lt_v[step]
            leader_name = "LT"
        else:
            leader_x = lc_x[step]
            leader_v = lc_v[step]
            leader_name = "LC"
        front_gap[step] = leader_x - ego_x[step]
        required_front_gap[step] = minimum_safe_gap(leader_v, ego_v[step], self.cfg)
        front_safety[step] = front_safety_margin(
            leader_x=leader_x,
            leader_v=leader_v,
            follower_x=ego_x[step],
            follower_v=ego_v[step],
            cfg=self.cfg,
        )
        ttc_front_vehicle[step] = leader_name
        closing_speed = ego_v[step] - leader_v
        if front_gap[step] < collision_threshold:
            ttc[step] = 0.0
        elif closing_speed > 1e-9:
            ttc[step] = front_gap[step] / closing_speed

    def _collision_vehicle_at_step(
        self,
        step: int,
        ego_x: np.ndarray,
        ego_lane: np.ndarray,
        front_gap: np.ndarray,
        ttc_front_vehicle: list[str],
        collision_threshold: float,
    ) -> Optional[str]:
        if not np.isfinite(ego_x[step]):
            return None
        if front_gap[step] < collision_threshold:
            return ttc_front_vehicle[step]
        if int(ego_lane[step]) == TARGET_LANE:
            ft_gap = ego_x[step] - float(self.scenario.true_ft.x[step])
            if ft_gap < collision_threshold:
                return "FT"
        return None

    def _record_decision_solve_time(
        self,
        step: int,
        simulation_time: float,
        plan,
        solve_records: list[SolveRecord],
        total_planning_time: float,
    ) -> None:
        selected_merge_time = None if plan.merge_step is None else plan.merge_step * self.cfg.dt
        self.decision_solve_records.append(
            DecisionSolveRecord(
                simulation_step=int(step),
                simulation_time=float(simulation_time),
                selected_merge_step=plan.merge_step,
                selected_merge_time=selected_merge_time,
                selected_status=plan.solver_status,
                selected_feasible=bool(plan.feasible),
                selected_objective_value=float(plan.cost),
                candidate_problem_count=len(solve_records),
                feasible_candidate_count=sum(1 for record in solve_records if record.feasible),
                gurobi_subproblem_time_sum=float(sum(record.solve_time for record in solve_records)),
                total_planning_time=float(total_planning_time),
            )
        )


def export_result_csv(
    result: SimulationResult,
    output_path: Path,
    algorithm_label: str = "",
    method_label: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "method",
                "time",
                "ego_x",
                "ego_v",
                "ego_a",
                "ego_lane",
                "lc_x",
                "lc_v",
                "lt_x_true",
                "lt_v_true",
                "ft_x",
                "ft_v",
                "front_gap",
                "required_front_gap",
                "front_safety_margin",
                "ttc",
                "ttc_front_vehicle",
                "collision_event",
                "collision_threshold",
            ]
        )
        for i in range(len(result.time)):
            collision_event = result.collision_index == i
            writer.writerow(
                [
                    algorithm_label,
                    method_label,
                    result.time[i],
                    result.ego_x[i],
                    result.ego_v[i],
                    result.ego_a[i],
                    result.ego_lane[i],
                    result.lc_x[i],
                    result.lc_v[i],
                    result.lt_x[i],
                    result.lt_v[i],
                    result.ft_x[i],
                    result.ft_v[i],
                    result.front_gap[i],
                    result.required_front_gap[i],
                    result.front_safety_margin[i],
                    _format_ttc(result.ttc[i]),
                    result.ttc_front_vehicle[i],
                    int(collision_event),
                    result.collision_threshold,
                ]
            )
    return output_path


def export_solve_times_csv(
    records: list[DecisionSolveRecord],
    output_path: Path,
    algorithm_label: str = "",
    method_label: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "method",
                "simulation_step",
                "simulation_time",
                "selected_merge_step",
                "selected_merge_time",
                "selected_status",
                "selected_feasible",
                "selected_objective_value",
                "candidate_problem_count",
                "feasible_candidate_count",
                "gurobi_subproblem_time_sum",
                "total_planning_time",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    algorithm_label,
                    method_label,
                    record.simulation_step,
                    record.simulation_time,
                    "" if record.selected_merge_step is None else record.selected_merge_step,
                    "" if record.selected_merge_time is None else record.selected_merge_time,
                    record.selected_status,
                    int(record.selected_feasible),
                    record.selected_objective_value,
                    record.candidate_problem_count,
                    record.feasible_candidate_count,
                    record.gurobi_subproblem_time_sum,
                    record.total_planning_time,
                ]
            )
    return output_path


def export_candidate_solve_times_csv(
    records: list[SolveRecord],
    output_path: Path,
    algorithm_label: str = "",
    method_label: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "method",
                "simulation_step",
                "simulation_time",
                "candidate_merge_step",
                "candidate_merge_time",
                "status",
                "feasible",
                "objective_value",
                "solve_time",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    algorithm_label,
                    method_label,
                    record.simulation_step,
                    record.simulation_time,
                    "" if record.candidate_merge_step is None else record.candidate_merge_step,
                    "" if record.candidate_merge_time is None else record.candidate_merge_time,
                    record.status,
                    int(record.feasible),
                    record.objective_value,
                    record.solve_time,
                ]
            )
    return output_path


def export_ttc_csv(
    result: SimulationResult,
    output_path: Path,
    algorithm_label: str = "",
    method_label: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm",
                "method",
                "time",
                "ego_lane",
                "front_vehicle",
                "ego_x",
                "ego_v",
                "front_x",
                "front_v",
                "front_gap",
                "closing_speed",
                "ttc",
            ]
        )
        for i in range(len(result.time)):
            if result.ttc_front_vehicle[i] == "LT":
                front_x = result.lt_x[i]
                front_v = result.lt_v[i]
                closing_speed = result.ego_v[i] - front_v
            elif result.ttc_front_vehicle[i] == "LC":
                front_x = result.lc_x[i]
                front_v = result.lc_v[i]
                closing_speed = result.ego_v[i] - front_v
            else:
                front_x = ""
                front_v = ""
                closing_speed = ""
            writer.writerow(
                [
                    algorithm_label,
                    method_label,
                    result.time[i],
                    result.ego_lane[i],
                    result.ttc_front_vehicle[i],
                    result.ego_x[i],
                    result.ego_v[i],
                    front_x,
                    front_v,
                    result.front_gap[i],
                    closing_speed,
                    _format_ttc(result.ttc[i]),
                ]
            )
    return output_path


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("inf")
    return float(np.mean(finite))


def finite_min(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("inf")
    return float(np.min(finite))


def _format_ttc(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{float(value):.10f}"
