from typing import List, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from security_lane_change.config import PlannerConfig
from security_lane_change.models import (
    CURRENT_LANE,
    TARGET_LANE,
    PredictionPacket,
    SolveRecord,
    TrajectoryPlan,
    VehicleState,
)
from security_lane_change.safety import build_protective_trajectory, minimum_safe_gap


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
    }
    return names.get(status, f"STATUS_{status}")


class LongitudinalTrajectoryOptimizer:
    """Gurobi lower-level optimizer for x, v and a under a fixed lane sequence."""

    def __init__(self, cfg: PlannerConfig, leader_policy: str = "security") -> None:
        self.cfg = cfg
        self.leader_policy = leader_policy
        self._speed_grid = np.linspace(cfg.limits.v_min, cfg.limits.v_max, 61).tolist()

    def optimize(
        self,
        ego_state: VehicleState,
        previous_accel: float,
        lane_sequence: np.ndarray,
        packet: PredictionPacket,
    ) -> TrajectoryPlan:
        horizon_steps = len(lane_sequence) - 1
        lc_leader = self._leader_trajectory(packet.lc, horizon_steps)
        lt_leader = self._leader_trajectory(packet.lt, horizon_steps)

        model = gp.Model("secure_lane_change_lower")
        model.Params.OutputFlag = 0

        x = model.addVars(horizon_steps + 1, lb=-GRB.INFINITY, name="x")
        v = model.addVars(
            horizon_steps + 1,
            lb=self.cfg.limits.v_min,
            ub=self.cfg.limits.v_max,
            name="v",
        )
        a = model.addVars(
            horizon_steps,
            lb=self.cfg.limits.a_min,
            ub=self.cfg.limits.a_max,
            name="a",
        )

        model.addConstr(x[0] == float(ego_state.x), name="x_initial")
        model.addConstr(v[0] == float(ego_state.v), name="v_initial")

        jerk_step = self.cfg.limits.j_max * self.cfg.dt
        for step in range(horizon_steps):
            model.addConstr(
                x[step + 1]
                == x[step] + v[step] * self.cfg.dt + 0.5 * a[step] * self.cfg.dt**2,
                name=f"dyn_x_{step}",
            )
            model.addConstr(
                v[step + 1] == v[step] + a[step] * self.cfg.dt,
                name=f"dyn_v_{step}",
            )
            if step == 0:
                model.addConstr(a[step] - previous_accel <= jerk_step, name="jerk_up_0")
                model.addConstr(previous_accel - a[step] <= jerk_step, name="jerk_dn_0")
            else:
                model.addConstr(a[step] - a[step - 1] <= jerk_step, name=f"jerk_up_{step}")
                model.addConstr(a[step - 1] - a[step] <= jerk_step, name=f"jerk_dn_{step}")

        self._add_front_constraints(model, x, v, lane_sequence, lc_leader, lt_leader)
        self._add_rear_target_constraints(model, x, v, lane_sequence, packet)
        self._add_optional_terminal_constraints(model, v, a, horizon_steps)

        objective = gp.QuadExpr()
        for step in range(1, horizon_steps + 1):
            objective += self.cfg.w_speed * (v[step] - self.cfg.desired_speed) * (
                v[step] - self.cfg.desired_speed
            )
        for step in range(horizon_steps):
            objective += self.cfg.w_accel * a[step] * a[step]
            if step == 0:
                objective += self.cfg.w_jerk * (a[step] - previous_accel) * (
                    a[step] - previous_accel
                )
            else:
                objective += self.cfg.w_jerk * (a[step] - a[step - 1]) * (
                    a[step] - a[step - 1]
                )
        model.setObjective(objective, GRB.MINIMIZE)

        model.optimize()
        status = _status_name(model.Status)
        solve_time = float(model.Runtime)
        feasible = model.SolCount > 0 and model.Status in {
            GRB.OPTIMAL,
            GRB.SUBOPTIMAL,
            GRB.TIME_LIMIT,
        }

        if not feasible:
            return self._infeasible_plan(
                ego_state,
                previous_accel,
                lane_sequence,
                status=status,
                solve_time=solve_time,
            )

        return TrajectoryPlan(
            x=np.asarray([x[step].X for step in range(horizon_steps + 1)], dtype=float),
            v=np.asarray([v[step].X for step in range(horizon_steps + 1)], dtype=float),
            a=np.asarray([a[step].X for step in range(horizon_steps)], dtype=float),
            lane=lane_sequence.astype(int),
            cost=float(model.ObjVal),
            feasible=True,
            solver_status=status,
            solve_time=solve_time,
        )

    def _leader_trajectory(self, received_trajectory, horizon_steps: int):
        if self.leader_policy == "security":
            return build_protective_trajectory(received_trajectory, horizon_steps, self.cfg)
        if self.leader_policy == "fully_trust":
            return received_trajectory
        raise ValueError(f"Unknown leader policy: {self.leader_policy}")

    def _add_front_constraints(
        self,
        model: gp.Model,
        x,
        v,
        lane_sequence: np.ndarray,
        lc_protect,
        lt_protect,
    ) -> None:
        for step in range(1, len(lane_sequence)):
            leader = lt_protect if int(lane_sequence[step]) == TARGET_LANE else lc_protect
            gap = model.addVar(lb=0.0, name=f"front_dmin_{step}")
            gap_values = [
                minimum_safe_gap(float(leader.v[step]), float(speed), self.cfg)
                for speed in self._speed_grid
            ]
            model.addGenConstrPWL(
                v[step],
                gap,
                self._speed_grid,
                gap_values,
                name=f"front_dmin_pwl_{step}",
            )
            model.addConstr(
                x[step] <= float(leader.x[step]) - gap,
                name=f"front_safety_{step}",
            )

    def _add_rear_target_constraints(
        self,
        model: gp.Model,
        x,
        v,
        lane_sequence: np.ndarray,
        packet: PredictionPacket,
    ) -> None:
        for step in range(1, len(lane_sequence)):
            if int(lane_sequence[step]) != TARGET_LANE:
                continue
            ft_x = float(packet.ft.x[step])
            ft_v = float(packet.ft.v[step])
            gap = model.addVar(lb=0.0, name=f"rear_dmin_target_{step}")
            gap_values = [
                self._rear_target_safe_gap(float(ego_speed_as_leader), ft_v)
                for ego_speed_as_leader in self._speed_grid
            ]
            model.addGenConstrPWL(
                v[step],
                gap,
                self._speed_grid,
                gap_values,
                name=f"rear_dmin_pwl_target_{step}",
            )
            model.addConstr(
                x[step] >= ft_x + gap,
                name=f"rear_safety_target_{step}",
            )

    def _rear_target_safe_gap(self, ego_speed_as_leader: float, ft_speed: float) -> float:
        base_gap = minimum_safe_gap(float(ego_speed_as_leader), float(ft_speed), self.cfg)
        buffer_time = float(getattr(self, "rear_closing_time_buffer_s", 0.0))
        if buffer_time <= 0.0:
            return base_gap
        closing_buffer = max(0.0, float(ft_speed) - float(ego_speed_as_leader)) * buffer_time
        return base_gap + closing_buffer

    def _add_optional_terminal_constraints(self, model: gp.Model, v, a, horizon_steps: int) -> None:
        terminal_min_speed = getattr(self, "terminal_min_speed", None)
        if terminal_min_speed is not None:
            model.addConstr(
                v[horizon_steps] >= max(self.cfg.limits.v_min, float(terminal_min_speed)),
                name="terminal_min_speed",
            )

        terminal_min_accel = getattr(self, "terminal_min_accel", None)
        if terminal_min_accel is not None and horizon_steps > 0:
            model.addConstr(
                a[horizon_steps - 1] >= max(self.cfg.limits.a_min, float(terminal_min_accel)),
                name="terminal_min_accel",
            )

    def _merge_step_from_sequence(self, lane_sequence: np.ndarray) -> Optional[int]:
        for step in range(1, len(lane_sequence)):
            if int(lane_sequence[step - 1]) == CURRENT_LANE and int(lane_sequence[step]) == TARGET_LANE:
                return step
        return None

    def _infeasible_plan(
        self,
        ego_state: VehicleState,
        previous_accel: float,
        lane_sequence: np.ndarray,
        status: str = "INFEASIBLE",
        solve_time: float = 0.0,
    ) -> TrajectoryPlan:
        return TrajectoryPlan(
            x=np.asarray([ego_state.x], dtype=float),
            v=np.asarray([ego_state.v], dtype=float),
            a=np.asarray([previous_accel], dtype=float),
            lane=lane_sequence.astype(int),
            cost=float("inf"),
            feasible=False,
            solver_status=status,
            solve_time=solve_time,
        )


class SecurityLaneChangePlanner:
    """Bi-level planner: upper lane-change timing, lower Gurobi trajectory planning."""

    def __init__(self, cfg: PlannerConfig, leader_policy: str = "security") -> None:
        self.cfg = cfg
        self.lower_optimizer = LongitudinalTrajectoryOptimizer(cfg, leader_policy=leader_policy)
        self.solve_records: List[SolveRecord] = []

    def plan(
        self,
        ego_state: VehicleState,
        current_lane: int,
        previous_accel: float,
        packet: PredictionPacket,
        simulation_step: int = 0,
        simulation_time: float = 0.0,
    ) -> TrajectoryPlan:
        if current_lane == TARGET_LANE:
            lane_sequence = np.full(self.cfg.horizon_steps + 1, TARGET_LANE, dtype=int)
            plan = self.lower_optimizer.optimize(ego_state, previous_accel, lane_sequence, packet)
            plan.merge_step = 0
            self._record_solve(simulation_step, simulation_time, 0, plan)
            return plan

        best_plan: Optional[TrajectoryPlan] = None
        for merge_step in self._candidate_merge_steps():
            lane_sequence = self._lane_sequence_for_candidate(merge_step)
            plan = self.lower_optimizer.optimize(ego_state, previous_accel, lane_sequence, packet)
            plan.merge_step = merge_step
            self._record_solve(simulation_step, simulation_time, merge_step, plan)
            if not plan.feasible:
                continue
            plan.cost += self._upper_level_cost(merge_step)
            if best_plan is None or plan.cost < best_plan.cost:
                best_plan = plan

        if best_plan is None:
            stay_current = np.full(self.cfg.horizon_steps + 1, CURRENT_LANE, dtype=int)
            return self.lower_optimizer._infeasible_plan(
                ego_state,
                previous_accel,
                stay_current,
                status="NO_FEASIBLE_CANDIDATE",
            )
        return best_plan

    def _record_solve(
        self,
        simulation_step: int,
        simulation_time: float,
        merge_step: Optional[int],
        plan: TrajectoryPlan,
    ) -> None:
        merge_time = None if merge_step is None else merge_step * self.cfg.dt
        self.solve_records.append(
            SolveRecord(
                simulation_step=int(simulation_step),
                simulation_time=float(simulation_time),
                candidate_merge_step=merge_step,
                candidate_merge_time=merge_time,
                status=plan.solver_status,
                feasible=bool(plan.feasible),
                objective_value=float(plan.cost),
                solve_time=float(plan.solve_time),
            )
        )

    def _candidate_merge_steps(self) -> List[Optional[int]]:
        steps: List[Optional[int]] = list(
            range(
                1,
                self.cfg.horizon_steps + 1,
                max(1, self.cfg.candidate_stride_steps),
            )
        )
        if steps[-1] != self.cfg.horizon_steps:
            steps.append(self.cfg.horizon_steps)
        steps.append(None)
        return steps

    def _lane_sequence_for_candidate(self, merge_step: Optional[int]) -> np.ndarray:
        lane = np.full(self.cfg.horizon_steps + 1, CURRENT_LANE, dtype=int)
        if merge_step is not None:
            lane[int(merge_step) :] = TARGET_LANE
        return lane

    def _upper_level_cost(self, merge_step: Optional[int]) -> float:
        if merge_step is None:
            lane_cost_time = self.cfg.no_merge_penalty_s
        else:
            lane_cost_time = merge_step * self.cfg.dt
        return self.cfg.w_lane_change_time * lane_cost_time
