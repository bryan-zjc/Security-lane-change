from typing import Dict, List, Optional, Tuple

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
from security_lane_change.planner import _status_name
from security_lane_change.safety import build_protective_trajectory, minimum_safe_gap


Candidate = Optional[int]


class PMCTSMergeCandidateSelector:
    """PMCTS-style top-K selector for upper-level lane-change timing."""

    def __init__(self, cfg: PlannerConfig, leader_policy: str = "security") -> None:
        self.cfg = cfg
        self.leader_policy = leader_policy

    def rank_candidates(
        self,
        ego_state: VehicleState,
        previous_accel: float,
        candidates: List[Candidate],
        upper_level_cost,
        packet: PredictionPacket,
        prior_merge_step: Candidate = None,
        use_trajectory_prior: bool = False,
    ) -> List[Candidate]:
        if not candidates:
            return []

        cheap_metrics = {
            candidate: self._cheap_rollout_metrics(
                ego_state,
                previous_accel,
                candidate,
                upper_level_cost,
                packet,
            )
            for candidate in candidates
        }
        cheap_scores = {
            candidate: score
            + self._trajectory_prior_bonus(candidate, prior_merge_step, min_margin, use_trajectory_prior)
            for candidate, (score, min_margin) in cheap_metrics.items()
        }
        visits: Dict[Candidate, int] = {candidate: 1 for candidate in candidates}
        rewards: Dict[Candidate, float] = dict(cheap_scores)

        total_visits = len(candidates)
        for _ in range(max(0, self.cfg.pmcts_iterations)):
            selected = max(
                candidates,
                key=lambda candidate: self._uct_score(
                    rewards[candidate] / visits[candidate],
                    visits[candidate],
                    total_visits,
                ),
            )
            rewards[selected] += cheap_scores[selected]
            visits[selected] += 1
            total_visits += 1

        return sorted(candidates, key=lambda candidate: rewards[candidate] / visits[candidate], reverse=True)

    def _uct_score(self, mean_reward: float, visits: int, total_visits: int) -> float:
        exploration = self.cfg.pmcts_exploration * np.sqrt(np.log(total_visits + 1.0) / visits)
        return mean_reward + exploration

    def _cheap_rollout_metrics(
        self,
        ego_state: VehicleState,
        previous_accel: float,
        merge_step: Candidate,
        upper_level_cost,
        packet: PredictionPacket,
    ) -> float:
        horizon_steps = self.cfg.horizon_steps
        lc_leader = self._leader_trajectory(packet.lc, horizon_steps)
        lt_leader = self._leader_trajectory(packet.lt, horizon_steps)
        lane_sequence = self._lane_sequence_for_candidate(merge_step)

        x = float(ego_state.x)
        v = float(ego_state.v)
        prev_a = float(previous_accel)
        min_margin = float("inf")
        total_cost = 0.0
        violation = 0.0

        for step in range(horizon_steps):
            jerk_step = self.cfg.limits.j_max * self.cfg.dt
            a_des = (self.cfg.desired_speed - v) / self.cfg.dt
            accel = float(
                np.clip(
                    a_des,
                    max(self.cfg.limits.a_min, prev_a - jerk_step),
                    min(self.cfg.limits.a_max, prev_a + jerk_step),
                )
            )
            if v + accel * self.cfg.dt < self.cfg.limits.v_min:
                accel = (self.cfg.limits.v_min - v) / self.cfg.dt
            v_next = np.clip(v + accel * self.cfg.dt, self.cfg.limits.v_min, self.cfg.limits.v_max)
            x_next = x + v * self.cfg.dt + 0.5 * accel * self.cfg.dt**2

            lane = int(lane_sequence[step + 1])
            leader = lt_leader if lane == TARGET_LANE else lc_leader
            margin = leader.x[step + 1] - x_next - minimum_safe_gap(leader.v[step + 1], v_next, self.cfg)
            min_margin = min(min_margin, float(margin))
            if margin < 0.0:
                violation += -float(margin)

            if lane == TARGET_LANE:
                rear_gap = self._rear_target_safe_gap(v_next, packet.ft.v[step + 1])
                rear_margin = x_next - packet.ft.x[step + 1] - rear_gap
                min_margin = min(min_margin, float(rear_margin))
                if rear_margin < 0.0:
                    violation += -float(rear_margin)

            total_cost += self.cfg.w_speed * (self.cfg.desired_speed - v_next) ** 2
            total_cost += self.cfg.w_accel * accel**2
            total_cost += self.cfg.w_jerk * (accel - prev_a) ** 2

            x = float(x_next)
            v = float(v_next)
            prev_a = accel

        total_cost += upper_level_cost(merge_step)
        safety_bonus = min(20.0, max(-20.0, min_margin))
        score = -total_cost - 1000.0 * violation + 2.0 * safety_bonus
        return score, min_margin

    def _trajectory_prior_bonus(
        self,
        candidate: Candidate,
        prior_merge_step: Candidate,
        min_margin: float,
        use_trajectory_prior: bool,
    ) -> float:
        if not use_trajectory_prior:
            return 0.0

        distance = self._candidate_distance(candidate, prior_merge_step)
        sigma = max(1e-6, float(self.cfg.fast_slc_prior_sigma_steps))
        proximity = np.exp(-distance / sigma)
        gate = self._safety_gate(min_margin)
        return float(self.cfg.fast_slc_prior_weight * proximity * gate)

    def _candidate_distance(self, candidate: Candidate, prior_merge_step: Candidate) -> float:
        far_distance = float(self.cfg.horizon_steps + 1)
        if candidate is None and prior_merge_step is None:
            return 0.0
        if candidate is None or prior_merge_step is None:
            return far_distance
        return abs(float(candidate) - float(prior_merge_step))

    def _safety_gate(self, min_margin: float) -> float:
        arg = float(self.cfg.fast_slc_prior_gate_gamma) * float(min_margin)
        arg = float(np.clip(arg, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-arg)))

    def _lane_sequence_for_candidate(self, merge_step: Candidate) -> np.ndarray:
        lane = np.full(self.cfg.horizon_steps + 1, CURRENT_LANE, dtype=int)
        if merge_step is not None:
            lane[int(merge_step) :] = TARGET_LANE
        return lane

    def _leader_trajectory(self, received_trajectory, horizon_steps: int):
        if self.leader_policy == "security":
            return build_protective_trajectory(received_trajectory, horizon_steps, self.cfg)
        if self.leader_policy == "fully_trust":
            return received_trajectory
        raise ValueError(f"Unknown leader policy: {self.leader_policy}")

    def _rear_target_safe_gap(self, ego_speed_as_leader: float, ft_speed: float) -> float:
        base_gap = minimum_safe_gap(float(ego_speed_as_leader), float(ft_speed), self.cfg)
        buffer_time = float(getattr(self, "rear_closing_time_buffer_s", 0.0))
        if buffer_time <= 0.0:
            return base_gap
        closing_buffer = max(0.0, float(ft_speed) - float(ego_speed_as_leader)) * buffer_time
        return base_gap + closing_buffer


class AffineDMinEnvelope:
    """Conservative affine upper envelope for speed-dependent d_min."""

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self._speed_samples = np.linspace(
            cfg.limits.v_min,
            cfg.limits.v_max,
            cfg.dmin_envelope_grid_size,
        )
        self._slope_candidates = np.linspace(-2.0, 8.0, cfg.dmin_envelope_slope_count)
        self._cache: Dict[Tuple[str, float, float, float], Tuple[float, float]] = {}

    def front_bound(self, leader_speed: float, speed_low: float, speed_high: float) -> Tuple[float, float]:
        key = ("front", round(float(leader_speed), 4), round(float(speed_low), 4), round(float(speed_high), 4))
        if key not in self._cache:
            speed_samples = self._interval_samples(speed_low, speed_high)
            values = np.asarray(
                [minimum_safe_gap(float(leader_speed), float(speed), self.cfg) for speed in speed_samples],
                dtype=float,
            )
            self._cache[key] = self._best_affine_upper_bound(speed_samples, values)
        return self._cache[key]

    def rear_bound(
        self,
        follower_speed: float,
        speed_low: float,
        speed_high: float,
        closing_buffer_time_s: float = 0.0,
    ) -> Tuple[float, float]:
        key = (
            "rear",
            round(float(follower_speed), 4),
            round(float(speed_low), 4),
            round(float(speed_high), 4),
            round(float(closing_buffer_time_s), 4),
        )
        if key not in self._cache:
            speed_samples = self._interval_samples(speed_low, speed_high)
            values = np.asarray(
                [
                    self._rear_target_safe_gap(
                        ego_speed_as_leader=float(speed),
                        ft_speed=float(follower_speed),
                        closing_buffer_time_s=closing_buffer_time_s,
                    )
                    for speed in speed_samples
                ],
                dtype=float,
            )
            self._cache[key] = self._best_affine_upper_bound(speed_samples, values)
        return self._cache[key]

    def _rear_target_safe_gap(
        self,
        ego_speed_as_leader: float,
        ft_speed: float,
        closing_buffer_time_s: float,
    ) -> float:
        base_gap = minimum_safe_gap(float(ego_speed_as_leader), float(ft_speed), self.cfg)
        if closing_buffer_time_s <= 0.0:
            return base_gap
        closing_buffer = max(0.0, float(ft_speed) - float(ego_speed_as_leader)) * float(closing_buffer_time_s)
        return base_gap + closing_buffer

    def _interval_samples(self, speed_low: float, speed_high: float) -> np.ndarray:
        low = max(self.cfg.limits.v_min, min(float(speed_low), float(speed_high)))
        high = min(self.cfg.limits.v_max, max(float(speed_low), float(speed_high)))
        if abs(high - low) < 1e-9:
            return np.asarray([low], dtype=float)
        return np.linspace(low, high, self.cfg.dmin_envelope_grid_size)

    def _best_affine_upper_bound(self, speed_samples: np.ndarray, values: np.ndarray) -> Tuple[float, float]:
        best_alpha = 0.0
        best_beta = float(np.max(values)) + 0.05
        best_mean_over = float("inf")

        for alpha in self._slope_candidates:
            beta = float(np.max(values - alpha * speed_samples)) + 0.05
            upper = alpha * speed_samples + beta
            mean_over = float(np.mean(upper - values))
            if mean_over < best_mean_over:
                best_alpha = float(alpha)
                best_beta = beta
                best_mean_over = mean_over

        return best_alpha, best_beta


class AcceleratedLongitudinalTrajectoryOptimizer:
    """Gurobi QP with d_min affine envelopes and warm starts."""

    def __init__(self, cfg: PlannerConfig, leader_policy: str = "security") -> None:
        self.cfg = cfg
        self.leader_policy = leader_policy
        self.envelope = AffineDMinEnvelope(cfg)

    def optimize(
        self,
        ego_state: VehicleState,
        previous_accel: float,
        lane_sequence: np.ndarray,
        packet: PredictionPacket,
        warm_start: Optional[TrajectoryPlan] = None,
    ) -> TrajectoryPlan:
        horizon_steps = len(lane_sequence) - 1
        lc_leader = self._leader_trajectory(packet.lc, horizon_steps)
        lt_leader = self._leader_trajectory(packet.lt, horizon_steps)

        model = gp.Model("accelerated_secure_lane_change_lower")
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

        self._add_front_constraints(model, x, v, lane_sequence, lc_leader, lt_leader, ego_state.v)
        self._add_rear_target_constraints(model, x, v, lane_sequence, packet, ego_state.v)
        self._add_optional_terminal_constraints(model, v, a, horizon_steps)
        self._apply_warm_start(x, v, a, ego_state, previous_accel, warm_start)

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
        initial_speed: float,
    ) -> None:
        for step in range(1, len(lane_sequence)):
            leader = lt_protect if int(lane_sequence[step]) == TARGET_LANE else lc_protect
            speed_low, speed_high = self._reachable_speed_interval(initial_speed, step)
            alpha, beta = self.envelope.front_bound(float(leader.v[step]), speed_low, speed_high)
            model.addConstr(
                x[step] + alpha * v[step] <= float(leader.x[step]) - beta,
                name=f"front_safety_affine_{step}",
            )

    def _add_rear_target_constraints(
        self,
        model: gp.Model,
        x,
        v,
        lane_sequence: np.ndarray,
        packet: PredictionPacket,
        initial_speed: float,
    ) -> None:
        for step in range(1, len(lane_sequence)):
            if int(lane_sequence[step]) != TARGET_LANE:
                continue
            speed_low, speed_high = self._reachable_speed_interval(initial_speed, step)
            alpha, beta = self.envelope.rear_bound(
                float(packet.ft.v[step]),
                speed_low,
                speed_high,
                closing_buffer_time_s=float(getattr(self, "rear_closing_time_buffer_s", 0.0)),
            )
            model.addConstr(
                x[step] - alpha * v[step] >= float(packet.ft.x[step]) + beta,
                name=f"rear_safety_affine_target_{step}",
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

    def _reachable_speed_interval(self, initial_speed: float, step: int) -> Tuple[float, float]:
        horizon_time = max(0, int(step)) * self.cfg.dt
        speed_low = float(initial_speed) + self.cfg.limits.a_min * horizon_time
        speed_high = float(initial_speed) + self.cfg.limits.a_max * horizon_time
        return (
            max(self.cfg.limits.v_min, speed_low),
            min(self.cfg.limits.v_max, speed_high),
        )

    def _apply_warm_start(
        self,
        x,
        v,
        a,
        ego_state: VehicleState,
        previous_accel: float,
        warm_start: Optional[TrajectoryPlan],
    ) -> None:
        horizon_steps = len(a)
        if warm_start is not None and warm_start.feasible and len(warm_start.x) >= 2:
            for step in range(horizon_steps + 1):
                src = min(step + 1, len(warm_start.x) - 1)
                x[step].Start = float(warm_start.x[src])
                v[step].Start = float(warm_start.v[src])
            for step in range(horizon_steps):
                src = min(step + 1, len(warm_start.a) - 1)
                a[step].Start = float(warm_start.a[src])
            x[0].Start = float(ego_state.x)
            v[0].Start = float(ego_state.v)
            return

        x_start = float(ego_state.x)
        v_start = float(ego_state.v)
        a_start = float(previous_accel)
        x[0].Start = x_start
        v[0].Start = v_start
        for step in range(horizon_steps):
            jerk_step = self.cfg.limits.j_max * self.cfg.dt
            desired_accel = (self.cfg.desired_speed - v_start) / self.cfg.dt
            a_start = float(
                np.clip(
                    desired_accel,
                    max(self.cfg.limits.a_min, a_start - jerk_step),
                    min(self.cfg.limits.a_max, a_start + jerk_step),
                )
            )
            if v_start + a_start * self.cfg.dt < self.cfg.limits.v_min:
                a_start = (self.cfg.limits.v_min - v_start) / self.cfg.dt
            x_start = x_start + v_start * self.cfg.dt + 0.5 * a_start * self.cfg.dt**2
            v_start = float(np.clip(v_start + a_start * self.cfg.dt, self.cfg.limits.v_min, self.cfg.limits.v_max))
            a[step].Start = a_start
            x[step + 1].Start = x_start
            v[step + 1].Start = v_start

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


class AcceleratedSecurityLaneChangePlanner:
    """PMCTS candidate selection + Gurobi warm start + affine d_min envelope."""

    algorithm_label = "accelerate algorithm"

    def __init__(
        self,
        cfg: PlannerConfig,
        use_trajectory_prior: bool = False,
        leader_policy: str = "security",
    ) -> None:
        self.cfg = cfg
        self.use_trajectory_prior = use_trajectory_prior
        self.leader_policy = leader_policy
        self.lower_optimizer = AcceleratedLongitudinalTrajectoryOptimizer(cfg, leader_policy=leader_policy)
        self.candidate_selector = PMCTSMergeCandidateSelector(cfg, leader_policy=leader_policy)
        self.solve_records: List[SolveRecord] = []
        self._warm_start_plan: Optional[TrajectoryPlan] = None
        self._previous_selected_merge_step: Candidate = None

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
            plan = self.lower_optimizer.optimize(
                ego_state,
                previous_accel,
                lane_sequence,
                packet,
                warm_start=self._warm_start_plan,
            )
            plan.merge_step = 0
            self._record_solve(simulation_step, simulation_time, 0, plan)
            if plan.feasible:
                self._warm_start_plan = plan
            return plan

        all_candidates = self._candidate_merge_steps()
        ranked_candidates = self.candidate_selector.rank_candidates(
            ego_state,
            previous_accel,
            all_candidates,
            self._upper_level_cost,
            packet,
            prior_merge_step=self._shifted_prior_merge_step(),
            use_trajectory_prior=self.use_trajectory_prior,
        )
        top_k = int(getattr(self, "top_k_override", self.cfg.accelerated_top_k))
        selected_candidates = ranked_candidates[: max(1, min(len(ranked_candidates), top_k))]

        best_plan = self._solve_candidate_set(
            selected_candidates,
            ego_state,
            previous_accel,
            packet,
            simulation_step,
            simulation_time,
        )

        if best_plan is None:
            fallback_candidates = ranked_candidates[len(selected_candidates) :]
            best_plan = self._solve_candidate_set(
                fallback_candidates,
                ego_state,
                previous_accel,
                packet,
                simulation_step,
                simulation_time,
                stop_at_first_feasible=True,
            )

        if best_plan is None:
            stay_current = np.full(self.cfg.horizon_steps + 1, CURRENT_LANE, dtype=int)
            return self.lower_optimizer._infeasible_plan(
                ego_state,
                previous_accel,
                stay_current,
                status="NO_FEASIBLE_CANDIDATE",
            )

        self._warm_start_plan = best_plan
        self._previous_selected_merge_step = best_plan.merge_step
        return best_plan

    def _shifted_prior_merge_step(self) -> Candidate:
        if self._previous_selected_merge_step is None:
            return None
        shifted = int(self._previous_selected_merge_step) - 1
        if shifted <= 0:
            return 1
        return min(shifted, self.cfg.horizon_steps)

    def _solve_candidate_set(
        self,
        candidates: List[Candidate],
        ego_state: VehicleState,
        previous_accel: float,
        packet: PredictionPacket,
        simulation_step: int,
        simulation_time: float,
        stop_at_first_feasible: bool = False,
    ) -> Optional[TrajectoryPlan]:
        best_plan: Optional[TrajectoryPlan] = None
        for merge_step in candidates:
            lane_sequence = self._lane_sequence_for_candidate(merge_step)
            plan = self.lower_optimizer.optimize(
                ego_state,
                previous_accel,
                lane_sequence,
                packet,
                warm_start=self._warm_start_plan,
            )
            plan.merge_step = merge_step
            self._record_solve(simulation_step, simulation_time, merge_step, plan)
            if not plan.feasible:
                continue
            plan.cost += self._upper_level_cost(merge_step)
            if best_plan is None or plan.cost < best_plan.cost:
                best_plan = plan
            if stop_at_first_feasible:
                break
        return best_plan

    def _record_solve(
        self,
        simulation_step: int,
        simulation_time: float,
        merge_step: Candidate,
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

    def _candidate_merge_steps(self) -> List[Candidate]:
        steps: List[Candidate] = list(
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

    def _lane_sequence_for_candidate(self, merge_step: Candidate) -> np.ndarray:
        lane = np.full(self.cfg.horizon_steps + 1, CURRENT_LANE, dtype=int)
        if merge_step is not None:
            lane[int(merge_step) :] = TARGET_LANE
        return lane

    def _upper_level_cost(self, merge_step: Candidate) -> float:
        if merge_step is None:
            lane_cost_time = self.cfg.no_merge_penalty_s
        else:
            lane_cost_time = merge_step * self.cfg.dt
        return self.cfg.w_lane_change_time * lane_cost_time


class FASTSLCPlanner(AcceleratedSecurityLaneChangePlanner):
    """FAST-SLC with rolling-horizon trajectory-prior PMCTS."""

    algorithm_label = "FAST-SLC"

    def __init__(self, cfg: PlannerConfig, leader_policy: str = "security") -> None:
        super().__init__(cfg, use_trajectory_prior=True, leader_policy=leader_policy)
