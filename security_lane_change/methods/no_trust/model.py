from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.models import (
    CURRENT_LANE,
    TARGET_LANE,
    PredictionPacket,
    TrajectoryPlan,
    VehicleState,
    VehicleTrajectory,
)


METHOD_LABEL = "no-trust"
METHOD_SLUG = "no_trust"
MOVING_LEADER_SPEED_EPS = 0.5
CREEP_GAP_BUFFER_M = 1.0
CREEP_SPEED_RATIO = 0.65
CREEP_MIN_SPEED_MPS = 0.2


@dataclass
class _NeighborStates:
    lc: VehicleState
    lt: VehicleState
    ft: VehicleState


class NoTrustIDMMOBILPlanner:
    """No-trust baseline based on delayed perception, IDM, and MOBIL."""

    algorithm_label = "IDM-MOBIL"

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self.solve_records = []
        self._mobil_confirmation_time = 0.0

    def build_planning_packet(self, scenario, step: int) -> PredictionPacket:
        detected_step = max(0, int(step) - self.cfg.detection_delay_steps)
        horizon_steps = self.cfg.horizon_steps
        return PredictionPacket(
            lc=self._constant_state_trajectory(scenario.true_lc.state_at(detected_step), horizon_steps),
            lt=self._constant_state_trajectory(scenario.true_lt.state_at(detected_step), horizon_steps),
            ft=self._constant_state_trajectory(scenario.true_ft.state_at(detected_step), horizon_steps),
        )

    def plan(
        self,
        ego_state: VehicleState,
        current_lane: int,
        previous_accel: float,
        packet: PredictionPacket,
        simulation_step: int = 0,
        simulation_time: float = 0.0,
    ) -> TrajectoryPlan:
        neighbors = _NeighborStates(
            lc=packet.lc.state_at(0),
            lt=packet.lt.state_at(0),
            ft=packet.ft.state_at(0),
        )

        if current_lane == TARGET_LANE:
            accel = self._idm_accel(ego_state, neighbors.lt)
            leader = neighbors.lt
            next_lane = TARGET_LANE
        else:
            accel_current = self._idm_accel(ego_state, neighbors.lc)
            accel_target = self._idm_accel(ego_state, neighbors.lt)
            should_change = self._mobil_decision(
                ego_state=ego_state,
                accel_current=accel_current,
                accel_target=accel_target,
                neighbors=neighbors,
            )
            should_change = self._confirm_mobil_decision(should_change)
            next_lane = TARGET_LANE if should_change else CURRENT_LANE
            accel = accel_target if should_change else accel_current
            leader = neighbors.lt if should_change else neighbors.lc

        accel = self._clip_accel(accel, previous_accel, ego_state, leader)
        v_next = max(self.cfg.limits.v_min, min(self.cfg.limits.v_max, ego_state.v + accel * self.cfg.dt))
        x_next = ego_state.x + ego_state.v * self.cfg.dt + 0.5 * accel * self.cfg.dt**2

        return TrajectoryPlan(
            x=np.asarray([ego_state.x, x_next], dtype=float),
            v=np.asarray([ego_state.v, v_next], dtype=float),
            a=np.asarray([accel], dtype=float),
            lane=np.asarray([current_lane, next_lane], dtype=int),
            cost=0.0,
            feasible=True,
            merge_step=1 if current_lane == CURRENT_LANE and next_lane == TARGET_LANE else None,
            solver_status="IDM_MOBIL",
            solve_time=0.0,
        )

    def _mobil_decision(
        self,
        ego_state: VehicleState,
        accel_current: float,
        accel_target: float,
        neighbors: _NeighborStates,
    ) -> bool:
        ft_before = self._idm_accel(neighbors.ft, neighbors.lt)
        ft_after = self._idm_accel(neighbors.ft, ego_state)
        safety_ok = ft_after >= self.cfg.mobil_safe_decel
        incentive = (accel_target - accel_current) + self.cfg.mobil_politeness * (ft_after - ft_before)
        incentive_ok = incentive > self.cfg.mobil_accel_threshold
        return bool(safety_ok and incentive_ok)

    def _confirm_mobil_decision(self, raw_decision: bool) -> bool:
        if not raw_decision:
            self._mobil_confirmation_time = 0.0
            return False

        required_time = max(0.0, float(self.cfg.mobil_confirmation_time_s))
        self._mobil_confirmation_time += self.cfg.dt
        return self._mobil_confirmation_time >= required_time

    def _idm_accel(self, follower: VehicleState, leader: VehicleState) -> float:
        gap = max(0.1, leader.x - follower.x)
        dv = follower.v - leader.v
        desired_gap = (
            self.cfg.idm_s0
            + follower.v * self.cfg.idm_time_headway
            + follower.v * dv / (2.0 * np.sqrt(self.cfg.limits.a_max * self.cfg.idm_comfort_decel))
        )
        desired_gap = max(self.cfg.idm_s0, desired_gap)
        accel = self.cfg.limits.a_max * (
            1.0
            - (max(0.0, follower.v) / max(0.1, self.cfg.desired_speed)) ** self.cfg.idm_delta
            - (desired_gap / gap) ** 2
        )
        return float(accel)

    def _clip_accel(
        self,
        accel: float,
        previous_accel: float,
        ego_state: VehicleState,
        leader: VehicleState,
    ) -> float:
        jerk_step = self.cfg.limits.j_max * self.cfg.dt
        lower = max(self.cfg.limits.a_min, previous_accel - jerk_step)
        upper = min(self.cfg.limits.a_max, previous_accel + jerk_step)
        creep_lower = self._moving_leader_creep_lower_bound(ego_state, leader)
        creep_active = creep_lower is not None
        if creep_lower is not None:
            lower = min(max(lower, creep_lower), upper)
        accel = float(
            np.clip(
                accel,
                lower,
                upper,
            )
        )
        if creep_active:
            min_next_speed = min(max(CREEP_MIN_SPEED_MPS, 0.25 * leader.v), leader.v)
            if ego_state.v + accel * self.cfg.dt < min_next_speed:
                accel = (min_next_speed - ego_state.v) / self.cfg.dt
        if ego_state.v + accel * self.cfg.dt < self.cfg.limits.v_min:
            accel = (self.cfg.limits.v_min - ego_state.v) / self.cfg.dt
        return float(np.clip(accel, self.cfg.limits.a_min, self.cfg.limits.a_max))

    def _moving_leader_creep_lower_bound(
        self,
        ego_state: VehicleState,
        leader: VehicleState,
    ) -> float | None:
        gap = leader.x - ego_state.x
        if leader.v <= MOVING_LEADER_SPEED_EPS:
            return None
        if gap <= self.cfg.min_static_gap + CREEP_GAP_BUFFER_M:
            return None

        speed_floor = max(CREEP_MIN_SPEED_MPS, CREEP_SPEED_RATIO * leader.v)
        speed_floor = min(speed_floor, leader.v, self.cfg.desired_speed)
        if ego_state.v >= speed_floor:
            return None
        return (speed_floor - ego_state.v) / self.cfg.dt

    def _constant_state_trajectory(self, state: VehicleState, horizon_steps: int) -> VehicleTrajectory:
        x = np.full(horizon_steps + 1, state.x, dtype=float)
        v = np.full(horizon_steps + 1, state.v, dtype=float)
        a = np.full(horizon_steps + 1, state.a, dtype=float)
        return VehicleTrajectory(x=x, v=v, a=a)


def create_planner(cfg: PlannerConfig, algorithm_mode: str):
    return NoTrustIDMMOBILPlanner(cfg)
