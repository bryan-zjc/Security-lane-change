import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.models import TARGET_LANE, PredictionPacket, VehicleState, VehicleTrajectory


class AttackedLaneChangeScenario:
    """An extreme attack case for the target-lane leader LT.

    At the beginning, LT is too close for a safe lane change. LT then
    accelerates and opens a target-lane gap. Once this gap becomes attractive,
    LT suddenly brakes to a full stop, while the transmitted trajectory still
    claims that LT will keep accelerating.
    """

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self.total_steps = cfg.simulation_steps + cfg.horizon_steps + 2
        self.scenario_label = "LT hard-brake attack"
        self.scenario_slug = "lt_hard_brake"
        self.has_network_attack = True
        self.attacked_vehicle = "LT"
        self.hard_brake_vehicle = "LT"
        self.ego_initial = VehicleState(x=0.0, v=11.5, a=0.0)
        self.lt_brake_start_s = 3.6
        self.hard_brake_start_s = self.lt_brake_start_s
        self.lt_accel_before_brake = 2.0
        self.lt_brake_accel = -9.0
        self.lt_brake_end_speed = 0.0
        self.lt_attack_accel = 5.0
        self.ft_follow_lt_when_unmerged = False
        self.ft_follow_stopped_ego_after_lt_collision = False
        self._post_collision_ego_leader: VehicleState | None = None
        self.true_lc = self._constant_speed_trajectory(x0=30.0, v0=7.0)
        self.nominal_ft = self._constant_speed_trajectory(x0=-45.0, v0=10.0)
        self.true_ft = self._copy_trajectory(self.nominal_ft)
        self.true_lt = self._lt_true_trajectory(x0=9.0, v0=10.0)

    def reset_dynamic_trajectories(self) -> None:
        self.true_ft = self._copy_trajectory(self.nominal_ft)
        self._post_collision_ego_leader = None

    def build_prediction_packet(self, start_step: int, horizon_steps: int) -> PredictionPacket:
        lc = self._slice_with_padding(self.true_lc, start_step, horizon_steps)
        ft = self._slice_with_padding(self.true_ft, start_step, horizon_steps)
        lt = self._attacked_lt_prediction(start_step, horizon_steps)
        return PredictionPacket(lc=lc, lt=lt, ft=ft)

    def update_ft_after_ego_step(
        self,
        step: int,
        ego_state: VehicleState,
        ego_lane: int,
    ) -> None:
        next_step = step + 1
        if next_step >= len(self.true_ft.x):
            return

        ft_state = self.true_ft.state_at(step)
        leader = self._ft_leader(step=step, ego_state=ego_state, ego_lane=ego_lane, ft_state=ft_state)
        if leader is None:
            self._copy_nominal_ft_step(next_step)
            return

        if leader.x > ft_state.x:
            accel = self._idm_accel(follower=ft_state, leader=leader)
        else:
            accel = self.cfg.limits.a_min
        self._propagate_ft_step(step, accel)

    def propagate_after_collision(
        self,
        collision_index: int | None,
        collision_front_vehicle: str | None,
        ego_x: np.ndarray,
        ego_v: np.ndarray,
        ego_a: np.ndarray,
        ego_lane: np.ndarray,
    ) -> None:
        if not bool(getattr(self, "ft_follow_stopped_ego_after_lt_collision", False)):
            return
        if collision_index is None or collision_front_vehicle != "LT":
            return
        if not np.isfinite(ego_x[collision_index]):
            return

        self._post_collision_ego_leader = VehicleState(
            x=float(ego_x[collision_index]),
            v=0.0,
            a=0.0,
        )
        end_step = min(self.cfg.simulation_steps, len(self.true_ft.x) - 1)
        for step in range(int(collision_index), end_step):
            next_step = step + 1
            ego_x[next_step] = self._post_collision_ego_leader.x
            ego_v[next_step] = 0.0
            ego_a[step] = 0.0
            ego_lane[next_step] = TARGET_LANE
            self.update_ft_after_ego_step(step, self._post_collision_ego_leader, TARGET_LANE)
        ego_a[end_step] = 0.0

    def _ft_leader(
        self,
        step: int,
        ego_state: VehicleState,
        ego_lane: int,
        ft_state: VehicleState,
    ) -> VehicleState | None:
        if self._post_collision_ego_leader is not None:
            return self._post_collision_ego_leader
        if ego_lane == TARGET_LANE and ego_state.x > ft_state.x:
            return ego_state
        if bool(getattr(self, "ft_follow_lt_when_unmerged", False)):
            lt_state = self.true_lt.state_at(step)
            if lt_state.x > ft_state.x:
                return lt_state
        return None

    def _constant_speed_trajectory(self, x0: float, v0: float) -> VehicleTrajectory:
        dt = self.cfg.dt
        steps = self.total_steps
        x = x0 + np.arange(steps + 1, dtype=float) * dt * v0
        v = np.full(steps + 1, v0, dtype=float)
        a = np.zeros(steps + 1, dtype=float)
        return VehicleTrajectory(x=x, v=v, a=a)

    def _copy_trajectory(self, trajectory: VehicleTrajectory) -> VehicleTrajectory:
        return VehicleTrajectory(
            x=trajectory.x.copy(),
            v=trajectory.v.copy(),
            a=trajectory.a.copy(),
        )

    def _copy_nominal_ft_step(self, step: int) -> None:
        self.true_ft.x[step] = self.nominal_ft.x[step]
        self.true_ft.v[step] = self.nominal_ft.v[step]
        self.true_ft.a[step] = self.nominal_ft.a[step]

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
        return self._clip_ft_accel(float(accel), follower)

    def _clip_ft_accel(self, accel: float, ft_state: VehicleState) -> float:
        accel = float(np.clip(accel, self.cfg.limits.a_min, self.cfg.limits.a_max))
        if ft_state.v + accel * self.cfg.dt < self.cfg.limits.v_min:
            accel = (self.cfg.limits.v_min - ft_state.v) / self.cfg.dt
        return float(np.clip(accel, self.cfg.limits.a_min, self.cfg.limits.a_max))

    def _propagate_ft_step(self, step: int, accel: float) -> None:
        dt = self.cfg.dt
        ft_state = self.true_ft.state_at(step)
        if ft_state.v + accel * dt < self.cfg.limits.v_min:
            accel = (self.cfg.limits.v_min - ft_state.v) / dt
        next_v = max(self.cfg.limits.v_min, min(self.cfg.limits.v_max, ft_state.v + accel * dt))
        next_x = ft_state.x + ft_state.v * dt + 0.5 * accel * dt * dt
        self.true_ft.x[step + 1] = next_x
        self.true_ft.v[step + 1] = next_v
        self.true_ft.a[step] = accel
        self.true_ft.a[step + 1] = accel

    def _lt_true_trajectory(self, x0: float, v0: float) -> VehicleTrajectory:
        dt = self.cfg.dt
        steps = self.total_steps
        x = np.zeros(steps + 1)
        v = np.zeros(steps + 1)
        a = np.zeros(steps + 1)
        x[0] = x0
        v[0] = v0

        for k in range(steps):
            t = k * dt
            if t >= self.lt_brake_start_s and v[k] > self.lt_brake_end_speed:
                a[k] = self.lt_brake_accel
            elif t < self.lt_brake_start_s:
                a[k] = self.lt_accel_before_brake
            else:
                a[k] = 0.0
            if a[k] < 0.0 and v[k] + a[k] * dt < self.lt_brake_end_speed:
                a[k] = (self.lt_brake_end_speed - v[k]) / dt
            if a[k] < 0.0:
                v_next = max(self.lt_brake_end_speed, v[k] + a[k] * dt)
            elif a[k] > 0.0:
                v_next = min(self.cfg.limits.v_max, v[k] + a[k] * dt)
            else:
                v_next = v[k]
            x[k + 1] = x[k] + v[k] * dt + 0.5 * a[k] * dt * dt
            v[k + 1] = v_next

        a[-1] = a[-2]
        return VehicleTrajectory(x=x, v=v, a=a)

    def _attacked_lt_prediction(self, start_step: int, horizon_steps: int) -> VehicleTrajectory:
        current = self.true_lt.state_at(start_step)
        dt = self.cfg.dt
        x = np.zeros(horizon_steps + 1)
        v = np.zeros(horizon_steps + 1)
        a = np.full(horizon_steps + 1, self.lt_attack_accel)
        x[0] = current.x
        v[0] = current.v

        for k in range(horizon_steps):
            v[k + 1] = min(self.cfg.limits.v_max, v[k] + self.lt_attack_accel * dt)
            x[k + 1] = x[k] + v[k] * dt + 0.5 * self.lt_attack_accel * dt * dt

        a[-1] = a[-2]
        return VehicleTrajectory(x=x, v=v, a=a)

    def _slice_with_padding(
        self,
        trajectory: VehicleTrajectory,
        start_step: int,
        horizon_steps: int,
    ) -> VehicleTrajectory:
        end = start_step + horizon_steps
        idx = np.arange(start_step, end + 1)
        idx = np.clip(idx, 0, len(trajectory.x) - 1)
        return VehicleTrajectory(
            x=trajectory.x[idx],
            v=trajectory.v[idx],
            a=trajectory.a[idx],
        )


class LCHardBrakeScenario(AttackedLaneChangeScenario):
    """LC suddenly brakes while the target lane remains blocked by a close LT."""

    def __init__(self, cfg: PlannerConfig) -> None:
        self.cfg = cfg
        self.total_steps = cfg.simulation_steps + cfg.horizon_steps + 2
        self.scenario_label = "LC hard-brake attack"
        self.scenario_slug = "lc_hard_brake"
        self.has_network_attack = True
        self.attacked_vehicle = "LC"
        self.hard_brake_vehicle = "LC"
        self.ego_initial = VehicleState(x=0.0, v=12.0, a=0.0)
        self.lc_brake_start_s = 1.2
        self.hard_brake_start_s = self.lc_brake_start_s
        self.lt_brake_start_s = self.lc_brake_start_s
        self.lc_brake_accel = self.cfg.limits.a_min
        self.lc_attack_accel = 3.0
        self.true_lc = self._lc_true_trajectory(x0=30.0, v0=10.0)
        self.true_lt = self._constant_speed_trajectory(x0=6.0, v0=8.0)
        self.nominal_ft = self._constant_speed_trajectory(x0=-45.0, v0=10.0)
        self.true_ft = self._copy_trajectory(self.nominal_ft)

    def reset_dynamic_trajectories(self) -> None:
        self.true_ft = self._copy_trajectory(self.nominal_ft)

    def build_prediction_packet(self, start_step: int, horizon_steps: int) -> PredictionPacket:
        lc = self._attacked_lc_prediction(start_step, horizon_steps)
        lt = self._slice_with_padding(self.true_lt, start_step, horizon_steps)
        ft = self._slice_with_padding(self.true_ft, start_step, horizon_steps)
        return PredictionPacket(lc=lc, lt=lt, ft=ft)

    def _lc_true_trajectory(self, x0: float, v0: float) -> VehicleTrajectory:
        dt = self.cfg.dt
        steps = self.total_steps
        x = np.zeros(steps + 1)
        v = np.zeros(steps + 1)
        a = np.zeros(steps + 1)
        x[0] = x0
        v[0] = v0

        for k in range(steps):
            t = k * dt
            if t >= self.lc_brake_start_s and v[k] > 0.0:
                a[k] = self.lc_brake_accel
            else:
                a[k] = 0.0
            if a[k] < 0.0 and v[k] + a[k] * dt < 0.0:
                a[k] = -v[k] / dt
            v_next = max(0.0, v[k] + a[k] * dt)
            x[k + 1] = x[k] + v[k] * dt + 0.5 * a[k] * dt * dt
            v[k + 1] = v_next

        a[-1] = a[-2]
        return VehicleTrajectory(x=x, v=v, a=a)

    def _attacked_lc_prediction(self, start_step: int, horizon_steps: int) -> VehicleTrajectory:
        current = self.true_lc.state_at(start_step)
        dt = self.cfg.dt
        x = np.zeros(horizon_steps + 1)
        v = np.zeros(horizon_steps + 1)
        a = np.full(horizon_steps + 1, self.lc_attack_accel)
        x[0] = current.x
        v[0] = current.v

        for k in range(horizon_steps):
            v[k + 1] = min(self.cfg.limits.v_max, v[k] + self.lc_attack_accel * dt)
            x[k + 1] = x[k] + v[k] * dt + 0.5 * self.lc_attack_accel * dt * dt

        a[-1] = a[-2]
        return VehicleTrajectory(x=x, v=v, a=a)
