import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.models import VehicleState, VehicleTrajectory


def propagate_state(state: VehicleState, accel: float, dt: float) -> VehicleState:
    if state.v + accel * dt < 0.0:
        accel = -state.v / dt
    v_next = max(0.0, state.v + accel * dt)
    x_next = state.x + state.v * dt + 0.5 * accel * dt * dt
    return VehicleState(x=x_next, v=v_next, a=accel)


def minimum_safe_gap(leader_speed: float, follower_speed: float, cfg: PlannerConfig) -> float:
    """Speed-dependent d_min from the paper, with a static vehicle-length floor."""
    v_l = max(0.0, float(leader_speed))
    v = max(0.0, float(follower_speed))
    h = cfg.detection_delay_s
    a_max = cfg.limits.a_max
    a_min = cfg.limits.a_min

    if v_l + h * a_max <= v:
        gap = h * v - (v * v) / (2.0 * a_min)
    elif v_l <= v <= v_l + h * a_max:
        gap = (
            h * v
            - ((v_l - v) ** 2) / (2.0 * a_max)
            + 0.5 * a_max * h * h
            - ((v_l + a_max * h) ** 2) / (2.0 * a_min)
        )
    else:
        gap = h * v + 0.5 * a_max * h * h - ((v + a_max * h) ** 2) / (2.0 * a_min)

    return max(cfg.min_static_gap, float(gap) + cfg.safety_buffer)


def front_safety_margin(
    leader_x: float,
    leader_v: float,
    follower_x: float,
    follower_v: float,
    cfg: PlannerConfig,
) -> float:
    return float(leader_x - follower_x) - minimum_safe_gap(leader_v, follower_v, cfg)


def build_protective_trajectory(
    received_trajectory: VehicleTrajectory,
    horizon_steps: int,
    cfg: PlannerConfig,
) -> VehicleTrajectory:
    """Construct the protective leader trajectory.

    The received future may be falsified. Following Eqs. (7)-(8), the safety
    bound at each prediction step is not imposed on the raw communicated
    future state. It is imposed on a one-step maximum-braking position
    propagated from the previously received state.
    """
    dt = cfg.dt
    a_brake = cfg.limits.a_min
    x = np.zeros(horizon_steps + 1)
    v = np.zeros(horizon_steps + 1)
    a = np.full(horizon_steps + 1, a_brake)
    x[0] = received_trajectory.x[0]
    v[0] = max(0.0, received_trajectory.v[0])

    for k in range(1, horizon_steps + 1):
        prev_idx = min(k - 1, len(received_trajectory.x) - 1)
        prev = VehicleState(
            x=float(received_trajectory.x[prev_idx]),
            v=float(received_trajectory.v[prev_idx]),
            a=float(received_trajectory.a[prev_idx]),
        )
        protected = propagate_state(prev, a_brake, dt)
        x[k] = protected.x
        v[k] = protected.v

    a[-1] = a[-2] if horizon_steps > 0 else a_brake
    return VehicleTrajectory(x=x, v=v, a=a)
