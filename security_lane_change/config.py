import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleLimits:
    a_min: float = -4.5
    a_max: float = 2.0
    v_min: float = 0.0
    v_max: float = 22.0
    j_max: float = 6.0


@dataclass(frozen=True)
class PlannerConfig:
    dt: float = 0.1
    planning_horizon_s: float = 2.5
    simulation_time_s: float = 12.0
    communication_delay_s: float = 0.0
    detection_delay_s: float = 0.5
    desired_speed: float = 15.0
    min_static_gap: float = 4.5
    safety_buffer: float = 1.0
    w_speed: float = 1.0
    w_accel: float = 0.35
    w_jerk: float = 0.08
    w_lane_change_time: float = 2.0
    no_merge_penalty_s: float = 40.0
    accel_grid_size: int = 13
    beam_width: int = 24
    candidate_stride_steps: int = 1
    accelerated_top_k: int = 5
    pmcts_iterations: int = 32
    pmcts_exploration: float = 0.6
    fast_slc_prior_weight: float = 60.0
    fast_slc_prior_sigma_steps: float = 3.0
    fast_slc_prior_gate_gamma: float = 0.6
    dmin_envelope_grid_size: int = 121
    dmin_envelope_slope_count: int = 101
    idm_delta: float = 4.0
    idm_s0: float = 10.0
    idm_time_headway: float = 3.0
    idm_comfort_decel: float = 2.5
    mobil_politeness: float = 0.4
    mobil_accel_threshold: float = 1.05
    mobil_confirmation_time_s: float = 0.4
    mobil_safe_decel: float = -3.0
    limits: VehicleLimits = VehicleLimits()

    @property
    def horizon_steps(self) -> int:
        return int(math.ceil(self.planning_horizon_s / self.dt))

    @property
    def simulation_steps(self) -> int:
        return int(round(self.simulation_time_s / self.dt))

    @property
    def detection_delay_steps(self) -> int:
        return int(round(self.detection_delay_s / self.dt))

    @property
    def communication_delay_steps(self) -> int:
        return int(round(self.communication_delay_s / self.dt))
