from security_lane_change.accelerated_planner import AcceleratedSecurityLaneChangePlanner, FASTSLCPlanner
from security_lane_change.config import PlannerConfig
from security_lane_change.planner import SecurityLaneChangePlanner


METHOD_LABEL = "fully-trust"
METHOD_SLUG = "fully_trust"

OPEN_LOOP_TERMINAL_MIN_SPEED_MPS = 1.0
OPEN_LOOP_TERMINAL_MIN_ACCEL_MPS2 = -0.5
NO_ATTACK_TERMINAL_MIN_SPEED_MPS = 0.2
NO_ATTACK_REAR_CLOSING_BUFFER_S = 0.0


def create_planner(cfg: PlannerConfig, algorithm_mode: str):
    if algorithm_mode == "FAST-SLC":
        return _with_open_loop_execution(FASTSLCPlanner(cfg, leader_policy="fully_trust"), cfg)
    if algorithm_mode == "accelerate_algorithm":
        return _with_open_loop_execution(
            AcceleratedSecurityLaneChangePlanner(cfg, leader_policy="fully_trust"),
            cfg,
        )
    if algorithm_mode == "gurobi_solve":
        return _with_open_loop_execution(SecurityLaneChangePlanner(cfg, leader_policy="fully_trust"), cfg)
    raise ValueError(f"Unknown algorithm mode for fully-trust model: {algorithm_mode}")


def _with_open_loop_execution(planner, cfg: PlannerConfig):
    planner.execution_steps = cfg.horizon_steps
    planner.execution_mode = "open-loop Np execution"
    planner.open_loop_only_when_attacked = True
    planner.configure_for_scenario = lambda scenario: _configure_for_scenario(planner, scenario)
    _configure_for_scenario(planner, scenario=None)
    return planner


def _configure_for_scenario(planner, scenario) -> None:
    has_attack = True if scenario is None else bool(getattr(scenario, "has_network_attack", True))
    _configure_ft_dynamics_for_scenario(scenario, enabled=has_attack)
    if has_attack:
        planner.execution_steps = planner.cfg.horizon_steps
        planner.execution_mode = "open-loop Np execution"
        planner.infeasible_accel_policy = "emergency_brake"
        _set_top_k_override(planner, None)
        _set_lower_optimizer_options(
            planner,
            terminal_min_speed=OPEN_LOOP_TERMINAL_MIN_SPEED_MPS,
            terminal_min_accel=OPEN_LOOP_TERMINAL_MIN_ACCEL_MPS2,
            rear_closing_time_buffer_s=float(planner.cfg.planning_horizon_s),
        )
        return

    planner.execution_steps = 1
    planner.execution_mode = "rolling horizon dt execution"
    planner.infeasible_accel_policy = "keep_speed"
    _set_top_k_override(planner, planner.cfg.horizon_steps + 1)
    _set_lower_optimizer_options(
        planner,
        terminal_min_speed=NO_ATTACK_TERMINAL_MIN_SPEED_MPS,
        terminal_min_accel=None,
        rear_closing_time_buffer_s=NO_ATTACK_REAR_CLOSING_BUFFER_S,
    )


def _configure_ft_dynamics_for_scenario(scenario, enabled: bool) -> None:
    if scenario is None:
        return
    is_lt_hard_brake = getattr(scenario, "scenario_slug", "") == "lt_hard_brake"
    use_dynamic_ft = bool(enabled and is_lt_hard_brake)
    if hasattr(scenario, "ft_follow_lt_when_unmerged"):
        scenario.ft_follow_lt_when_unmerged = use_dynamic_ft
    if hasattr(scenario, "ft_follow_stopped_ego_after_lt_collision"):
        scenario.ft_follow_stopped_ego_after_lt_collision = use_dynamic_ft


def _set_lower_optimizer_options(
    planner,
    terminal_min_speed: float | None,
    terminal_min_accel: float | None,
    rear_closing_time_buffer_s: float,
) -> None:
    lower_optimizer = getattr(planner, "lower_optimizer", None)
    if lower_optimizer is not None:
        _set_or_delete(lower_optimizer, "terminal_min_speed", terminal_min_speed)
        _set_or_delete(lower_optimizer, "terminal_min_accel", terminal_min_accel)
        lower_optimizer.rear_closing_time_buffer_s = float(rear_closing_time_buffer_s)

    candidate_selector = getattr(planner, "candidate_selector", None)
    if candidate_selector is not None:
        candidate_selector.rear_closing_time_buffer_s = float(rear_closing_time_buffer_s)


def _set_top_k_override(planner, value: int | None) -> None:
    if value is None:
        if hasattr(planner, "top_k_override"):
            delattr(planner, "top_k_override")
        return
    planner.top_k_override = int(value)


def _set_or_delete(obj, name: str, value) -> None:
    if value is None:
        if hasattr(obj, name):
            delattr(obj, name)
        return
    setattr(obj, name, value)
