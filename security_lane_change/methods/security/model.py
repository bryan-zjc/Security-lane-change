from security_lane_change.accelerated_planner import AcceleratedSecurityLaneChangePlanner, FASTSLCPlanner
from security_lane_change.config import PlannerConfig
from security_lane_change.planner import SecurityLaneChangePlanner


METHOD_LABEL = "security"
METHOD_SLUG = "security"


def create_planner(cfg: PlannerConfig, algorithm_mode: str):
    if algorithm_mode == "FAST-SLC":
        return FASTSLCPlanner(cfg, leader_policy="security")
    if algorithm_mode == "accelerate_algorithm":
        return AcceleratedSecurityLaneChangePlanner(cfg, leader_policy="security")
    if algorithm_mode == "gurobi_solve":
        return SecurityLaneChangePlanner(cfg, leader_policy="security")
    raise ValueError(f"Unknown algorithm mode for security model: {algorithm_mode}")
