from dataclasses import dataclass
from typing import List, Optional

import numpy as np


CURRENT_LANE = 0
TARGET_LANE = 1


@dataclass(frozen=True)
class VehicleState:
    x: float
    v: float
    a: float = 0.0


@dataclass
class VehicleTrajectory:
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.v = np.asarray(self.v, dtype=float)
        self.a = np.asarray(self.a, dtype=float)
        if not (len(self.x) == len(self.v) == len(self.a)):
            raise ValueError("x, v and a must have the same length.")

    def state_at(self, index: int) -> VehicleState:
        idx = max(0, min(int(index), len(self.x) - 1))
        return VehicleState(
            x=float(self.x[idx]),
            v=float(self.v[idx]),
            a=float(self.a[idx]),
        )


@dataclass
class PredictionPacket:
    lc: VehicleTrajectory
    lt: VehicleTrajectory
    ft: VehicleTrajectory


@dataclass
class TrajectoryPlan:
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    lane: np.ndarray
    cost: float
    feasible: bool
    merge_step: Optional[int] = None
    solver_status: str = ""
    solve_time: float = 0.0


@dataclass
class SolveRecord:
    simulation_step: int
    simulation_time: float
    candidate_merge_step: Optional[int]
    candidate_merge_time: Optional[float]
    status: str
    feasible: bool
    objective_value: float
    solve_time: float


@dataclass
class DecisionSolveRecord:
    simulation_step: int
    simulation_time: float
    selected_merge_step: Optional[int]
    selected_merge_time: Optional[float]
    selected_status: str
    selected_feasible: bool
    selected_objective_value: float
    candidate_problem_count: int
    feasible_candidate_count: int
    gurobi_subproblem_time_sum: float
    total_planning_time: float


@dataclass
class SimulationResult:
    time: np.ndarray
    ego_x: np.ndarray
    ego_v: np.ndarray
    ego_a: np.ndarray
    ego_lane: np.ndarray
    lc_x: np.ndarray
    lc_v: np.ndarray
    lt_x: np.ndarray
    lt_v: np.ndarray
    ft_x: np.ndarray
    ft_v: np.ndarray
    front_gap: np.ndarray
    required_front_gap: np.ndarray
    front_safety_margin: np.ndarray
    ttc: np.ndarray
    ttc_front_vehicle: List[str]
    merge_time: Optional[float]
    collision_time: Optional[float]
    collision_index: Optional[int]
    collision_front_vehicle: Optional[str]
    collision_threshold: float
    first_attacked_prediction: VehicleTrajectory
    attacked_vehicle: str
    prediction_label: str = "attacked prediction at t=0"
