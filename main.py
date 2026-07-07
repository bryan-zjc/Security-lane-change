import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.methods.fully_trust import model as fully_trust_model
from security_lane_change.methods.no_trust import model as no_trust_model
from security_lane_change.methods.security import model as security_model
from security_lane_change.plotting import plot_attack_method_comparison, plot_result
from security_lane_change.safety import minimum_safe_gap
from security_lane_change.scenario import AttackedLaneChangeScenario, LCHardBrakeScenario
from security_lane_change.simulator import (
    RecedingHorizonSimulator,
    export_candidate_solve_times_csv,
    export_result_csv,
    export_solve_times_csv,
    export_ttc_csv,
    finite_mean,
    finite_min,
)


METHOD_MODE = "all"  # "security", "fully_trust", "no_trust", "all"
ALGORITHM_MODE = "FAST-SLC"  # "gurobi_solve", "accelerate_algorithm", "FAST-SLC"
SCENARIO_MODE = "lt_hard_brake"  # "lc_hard_brake", "lt_hard_brake"

ALGORITHM_LABELS = {
    "gurobi_solve": "Gurobi solve",
    "accelerate_algorithm": "accelerate algorithm",
    "FAST-SLC": "FAST-SLC",
}

METHODS = {
    "security": security_model,
    "fully_trust": fully_trust_model,
    "no_trust": no_trust_model,
}

SCENARIOS = {
    "lc_hard_brake": LCHardBrakeScenario,
    "lt_hard_brake": AttackedLaneChangeScenario,
}


def main() -> None:
    args = parse_args()
    selected_methods = list(METHODS) if args.method == "all" else [args.method]

    summaries = []
    for method_mode in selected_methods:
        summaries.append(run_one_method(method_mode, args.algorithm, args.scenario))

    if len(summaries) > 1:
        scenario_slug = summaries[0]["scenario_slug"]
        summary_target = Path("results") / "security_lane_change" / scenario_slug / "method_comparison_summary.csv"
        try:
            summary_path = export_summary_csv(summaries, summary_target)
        except PermissionError:
            summary_path = export_summary_csv(
                summaries,
                summary_target.with_name(f"{summary_target.stem}_new{summary_target.suffix}"),
            )
        print(f"Method comparison summary saved to: {summary_path}")
        comparison_paths = plot_attack_method_comparison(
            method_results=[(summary["method"], summary["_result"]) for summary in summaries],
            cfg=summaries[0]["_cfg"],
            output_dir=Path("results") / "security_lane_change" / scenario_slug / "method_comparison",
            hard_brake_vehicle=summaries[0]["hard_brake_vehicle"],
            scenario_label=summaries[0]["scenario"],
        )
        for path in comparison_paths:
            print(f"Method comparison figure saved to: {path}")


def run_one_method(method_mode: str, algorithm_mode: str, scenario_mode: str) -> dict[str, Any]:
    cfg = PlannerConfig()
    scenario = SCENARIOS[scenario_mode](cfg)
    method_module = METHODS[method_mode]
    planner = method_module.create_planner(cfg, algorithm_mode)

    scenario_label = getattr(scenario, "scenario_label", scenario_mode)
    scenario_slug = getattr(scenario, "scenario_slug", scenario_mode)
    method_label = method_module.METHOD_LABEL
    method_slug = method_module.METHOD_SLUG
    algorithm_label = getattr(planner, "algorithm_label", ALGORITHM_LABELS.get(algorithm_mode, algorithm_mode))
    algorithm_slug = algorithm_label.lower().replace(" ", "_").replace("-", "_")

    simulator = RecedingHorizonSimulator(cfg, scenario, planner)
    result = simulator.run()

    output_dir = Path("results") / "security_lane_change" / scenario_slug / method_slug / algorithm_slug
    csv_path = export_result_csv(
        result,
        output_dir / f"trajectory_{method_slug}_{algorithm_slug}.csv",
        algorithm_label=algorithm_label,
        method_label=method_label,
    )
    solve_times_path = export_solve_times_csv(
        simulator.decision_solve_records,
        output_dir / f"solve_times_{method_slug}_{algorithm_slug}.csv",
        algorithm_label=algorithm_label,
        method_label=method_label,
    )
    candidate_solve_times_path = export_candidate_solve_times_csv(
        getattr(planner, "solve_records", []),
        output_dir / f"candidate_solve_times_{method_slug}_{algorithm_slug}.csv",
        algorithm_label=algorithm_label,
        method_label=method_label,
    )
    ttc_path = export_ttc_csv(
        result,
        output_dir / f"ttc_timeseries_{method_slug}_{algorithm_slug}.csv",
        algorithm_label=algorithm_label,
        method_label=method_label,
    )
    fig_path = plot_result(
        result,
        cfg,
        output_dir / f"trajectory_{method_slug}_{algorithm_slug}.png",
        hard_brake_start_s=getattr(scenario, "hard_brake_start_s", scenario.lt_brake_start_s),
        hard_brake_vehicle=getattr(scenario, "hard_brake_vehicle", "LT"),
        algorithm_label=algorithm_label,
        method_label=method_label,
        scenario_label=scenario_label,
    )

    summary = build_summary(
        method_label=method_label,
        method_slug=method_slug,
        scenario_label=scenario_label,
        scenario_slug=scenario_slug,
        algorithm_label=algorithm_label,
        algorithm_slug=algorithm_slug,
        cfg=cfg,
        scenario=scenario,
        simulator=simulator,
        planner=planner,
        result=result,
    )

    print_summary(summary)
    print(f"CSV saved to: {csv_path}")
    print(f"Decision solve times CSV saved to: {solve_times_path}")
    print(f"Candidate solve times CSV saved to: {candidate_solve_times_path}")
    print(f"TTC CSV saved to: {ttc_path}")
    print(f"Figure saved to: {fig_path}")
    summary["_result"] = result
    summary["_cfg"] = cfg
    return summary


def build_summary(
    method_label: str,
    method_slug: str,
    scenario_label: str,
    scenario_slug: str,
    algorithm_label: str,
    algorithm_slug: str,
    cfg: PlannerConfig,
    scenario: AttackedLaneChangeScenario,
    simulator: RecedingHorizonSimulator,
    planner,
    result,
) -> dict[str, Any]:
    decision_solve_times = np.asarray(
        [record.total_planning_time for record in simulator.decision_solve_records],
        dtype=float,
    )
    candidate_solve_times = np.asarray(
        [record.solve_time for record in getattr(planner, "solve_records", [])],
        dtype=float,
    )
    valid_mask = np.isfinite(result.ego_x)
    before_mask = (result.ego_lane == 0) & valid_mask
    after_mask = (result.ego_lane == 1) & valid_mask
    hard_brake_vehicle = getattr(scenario, "hard_brake_vehicle", "LT")
    hard_brake_start_s = getattr(scenario, "hard_brake_start_s", scenario.lt_brake_start_s)
    hard_brake_traj = scenario.true_lc if hard_brake_vehicle == "LC" else scenario.true_lt
    stop_candidates = np.where(hard_brake_traj.v[: cfg.simulation_steps + 1] <= 1e-6)[0]
    hard_brake_stop_time = None if len(stop_candidates) == 0 else float(stop_candidates[0] * cfg.dt)
    collision_time = result.collision_time
    valid_indices = np.where(valid_mask)[0]
    last_valid_index = int(valid_indices[-1]) if len(valid_indices) else 0
    final_lane = "target" if int(result.ego_lane[last_valid_index]) == 1 else "current"
    initial_target_gap = scenario.true_lt.x[0] - scenario.ego_initial.x
    initial_required_gap = minimum_safe_gap(scenario.true_lt.v[0], scenario.ego_initial.v, cfg)
    after_brake_mask = (result.time >= hard_brake_start_s) & valid_mask

    execution_steps = effective_execution_steps(planner, scenario)
    return {
        "scenario": scenario_label,
        "scenario_slug": scenario_slug,
        "method": method_label,
        "method_slug": method_slug,
        "algorithm": algorithm_label,
        "algorithm_slug": algorithm_slug,
        "dt": cfg.dt,
        "planning_horizon_s": cfg.planning_horizon_s,
        "horizon_steps": cfg.horizon_steps,
        "execution_steps": execution_steps,
        "execution_interval_s": execution_steps * cfg.dt,
        "initial_target_gap": initial_target_gap,
        "initial_required_gap": initial_required_gap,
        "hard_brake_vehicle": hard_brake_vehicle,
        "hard_brake_time": hard_brake_start_s,
        "hard_brake_stop_time": hard_brake_stop_time,
        "merge_time": result.merge_time,
        "final_lane": final_lane,
        "collision_time": collision_time,
        "collision_front_vehicle": result.collision_front_vehicle,
        "collision_threshold": result.collision_threshold,
        "min_front_gap": finite_min(result.front_gap),
        "min_front_safety_margin": finite_min(result.front_safety_margin),
        "min_safety_margin_after_hard_brake": finite_min(result.front_safety_margin[after_brake_mask]),
        "mean_planning_time": float(np.mean(decision_solve_times)) if len(decision_solve_times) else float("nan"),
        "mean_candidate_gurobi_time": (
            float(np.mean(candidate_solve_times)) if len(candidate_solve_times) else float("nan")
        ),
        "ttc_mean": finite_mean(result.ttc),
        "ttc_min": finite_min(result.ttc),
        "ttc_before_lc_mean": finite_mean(result.ttc[before_mask]),
        "ttc_before_lc_min": finite_min(result.ttc[before_mask]),
        "ttc_after_lt_mean": finite_mean(result.ttc[after_mask]),
        "ttc_after_lt_min": finite_min(result.ttc[after_mask]),
    }


def effective_execution_steps(planner, scenario) -> int:
    configured_steps = int(getattr(planner, "execution_steps", 1))
    if getattr(planner, "open_loop_only_when_attacked", False):
        if not bool(getattr(scenario, "has_network_attack", True)):
            return 1
    return configured_steps


def print_summary(summary: dict[str, Any]) -> None:
    merge_text = "not merged" if summary["merge_time"] is None else f"{summary['merge_time']:.2f} s"
    merge_status = (
        "not completed"
        if summary["merge_time"] is None
        else f"completed at {summary['merge_time']:.2f} s"
    )
    stop_text = (
        "not stopped"
        if summary["hard_brake_stop_time"] is None
        else f"{summary['hard_brake_stop_time']:.2f} s"
    )
    if summary["collision_time"] is None:
        collision_text = "none"
    else:
        collision_text = (
            f"{summary['collision_time']:.2f} s "
            f"with {summary['collision_front_vehicle']}"
        )
    print("")
    print(
        "Lane-change simulation completed "
        f"({summary['scenario']} | {summary['method']} | {summary['algorithm']})."
    )
    print(f"Lane-change completion: {merge_status}")
    print(
        "Plan execution interval: "
        f"{summary['execution_interval_s']:.2f} s "
        f"({summary['execution_steps']} step(s)); horizon: {summary['planning_horizon_s']:.1f} s"
    )
    print(
        "Initial target-lane front gap: "
        f"{summary['initial_target_gap']:.2f} m; required: {summary['initial_required_gap']:.2f} m"
    )
    print(f"{summary['hard_brake_vehicle']} true hard-brake time: {summary['hard_brake_time']:.2f} s")
    print(f"{summary['hard_brake_vehicle']} stop time: {stop_text}")
    print(f"Merge time: {merge_text}; final lane: {summary['final_lane']}")
    print(f"Collision time: {collision_text}; threshold: {summary['collision_threshold']:.2f} m")
    print(f"Minimum actual front gap: {summary['min_front_gap']:.2f} m")
    print(f"Minimum front safety margin: {summary['min_front_safety_margin']:.2f} m")
    print(
        f"Minimum safety margin after {summary['hard_brake_vehicle']} hard brake: "
        f"{summary['min_safety_margin_after_hard_brake']:.2f} m"
    )
    print(f"Mean planning time per decision step: {summary['mean_planning_time']:.6f} s")
    print(f"Mean Gurobi time per candidate subproblem: {summary['mean_candidate_gurobi_time']:.6f} s")
    print(f"TTC mean/min over finite values: {summary['ttc_mean']:.3f} s / {summary['ttc_min']:.3f} s")
    print(
        "TTC before lane change with LC mean/min: "
        f"{summary['ttc_before_lc_mean']:.3f} s / {summary['ttc_before_lc_min']:.3f} s"
    )
    print(
        "TTC after lane change with LT mean/min: "
        f"{summary['ttc_after_lt_mean']:.3f} s / {summary['ttc_after_lt_min']:.3f} s"
    )


def export_summary_csv(summaries: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "method",
        "algorithm",
        "dt",
        "planning_horizon_s",
        "horizon_steps",
        "execution_steps",
        "execution_interval_s",
        "merge_time",
        "hard_brake_vehicle",
        "hard_brake_time",
        "hard_brake_stop_time",
        "collision_time",
        "collision_front_vehicle",
        "collision_threshold",
        "final_lane",
        "min_front_gap",
        "min_front_safety_margin",
        "min_safety_margin_after_hard_brake",
        "mean_planning_time",
        "mean_candidate_gurobi_time",
        "ttc_mean",
        "ttc_min",
        "ttc_before_lc_mean",
        "ttc_before_lc_min",
        "ttc_after_lt_mean",
        "ttc_after_lt_min",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary[field] for field in fields})
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Security lane-change trajectory planning demo.")
    parser.add_argument(
        "--method",
        choices=["security", "fully_trust", "no_trust", "all"],
        default=METHOD_MODE,
        help="Choose one strategy or run all three strategies.",
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHM_LABELS),
        default=ALGORITHM_MODE,
        help=(
            "Solver for security and fully-trust models. "
            "No-trust always uses IDM-MOBIL because it is a reactive baseline."
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default=SCENARIO_MODE,
        help="Choose the attack scenario.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
