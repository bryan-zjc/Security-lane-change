from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from security_lane_change.config import PlannerConfig
from security_lane_change.models import CURRENT_LANE, TARGET_LANE, SimulationResult


def plot_result(
    result: SimulationResult,
    cfg: PlannerConfig,
    output_path: Path,
    hard_brake_start_s: float,
    hard_brake_vehicle: str = "LT",
    algorithm_label: str = "",
    method_label: str = "",
    scenario_label: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_parts = [part for part in [scenario_label, method_label, algorithm_label] if part]

    ego_color = "black"
    lc_color = "tab:blue"
    lt_color = "#f2c94c"
    ft_color = "tab:green"

    position_path = output_path
    speed_path = output_path.with_name(f"{output_path.stem}_speed{output_path.suffix}")
    gap_path = output_path.with_name(f"{output_path.stem}_gap{output_path.suffix}")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    if title_parts:
        fig.suptitle(f"Security lane-change trajectory ({' | '.join(title_parts)})")

    ax.plot(result.time, result.ego_x, color=ego_color, label="Ego CAV trajectory", linewidth=2.2)
    _plot_lane_weighted(
        ax,
        result.time,
        result.lc_x,
        result.ego_lane,
        normal_lane=CURRENT_LANE,
        color=lc_color,
        label="LC true trajectory",
        linewidth=1.5,
    )
    _plot_lane_weighted(
        ax,
        result.time,
        result.lt_x,
        result.ego_lane,
        normal_lane=TARGET_LANE,
        color=lt_color,
        label="LT true trajectory",
        linewidth=1.8,
    )
    _plot_lane_weighted(
        ax,
        result.time,
        result.ft_x,
        result.ego_lane,
        normal_lane=TARGET_LANE,
        color=ft_color,
        label="FT true trajectory",
        linewidth=1.5,
    )
    if result.merge_time is not None:
        ax.axvline(
            result.merge_time,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label="Lane change time step",
        )
    if result.collision_index is not None:
        idx = result.collision_index
        ax.scatter(
            result.time[idx],
            result.ego_x[idx],
            s=42,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.8,
            label="collision",
            zorder=5,
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, position_path, title_parts)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    if title_parts:
        fig.suptitle(f"Speed profile ({' | '.join(title_parts)})")
    ax.plot(result.time, result.ego_v, color=ego_color, label="Ego CAV speed", linewidth=2.0)
    if hard_brake_vehicle == "LC":
        ax.plot(result.time, result.lc_v, color=lc_color, label="LC true speed", linewidth=1.8)
    else:
        ax.plot(result.time, result.lt_v, color=lt_color, label="LT true speed", linewidth=1.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    if result.merge_time is not None:
        ax.axvline(result.merge_time, color="black", linestyle=":", linewidth=1.4)
    if result.collision_index is not None:
        idx = result.collision_index
        ax.scatter(
            result.time[idx],
            result.ego_v[idx],
            s=42,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.8,
            label="collision",
            zorder=5,
        )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, speed_path, title_parts)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    if title_parts:
        fig.suptitle(f"Front gap ({' | '.join(title_parts)})")
    ax.plot(result.time, result.front_gap, color="#83c76d", label="Actual gap with leading vehicle", linewidth=2.0)
    ax.plot(result.time, result.required_front_gap, color="#b41f1f", label="Required $d_{\\min}$ with leading vehicle", linewidth=1.8)
    # ax.plot(result.time, result.front_safety_margin, label="safety margin", linewidth=1.5)
    if result.collision_index is not None:
        idx = result.collision_index
        ax.scatter(
            result.time[idx],
            result.front_gap[idx],
            s=42,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.8,
            zorder=5,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    if result.merge_time is not None:
        ax.axvline(result.merge_time, color="black", linestyle=":", linewidth=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("gap (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    _save_figure(fig, gap_path, title_parts)

    return position_path


def plot_attack_method_comparison(
    method_results: list[tuple[str, SimulationResult]],
    cfg: PlannerConfig,
    output_dir: Path,
    hard_brake_vehicle: str,
    scenario_label: str = "",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    position_path = output_dir / "method_comparison_position.png"
    speed_path = output_dir / "method_comparison_speed.png"
    gap_path = output_dir / "method_comparison_gap.png"

    _plot_attack_position_comparison(
        method_results=method_results,
        cfg=cfg,
        output_path=position_path,
        hard_brake_vehicle=hard_brake_vehicle,
        scenario_label=scenario_label,
    )
    _plot_attack_speed_comparison(
        method_results=method_results,
        cfg=cfg,
        output_path=speed_path,
        hard_brake_vehicle=hard_brake_vehicle,
        scenario_label=scenario_label,
    )
    _plot_attack_gap_comparison(
        method_results=method_results,
        output_path=gap_path,
        scenario_label=scenario_label,
    )
    return [position_path, speed_path, gap_path]


def _plot_attack_position_comparison(
    method_results: list[tuple[str, SimulationResult]],
    cfg: PlannerConfig,
    output_path: Path,
    hard_brake_vehicle: str,
    scenario_label: str,
) -> None:
    ego_color = "black"
    lc_color = "tab:blue"
    lt_color = "#f2c94c"
    ft_color = "tab:green"
    attack_color = lc_color if hard_brake_vehicle == "LC" else lt_color
    attack_lane = CURRENT_LANE if hard_brake_vehicle == "LC" else TARGET_LANE

    fig, axes = _comparison_axes(method_results, f"Position trajectory ({scenario_label})")
    for ax, (method_label, result) in zip(axes, method_results):
        ego_x = _truncate_after_collision(result.ego_x, result)
        ax.plot(result.time, ego_x, color=ego_color, label="Ego CAV trajectory", linewidth=2.2)
        _plot_lane_weighted(
            ax,
            result.time,
            result.lc_x,
            result.ego_lane,
            normal_lane=CURRENT_LANE,
            color=lc_color,
            label="LC true trajectory",
            linewidth=1.5,
        )
        _plot_lane_weighted(
            ax,
            result.time,
            result.lt_x,
            result.ego_lane,
            normal_lane=TARGET_LANE,
            color=lt_color,
            label="LT true trajectory",
            linewidth=1.8,
        )
        _plot_lane_weighted(
            ax,
            result.time,
            result.ft_x,
            result.ego_lane,
            normal_lane=TARGET_LANE,
            color=ft_color,
            label="FT true trajectory",
            linewidth=1.5,
        )
        pred_t, pred_lane = _initial_prediction_time_and_lane(result, cfg)
        _plot_lane_weighted(
            ax,
            pred_t,
            result.first_attacked_prediction.x[: len(pred_t)],
            pred_lane,
            normal_lane=attack_lane,
            color=attack_color,
            label=f"Attacked {hard_brake_vehicle} trajectory",
            linewidth=1.5,
            linestyle="--",
        )
        if result.collision_index is not None:
            idx = result.collision_index
            ax.scatter(
                result.time[idx],
                result.ego_x[idx],
                s=42,
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.8,
                label="collision",
                zorder=5,
            )
        _decorate_attack_axis(ax, result, method_label, ylabel="position (m)")
    _save_comparison_figure(fig, axes, output_path)


def _plot_attack_speed_comparison(
    method_results: list[tuple[str, SimulationResult]],
    cfg: PlannerConfig,
    output_path: Path,
    hard_brake_vehicle: str,
    scenario_label: str,
) -> None:
    ego_color = "black"
    lc_color = "tab:blue"
    lt_color = "#f2c94c"
    attack_color = lc_color if hard_brake_vehicle == "LC" else lt_color
    attack_lane = CURRENT_LANE if hard_brake_vehicle == "LC" else TARGET_LANE

    fig, axes = _comparison_axes(method_results, f"Speed profile ({scenario_label})")
    for ax, (method_label, result) in zip(axes, method_results):
        ego_v = _truncate_after_collision(result.ego_v, result)
        ax.plot(result.time, ego_v, color=ego_color, label="Ego CAV speed", linewidth=2.0)
        if hard_brake_vehicle == "LC":
            ax.plot(result.time, result.lc_v, color=lc_color, label="LC true speed", linewidth=1.8)
        else:
            ax.plot(result.time, result.lt_v, color=lt_color, label="LT true speed", linewidth=1.8)
        pred_t, pred_lane = _initial_prediction_time_and_lane(result, cfg)
        _plot_lane_weighted(
            ax,
            pred_t,
            result.first_attacked_prediction.v[: len(pred_t)],
            pred_lane,
            normal_lane=attack_lane,
            color=attack_color,
            label=f"Attacked {hard_brake_vehicle} speed",
            linewidth=1.5,
            linestyle="--",
        )
        if result.collision_index is not None:
            idx = result.collision_index
            ax.scatter(
                result.time[idx],
                result.ego_v[idx],
                s=42,
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.8,
                label="collision",
                zorder=5,
            )
        _decorate_attack_axis(ax, result, method_label, ylabel="speed (m/s)")
    _save_comparison_figure(fig, axes, output_path)


def _plot_attack_gap_comparison(
    method_results: list[tuple[str, SimulationResult]],
    output_path: Path,
    scenario_label: str,
) -> None:
    fig, axes = _comparison_axes(method_results, f"Front gap ({scenario_label})")
    for ax, (method_label, result) in zip(axes, method_results):
        front_gap = _truncate_after_collision(result.front_gap, result)
        required_front_gap = _truncate_after_collision(result.required_front_gap, result)
        ax.plot(result.time, front_gap, color="#83c76d", label="Actual gap with leading vehicle", linewidth=2.0)
        ax.plot(result.time, required_front_gap, color="#b41f1f", label="Required $d_{\\min}$ with leading vehicle", linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=1.0)
        if result.collision_index is not None:
            idx = result.collision_index
            ax.scatter(
                result.time[idx],
                result.front_gap[idx],
                s=42,
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.8,
                zorder=5,
            )
        _decorate_attack_axis(ax, result, method_label, ylabel="gap (m)")
    _save_comparison_figure(fig, axes, output_path)


def _truncate_after_collision(values: np.ndarray, result: SimulationResult) -> np.ndarray:
    plotted = np.asarray(values, dtype=float).copy()
    if result.collision_index is not None:
        plotted[int(result.collision_index) + 1 :] = np.nan
    return plotted


def _plot_lane_weighted(
    ax,
    time: np.ndarray,
    values: np.ndarray,
    ego_lane: np.ndarray,
    normal_lane: int,
    color: str,
    label: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    lane_for_plot = _lane_for_plotting(ego_lane)
    normal_mask = lane_for_plot == normal_lane
    faded_mask = (lane_for_plot == CURRENT_LANE) | (lane_for_plot == TARGET_LANE)
    faded_mask &= ~normal_mask
    has_normal_segment = bool(np.any(normal_mask & np.isfinite(values)))
    ax.plot(
        time,
        _masked_values(values, faded_mask),
        color=color,
        label="_nolegend_" if has_normal_segment else label,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=0.25,
    )
    ax.plot(
        time,
        _masked_values(values, normal_mask),
        color=color,
        label=label if has_normal_segment else "_nolegend_",
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=1.0,
    )


def _comparison_axes(method_results: list[tuple[str, SimulationResult]], title: str):
    fig, axes = plt.subplots(
        1,
        len(method_results),
        figsize=(4.2 * len(method_results), 4.2),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title)
    return fig, axes


def _decorate_attack_axis(ax, result: SimulationResult, method_label: str, ylabel: str) -> None:
    ax.set_title(method_label, fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    merge_time = _merge_time_for_plot(result)
    if merge_time is not None:
        ax.axvline(merge_time, color="black", linestyle=":", linewidth=1.4, label="Lane change time step")
    ax.grid(True, alpha=0.25)


def _merge_time_for_plot(result: SimulationResult) -> float | None:
    if result.merge_time is not None:
        return float(result.merge_time)

    time = np.asarray(result.time, dtype=float)
    lane = _lane_for_plotting(result.ego_lane)
    if len(time) != len(lane) or len(time) < 2:
        return None

    transitions = np.where((lane[:-1] == CURRENT_LANE) & (lane[1:] == TARGET_LANE))[0]
    if len(transitions) == 0:
        return None
    return float(time[int(transitions[0]) + 1])


def _initial_prediction_time_and_lane(result: SimulationResult, cfg: PlannerConfig) -> tuple[np.ndarray, np.ndarray]:
    prediction_window_len = cfg.horizon_steps + 1
    pred_len = min(
        prediction_window_len,
        len(result.first_attacked_prediction.x),
        len(result.time),
        len(result.ego_lane),
    )
    pred_t = result.time[0] + cfg.dt * np.arange(pred_len, dtype=float)
    pred_lane = result.ego_lane[:pred_len]
    return pred_t, pred_lane


def _save_comparison_figure(fig, axes, output_path: Path) -> None:
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label == "_nolegend_" or label in labels:
                continue
            handles.append(handle)
            labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(values, dtype=float).copy()
    masked[~mask] = np.nan
    return masked


def _lane_for_plotting(ego_lane: np.ndarray) -> np.ndarray:
    lane = np.asarray(ego_lane, dtype=int).copy()
    valid = (lane == CURRENT_LANE) | (lane == TARGET_LANE)
    if not np.any(valid):
        return lane

    first_valid = int(np.where(valid)[0][0])
    lane[:first_valid] = lane[first_valid]
    for idx in range(first_valid + 1, len(lane)):
        if lane[idx] not in (CURRENT_LANE, TARGET_LANE):
            lane[idx] = lane[idx - 1]
    return lane


def _save_figure(fig, output_path: Path, title_parts: list[str]) -> None:
    if title_parts:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
