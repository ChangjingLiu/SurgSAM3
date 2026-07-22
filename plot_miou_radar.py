#!/usr/bin/env python3
"""Plot per-class IoU and Dice radar charts from eval JSON files in outputs/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ("cholec", "endovis", "cadis")
METRICS = ("iou", "dice")
METRIC_FIELD = {
    "iou": "per_class_iou",
    "dice": "per_class_dice",
}
METRIC_TITLE = {
    "iou": "IoU",
    "dice": "Dice",
}

MODEL_SPECS: list[tuple[str, str, dict]] = [
    ("lora", "SAM3 LoRA", {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.0, "fill_alpha": 0.18}),
    ("base", "SAM3", {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "fill_alpha": 0.12}),
    (
        "medical_sam3_2d",
        "Medical-SAM3 2D",
        {"color": "#d62728", "linestyle": "-", "linewidth": 2.5, "fill_alpha": 0.20},
    ),
    (
        "medical_sam3_3d",
        "Medical-SAM3 3D",
        {"color": "#ff7f0e", "linestyle": "-.", "linewidth": 2.0, "fill_alpha": 0.15},
    ),
]

SUFFIX = {
    "lora": "",
    "base": "_base",
    "medical_sam3_2d": "_medical_sam3_2d",
    "medical_sam3_3d": "_medical_sam3_3d",
}

DATASET_TITLES = {
    "cholec": "Cholec",
    "endovis": "EndoVis",
    "cadis": "CaDIS",
}


def json_path(outputs_dir: Path, dataset: str, model_key: str) -> Path:
    return outputs_dir / f"miou_{dataset}_test{SUFFIX[model_key]}.json"


def iou_percent_to_dice_percent(iou_pct: float) -> float:
    """Approximate Dice from IoU%% when legacy JSON lacks per_class_dice."""
    if iou_pct <= 0:
        return 0.0
    iou = iou_pct / 100.0
    return 200.0 * iou / (1.0 + iou)


def load_per_class(
    path: Path,
    metric: str,
    level: str = "overall",
    segment: str | None = None,
) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if level == "overall":
        block = data["overall"]
    else:
        if segment is None:
            raise ValueError("--segment is required when --level=by_segment")
        block = data["by_segment"][segment]

    field = METRIC_FIELD[metric]
    if field in block:
        return dict(block[field])

    if metric == "dice" and "per_class_iou" in block:
        return {k: iou_percent_to_dice_percent(v) for k, v in block["per_class_iou"].items()}

    return {}


def wrap_label(label: str, max_len: int = 16) -> str:
    if len(label) <= max_len:
        return label
    parts = label.replace("-", " ").replace("/", " ").split()
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip()
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)
    return "\n".join(lines)


def filter_axes_by_threshold(
    labels: list[str],
    series: dict[str, list[float]],
    min_score: float,
) -> tuple[list[int], list[str], dict[str, list[float]]]:
    if min_score <= 0 or not labels:
        keep_idx = list(range(len(labels)))
        return keep_idx, labels, series

    keep_idx = [
        i
        for i in range(len(labels))
        if max(vals[i] for vals in series.values()) >= min_score
    ]
    new_labels = [labels[i] for i in keep_idx]
    new_series = {name: [vals[i] for i in keep_idx] for name, vals in series.items()}
    return keep_idx, new_labels, new_series


def subset_series_by_indices(
    labels: list[str],
    series: dict[str, list[float]],
    keep_idx: list[int],
) -> tuple[list[str], dict[str, list[float]]]:
    if not keep_idx:
        return [], {name: [] for name in series}
    if len(labels) == len(keep_idx) and keep_idx == list(range(len(labels))):
        return labels, series
    new_labels = [labels[i] for i in keep_idx]
    new_series = {name: [vals[i] for i in keep_idx] for name, vals in series.items()}
    return new_labels, new_series


def collect_series(
    outputs_dir: Path,
    dataset: str,
    model_keys: list[str],
    metric: str,
    level: str,
    segment: str | None,
) -> tuple[list[str], dict[str, list[float]]]:
    raw: dict[str, dict[str, float]] = {}
    for model_key in model_keys:
        path = json_path(outputs_dir, dataset, model_key)
        if not path.exists():
            continue
        label = next(name for key, name, _ in MODEL_SPECS if key == model_key)
        raw[label] = load_per_class(path, metric, level, segment)

    if not raw:
        return [], {}

    labels = sorted({cls for per_class in raw.values() for cls in per_class}, key=str.lower)
    series = {name: [per_class.get(cls, 0.0) for cls in labels] for name, per_class in raw.items()}
    return labels, series


def collect_combined_series(
    outputs_dir: Path,
    datasets: list[str],
    model_keys: list[str],
    metric: str,
    level: str,
    segment: str | None,
) -> tuple[list[str], dict[str, list[float]]]:
    raw: dict[str, dict[str, float]] = {}
    axis_keys: list[str] = []

    for dataset in datasets:
        ds_title = DATASET_TITLES.get(dataset, dataset)
        per_model: dict[str, dict[str, float]] = {}
        for model_key in model_keys:
            path = json_path(outputs_dir, dataset, model_key)
            if not path.exists():
                continue
            model_label = next(name for key, name, _ in MODEL_SPECS if key == model_key)
            per_model[model_label] = load_per_class(path, metric, level, segment)

        if not per_model:
            continue

        classes = sorted({c for pc in per_model.values() for c in pc}, key=str.lower)
        for cls in classes:
            axis_key = f"{ds_title}/{cls}"
            axis_keys.append(axis_key)
            for model_label, per_class in per_model.items():
                raw.setdefault(model_label, {})[axis_key] = per_class.get(cls, 0.0)

    if not axis_keys:
        return [], {}

    series = {name: [raw[name].get(k, 0.0) for k in axis_keys] for name in raw}
    display_labels = [key.replace("/", "\n", 1) for key in axis_keys]
    return display_labels, series


LORA_LABEL = next(name for key, name, _ in MODEL_SPECS if key == "lora")


def plot_radar(
    ax: plt.Axes,
    labels: list[str],
    series: dict[str, list[float]],
    style_map: dict[str, dict],
    show_values: bool,
    ylim: float,
    label_fontsize: float = 7,
    value_fontsize: float = 5.5,
    annotate_models: set[str] | None = None,
) -> None:
    n = len(labels)
    if n == 0:
        ax.set_visible(False)
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([wrap_label(lab) for lab in labels], fontsize=label_fontsize)
    ax.set_ylim(0, ylim)
    step = 20 if ylim >= 80 else 10
    ax.set_yticks(list(range(step, int(ylim) + 1, step)))
    ax.set_yticklabels([str(v) for v in range(step, int(ylim) + 1, step)], fontsize=7)
    ax.grid(True, linestyle=":", alpha=0.45)

    for name, values in series.items():
        style = style_map[name]
        vals_closed = np.concatenate([values, values[:1]])
        ax.plot(
            angles_closed,
            vals_closed,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax.fill(angles_closed, vals_closed, color=style["color"], alpha=style["fill_alpha"])

        if show_values and (annotate_models is None or name in annotate_models):
            for ang, val in zip(angles, values):
                radius = min(val + ylim * 0.04, ylim - 1)
                ax.text(
                    ang,
                    radius,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color=style["color"],
                )


def discover_available_models(outputs_dir: Path, dataset: str) -> list[str]:
    return [key for key in SUFFIX if json_path(outputs_dir, dataset, key).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IoU/Dice radar charts from eval JSON files")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/miou_dice_radar_combined.png"))
    parser.add_argument("--level", choices=("overall", "by_segment"), default="overall")
    parser.add_argument("--segment", type=str, default=None, help="e.g. seq_5, 01_00080")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[key for key, _, _ in MODEL_SPECS],
        choices=[key for key, _, _ in MODEL_SPECS],
    )
    parser.add_argument(
        "--combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge all datasets into one radar chart per metric (default: True)",
    )
    parser.add_argument(
        "--min-max-iou",
        type=float,
        default=15.0,
        help="Drop class axes if max IoU across models is below this (default: 15, 0=disable)",
    )
    parser.add_argument(
        "--metric",
        choices=("both", "iou", "dice"),
        default="both",
        help="Which metric(s) to plot (default: both IoU and Dice side by side)",
    )
    parser.add_argument("--ylim", type=float, default=100.0, help="Radial axis max (default: 100)")
    parser.add_argument(
        "--show-values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show numeric labels on the chart (default: off)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    metrics_to_plot = list(METRICS) if args.metric == "both" else [args.metric]
    style_map = {label: style for _, label, style in MODEL_SPECS}
    key_to_label = {key: label for key, label, _ in MODEL_SPECS}
    selected_model_labels = [key_to_label[k] for k in args.models]

    if args.combined:
        full_labels_iou, series_iou = collect_combined_series(
            args.outputs_dir, args.datasets, args.models, "iou", args.level, args.segment
        )
        before = len(full_labels_iou)
        keep_idx, filter_labels, filter_series_iou = filter_axes_by_threshold(
            full_labels_iou, series_iou, args.min_max_iou
        )
        filter_series_iou = {
            k: v for k, v in filter_series_iou.items() if k in selected_model_labels
        }

        if not filter_series_iou:
            raise SystemExit(f"No JSON files found under {args.outputs_dir}")

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(
            1,
            n_metrics,
            figsize=(5.0 * n_metrics, 6.2),
            subplot_kw={"polar": True},
            squeeze=False,
        )

        annotate = {LORA_LABEL} if args.show_values else None

        for col, metric in enumerate(metrics_to_plot):
            ax = axes[0, col]
            labels_m, series_m = collect_combined_series(
                args.outputs_dir, args.datasets, args.models, metric, args.level, args.segment
            )
            _, series_m = subset_series_by_indices(labels_m, series_m, keep_idx)
            series_m = {k: v for k, v in series_m.items() if k in selected_model_labels}

            plot_radar(
                ax,
                filter_labels,
                series_m,
                style_map,
                show_values=args.show_values,
                ylim=args.ylim,
                label_fontsize=6.5 if col == 0 else 0,
                value_fontsize=5,
                annotate_models=annotate,
            )
            if col > 0:
                ax.set_xticklabels([])

        plt.tight_layout(rect=[0, 0.08, 1, 0.76])

        title_y = max(ax.get_position().y1 for ax in axes[0]) + 0.085
        for col, metric in enumerate(metrics_to_plot):
            pos = axes[0, col].get_position()
            fig.text(
                pos.x0 + pos.width / 2,
                title_y,
                f"Per-class {METRIC_TITLE[metric]} (%)",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels_leg,
                loc="lower center",
                ncol=min(len(handles), 4),
                bbox_to_anchor=(0.5, -0.04),
                fontsize=9,
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved radar chart to {args.out}")
        print(f"  metrics: {metrics_to_plot}")
        print(f"  classes: {before} → {len(filter_labels)} (min-max-iou={args.min_max_iou})")
        print(f"  models: {[k for k in args.models if key_to_label[k] in filter_series_iou]}")
        return

    # Separate panel per dataset (single metric or first metric only for simplicity)
    metric = metrics_to_plot[0]
    datasets_with_data = []
    for dataset in args.datasets:
        labels, series = collect_series(
            args.outputs_dir, dataset, args.models, metric, args.level, args.segment
        )
        if not series:
            continue
        before = len(labels)
        if metric == "iou":
            _, labels, series = filter_axes_by_threshold(labels, series, args.min_max_iou)
        datasets_with_data.append((dataset, labels, series, before))

    if not datasets_with_data:
        raise SystemExit(f"No JSON files found under {args.outputs_dir}")

    n = len(datasets_with_data)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5.2 * n, 5.4),
        subplot_kw={"polar": True},
        squeeze=False,
    )

    for col, (dataset, labels, series, before) in enumerate(datasets_with_data):
        ax = axes[0, col]
        title = DATASET_TITLES.get(dataset, dataset.upper())
        if args.level == "by_segment":
            title += f"\n({args.segment})"
        if args.min_max_iou > 0 and metric == "iou":
            title += f"\n{before}→{len(labels)} cls"
        ax.set_title(title, fontsize=11, pad=18)
        plot_radar(
            ax,
            labels,
            {k: v for k, v in series.items() if k in selected_model_labels},
            style_map,
            show_values=args.show_values,
            ylim=args.ylim,
        )

    level_text = "overall" if args.level == "overall" else f"segment={args.segment}"
    fig.suptitle(f"Per-class {METRIC_TITLE[metric]} (%) — {level_text}", fontsize=14, y=1.02)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_leg,
            loc="lower center",
            ncol=min(len(handles), 4),
            bbox_to_anchor=(0.5, -0.06),
            fontsize=9,
        )

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved radar chart to {args.out}")


if __name__ == "__main__":
    main()
