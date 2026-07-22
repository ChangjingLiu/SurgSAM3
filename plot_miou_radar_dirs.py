#!/usr/bin/env python3
"""Plot IoU/Dice radar charts when each method stores eval JSON in its own directory.

Unlike plot_miou_radar.py (single --outputs-dir), pass one directory per method, e.g.:

  python plot_miou_radar_dirs.py \\
    --sam3-dir outputs_sam3 \\
    --sam3-lora-dir outputs_r16_single_prompt_base_lora \\
    --medical-sam3-2d-dir outputs_medical_sam3_2d \\
    --medical-sam3-3d-dir outputs_sam3_3d \\
    --medical-sam3-2d-lora-dir outputs_r16_single_prompt_medical_sam3_lora \\
    --medical-sam3-2d-lora-suffix _medical_sam3_base \\
    --combined --metric both --out outputs/radar_multi_dir.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

DATASET_TITLES = {
    "cholec": "Cholec",
    "endovis": "EndoVis",
    "cadis": "CaDIS",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    dir_flag: str
    suffix_flag: str
    default_suffix: str
    style: dict


METHOD_SPECS: list[MethodSpec] = [
    MethodSpec(
        "sam3",
        "SAM3",
        "sam3_dir",
        "sam3_suffix",
        "",
        {"color": "#1f77b4", "linestyle": "--", "linewidth": 2.0, "fill_alpha": 0.12},
    ),
    MethodSpec(
        "sam3_lora",
        "SAM3 LoRA",
        "sam3_lora_dir",
        "sam3_lora_suffix",
        "_lora",
        {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.0, "fill_alpha": 0.18},
    ),
    MethodSpec(
        "medical_sam3_2d",
        "Medical-SAM3 2D",
        "medical_sam3_2d_dir",
        "medical_sam3_2d_suffix",
        "_medical_sam3_2d",
        {"color": "#d62728", "linestyle": "-", "linewidth": 2.5, "fill_alpha": 0.20},
    ),
    MethodSpec(
        "medical_sam3_3d",
        "Medical-SAM3 3D",
        "medical_sam3_3d_dir",
        "medical_sam3_3d_suffix",
        "_medical_sam3_3d",
        {"color": "#ff7f0e", "linestyle": "-.", "linewidth": 2.0, "fill_alpha": 0.15},
    ),
    MethodSpec(
        "medical_sam3_2d_lora",
        "Medical-SAM3 2D LoRA",
        "medical_sam3_2d_lora_dir",
        "medical_sam3_2d_lora_suffix",
        "_medical_sam3_lora",
        {"color": "#9467bd", "linestyle": "-", "linewidth": 2.0, "fill_alpha": 0.18},
    ),
]

METHOD_KEYS = [spec.key for spec in METHOD_SPECS]


def resolve_json_path(method_dir: Path, dataset: str, suffix: str) -> Path | None:
    """Resolve miou JSON for one dataset; exact name first, then single glob match."""
    exact = method_dir / f"miou_{dataset}_test{suffix}.json"
    if exact.is_file():
        return exact

    pattern = f"miou_{dataset}_test*.json"
    matches = sorted(method_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise FileNotFoundError(
            f"Ambiguous eval JSON for dataset '{dataset}' in {method_dir}: {names}. "
            f"Set an explicit --*-suffix or keep only one file matching {pattern}."
        )
    return None


def iou_percent_to_dice_percent(iou_pct: float) -> float:
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


def build_method_sources(args: argparse.Namespace) -> dict[str, tuple[Path, str]]:
    """Map method key -> (directory, filename suffix)."""
    sources: dict[str, tuple[Path, str]] = {}
    for spec in METHOD_SPECS:
        method_dir = getattr(args, spec.dir_flag)
        if method_dir is None:
            continue
        suffix = getattr(args, spec.suffix_flag)
        if suffix is None:
            suffix = spec.default_suffix
        sources[spec.key] = (Path(method_dir), suffix)
    return sources


def collect_series(
    method_sources: dict[str, tuple[Path, str]],
    dataset: str,
    method_keys: list[str],
    metric: str,
    level: str,
    segment: str | None,
) -> tuple[list[str], dict[str, list[float]]]:
    raw: dict[str, dict[str, float]] = {}
    for method_key in method_keys:
        if method_key not in method_sources:
            continue
        method_dir, suffix = method_sources[method_key]
        path = resolve_json_path(method_dir, dataset, suffix)
        if path is None:
            continue
        label = next(spec.label for spec in METHOD_SPECS if spec.key == method_key)
        raw[label] = load_per_class(path, metric, level, segment)

    if not raw:
        return [], {}

    labels = sorted({cls for per_class in raw.values() for cls in per_class}, key=str.lower)
    series = {name: [per_class.get(cls, 0.0) for cls in labels] for name, per_class in raw.items()}
    return labels, series


def collect_combined_series(
    method_sources: dict[str, tuple[Path, str]],
    datasets: list[str],
    method_keys: list[str],
    metric: str,
    level: str,
    segment: str | None,
) -> tuple[list[str], dict[str, list[float]]]:
    raw: dict[str, dict[str, float]] = {}
    axis_keys: list[str] = []

    for dataset in datasets:
        ds_title = DATASET_TITLES.get(dataset, dataset)
        per_model: dict[str, dict[str, float]] = {}
        for method_key in method_keys:
            if method_key not in method_sources:
                continue
            method_dir, suffix = method_sources[method_key]
            path = resolve_json_path(method_dir, dataset, suffix)
            if path is None:
                continue
            model_label = next(spec.label for spec in METHOD_SPECS if spec.key == method_key)
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
    ax.tick_params(axis="x", pad=12)   # 默认约 4–8，可试 10–18
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


def add_method_args(parser: argparse.ArgumentParser) -> None:
    for spec in METHOD_SPECS:
        flag = spec.dir_flag.replace("_", "-")
        parser.add_argument(
            f"--{flag}",
            type=Path,
            default=None,
            help=f"Directory with eval JSON for {spec.label}",
        )
        suffix_flag = spec.suffix_flag.replace("_", "-")
        parser.add_argument(
            f"--{suffix_flag}",
            type=str,
            default=None,
            help=(
                f"Filename suffix for {spec.label} "
                f"(default: '{spec.default_suffix}' → miou_<dataset>_test{spec.default_suffix}.json)"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot IoU/Dice radar charts from per-method eval JSON directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_method_args(parser)
    parser.add_argument("--out", type=Path, default=Path("outputs/miou_dice_radar_multi_dir.png"))
    parser.add_argument("--level", choices=("overall", "by_segment"), default="overall")
    parser.add_argument("--segment", type=str, default=None, help="e.g. seq_5, 01_00080")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_KEYS),
        choices=METHOD_KEYS,
        help="Subset of methods to plot (dirs must be provided for each)",
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
    )
    parser.add_argument("--ylim", type=float, default=100.0)
    parser.add_argument(
        "--show-values",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    method_sources = build_method_sources(args)
    if not method_sources:
        parser.error(
            "Provide at least one method directory "
            "(e.g. --sam3-dir, --sam3-lora-dir, --medical-sam3-2d-dir)"
        )

    selected_keys = [k for k in args.methods if k in method_sources]
    if not selected_keys:
        parser.error("None of the requested --methods have a directory argument set")

    metrics_to_plot = list(METRICS) if args.metric == "both" else [args.metric]
    style_map = {spec.label: spec.style for spec in METHOD_SPECS}
    key_to_label = {spec.key: spec.label for spec in METHOD_SPECS}
    selected_labels = [key_to_label[k] for k in selected_keys]

    print("Method directories:")
    for key in selected_keys:
        method_dir, suffix = method_sources[key]
        print(f"  {key_to_label[key]}: {method_dir} (suffix={suffix!r})")

    lora_label = key_to_label.get("sam3_lora", "SAM3 LoRA")

    if args.combined:
        full_labels_iou, series_iou = collect_combined_series(
            method_sources, args.datasets, selected_keys, "iou", args.level, args.segment
        )
        before = len(full_labels_iou)
        keep_idx, filter_labels, filter_series_iou = filter_axes_by_threshold(
            full_labels_iou, series_iou, args.min_max_iou
        )
        filter_series_iou = {
            k: v for k, v in filter_series_iou.items() if k in selected_labels
        }

        if not filter_series_iou:
            raise SystemExit("No eval JSON files found for the given method directories")

        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(
            1,
            n_metrics,
            figsize=(5.0 * n_metrics, 6.2),
            subplot_kw={"polar": True},
            squeeze=False,
        )

        annotate = {lora_label} if args.show_values else None

        for col, metric in enumerate(metrics_to_plot):
            ax = axes[0, col]
            labels_m, series_m = collect_combined_series(
                method_sources, args.datasets, selected_keys, metric, args.level, args.segment
            )
            _, series_m = subset_series_by_indices(labels_m, series_m, keep_idx)
            series_m = {k: v for k, v in series_m.items() if k in selected_labels}

            plot_radar(
                ax,
                filter_labels,
                series_m,
                style_map,
                show_values=args.show_values,
                ylim=args.ylim,
                label_fontsize=6.5,
                value_fontsize=5,
                annotate_models=annotate,
            )

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
        print(f"  methods: {selected_keys}")
        return

    metric = metrics_to_plot[0]
    datasets_with_data = []
    for dataset in args.datasets:
        labels, series = collect_series(
            method_sources, dataset, selected_keys, metric, args.level, args.segment
        )
        if not series:
            continue
        before = len(labels)
        if metric == "iou":
            _, labels, series = filter_axes_by_threshold(labels, series, args.min_max_iou)
        datasets_with_data.append((dataset, labels, series, before))

    if not datasets_with_data:
        raise SystemExit("No eval JSON files found for the given method directories")

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
            {k: v for k, v in series.items() if k in selected_labels},
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
