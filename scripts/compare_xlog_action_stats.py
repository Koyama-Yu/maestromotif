#!/usr/bin/env python3
"""
Compare xlogfile-based action stats across experiments and plot mean/median bar charts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTION_METRICS = [
    "eat_count",
    "quaff_count",
    "read_count",
]

COLOR_MAP = {
    "itemuse_extrinsic": "#4C78A8",  # blue
    "itemuse_only_llm": "#E45756",   # red
    "pureRL": "#F58518",             # orange
    "autoascend": "#54A24B",         # green
}


def load_summary(path: Path) -> dict:
    return pd.read_json(path, typ="series").to_dict()


def load_autoascend_metrics(metrics_path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(metrics_path)
    mapping = {
        "eat_count": "action_count_eat",
        "quaff_count": "action_count_quaff",
        "read_count": "action_count_read",
    }
    stats: Dict[str, Dict[str, float]] = {}
    for metric, col in mapping.items():
        if col not in df.columns:
            stats[metric] = {"mean": None, "median": None}
            continue
        series = df[col].dropna()
        if series.empty:
            stats[metric] = {"mean": None, "median": None}
            continue
        stats[metric] = {"mean": float(series.mean()), "median": float(series.median())}
    return stats


def load_episode_counts(path: Path) -> pd.DataFrame:
    csv_path = path.with_name("xlog_action_counts.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    return pd.read_csv(csv_path)


def plot_bars(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    *,
    out_dir: Path,
    stat_key: str,
) -> None:
    experiments = list(summary.keys())
    for metric in metrics:
        values = [summary[exp][metric][stat_key] for exp in experiments]
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(experiments)), 4))
        colors = [COLOR_MAP.get(exp, "#4C78A8") for exp in experiments]
        bars = ax.bar(experiments, values, color=colors)
        ax.set_title(f"{metric} ({stat_key})")
        ax.set_ylabel(metric)
        ax.set_xticklabels(experiments, rotation=45, ha="right")
        for bar, value in zip(bars, values):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                label = "nan"
            elif abs(value) >= 100:
                label = f"{value:.0f}"
            elif abs(value) >= 10:
                label = f"{value:.1f}"
            else:
                label = f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
        fig.tight_layout()
        out_path = out_dir / f"{metric}_{stat_key}.png"
        fig.savefig(out_path)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare xlog action stats across experiments."
    )
    ap.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing experiment folders (e.g., train_dir/skill_policy)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("xlog_action_compare"),
        help="Directory for summary and plots",
    )
    ap.add_argument(
        "--glob",
        default="*/runtime_metrics/xlog_action_summary.json",
        help="Glob pattern to find xlog_action_summary.json files under root_dir",
    )
    args = ap.parse_args()

    summary_paths = sorted(args.root_dir.glob(args.glob))
    if not summary_paths:
        raise SystemExit(
            f"No xlog_action_summary.json found under {args.root_dir} with glob {args.glob}"
        )

    stats_by_exp: Dict[str, Dict[str, Dict[str, float]]] = {}
    counts_by_exp: Dict[str, pd.DataFrame] = {}
    for path in summary_paths:
        exp_name = path.parts[-3]
        data = load_summary(path)
        per_episode = data.get("per_episode_stats", {})
        stats_by_exp[exp_name] = {}
        for metric in ACTION_METRICS:
            stats_by_exp[exp_name][metric] = per_episode.get(
                metric, {"mean": None, "median": None}
            )
        try:
            counts_by_exp[exp_name] = load_episode_counts(path)
        except FileNotFoundError:
            counts_by_exp[exp_name] = pd.DataFrame()

    # Optional: autoascend metrics from project root
    project_root = args.root_dir.parent.parent
    autoascend_dir = project_root / "autoascend"
    autoascend_csv = autoascend_dir / "metrics.csv"
    autoascend_json = autoascend_dir / "metrics.json"
    if autoascend_csv.exists():
        stats_by_exp["autoascend"] = load_autoascend_metrics(autoascend_csv)
        counts_by_exp.setdefault("autoascend", pd.DataFrame())
    elif autoascend_json.exists():
        import json as _json
        data = _json.loads(autoascend_json.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        if episodes:
            df = pd.DataFrame(episodes)
            stats_by_exp["autoascend"] = {}
            for metric, key in {
                "eat_count": "eat",
                "quaff_count": "quaff",
                "read_count": "read",
            }.items():
                series = df.get("action_counts")
                if series is None:
                    stats_by_exp["autoascend"][metric] = {"mean": None, "median": None}
                    continue
                series = series.apply(lambda x: x.get(key) if isinstance(x, dict) else None)
                series = pd.Series(series).dropna()
                if series.empty:
                    stats_by_exp["autoascend"][metric] = {"mean": None, "median": None}
                else:
                    stats_by_exp["autoascend"][metric] = {
                        "mean": float(series.mean()),
                        "median": float(series.median()),
                    }
            counts_by_exp.setdefault("autoascend", pd.DataFrame())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for exp, stats in stats_by_exp.items():
        row = {"experiment": exp}
        for metric in ACTION_METRICS:
            row[f"{metric}_mean"] = stats[metric].get("mean")
            row[f"{metric}_median"] = stats[metric].get("median")
        summary_rows.append(row)
    summary_csv = args.output_dir / "xlog_action_summary_compare.csv"
    pd.DataFrame(summary_rows).sort_values("experiment").to_csv(summary_csv, index=False)
    print(f"Wrote summary CSV to {summary_csv}")

    plot_bars(stats_by_exp, ACTION_METRICS, out_dir=args.output_dir, stat_key="mean")
    plot_bars(stats_by_exp, ACTION_METRICS, out_dir=args.output_dir, stat_key="median")
    print(f"Wrote plots to {args.output_dir}")

    # Distribution plots per action per experiment
    for exp_name, df in counts_by_exp.items():
        if df.empty:
            continue
        for action in ("eat", "read", "quaff"):
            if action not in df.columns:
                continue
            series = df[action].dropna()
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            max_val = series.max()
            p99 = series.quantile(0.99)
            if max_val > 50 and series.median() == 0:
                upper = max(1, int(p99))
                clipped = series[series <= upper]
                overflow = int((series > upper).sum())
                ax.hist(clipped, bins=min(50, max(10, upper)), color="#4C78A8", alpha=0.8)
                ax.set_xlim(0, upper)
                if overflow:
                    ax.text(
                        0.98,
                        0.95,
                        f"overflow (> {upper}): {overflow}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            else:
                ax.hist(series, bins=50, color="#4C78A8", alpha=0.8)
            mean_val = series.mean()
            median_val = series.median()
            ax.axvline(mean_val, color="#F58518", linestyle="--", label=f"mean={mean_val:.2f}")
            ax.axvline(median_val, color="#54A24B", linestyle="-", label=f"median={median_val:.2f}")
            ax.set_title(f"{exp_name} {action} count distribution")
            ax.set_xlabel(f"{action} count per episode")
            ax.set_ylabel("episodes")
            ax.legend()
            fig.tight_layout()
            out_path = args.output_dir / f"{exp_name}_{action}_dist.png"
            fig.savefig(out_path)
            plt.close(fig)

            # Log-scale version for better visibility of sparse counts
            fig, ax = plt.subplots(figsize=(6, 4))
            if max_val > 50 and series.median() == 0:
                upper = max(1, int(p99))
                clipped = series[series <= upper]
                overflow = int((series > upper).sum())
                ax.hist(clipped, bins=min(50, max(10, upper)), color="#4C78A8", alpha=0.8)
                ax.set_xlim(0, upper)
                if overflow:
                    ax.text(
                        0.98,
                        0.95,
                        f"overflow (> {upper}): {overflow}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                    )
            else:
                ax.hist(series, bins=50, color="#4C78A8", alpha=0.8)
            ax.set_yscale("log")
            ax.axvline(mean_val, color="#F58518", linestyle="--", label=f"mean={mean_val:.2f}")
            ax.axvline(median_val, color="#54A24B", linestyle="-", label=f"median={median_val:.2f}")
            ax.set_title(f"{exp_name} {action} count distribution (log y)")
            ax.set_xlabel(f"{action} count per episode")
            ax.set_ylabel("episodes (log)")
            ax.legend()
            fig.tight_layout()
            out_path = args.output_dir / f"{exp_name}_{action}_dist_log.png"
            fig.savefig(out_path)
            plt.close(fig)


if __name__ == "__main__":
    main()
