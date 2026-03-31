#!/usr/bin/env python3
"""
Compare runtime metrics across experiments and plot mean/median bar charts.
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


METRICS = [
    "eat_freq",
    "read_freq",
    "drop_freq",
    "quaff_freq",
    "zap_freq",
    "hunger_recovery",
    "non_hungry_rate",
]

COUNT_METRICS = [
    "eat_count",
    "read_count",
    "drop_count",
    "quaff_count",
    "zap_count",
]



def load_experiment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "steps" not in df.columns:
        raise ValueError(f"steps column missing in {path}")
    steps = df["steps"].replace(0, np.nan)
    for action in ("eat", "read", "drop", "quaff", "zap"):
        col = f"{action}_freq"
        if action in df.columns:
            df[col] = df[action] / steps
        else:
            df[col] = np.nan
    return df


def summarize(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        if metric not in df.columns:
            stats[metric] = {"mean": float("nan"), "median": float("nan")}
            continue
        series = df[metric].dropna()
        if series.empty:
            stats[metric] = {"mean": float("nan"), "median": float("nan")}
            continue
        stats[metric] = {"mean": float(series.mean()), "median": float(series.median())}
    for action in ("eat", "read", "drop", "quaff", "zap"):
        if action in df.columns:
            series = df[action].dropna()
        else:
            series = pd.Series(dtype=float)
        metric = f"{action}_count"
        if series.empty:
            stats[metric] = {"mean": float("nan"), "median": float("nan")}
            continue
        stats[metric] = {"mean": float(series.mean()), "median": float(series.median())}
    return stats


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
        bars = ax.bar(experiments, values, color="#4C78A8")
        ax.set_title(f"{metric} ({stat_key})")
        ax.set_ylabel(metric)
        ax.set_xticklabels(experiments, rotation=45, ha="right")
        for bar, value in zip(bars, values):
            if np.isnan(value):
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
    ap = argparse.ArgumentParser(description="Compare runtime metrics across experiments.")
    ap.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing experiment folders (e.g., train_dir/skill_policy)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runtime_metrics_compare"),
        help="Directory for summary and plots",
    )
    ap.add_argument(
        "--glob",
        default="*/runtime_metrics/aggregate.csv",
        help="Glob pattern to find aggregate.csv files under root_dir",
    )
    args = ap.parse_args()

    root_dir = args.root_dir
    csv_paths = sorted(root_dir.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"No aggregate.csv found under {root_dir} with glob {args.glob}")

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for csv_path in csv_paths:
        exp_name = csv_path.parts[-3]
        df = load_experiment_csv(csv_path)
        summary[exp_name] = summarize(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "runtime_metrics_summary.csv"
    rows = []
    for exp, stats in summary.items():
        row = {"experiment": exp}
        for metric in METRICS + COUNT_METRICS:
            row[f"{metric}_mean"] = stats[metric]["mean"]
            row[f"{metric}_median"] = stats[metric]["median"]
        rows.append(row)
    pd.DataFrame(rows).sort_values("experiment").to_csv(summary_path, index=False)
    print(f"Wrote summary CSV to {summary_path}")

    plot_bars(summary, METRICS + COUNT_METRICS, out_dir=args.output_dir, stat_key="mean")
    plot_bars(summary, METRICS + COUNT_METRICS, out_dir=args.output_dir, stat_key="median")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
