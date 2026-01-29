#!/usr/bin/env python3
"""
Aggregate runtime metrics JSONL files produced during training/evaluation.

Outputs:
  - summary JSON with overall counts and top-N episodes
  - optional CSV with per-episode rows
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def top_by_metric(
    episodes: List[Dict[str, Any]],
    key: str,
    topn: int,
    *,
    nested: bool = False,
    nested_key: str | None = None,
) -> List[Dict[str, Any]]:
    if nested:
        ranked = sorted(
            episodes,
            key=lambda r: (
                (r.get(key, {}) or {}).get(nested_key) is not None,
                (r.get(key, {}) or {}).get(nested_key, 0),
            ),
            reverse=True,
        )
    else:
        ranked = sorted(
            episodes,
            key=lambda r: (r.get(key) is not None, r.get(key, 0)),
            reverse=True,
        )
    return ranked[:topn]


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate runtime_metrics JSONL files.")
    ap.add_argument(
        "metrics_dir",
        type=Path,
        help="Directory containing runtime_metrics_*.jsonl files",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_metrics_aggregate.json"),
        help="Output JSON summary path",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for per-episode rows",
    )
    ap.add_argument("--top", type=int, default=20, help="Top-N episodes to include")
    args = ap.parse_args()

    metrics_dir = args.metrics_dir
    jsonl_files = sorted(metrics_dir.glob("runtime_metrics_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No runtime_metrics_*.jsonl found in {metrics_dir}")

    episodes: List[Dict[str, Any]] = []
    total_steps = 0
    action_counts = Counter()

    for path in jsonl_files:
        for rec in load_jsonl(path):
            episodes.append(rec)
            steps = int(rec.get("steps") or 0)
            total_steps += steps
            for name, count in (rec.get("action_counts") or {}).items():
                action_counts[name] += int(count)

    action_freq = {}
    if total_steps > 0:
        for name, count in action_counts.items():
            action_freq[name] = count / total_steps

    def metric_values(
        metric_key: str, *, nested: bool = False, nested_key: str | None = None
    ) -> List[float]:
        values = []
        for rec in episodes:
            if nested:
                val = (rec.get(metric_key, {}) or {}).get(nested_key)
            else:
                val = rec.get(metric_key)
            if val is None:
                continue
            try:
                values.append(float(val))
            except Exception:
                continue
        return values

    def summarize_series(values: List[float]) -> Dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None}
        return {"mean": mean(values), "median": median(values)}

    def action_count_values(action: str) -> List[float]:
        values = []
        for rec in episodes:
            counts = rec.get("action_counts") or {}
            val = counts.get(action)
            if val is None:
                continue
            try:
                values.append(float(val))
            except Exception:
                continue
        return values

    per_episode_stats = {
        "eat_freq": summarize_series(metric_values("action_freq", nested=True, nested_key="eat")),
        "read_freq": summarize_series(metric_values("action_freq", nested=True, nested_key="read")),
        "drop_freq": summarize_series(metric_values("action_freq", nested=True, nested_key="drop")),
        "quaff_freq": summarize_series(metric_values("action_freq", nested=True, nested_key="quaff")),
        "zap_freq": summarize_series(metric_values("action_freq", nested=True, nested_key="zap")),
        "eat_count": summarize_series(action_count_values("eat")),
        "read_count": summarize_series(action_count_values("read")),
        "drop_count": summarize_series(action_count_values("drop")),
        "quaff_count": summarize_series(action_count_values("quaff")),
        "zap_count": summarize_series(action_count_values("zap")),
        "hunger_recovery": summarize_series(metric_values("hunger_recovery")),
        "non_hungry_rate": summarize_series(metric_values("non_hungry_rate")),
    }

    summary = {
        "total_files": len(jsonl_files),
        "total_episodes": len(episodes),
        "total_steps": total_steps,
        "action_counts": dict(action_counts),
        "action_freq": action_freq,
        "per_episode_stats": per_episode_stats,
        "top": {
            "eat_freq": top_by_metric(
                episodes, "action_freq", args.top, nested=True, nested_key="eat"
            ),
            "read_freq": top_by_metric(
                episodes, "action_freq", args.top, nested=True, nested_key="read"
            ),
            "hunger_recovery": top_by_metric(episodes, "hunger_recovery", args.top),
            "non_hungry_rate": top_by_metric(episodes, "non_hungry_rate", args.top),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote summary JSON to {args.output}")

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "episode_id",
                "gameid",
                "worker_idx",
                "env_idx",
                "episode_idx",
                "steps",
                "eat",
                "read",
                "drop",
                "quaff",
                "zap",
                "hunger_recovery",
                "non_hungry_rate",
                "ttyrec",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in episodes:
                actions = rec.get("action_counts", {}) or {}
                writer.writerow(
                    {
                        "episode_id": rec.get("episode_id"),
                        "gameid": rec.get("gameid"),
                        "worker_idx": rec.get("worker_idx"),
                        "env_idx": rec.get("env_idx"),
                        "episode_idx": rec.get("episode_idx"),
                        "steps": rec.get("steps"),
                        "eat": actions.get("eat", 0),
                        "read": actions.get("read", 0),
                        "drop": actions.get("drop", 0),
                        "quaff": actions.get("quaff", 0),
                        "zap": actions.get("zap", 0),
                        "hunger_recovery": rec.get("hunger_recovery"),
                        "non_hungry_rate": rec.get("non_hungry_rate"),
                        "ttyrec": rec.get("ttyrec"),
                    }
                )
        print(f"Wrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
