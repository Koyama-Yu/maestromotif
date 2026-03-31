#!/usr/bin/env python3
"""
Aggregate action counts (eat/read/drop/quaff/zap) from xlogfiles.

Writes:
  - summary JSON
  - optional CSV with per-episode action counts
  - optional JSONL with per-episode entries
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional


TARGET_ACTIONS = ("eat", "read", "drop", "quaff", "zap")


def iter_xlogfiles(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".xlogfile"):
                yield Path(dirpath) / fname


def parse_xlog_line(line: str) -> Optional[Dict[str, object]]:
    inv_by_name_raw = None
    ttyrecname = None
    for token in line.rstrip().split("\t"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "inv_by_name":
            inv_by_name_raw = value
        elif key == "ttyrecname":
            ttyrecname = value
    if inv_by_name_raw is None:
        return None
    try:
        inv_by_name = json.loads(inv_by_name_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(inv_by_name, dict):
        return None

    action_counts = Counter()
    for _, payload in inv_by_name.items():
        if not isinstance(payload, dict):
            continue
        actions = payload.get("actions")
        if not isinstance(actions, dict):
            continue
        for action, count in actions.items():
            if action not in TARGET_ACTIONS:
                continue
            try:
                action_counts[action] += int(count)
            except Exception:
                continue

    return {
        "ttyrecname": ttyrecname,
        "action_counts": dict(action_counts),
    }


def summarize_series(values: List[int]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {"mean": mean(values), "median": median(values)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate xlog action counts.")
    ap.add_argument("input_dir", type=Path, help="Directory containing xlogfiles")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("xlog_action_summary.json"),
        help="Summary JSON output path",
    )
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument("--jsonl", type=Path, default=None, help="Optional JSONL output path")
    ap.add_argument("--top", type=int, default=20, help="Top-N episodes per action")
    args = ap.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Directory not found: {input_dir}")

    totals = Counter()
    per_action_values: Dict[str, List[int]] = {a: [] for a in TARGET_ACTIONS}
    top_heaps: Dict[str, List[tuple[int, int, Dict[str, object]]]] = {
        a: [] for a in TARGET_ACTIONS
    }
    top_counter = 0

    csv_handle = None
    jsonl_handle = None
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        csv_handle = args.csv.open("w", encoding="utf-8", newline="")
        csv_handle.write(
            "episode_id,source_logfile,line_number,ttyrecname,eat,read,drop,quaff,zap,total_actions\n"
        )
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = args.jsonl.open("w", encoding="utf-8")

    total_episodes = 0

    for log_path in iter_xlogfiles(input_dir):
        rel_path = log_path.relative_to(input_dir)
        with log_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                parsed = parse_xlog_line(line)
                if parsed is None:
                    continue
                total_episodes += 1
                action_counts = {a: int(parsed["action_counts"].get(a, 0)) for a in TARGET_ACTIONS}
                total_actions = sum(action_counts.values())
                episode_id = f"{rel_path}:{line_no}"

                for action, count in action_counts.items():
                    totals[action] += count
                    per_action_values[action].append(count)

                    if args.top > 0:
                        top_counter += 1
                        entry = {
                            "episode_id": episode_id,
                            "source_logfile": str(rel_path),
                            "line_number": line_no,
                            "ttyrecname": parsed.get("ttyrecname"),
                            "action": action,
                            "count": count,
                        }
                        heap = top_heaps[action]
                        if len(heap) < args.top:
                            heap.append((count, top_counter, entry))
                            heap.sort(key=lambda x: x[0])
                        elif heap[0][0] < count:
                            heap[0] = (count, top_counter, entry)
                            heap.sort(key=lambda x: x[0])

                if csv_handle:
                    csv_handle.write(
                        f"{episode_id},{rel_path},{line_no},{parsed.get('ttyrecname') or ''},"
                        f"{action_counts['eat']},{action_counts['read']},{action_counts['drop']},"
                        f"{action_counts['quaff']},{action_counts['zap']},{total_actions}\n"
                    )
                if jsonl_handle:
                    jsonl_handle.write(
                        json.dumps(
                            {
                                "episode_id": episode_id,
                                "source_logfile": str(rel_path),
                                "line_number": line_no,
                                "ttyrecname": parsed.get("ttyrecname"),
                                "action_counts": action_counts,
                                "total_actions": total_actions,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

    if csv_handle:
        csv_handle.close()
    if jsonl_handle:
        jsonl_handle.close()

    per_episode_stats = {
        f"{action}_count": summarize_series(per_action_values[action])
        for action in TARGET_ACTIONS
    }

    top = {}
    for action, heap in top_heaps.items():
        top[action] = [entry for _, _, entry in sorted(heap, reverse=True)]

    summary = {
        "total_episodes": total_episodes,
        "action_counts": dict(totals),
        "per_episode_stats": per_episode_stats,
        "top": top,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote summary JSON to {args.output}")
    if args.csv:
        print(f"Wrote CSV to {args.csv}")
    if args.jsonl:
        print(f"Wrote JSONL to {args.jsonl}")


if __name__ == "__main__":
    main()
