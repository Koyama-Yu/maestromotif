#!/usr/bin/env python3
"""
Compute per-episode action frequencies and hunger metrics from NLE ttyrecs.

Outputs:
  - JSON summary (overall + per-episode)
  - Optional CSV per-episode metrics
  - Top-N ttyrec files by eat/read frequency and hunger resolution
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import nle.dataset as nld
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "nle.dataset is required. Install NLE or run inside the NLE environment."
    ) from exc


DEFAULT_HUNGER_IDX = 21

ACTION_KEYCODES = {
    "eat": {ord("e"), ord("E")},
    "read": {ord("r"), ord("R")},
    "drop": {ord("d"), ord("D")},
    "quaff": {ord("q"), ord("Q")},
    "zap": {ord("z"), ord("Z")},
}


@dataclass
class EpisodeStats:
    gameid: int
    action_counts: Counter = field(default_factory=Counter)
    steps: int = 0
    non_hungry_steps: int = 0
    hunger_recovery: float = 0.0
    last_hunger: Optional[int] = None
    ttyrec: Optional[str] = None

    def update_hunger(
        self, hunger_value: int, *, non_hungry_max: int, higher_is_worse: bool
    ) -> None:
        if hunger_value <= non_hungry_max:
            self.non_hungry_steps += 1
        if self.last_hunger is not None:
            if higher_is_worse:
                self.hunger_recovery += max(self.last_hunger - hunger_value, 0)
            else:
                self.hunger_recovery += max(hunger_value - self.last_hunger, 0)
        self.last_hunger = hunger_value
        self.steps += 1


def find_ttyrec_name(meta: Dict[str, Any]) -> Optional[str]:
    preferred_keys = [
        "ttyrecname",
        "ttyrec",
        "ttyrec_file",
        "ttyrecfile",
        "filename",
        "path",
        "recording",
    ]
    for key in preferred_keys:
        if key in meta and meta[key]:
            return str(meta[key])
    for key, value in meta.items():
        if "ttyrec" in key.lower() and value:
            return str(value)
    return None


def resolve_done_key(batch: Dict[str, Any]) -> Optional[str]:
    for key in ("done", "terminal", "dones", "terminals", "is_terminal"):
        if key in batch:
            return key
    return None


def iter_frames(
    batch: Dict[str, Any],
    *,
    blstats_key: str,
    hunger_index: int,
    keypress_key: str,
) -> Iterable[Tuple[int, int, int, int]]:
    """
    Yield (gameid, keypress, hunger_value, done_flag) per frame in batch.
    """
    gameids = batch["gameids"]
    keypresses = batch.get(keypress_key)
    if keypresses is None:
        raise KeyError(
            f"Missing key '{keypress_key}' in batch. Available keys: {sorted(batch.keys())}"
        )
    blstats = batch.get(blstats_key)
    if blstats is None:
        raise KeyError(
            f"Missing key '{blstats_key}' in batch. Available keys: {sorted(batch.keys())}"
        )
    if blstats.shape[-1] <= hunger_index:
        raise ValueError(
            f"{blstats_key} has length {blstats.shape[-1]} < hunger index {hunger_index}"
        )

    done_key = resolve_done_key(batch)
    done_flags = batch.get(done_key) if done_key else None

    batch_size = keypresses.shape[0]
    seq_len = keypresses.shape[1]
    for b in range(batch_size):
        for t in range(seq_len):
            done_flag = bool(done_flags[b, t]) if done_flags is not None else False
            gameid = int(gameids[b, t])
            keypress = int(keypresses[b, t])
            hunger_value = int(blstats[b, t, hunger_index])
            yield gameid, keypress, hunger_value, int(done_flag)
            if done_flag:
                break


def iter_frames_no_hunger(
    batch: Dict[str, Any],
    *,
    keypress_key: str,
) -> Iterable[Tuple[int, int, int]]:
    """
    Yield (gameid, keypress, done_flag) per frame in batch (no blstats).
    """
    gameids = batch["gameids"]
    keypresses = batch.get(keypress_key)
    if keypresses is None:
        raise KeyError(
            f"Missing key '{keypress_key}' in batch. Available keys: {sorted(batch.keys())}"
        )
    done_key = resolve_done_key(batch)
    done_flags = batch.get(done_key) if done_key else None

    batch_size = keypresses.shape[0]
    seq_len = keypresses.shape[1]
    for b in range(batch_size):
        for t in range(seq_len):
            done_flag = bool(done_flags[b, t]) if done_flags is not None else False
            gameid = int(gameids[b, t])
            keypress = int(keypresses[b, t])
            yield gameid, keypress, int(done_flag)
            if done_flag:
                break


def summarize_actions(counter: Counter, total_steps: int) -> Dict[str, Any]:
    summary = {}
    for action, keycodes in ACTION_KEYCODES.items():
        count = sum(counter[code] for code in keycodes)
        freq = (count / total_steps) if total_steps else 0.0
        summary[action] = {"count": count, "per_step": freq}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze ttyrec datasets for action frequency and hunger resolution metrics."
        )
    )
    ap.add_argument("--dataset", required=True, help="Dataset name registered in DB")
    ap.add_argument("--db", default="ttyrecs.db", help="NLE DB filename")
    ap.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size")
    ap.add_argument(
        "--blstats-key",
        default="blstats",
        help="Batch key name for blstats (default: blstats)",
    )
    ap.add_argument(
        "--keypress-key",
        default="keypresses",
        help="Batch key name for keypresses (default: keypresses)",
    )
    ap.add_argument(
        "--hunger-index",
        type=int,
        default=DEFAULT_HUNGER_IDX,
        help="Index of hunger value in blstats (default: 21)",
    )
    ap.add_argument(
        "--no-hunger",
        action="store_true",
        help="Skip hunger metrics (use when blstats is unavailable)",
    )
    ap.add_argument(
        "--non-hungry-max",
        type=int,
        default=1,
        help="Max hunger value considered non-hungry (default: 1)",
    )
    ap.add_argument(
        "--higher-is-worse",
        action="store_true",
        default=True,
        help="Treat larger hunger values as worse (default)",
    )
    ap.add_argument(
        "--higher-is-better",
        action="store_false",
        dest="higher_is_worse",
        help="Treat larger hunger values as better",
    )
    ap.add_argument("--top", type=int, default=20, help="Top-N episodes to report")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("ttyrec_metrics.json"),
        help="Output JSON file",
    )
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    args = ap.parse_args()

    dataset = nld.TtyrecDataset(
        args.dataset,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        dbfilename=args.db,
        shuffle=False,
        loop_forever=False,
    )

    per_episode: Dict[int, EpisodeStats] = {}
    meta_cache: Dict[int, Optional[str]] = {}
    action_keypress_counts = Counter()
    total_steps = 0

    for batch in dataset:
        if args.no_hunger:
            try:
                frames_no_hunger = iter_frames_no_hunger(
                    batch, keypress_key=args.keypress_key
                )
            except KeyError as exc:
                raise SystemExit(str(exc)) from exc
            for gameid, keypress, done_flag in frames_no_hunger:
                if gameid == 0:
                    continue
                if gameid not in per_episode:
                    ttyrec = meta_cache.get(gameid)
                    if ttyrec is None:
                        meta = dict(dataset.get_meta(gameid))
                        ttyrec = find_ttyrec_name(meta)
                        meta_cache[gameid] = ttyrec
                    per_episode[gameid] = EpisodeStats(gameid=gameid, ttyrec=ttyrec)
                stats = per_episode[gameid]
                stats.steps += 1
                action_keypress_counts[keypress] += 1
                total_steps += 1
                for action, keycodes in ACTION_KEYCODES.items():
                    if keypress in keycodes:
                        stats.action_counts[action] += 1
                        break
                if done_flag:
                    stats.last_hunger = None
            continue

        try:
            frames = iter_frames(
                batch,
                blstats_key=args.blstats_key,
                hunger_index=args.hunger_index,
                keypress_key=args.keypress_key,
            )
        except KeyError as exc:
            raise SystemExit(
                f"{exc}. If your dataset does not include blstats, use --no-hunger."
            ) from exc

        for gameid, keypress, hunger_value, done_flag in frames:
            if gameid == 0:
                continue
            if gameid not in per_episode:
                ttyrec = meta_cache.get(gameid)
                if ttyrec is None:
                    meta = dict(dataset.get_meta(gameid))
                    ttyrec = find_ttyrec_name(meta)
                    meta_cache[gameid] = ttyrec
                per_episode[gameid] = EpisodeStats(gameid=gameid, ttyrec=ttyrec)
            stats = per_episode[gameid]
            stats.update_hunger(
                hunger_value,
                non_hungry_max=args.non_hungry_max,
                higher_is_worse=args.higher_is_worse,
            )
            action_keypress_counts[keypress] += 1
            total_steps += 1
            for action, keycodes in ACTION_KEYCODES.items():
                if keypress in keycodes:
                    stats.action_counts[action] += 1
                    break

            if done_flag:
                stats.last_hunger = None

    episodes = []
    for stats in per_episode.values():
        non_hungry_rate = stats.non_hungry_steps / stats.steps if stats.steps else 0.0
        episodes.append(
            {
                "gameid": stats.gameid,
                "ttyrec": stats.ttyrec,
                "steps": stats.steps,
                "action_counts": dict(stats.action_counts),
                "hunger_recovery": stats.hunger_recovery,
                "non_hungry_rate": non_hungry_rate,
            }
        )

    def top_by_action(action: str, topn: int) -> list[dict[str, Any]]:
        ranked = sorted(
            episodes,
            key=lambda r: r.get("action_counts", {}).get(action, 0),
            reverse=True,
        )
        return ranked[:topn]

    def top_by_metric(key: str, topn: int) -> list[dict[str, Any]]:
        ranked = sorted(episodes, key=lambda r: r.get(key, 0), reverse=True)
        return ranked[:topn]

    summary = {
        "total_episodes": len(episodes),
        "total_steps": total_steps,
        "action_frequencies": summarize_actions(action_keypress_counts, total_steps),
        "hunger": {
            "non_hungry_max": args.non_hungry_max,
            "higher_is_worse": args.higher_is_worse,
            "available": not args.no_hunger,
            "avg_non_hungry_rate": (
                sum(r["non_hungry_rate"] for r in episodes) / len(episodes)
                if episodes and not args.no_hunger
                else None
            ),
            "avg_hunger_recovery": (
                sum(r["hunger_recovery"] for r in episodes) / len(episodes)
                if episodes and not args.no_hunger
                else None
            ),
        },
    }

    output = {
        "summary": summary,
        "top": {
            "eat": top_by_action("eat", args.top),
            "read": top_by_action("read", args.top),
            "non_hungry_rate": []
            if args.no_hunger
            else top_by_metric("non_hungry_rate", args.top),
            "hunger_recovery": []
            if args.no_hunger
            else top_by_metric("hunger_recovery", args.top),
        },
        "episodes": episodes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if args.csv:
        import csv

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "gameid",
                "ttyrec",
                "steps",
                "eat",
                "read",
                "drop",
                "quaff",
                "zap",
                "hunger_recovery",
                "non_hungry_rate",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in episodes:
                actions = row.get("action_counts", {})
                writer.writerow(
                    {
                        "gameid": row.get("gameid"),
                        "ttyrec": row.get("ttyrec"),
                        "steps": row.get("steps"),
                        "eat": actions.get("eat", 0),
                        "read": actions.get("read", 0),
                        "drop": actions.get("drop", 0),
                        "quaff": actions.get("quaff", 0),
                        "zap": actions.get("zap", 0),
                        "hunger_recovery": row.get("hunger_recovery", 0.0),
                        "non_hungry_rate": row.get("non_hungry_rate", 0.0),
                    }
                )

    print(f"Wrote JSON metrics to {args.output}")
    if args.csv:
        print(f"Wrote CSV metrics to {args.csv}")


if __name__ == "__main__":
    main()
