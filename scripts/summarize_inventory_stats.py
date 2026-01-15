#!/usr/bin/env python3
"""Aggregate inventory statistics JSON files and print top-N rankings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_records(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def iter_field(records: Iterable[Dict], key: str) -> Iterable[Dict]:
    for record in records:
        if not isinstance(record, dict):
            continue
        payload = record.get(key)
        if isinstance(payload, dict):
            yield payload


def to_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def aggregate_simple_counters(
    records: Iterable[Dict[str, int]], normalize: bool = False
) -> Counter:
    counter: Counter = Counter()
    for entry in records:
        for name, count in entry.items():
            key = normalize_item_name(name) if normalize else name
            counter[key] += to_int(count)
    return counter


def aggregate_acquired(records: Iterable[Dict[str, dict]], normalize: bool = False) -> Counter:
    counter: Counter = Counter()
    for entry in records:
        for name, payload in entry.items():
            key = normalize_item_name(name) if normalize else name
            if isinstance(payload, dict):
                counter[key] += to_int(payload.get("acquired", 0) or 0)
            elif isinstance(payload, (int, float)):
                counter[key] += to_int(payload)
    return counter


def aggregate_usage(
    records: Iterable[Dict[str, dict]], normalize: bool = False
) -> Tuple[Counter, Dict[str, Counter]]:
    overall = Counter()
    per_action: Dict[str, Counter] = defaultdict(Counter)
    for entry in records:
        for name, payload in entry.items():
            key = normalize_item_name(name) if normalize else name
            actions = payload.get("actions", {}) if isinstance(payload, dict) else {}
            for action, count in actions.items():
                amount = to_int(count)
                if amount <= 0:
                    continue
                overall[key] += amount
                per_action[action][key] += amount
    return overall, per_action


def compute_effective_usage(overall: Counter, drop_counter: Counter) -> Counter:
    effective = Counter()
    for item in set(overall) | set(drop_counter):
        effective[item] = overall.get(item, 0) - drop_counter.get(item, 0)
    return effective


def compute_ratio(numer: Counter, denom: Counter, scale: float = 1.0) -> Counter:
    ratio = Counter()
    for item, num in numer.items():
        den = denom.get(item, 0)
        if den > 0:
            ratio[item] = (num / den) * scale
    return ratio


def clamp_to_denom(numer: Counter, denom: Counter) -> Counter:
    clamped = Counter()
    for item, num in numer.items():
        clamped[item] = min(num, denom.get(item, 0))
    return clamped


def normalize_item_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""

    s = name.strip()
    s = re.sub(r"^(?:\d+[\s-]+|an?\s+)", "", s, flags=re.IGNORECASE)

    def singular(token: str) -> str:
        lower = token.lower()
        if lower.endswith("ies"):
            return token[:-1]
        if lower.endswith("s") and not lower.endswith("ss") and len(token) > 2:
            return token[:-1]
        return token

    if "gold pieces" in s:
        s = s.replace("gold pieces", "gold piece")

    parts = s.split()
    drop_tokens = {
        "corroded",
        "rusty",
        "very",
        "thoroughly",
        "burnt",
        "blessed",
        "uncursed",
        "cursed",
    }
    parts = [p for p in parts if p.lower() not in drop_tokens]

    for i in range(len(parts) - 1):
        if parts[i + 1].lower() == "of":
            parts[i] = singular(parts[i])

    if parts:
        parts[-1] = singular(parts[-1])
    return " ".join(parts)


def top_entries(counter: Counter, top_n: int) -> List[Tuple[str, float]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def counter_to_sorted_list(counter: Counter) -> List[Dict[str, float]]:
    return [{"name": name, "count": count} for name, count in top_entries(counter, len(counter))]


def print_section(
    title: str,
    counter: Counter,
    top_n: int,
    formatter=str,
) -> List[Tuple[str, float]]:
    print(title)
    if not counter:
        print("  (no data)")
        print()
        return []

    entries = top_entries(counter, top_n)
    for idx, (name, count) in enumerate(entries, start=1):
        print(f" {idx:2d}. {name} ({formatter(count)})")
    print()
    return entries


def print_usage_sections(
    title_prefix: str,
    overall: Counter,
    per_action: Dict[str, Counter],
    top_n: int,
    actions: Optional[List[str]],
) -> List[Tuple[str, List[Tuple[str, float]]]]:
    sections = []
    sections.append((
        f"{title_prefix} (overall usage)",
        print_section(f"{title_prefix} (overall usage)", overall, top_n),
    ))

    if actions:
        for action in actions:
            counter = per_action.get(action, Counter())
            title = f"{title_prefix} (action: {action})"
            sections.append((title, print_section(title, counter, top_n)))
    else:
        for action, counter in sorted(
            per_action.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])
        ):
            title = f"{title_prefix} (action: {action})"
            sections.append((title, print_section(title, counter, top_n)))
    return sections


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", title.strip().lower())
    return slug.strip("_") or "section"


def write_csv(path: Path, entries: List[Tuple[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "name", "count"])
        for idx, (name, count) in enumerate(entries, start=1):
            writer.writerow([idx, name, count])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank frequent items from NetHack inventory data."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to inventory JSON (e.g. inventory_summary.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top entries to show for each attribute (default: 10).",
    )
    parser.add_argument(
        "--actions",
        nargs="*",
        default=None,
        help="Optional action names to display (default: show all actions).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Optional directory to export each ranking as CSV (created if missing)",
    )
    parser.add_argument(
        "--output",
        help="Path to write full rankings as JSON (all items, not truncated).",
    )
    parser.add_argument(
        "--output-effective",
        help="Path to write effective usage rankings as JSON (all items).",
    )
    args = parser.parse_args()

    records = load_records(args.json_path)

    pickups_by_name = aggregate_simple_counters(
        iter_field(records, "inv_pickups_by_name")
    )
    pickups_by_class = aggregate_simple_counters(
        iter_field(records, "inv_pickups_by_class")
    )
    inv_by_name = aggregate_acquired(iter_field(records, "inv_by_name"))
    inv_by_category = aggregate_acquired(iter_field(records, "inv_by_category"))
    usage_overall, usage_per_action = aggregate_usage(iter_field(records, "inv_by_name"))
    effective_usage = compute_effective_usage(
        usage_overall, usage_per_action.get("drop", Counter())
    )
    effective_usage_clamped = clamp_to_denom(effective_usage, inv_by_name)
    ratio_eff_per_acquired = compute_ratio(effective_usage_clamped, inv_by_name)
    ratio_eff_per_usage = compute_ratio(effective_usage, usage_overall, scale=100.0)
    ratio_eff_per_acquired_pct = compute_ratio(
        effective_usage_clamped, inv_by_name, scale=100.0
    )

    usage_overall_cat, usage_per_action_cat = aggregate_usage(
        iter_field(records, "inv_by_category")
    )
    effective_usage_cat = compute_effective_usage(
        usage_overall_cat, usage_per_action_cat.get("drop", Counter())
    )
    effective_usage_cat_clamped = clamp_to_denom(effective_usage_cat, inv_by_category)
    ratio_eff_cat_per_acquired = compute_ratio(
        effective_usage_cat_clamped, inv_by_category
    )
    ratio_eff_cat_per_usage = compute_ratio(
        effective_usage_cat, usage_overall_cat, scale=100.0
    )
    ratio_eff_cat_per_acquired_pct = compute_ratio(
        effective_usage_cat_clamped, inv_by_category, scale=100.0
    )

    pickups_by_name_norm = aggregate_simple_counters(
        iter_field(records, "inv_pickups_by_name"), normalize=True
    )
    inv_by_name_norm = aggregate_acquired(
        iter_field(records, "inv_by_name"), normalize=True
    )
    usage_overall_norm, usage_per_action_norm = aggregate_usage(
        iter_field(records, "inv_by_name"), normalize=True
    )
    effective_usage_norm = compute_effective_usage(
        usage_overall_norm, usage_per_action_norm.get("drop", Counter())
    )
    effective_usage_norm_clamped = clamp_to_denom(
        effective_usage_norm, inv_by_name_norm
    )
    ratio_eff_per_acquired_norm = compute_ratio(
        effective_usage_norm_clamped, inv_by_name_norm
    )
    ratio_eff_per_usage_norm = compute_ratio(
        effective_usage_norm, usage_overall_norm, scale=100.0
    )
    ratio_eff_per_acquired_pct_norm = compute_ratio(
        effective_usage_norm_clamped, inv_by_name_norm, scale=100.0
    )

    usage_overall_cat_norm, usage_per_action_cat_norm = aggregate_usage(
        iter_field(records, "inv_by_category"), normalize=True
    )
    effective_usage_cat_norm = compute_effective_usage(
        usage_overall_cat_norm, usage_per_action_cat_norm.get("drop", Counter())
    )
    effective_usage_cat_norm_clamped = clamp_to_denom(
        effective_usage_cat_norm, inv_by_category
    )
    ratio_eff_cat_per_acquired_norm = compute_ratio(
        effective_usage_cat_norm_clamped, inv_by_category
    )
    ratio_eff_cat_per_usage_norm = compute_ratio(
        effective_usage_cat_norm, usage_overall_cat_norm, scale=100.0
    )
    ratio_eff_cat_per_acquired_pct_norm = compute_ratio(
        effective_usage_cat_norm_clamped, inv_by_category, scale=100.0
    )

    sections: List[Tuple[str, List[Tuple[str, float]]]] = []

    def add_section(
        title: str, counter: Counter, top_n: int, formatter=str
    ) -> None:
        entries = print_section(title, counter, top_n, formatter)
        sections.append((title, entries))

    add_section(f"Top {args.top} inv_pickups_by_name (raw)", pickups_by_name, args.top)
    add_section(f"Top {args.top} inv_pickups_by_class", pickups_by_class, args.top)
    add_section(
        f"Top {args.top} inv_by_name (acquired, raw)", inv_by_name, args.top
    )
    add_section(
        f"Top {args.top} inv_by_category (acquired, raw)", inv_by_category, args.top
    )
    sections.extend(
        print_usage_sections(
            f"Top {args.top} inv_by_name (usage, raw)",
            usage_overall,
            usage_per_action,
            args.top,
            args.actions,
        )
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage = usage - drop, raw)",
        effective_usage,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage per acquired, raw)",
        ratio_eff_per_acquired,
        args.top,
        formatter=lambda c: f"{c:.3f}",
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage % of acquired, raw)",
        ratio_eff_per_acquired_pct,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage % of total usage, raw)",
        ratio_eff_per_usage,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_category (usage, raw)",
        usage_overall_cat,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage = usage - drop, raw)",
        effective_usage_cat,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage per acquired, raw)",
        ratio_eff_cat_per_acquired,
        args.top,
        formatter=lambda c: f"{c:.3f}",
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage % of acquired, raw)",
        ratio_eff_cat_per_acquired_pct,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage % of total usage, raw)",
        ratio_eff_cat_per_usage,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )

    add_section(
        f"Top {args.top} inv_pickups_by_name (normalized)",
        pickups_by_name_norm,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_name (acquired, normalized)",
        inv_by_name_norm,
        args.top,
    )
    sections.extend(
        print_usage_sections(
            f"Top {args.top} inv_by_name (usage, normalized)",
            usage_overall_norm,
            usage_per_action_norm,
            args.top,
            args.actions,
        )
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage = usage - drop, normalized)",
        effective_usage_norm,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage per acquired, normalized)",
        ratio_eff_per_acquired_norm,
        args.top,
        formatter=lambda c: f"{c:.3f}",
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage % of acquired, normalized)",
        ratio_eff_per_acquired_pct_norm,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_name (effective usage % of total usage, normalized)",
        ratio_eff_per_usage_norm,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_category (usage, normalized)",
        usage_overall_cat_norm,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage = usage - drop, normalized)",
        effective_usage_cat_norm,
        args.top,
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage per acquired, normalized)",
        ratio_eff_cat_per_acquired_norm,
        args.top,
        formatter=lambda c: f"{c:.3f}",
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage % of acquired, normalized)",
        ratio_eff_cat_per_acquired_pct_norm,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )
    add_section(
        f"Top {args.top} inv_by_category (effective usage % of total usage, normalized)",
        ratio_eff_cat_per_usage_norm,
        args.top,
        formatter=lambda c: f"{c:.1f}%",
    )

    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        for title, entries in sections:
            path = args.csv_dir / f"{slugify(title)}.csv"
            write_csv(path, entries)

    if args.output:
        results = {
            "raw": {
                "inv_pickups_by_name": counter_to_sorted_list(pickups_by_name),
                "inv_pickups_by_class": counter_to_sorted_list(pickups_by_class),
                "inv_by_name_acquired": counter_to_sorted_list(inv_by_name),
                "inv_by_category_acquired": counter_to_sorted_list(inv_by_category),
                "effective_usage": counter_to_sorted_list(effective_usage),
                "effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired
                ),
                "effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_pct
                ),
                "effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_per_usage
                ),
                "category_usage_overall": counter_to_sorted_list(usage_overall_cat),
                "category_effective_usage": counter_to_sorted_list(effective_usage_cat),
                "category_effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_cat_per_acquired
                ),
                "category_effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_cat_per_acquired_pct
                ),
                "category_effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_cat_per_usage
                ),
                "usage_overall": counter_to_sorted_list(usage_overall),
                "usage_per_action": {
                    action: counter_to_sorted_list(counter)
                    for action, counter in usage_per_action.items()
                },
            },
            "normalized": {
                "inv_pickups_by_name": counter_to_sorted_list(pickups_by_name_norm),
                "inv_by_name_acquired": counter_to_sorted_list(inv_by_name_norm),
                "effective_usage": counter_to_sorted_list(effective_usage_norm),
                "effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_norm
                ),
                "effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_pct_norm
                ),
                "effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_per_usage_norm
                ),
                "category_usage_overall": counter_to_sorted_list(usage_overall_cat_norm),
                "category_effective_usage": counter_to_sorted_list(effective_usage_cat_norm),
                "category_effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_cat_per_acquired_norm
                ),
                "category_effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_cat_per_acquired_pct_norm
                ),
                "category_effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_cat_per_usage_norm
                ),
                "usage_overall": counter_to_sorted_list(usage_overall_norm),
                "usage_per_action": {
                    action: counter_to_sorted_list(counter)
                    for action, counter in usage_per_action_norm.items()
                },
            },
        }
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nFull rankings written to {args.output}")

    if args.output_effective:
        effective = {
            "raw": {
                "effective_usage": counter_to_sorted_list(effective_usage),
                "effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired
                ),
                "effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_pct
                ),
                "effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_per_usage
                ),
            },
            "normalized": {
                "effective_usage": counter_to_sorted_list(effective_usage_norm),
                "effective_usage_per_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_norm
                ),
                "effective_usage_pct_of_acquired": counter_to_sorted_list(
                    ratio_eff_per_acquired_pct_norm
                ),
                "effective_usage_pct_of_usage": counter_to_sorted_list(
                    ratio_eff_per_usage_norm
                ),
            },
        }
        Path(args.output_effective).write_text(
            json.dumps(effective, ensure_ascii=False, indent=2)
        )
        print(f"Effective rankings written to {args.output_effective}")


if __name__ == "__main__":
    main()
