#!/usr/bin/env python3
"""Aggregate inventory statistics JSON files and print top-N rankings."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Item-usage actions that are typically meaningful for inventory analysis.
DEFAULT_ACTIONS = [
    "drop",
    "throw",
    "eat",
    "quaff",
    "read",
    "zap",
    "wear",
    "wield",
    "apply",
    "puton",
    "takeoff",
    "remove",
    "equip",
    "invoke",
    "engrave",
]


EQUIPMENT_PATTERNS: Iterable[str] = (
    r"\s*\(being worn\)\s*",
    r"\s*\(wielded\)\s*",
    r"\s*\(weapon in hand\)\s*",
    r"\s*\(alternate weapon\)\s*",
    r"\s*\(on left hand\)\s*",
    r"\s*\(on right hand\)\s*",
    r"\s*\(in quiver\)\s*",
    r"\s*\(embedded in your skin\)\s*",
)

PAREN_COLON = re.compile(r"\s*\(\d+:\d+\)\s*")
ENCHANT_RE = re.compile(r"\s*[+-]\d+\s+")

IRREGULAR_PLURALS: Dict[str, str] = {
    "axes": "axe",
    "knives": "knife",
    "leaves": "leaf",
    "wolves": "wolf",
    "loaves": "loaf",
    "lives": "life",
    "elves": "elf",
    "dwarves": "dwarf",
    "scarves": "scarf",
    "staves": "staff",
    "teeth": "tooth",
    "feet": "foot",
    "geese": "goose",
    "mice": "mouse",
    "dice": "die",
    "thieves": "thief",
    "wives": "wife",
}

NO_SINGULARIZE = {
    "boots",
    "gauntlets",
    "gloves",
    "lenses",
    "glasses",
    "scales",
    "tricks",
    "pants",
    "trousers",
    "shorts",
    "sandals",
    "shoes",
    "series",
    "species",
}


def singularize_word(word: str) -> str:
    """Best-effort singularization for NetHack style item names."""
    if not word or word in NO_SINGULARIZE:
        return word
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if "-" in word:
        parts = word.split("-")
        parts[-1] = singularize_word(parts[-1])
        return "-".join(parts)
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 3:
        return word[:-3] + "f"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")) and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_item_name(name: str) -> str:
    """Normalize item names so stacks collapse to the same key."""
    if not name or not isinstance(name, str):
        return ""

    text = name.strip().lower()
    for pattern in EQUIPMENT_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = PAREN_COLON.sub(" ", text)
    text = ENCHANT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    suffixes = []
    while text.endswith(")"):
        idx = text.rfind("(")
        if idx == -1:
            break
        suffixes.append(text[idx:].strip())
        text = text[:idx].strip()
    suffix = " ".join(reversed(suffixes)).strip()

    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"^some\s+", "", text)
    text = re.sub(r"^(?:an?|the)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text.startswith(("pair of ", "set of ")):
        base = text
    else:
        words = text.split()
        for idx, token in enumerate(words):
            if token == "of":
                continue
            words[idx] = singularize_word(token)
        base = " ".join(words)

    result = base.strip()
    if suffix:
        result = f"{result} {suffix}".strip()
    return result


class InventoryAggregator:
    def __init__(self) -> None:
        self.pickups_by_name = Counter()
        self.pickups_by_name_norm = Counter()
        self.pickups_by_class = Counter()
        self.acquired_by_name = Counter()
        self.acquired_by_name_norm = Counter()
        self.acquired_by_category = Counter()
        self.usage_by_name = Counter()
        self.usage_norm_by_name = Counter()
        self.usage_by_action: Dict[str, Counter] = defaultdict(Counter)
        self.usage_norm_by_action: Dict[str, Counter] = defaultdict(Counter)

    def ingest(self, record: Dict) -> None:
        for name, count in (record.get("inv_pickups_by_name") or {}).items():
            amount = int(count)
            self.pickups_by_name[name] += amount
            norm = normalize_item_name(name)
            if norm:
                self.pickups_by_name_norm[norm] += amount

        for cls, count in (record.get("inv_pickups_by_class") or {}).items():
            self.pickups_by_class[cls] += int(count)

        for name, payload in (record.get("inv_by_name") or {}).items():
            acquired = int(payload.get("acquired", 0))
            self.acquired_by_name[name] += acquired
            norm = normalize_item_name(name)
            if norm:
                self.acquired_by_name_norm[norm] += acquired

            actions = payload.get("actions") or {}
            total_usage = 0
            for action, count in actions.items():
                amount = int(count)
                if amount <= 0:
                    continue
                total_usage += amount
                self.usage_by_action[action][name] += amount
                if norm:
                    self.usage_norm_by_action[action][norm] += amount
            if total_usage:
                self.usage_by_name[name] += total_usage
                if norm:
                    self.usage_norm_by_name[norm] += total_usage

        for cat, payload in (record.get("inv_by_category") or {}).items():
            self.acquired_by_category[cat] += int(payload.get("acquired", 0))

    def finalize(self, records: Iterable[Dict]) -> None:
        for rec in records:
            if isinstance(rec, dict):
                self.ingest(rec)


def load_records(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def get_top_entries(counter: Counter, topn: int) -> List[Tuple[str, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:topn]


def print_top(title: str, entries: List[Tuple[str, int]], topn: int) -> None:
    print(f"Top {topn} {title}")
    if not entries:
        print("  (no data)")
    else:
        for idx, (name, count) in enumerate(entries, start=1):
            print(f"  {idx:2d}. {name} ({count})")
    print()


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", title.strip().lower())
    return slug.strip("_") or "section"


def write_csv(path: Path, entries: List[Tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "name", "count"])
        for idx, (name, count) in enumerate(entries, start=1):
            writer.writerow([idx, name, count])


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank inventory statistics stored in JSON.")
    ap.add_argument("json_path", type=Path, help="Path to inventory JSON (e.g. result_test.json)")
    ap.add_argument("--top", type=int, default=10, help="How many entries to print per section")
    ap.add_argument(
        "--actions",
        nargs="*",
        default=list(DEFAULT_ACTIONS),
        help="Action names to display (default: common item-usage actions)",
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        help="Optional directory to also export each ranking as CSV (created if missing)",
    )
    args = ap.parse_args()

    records = load_records(args.json_path)
    agg = InventoryAggregator()
    agg.finalize(records)

    sections: List[Tuple[str, List[Tuple[str, int]]]] = []

    def add_section(title: str, counter: Counter) -> None:
        entries = get_top_entries(counter, args.top)
        sections.append((title, entries))
        print_top(title, entries, args.top)

    add_section("inv_pickups_by_name (raw)", agg.pickups_by_name)
    add_section("inv_pickups_by_class", agg.pickups_by_class)
    add_section("inv_by_name (acquired, raw)", agg.acquired_by_name)
    add_section("inv_by_category (acquired, raw)", agg.acquired_by_category)
    add_section("inv_by_name (usage, raw) (overall usage)", agg.usage_by_name)
    for action in args.actions:
        title = f"inv_by_name (usage, raw) (action: {action})"
        add_section(title, agg.usage_by_action.get(action, Counter()))

    add_section("inv_pickups_by_name (normalized)", agg.pickups_by_name_norm)
    add_section("inv_by_name (acquired, normalized)", agg.acquired_by_name_norm)
    add_section("inv_by_name (usage, normalized) (overall usage)", agg.usage_norm_by_name)
    for action in args.actions:
        title = f"inv_by_name (usage, normalized) (action: {action})"
        add_section(title, agg.usage_norm_by_action.get(action, Counter()))

    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        for title, entries in sections:
            path = args.csv_dir / f"{slugify(title)}.csv"
            write_csv(path, entries)


if __name__ == "__main__":
    main()
