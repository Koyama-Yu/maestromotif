import os
import json
from typing import Optional

TARGET_KEYS = {
    "inv_pickups_by_name",
    "inv_pickups_by_class",
    "inv_by_name",
    "inv_by_category",
}

OCLASS_TO_CATEGORY = {
    0: "unknown",
    1: "weapons",
    2: "armor",
    3: "rings",
    4: "amulets",
    5: "tools",
    6: "comestibles",
    7: "potions",
    8: "scrolls",
    9: "spellbooks",
    10: "wands",
    11: "coins",
    12: "gems",
    13: "boulders",
}

ACTION_CATEGORY_MAP = {
    "eat": {"comestibles"},
    "quaff": {"potions"},
    "read": {"scrolls"},
    "zap": {"wands"},
    "drop": set(OCLASS_TO_CATEGORY.values()),
}
ALLOWED_ACTIONS = set(ACTION_CATEGORY_MAP.keys())

CATEGORY_SYNONYMS = {
    "food": "comestibles",
    "comestible": "comestibles",
    "potion": "potions",
    "scroll": "scrolls",
    "wand": "wands",
}

NAME_CATEGORY_HINTS = {
    "potion": "potions",
    "scroll": "scrolls",
    "wand": "wands",
    "food": "comestibles",
    "corpse": "comestibles",
    "ration": "comestibles",
    "tin": "comestibles",
    "egg": "comestibles",
    "meat": "comestibles",
}


def normalize_category(category: Optional[str]) -> Optional[str]:
    if not category:
        return None
    cat = category.strip().lower()
    return CATEGORY_SYNONYMS.get(cat, cat)


def infer_category_from_name(item_name: str) -> Optional[str]:
    if not item_name:
        return None
    name = item_name.lower()
    for hint, category in NAME_CATEGORY_HINTS.items():
        if hint in name:
            return category
    return None


def extract_item_category(item_name: str, payload: object) -> Optional[str]:
    if isinstance(payload, dict):
        category = payload.get("category")
        if isinstance(category, str):
            return normalize_category(category)
        item_class = payload.get("class")
        if isinstance(item_class, int):
            return OCLASS_TO_CATEGORY.get(item_class, "unknown")
        if isinstance(item_class, str) and item_class.isdigit():
            return OCLASS_TO_CATEGORY.get(int(item_class), "unknown")
    return infer_category_from_name(item_name)


def filter_actions(actions: dict, category: Optional[str]) -> dict:
    filtered = {}
    for action, count in actions.items():
        if action not in ALLOWED_ACTIONS:
            continue
        if action == "drop":
            filtered[action] = count
            continue
        if category in ACTION_CATEGORY_MAP[action]:
            filtered[action] = count
    return filtered


def filter_inv_by_name(payload: dict) -> dict:
    filtered_payload = {}
    for item_name, entry in payload.items():
        if not isinstance(entry, dict):
            filtered_payload[item_name] = entry
            continue
        actions = entry.get("actions")
        if isinstance(actions, dict):
            category = extract_item_category(item_name, entry)
            filtered_actions = filter_actions(actions, category)
            entry = dict(entry)
            if filtered_actions:
                entry["actions"] = filtered_actions
            else:
                entry.pop("actions", None)
        filtered_payload[item_name] = entry
    return filtered_payload


def filter_inv_by_category(payload: dict) -> dict:
    filtered_payload = {}
    for category_name, entry in payload.items():
        if not isinstance(entry, dict):
            filtered_payload[category_name] = entry
            continue
        actions = entry.get("actions")
        if isinstance(actions, dict):
            category = normalize_category(category_name)
            filtered_actions = filter_actions(actions, category)
            entry = dict(entry)
            if filtered_actions:
                entry["actions"] = filtered_actions
            else:
                entry.pop("actions", None)
        filtered_payload[category_name] = entry
    return filtered_payload


def parse_xlog_line(line: str):
    """
    1行の xlogfile をパースして必要なキーだけ抜き出す
    """
    result = {}
    fields = line.strip().split("\t")

    for field in fields:
        if "=" not in field:
            continue
        key, value = field.split("=", 1)

        if key in TARGET_KEYS:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                # JSONとして壊れている場合は None にしておく
                result[key] = None

    if result:
        if isinstance(result.get("inv_by_name"), dict):
            result["inv_by_name"] = filter_inv_by_name(result["inv_by_name"])
        if isinstance(result.get("inv_by_category"), dict):
            result["inv_by_category"] = filter_inv_by_category(result["inv_by_category"])

    return result if result else None


def collect_from_folder(root_dir: str):
    collected = []

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith(".xlogfile"):
                continue

            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    parsed = parse_xlog_line(line)
                    if parsed is None:
                        continue

                    record = {
                        "source_file": fname,
                        "line": i,
                        **parsed
                    }
                    collected.append(record)

    return collected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="xlogfile が入っているフォルダ")
    parser.add_argument("-o", "--output", default="inventory_logs.json")
    args = parser.parse_args()

    data = collect_from_folder(args.input_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"{len(data)} records written to {args.output}")
