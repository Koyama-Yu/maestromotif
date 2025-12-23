import os
import json

TARGET_KEYS = {
    "inv_pickups_by_name",
    "inv_pickups_by_class",
    "inv_by_name",
    "inv_by_category",
}

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
