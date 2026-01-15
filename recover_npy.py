import argparse
import re
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Prompt log file path")
    parser.add_argument("--content_filter_log", default=None,
                        help="Content filter log file path")
    parser.add_argument("--out", required=True, help="Output .npy file path")
    parser.add_argument("--length", type=int, default=None,
                        help="Optional fixed length for output array")
    parser.add_argument("--base_npy", default=None,
                        help="Optional base .npy to initialize array")
    parser.add_argument("--base_csv", default=None,
                        help="Optional base .csv to initialize array")
    parser.add_argument("--min_index", type=int, default=0,
                        help="Minimum index to overwrite from the log")
    parser.add_argument("--content_filter_indices", default=None,
                        help="Comma-separated indices to mark as content filter")
    return parser.parse_args()


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()
    index_re = re.compile(r"^Index:(\d+)", re.MULTILINE)
    assistant_re = re.compile(r"assistant\s*:", re.IGNORECASE)
    label_re = re.compile(r"best_description\"?\s*:\s*(1|2|None)", re.IGNORECASE)

    text = load_text(args.log)
    matches = list(index_re.finditer(text))
    if not matches:
        raise SystemExit("no indices found in log")

    labels_by_index = {}
    content_filtered = set()
    if args.content_filter_indices:
        content_filtered = {int(x) for x in args.content_filter_indices.split(",") if x.strip()}
    max_idx = -1
    for i, match in enumerate(matches):
        idx = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        assistant_matches = list(assistant_re.finditer(block))
        if assistant_matches:
            last_assistant = assistant_matches[-1]
            assistant_text = block[last_assistant.end():]
            label_matches = label_re.findall(assistant_text)
            if label_matches:
                lab = label_matches[-1]
                if lab.lower() == "none":
                    val = 2  # TIE
                elif lab == "1":
                    val = 0  # FIRST
                else:
                    val = 1  # SECOND
                labels_by_index[idx] = val
        if idx > max_idx:
            max_idx = idx

    base = None
    if args.base_npy:
        base = np.load(args.base_npy).astype(np.int32)
    elif args.base_csv:
        base = np.loadtxt(args.base_csv, delimiter=",").astype(np.int32)
        if base.ndim > 1:
            base = base.flatten()

    if args.length is not None:
        out_len = args.length
    elif base is not None:
        out_len = len(base)
    else:
        out_len = max_idx + 1
    if out_len <= 0:
        raise SystemExit("no indices found in log")

    arr = np.full((out_len,), 3, dtype=np.int32)  # UNKOWN=3
    if base is not None:
        arr[:min(len(base), out_len)] = base[:min(len(base), out_len)]
    for i, v in labels_by_index.items():
        if i < args.min_index:
            continue
        if i < out_len:
            arr[i] = v
    for i in content_filtered:
        if i < args.min_index:
            continue
        if i < out_len:
            arr[i] = 4  # CONTENT_FILTER=4

    if args.content_filter_log:
        try:
            cf_text = load_text(args.content_filter_log)
            cf_indices = [int(m.group(1)) for m in index_re.finditer(cf_text)]
            for i in cf_indices:
                if i < args.min_index:
                    continue
                if i < out_len:
                    arr[i] = 4  # CONTENT_FILTER=4
        except FileNotFoundError:
            pass

    np.save(args.out, arr)
    unique, counts = np.unique(arr, return_counts=True)
    print("saved", args.out, "shape", arr.shape)
    print("value_counts", dict(zip(unique.tolist(), counts.tolist())))


if __name__ == "__main__":
    main()
