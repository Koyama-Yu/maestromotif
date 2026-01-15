import argparse
import os
import re
import statistics
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

SCORE_KEYS = ("points", "score")
TURN_KEYS = ("turns", "turn")
MAX_DEPTH_KEYS = ("maxlvl", "max_depth", "maxdepth", "depth")
END_REASON_KEYS = ("death", "end_reason", "endreason")


def parse_number(value: str) -> Optional[float]:
    try:
        return float(int(value))
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None


def get_first_number(fields: Dict[str, str], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in fields:
            return parse_number(fields[key])
    return None


def parse_xlog_line(line: str) -> Optional[Tuple[float, float, float, Optional[str]]]:
    """
    1行の xlogfile をパースして必要な統計キーだけ抜き出す
    """
    fields: Dict[str, str] = {}
    for token in line.strip().split("\t"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value

    score = get_first_number(fields, SCORE_KEYS)
    turns = get_first_number(fields, TURN_KEYS)
    max_depth = get_first_number(fields, MAX_DEPTH_KEYS)
    end_reason = fields.get(END_REASON_KEYS[0])
    if end_reason is None:
        for key in END_REASON_KEYS[1:]:
            if key in fields:
                end_reason = fields[key]
                break

    if score is None or turns is None or max_depth is None:
        return None

    return score, turns, max_depth, end_reason


def collect_from_path(
    input_path: str,
) -> Tuple[str, List[float], List[float], List[float], List[str]]:
    scores: List[float] = []
    turns_list: List[float] = []
    max_depths: List[float] = []
    end_reasons: List[str] = []

    if os.path.isfile(input_path):
        source = os.path.basename(input_path)
        paths = [input_path]
    else:
        source = os.path.basename(os.path.abspath(input_path))
        paths = []
        for root, _, files in os.walk(input_path):
            for fname in files:
                if fname.endswith(".xlogfile"):
                    paths.append(os.path.join(root, fname))

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_xlog_line(line)
                if parsed is None:
                    continue
                score, turns, max_depth, end_reason = parsed
                scores.append(score)
                turns_list.append(turns)
                max_depths.append(max_depth)
                if end_reason:
                    end_reasons.append(end_reason)

    return source, scores, turns_list, max_depths, end_reasons


def summarize(values: List[float]) -> Tuple[float, float, float]:
    mean_value = statistics.mean(values)
    median_value = statistics.median(values)
    std_value = statistics.pstdev(values)
    return mean_value, median_value, std_value


def normalize_end_reason(reason: str) -> Tuple[str, bool]:
    """Normalize similar end reasons."""
    text = reason.lower().strip().rstrip(".")

    if "timeout" in text:
        return "timeout", False
    if "quit" in text:
        return "aborted: quit", False
    if "starved to death" in text:
        return "starved to death", False
    if "petrified" in text or "turned to stone" in text:
        return "petrified", False
    if "choked" in text:
        return "choked", False
    if "poisoned by a rotted" in text:
        return "death: was poisoned. poisoned by a rotted corpse", False

    killed_match = re.search(r"killed by ([^.,]+)", text)
    if killed_match:
        subject = killed_match.group(1).strip()
        subject = re.sub(r"^(an|a|the) ", "", subject)

        keyword_map = [
            ("bolt of", "killed by bolt"),
            ("death ray", "killed by death ray"),
            ("magic missile", "killed by magic missile"),
            ("wand", "killed by wand"),
            ("trap", "killed by trap"),
            ("boulder", "killed by boulder"),
            ("rock", "killed by boulder"),
            ("poison gas", "killed by poison gas"),
            ("drawbridge", "killed by drawbridge"),
            ("falling", "killed by falling"),
            ("poison", "killed by poison"),
        ]
        for keyword, label in keyword_map:
            if keyword in subject:
                return label, False

        return "killed by monster", True

    return text, False


def print_ranking(counts: Counter, title: str) -> None:
    total = sum(counts.values())
    print(f"\n{title}:")
    try:
        for rank, (reason, count) in enumerate(counts.most_common(10), start=1):
            if total == 0:
                break
            percent = (count / total) * 100
            print(f"{rank:3d}. {reason:<40} {percent:6.2f}%")
    except BrokenPipeError:
        sys.exit(0)


def summarize_end_reasons(reasons: Iterable[str]) -> None:
    normalized = [normalize_end_reason(r) for r in reasons]

    def is_game_result(label: str) -> bool:
        return label not in {"timeout", "aborted: quit"}

    overall = Counter(label for label, _ in normalized if is_game_result(label))
    monster = Counter(
        label for label, is_monster in normalized if is_monster and is_game_result(label)
    )
    non_monster = Counter(
        label
        for label, is_monster in normalized
        if not is_monster and is_game_result(label)
    )
    raw = Counter(
        r
        for r, (label, _) in zip(reasons, normalized)
        if is_game_result(label)
    )

    print_ranking(overall, "End reason ranking (normalized)")
    if monster:
        print_ranking(monster, "Normalized (monsters only)")
    if non_monster:
        print_ranking(non_monster, "Normalized (non-monster)")
    print_ranking(raw, "End reason ranking (raw, unnormalized)")


def print_summary(
    source: str,
    scores: List[float],
    turns_list: List[float],
    max_depths: List[float],
    end_reasons: List[str],
) -> None:
    if not scores:
        raise SystemExit("No valid records found.")

    score_mean, score_median, score_std = summarize(scores)
    turn_mean, turn_median, turn_std = summarize(turns_list)
    depth_mean, depth_median, depth_std = summarize(max_depths)

    print(f"Source: {source}")
    print(f"score_mean    : {score_mean:.1f} +/- {score_std:.1f}")
    print(f"score_median  : {score_median:.1f}")
    print(f"turn_mean     : {turn_mean:.1f} +/- {turn_std:.1f}")
    print(f"turn_median   : {turn_median:.1f}")
    print(f"max_depth_mean    : {depth_mean:.1f} +/- {depth_std:.1f}")
    print(f"max_depth_median  : {depth_median:.1f}")
    if end_reasons:
        summarize_end_reasons(end_reasons)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="xlogfile が入っているフォルダ、または xlogfile 自体")
    args = parser.parse_args(argv)

    source, scores, turns_list, max_depths, end_reasons = collect_from_path(
        args.input_path
    )
    print_summary(source, scores, turns_list, max_depths, end_reasons)


if __name__ == "__main__":
    main()
