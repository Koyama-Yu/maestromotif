#!/usr/bin/env python3
"""
Extract inventory-related fields from NetHack xlogfiles stored under a ttyrec directory.

The parser copies the JSON payloads for the relevant inventory fields directly from
each log line, so very large datasets can be processed without repeatedly reparsing
each nested JSON blob.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

FIELD_KEYS: List[str] = [
    "inv_pickups_by_name",
    "inv_pickups_by_class",
    "inv_by_name",
    "inv_by_category",
]


def iter_xlogfiles(root: Path, *, file_list: Sequence[Path] | None) -> Iterator[Path]:
    """
    Yield .xlogfile paths under the given directory.

    Providing a precomputed file_list avoids the expensive directory scan when the
    ttyrec folder holds hundreds of thousands of ttyrecording files.
    """
    if file_list is not None:
        for path in file_list:
            yield path
        return
    yield from sorted(root.glob("*.xlogfile"))


def parse_line(
    line: str, *, include_ttyrec: bool = True
) -> Dict[str, str] | None:
    """
    Extract the raw JSON payloads for the requested inventory fields from a log line.

    Returns None when any of the required fields are missing on that line.
    """
    values: Dict[str, str] = {}
    for token in line.rstrip().split("\t"):
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        if key in FIELD_KEYS:
            values[key] = raw_value
        elif include_ttyrec and key == "ttyrecname":
            values[key] = raw_value
    if all(field in values for field in FIELD_KEYS):
        return values
    return None


def build_entry_json(
    *, source_logfile: str, line_number: int, values: Dict[str, str]
) -> str:
    """
    Assemble a JSON object string using the metadata plus copied inventory payloads.

    The metadata values are encoded with json.dumps for safety, while the existing
    inventory blobs are inserted directly as they already contain valid JSON.
    """
    parts = [
        '"source_logfile":' + json.dumps(source_logfile),
        '"line_number":' + str(line_number),
    ]
    if "ttyrecname" in values:
        parts.append('"ttyrecname":' + json.dumps(values["ttyrecname"]))
    for field in FIELD_KEYS:
        parts.append(f'"{field}":' + values[field])
    return "{" + ",".join(parts) + "}"


def extract_inventory(
    ttyrec_dir: Path,
    output_path: Path,
    *,
    file_list: Sequence[Path] | None = None,
    verbose: bool = True,
) -> None:
    """Stream the extracted entries into a JSON array stored at output_path."""
    entries_written = 0
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("[\n")
        first = True
        for log_path in iter_xlogfiles(ttyrec_dir, file_list=file_list):
            rel_path = log_path.relative_to(ttyrec_dir)
            with log_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    parsed = parse_line(line)
                    if parsed is None:
                        continue
                    entry_json = build_entry_json(
                        source_logfile=str(rel_path),
                        line_number=line_number,
                        values=parsed,
                    )
                    if not first:
                        outfile.write(",\n")
                    outfile.write(entry_json)
                    first = False
                    entries_written += 1
        outfile.write("\n]\n")
    if verbose:
        print(
            f"Wrote {entries_written} entries to {output_path} "
            f"from directory {ttyrec_dir}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract inventory-related JSON blobs from NetHack xlogfiles under a "
            "ttyrec directory."
        )
    )
    parser.add_argument(
        "ttyrec_dir",
        type=Path,
        help="Directory containing .xlogfile logs (e.g., .../ttyrecs/A)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file path. Defaults to ttyrec_dir/inventory_summary.json "
            "if omitted."
        ),
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited file listing .xlogfile paths. "
            "Use this when the directory contains so many files that globbing is slow."
        ),
    )
    args = parser.parse_args(argv)
    ttyrec_dir = args.ttyrec_dir.resolve()
    if not ttyrec_dir.exists():
        raise SystemExit(f"Directory not found: {ttyrec_dir}")
    output_path = (
        args.output.resolve() if args.output else ttyrec_dir / "inventory_summary.json"
    )
    file_list_paths: Sequence[Path] | None = None
    if args.file_list:
        with args.file_list.open("r", encoding="utf-8") as handle:
            file_list_paths = []
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                raw_path = Path(raw)
                if raw_path.is_absolute():
                    candidate = raw_path
                else:
                    candidate = Path.cwd() / raw_path
                    if not candidate.exists():
                        candidate = ttyrec_dir / raw_path
                if not candidate.exists():
                    raise FileNotFoundError(f"Listed file not found: {raw}")
                file_list_paths.append(candidate)
    extract_inventory(ttyrec_dir, output_path, file_list=file_list_paths)


if __name__ == "__main__":
    main()
