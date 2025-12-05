#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List, Optional

import nle.dataset as nld

DEFAULT_DB = "ttyrecs_test.db"

def ensure_db(dbfilename: str):
    if not nld.db.exists(dbfilename):
        nld.db.create(dbfilename)

def register_directory(dir_path: str, name: str, dbfilename: str, fmt: str = "nle"):
    ensure_db(dbfilename)
    if fmt == "nle":
        nld.add_nledata_directory(dir_path, name, dbfilename)
    elif fmt == "altorg":
        nld.add_altorg_directory(dir_path, name, dbfilename)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    print(f"Registered dataset '{name}' from '{dir_path}' into '{dbfilename}'")

def export_metadata(
    dataset_name: str,
    dbfilename: str,
    out_csv: str,
    subselect_sql: Optional[str] = None,
    subselect_sql_args: Optional[List[str]] = None,
):
    dataset = nld.TtyrecDataset(
        dataset_name,
        batch_size=1,
        seq_length=1,
        dbfilename=dbfilename,
        shuffle=False,
        loop_forever=False,
        subselect_sql=subselect_sql,
        subselect_sql_args=tuple(subselect_sql_args) if subselect_sql_args else None,
    )

    gameids = list(dataset._gameids)
    if not gameids:
        print("No games found for export.")
        return

    rows = []
    keys = set()
    for gid in gameids:
        try:
            meta = dict(dataset.get_meta(gid))
        except Exception as e:
            print(f"Warning: failed to get meta for gameid={gid}: {e}")
            continue
        meta["gameid"] = gid
        rows.append(meta)
        keys.update(meta.keys())

    keys = sorted(keys)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    print(f"Exported {len(rows)} rows to {out_csv}")

def main():
    p = argparse.ArgumentParser(description="NLE metadata exporter")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register", help="Register a replay directory into the DB")
    pr.add_argument("--dir", required=True, help="Path to replay directory (unzipped)")
    pr.add_argument("--name", required=True, help="Dataset name to register")
    pr.add_argument("--db", default=DEFAULT_DB, help="DB filename (default: ttyrecs.db)")
    pr.add_argument("--format", choices=["nle", "altorg"], default="nle",
                    help="Directory layout type: nle (nle_data/savedir) or altorg (NAO)")

    pe = sub.add_parser("export", help="Export metadata of a dataset to CSV")
    pe.add_argument("--name", required=True, help="Registered dataset name")
    pe.add_argument("--out", required=True, help="Output CSV path")
    pe.add_argument("--db", default=DEFAULT_DB, help="DB filename (default: ttyrecs.db)")
    pe.add_argument("--sql", default=None, help="Optional subselect SQL e.g. 'SELECT gameid FROM games WHERE role=? AND race=?'")
    pe.add_argument("--arg", action="append", default=[], help="Args for --sql (repeatable)")

    args = p.parse_args()

    if args.cmd == "register":
        register_directory(args.dir, args.name, args.db, fmt=args.format)
    elif args.cmd == "export":
        export_metadata(args.name, args.db, args.out, args.sql, args.arg)

if __name__ == "__main__":
    main()