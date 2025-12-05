#!/usr/bin/env python3
import os, json, argparse, glob, re
from collections import Counter
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter  # フォールバック

def merge_item_dict(dst: Dict[str, Dict[str, Any]], src: Dict[str, Dict[str, Any]]):
    for name, payload in (src or {}).items():
        if name not in dst:
            dst[name] = {"acquired": 0, "actions": Counter()}
        dst[name]["acquired"] += int(payload.get("acquired", 0))
        dst[name]["actions"].update(payload.get("actions", {}) or {})

def aggregate(json_dir: str):
    items, base_items, categories = {}, {}, {}
    total_episodes, files = 0, sorted(glob.glob(os.path.join(json_dir, "*.json")))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        total_episodes += int(data.get("total_episodes", 0))
        merge_item_dict(items, data.get("item_actions"))
        merge_item_dict(base_items, data.get("base_item_actions"))
        merge_item_dict(categories, data.get("category_actions"))
    return total_episodes, items, base_items, categories

def to_rows(d: Dict[str, Dict[str, Any]]):
    rows = []
    for k, v in d.items():
        rows.append({"name": k, "acquired": int(v.get("acquired", 0))})
    rows.sort(key=lambda r: (-r["acquired"], r["name"]))
    return rows

def sanitize(tag: str) -> str:
    tag = re.sub(r"[^\w\-/\.]+", "_", tag)
    return tag[:128]

def plot_top_bar(rows, title, topn=30):
    top = rows[:topn]
    names = [r["name"] for r in top]
    vals = [r["acquired"] for r in top]
    fig = plt.figure(figsize=(10, max(4, 0.3 * len(top))))
    plt.barh(names[::-1], vals[::-1], color="#4C78A8")
    plt.xlabel("acquired")
    plt.title(title)
    plt.tight_layout()
    return fig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True, help="…/item_stats/final")
    ap.add_argument("--logdir", default="runs/item_stats", help="TensorBoardログ出力先")
    ap.add_argument("--run-name", default="item_stats", help="TensorBoardのラン名")
    ap.add_argument("--topn", type=int, default=50, help="個別スカラーを書き出す上位件数")
    args = ap.parse_args()

    total_ep, items, base_items, categories = aggregate(args.json_dir)
    rows_items = to_rows(items)
    rows_base  = to_rows(base_items)
    rows_cat   = to_rows(categories)

    run_dir = os.path.join(args.logdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    w = SummaryWriter(log_dir=run_dir)

    # 総エピソード
    w.add_scalar("summary/total_episodes", total_ep, 0)

    # 上位Nをスカラー出力（タグ数の増え過ぎを防止）
    for r in rows_items[:args.topn]:
        w.add_scalar(f"items_acquired/{sanitize(r['name'])}", r["acquired"], 0)
    for r in rows_base[:args.topn]:
        w.add_scalar(f"base_items_acquired/{sanitize(r['name'])}", r["acquired"], 0)
    for r in rows_cat[:args.topn]:
        w.add_scalar(f"categories_acquired/{sanitize(r['name'])}", r["acquired"], 0)

    # 画像（棒グラフ）
    fig1 = plot_top_bar(rows_items, "Top items (acquired)", topn=min(30, len(rows_items)))
    fig2 = plot_top_bar(rows_base, "Top base items (acquired)", topn=min(30, len(rows_base)))
    fig3 = plot_top_bar(rows_cat, "Top categories (acquired)", topn=min(30, len(rows_cat)))
    w.add_figure("fig/top_items", fig1, 0)
    w.add_figure("fig/top_base_items", fig2, 0)
    w.add_figure("fig/top_categories", fig3, 0)
    plt.close(fig1); plt.close(fig2); plt.close(fig3)

    w.flush()
    w.close()
    print(f"✅ Exported to TensorBoard: {run_dir}")

if __name__ == "__main__":
    main()