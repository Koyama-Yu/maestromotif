import os
import csv
import json
import re
from collections import Counter
from datetime import datetime

import numpy as np


OCLASS_TO_CATEGORY = {
    0: 'unknown',
    1: 'weapons',
    2: 'armor',
    3: 'rings',
    4: 'amulets',
    5: 'tools',
    6: 'comestibles',
    7: 'potions',
    8: 'scrolls',
    9: 'spellbooks',
    10: 'wands',
    11: 'coins',
    12: 'gems',
    13: 'boulders',
}

CATEGORIES = [c for c in OCLASS_TO_CATEGORY.values() if c != 'unknown']


def _safe_key(s: str, limit: int = 40) -> str:
    s = s.lower()
    s = re.sub(r'[^a-z0-9_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:limit] if len(s) > limit else s


def _parse_inv_strs(inv_strs: np.ndarray) -> list[str]:
    """(55,80) uint8配列からインベントリ名を抽出して長さ55のリストにする。空は''。"""
    names: list[str] = [''] * (inv_strs.shape[0] if hasattr(inv_strs, 'shape') else 55)
    if hasattr(inv_strs, 'shape') and len(inv_strs.shape) == 2:
        for i in range(inv_strs.shape[0]):
            row = inv_strs[i]
            if row is None:
                continue
            valid = row[row != 0].tolist()
            try:
                s = ''.join(chr(b) for b in valid if 32 <= b <= 126).strip()
            except Exception:
                s = ''
            if not s:
                continue
            # 形式: "a - potion of healing" または "a potion of healing"
            if ' - ' in s:
                s = s.split(' - ', 1)[1].strip()
            elif len(s) > 2 and s[0].isalpha() and s[1] == ' ':
                s = s[2:].strip()
            names[i] = s
    else:
        # 後方互換: 1次元やリスト
        for i, v in enumerate(inv_strs[:len(names)]):
            if isinstance(v, (bytes, bytearray)):
                try:
                    s = v.decode('utf-8', errors='ignore')
                except Exception:
                    s = ''
            else:
                s = str(v)
            s = s.strip('\x00').strip()
            if not s:
                continue
            if ' - ' in s:
                s = s.split(' - ', 1)[1].strip()
            elif len(s) > 2 and s[0].isalpha() and s[1] == ' ':
                s = s[2:].strip()
            names[i] = s
    return names


def _classes_to_categories(inv_oclasses: np.ndarray) -> list[str]:
    cats: list[str] = []
    for cls in (inv_oclasses.tolist() if hasattr(inv_oclasses, 'tolist') else list(inv_oclasses)):
        cat = OCLASS_TO_CATEGORY.get(int(cls), 'unknown') if cls is not None else 'unknown'
        cats.append(cat)
    return cats


class ItemTracker:
    """最小構成のアイテム統計トラッカー。"""

    def __init__(self, experiment_name: str, mode: str = 'train'):
        self.experiment_name = experiment_name
        self.mode = mode  # 'train' or 'eval'

        # 前回スナップショット（現在インベントリ）
        self.prev_items: Counter[str] = Counter()
        self.prev_cats: Counter[str] = Counter()

        # エピソード集計
        self.ep_acq_by_item: Counter[str] = Counter()
        self.ep_used_by_item: Counter[str] = Counter()
        self.ep_acq_by_cat: Counter[str] = Counter()
        self.ep_used_by_cat: Counter[str] = Counter()

        # セッション集計
        self.sess_acq_by_item: Counter[str] = Counter()
        self.sess_used_by_item: Counter[str] = Counter()
        self.sess_acq_by_cat: Counter[str] = Counter()
        self.sess_used_by_cat: Counter[str] = Counter()

        self.episodes_meta: list[dict] = []

    def reset_episode(self):
        self.ep_acq_by_item.clear()
        self.ep_used_by_item.clear()
        self.ep_acq_by_cat.clear()
        self.ep_used_by_cat.clear()

    def reset_session(self):
        self.reset_episode()
        self.sess_acq_by_item.clear()
        self.sess_used_by_item.clear()
        self.sess_acq_by_cat.clear()
        self.sess_used_by_cat.clear()
        self.episodes_meta.clear()
        self.prev_items.clear()
        self.prev_cats.clear()

    def update_from_obs(self, obs: dict):
        """obsから現在のインベントリを構築し、前回との差分でacq/usedを更新。"""
        inv_strs = obs.get('inv_strs', np.empty((0,)))
        inv_oclasses = obs.get('inv_oclasses', np.empty((0,)))
        names = _parse_inv_strs(inv_strs) if len(inv_strs) > 0 else [''] * 55
        cats = _classes_to_categories(inv_oclasses) if len(inv_oclasses) > 0 else ['unknown'] * len(names)

        # 現在スナップショットを作成
        cur_items = Counter()
        cur_cats = Counter()
        for name, cat in zip(names, cats):
            if not name and cat == 'unknown':
                continue
            if name:
                cur_items[name] += 1
            if cat in CATEGORIES:
                cur_cats[cat] += 1

        # 差分: 取得 = 現在 - 前回 の正差分、使用 = 前回 - 現在 の正差分
        # アイテム名
        for itm, cnt in (cur_items - self.prev_items).items():
            if cnt > 0:
                self.ep_acq_by_item[itm] += cnt
                self.sess_acq_by_item[itm] += cnt
        for itm, cnt in (self.prev_items - cur_items).items():
            if cnt > 0:
                self.ep_used_by_item[itm] += cnt
                self.sess_used_by_item[itm] += cnt

        # カテゴリ
        for cat, cnt in (cur_cats - self.prev_cats).items():
            if cnt > 0:
                self.ep_acq_by_cat[cat] += cnt
                self.sess_acq_by_cat[cat] += cnt
        for cat, cnt in (self.prev_cats - cur_cats).items():
            if cnt > 0:
                self.ep_used_by_cat[cat] += cnt
                self.sess_used_by_cat[cat] += cnt

        # スナップショット更新
        self.prev_items = cur_items
        self.prev_cats = cur_cats

    def update_usage_from_message(self, msg: str | bytes):
        """任意: メッセージから使用を補強（検出できた場合のみ加算）。"""
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode('utf-8', errors='ignore').lower()
        else:
            msg = str(msg).lower()

        # ざっくりカテゴリのみ
        if 'you drink' in msg or 'you quaff' in msg:
            self.ep_used_by_cat['potions'] += 1
            self.sess_used_by_cat['potions'] += 1
        if 'you read' in msg and 'scroll' in msg:
            self.ep_used_by_cat['scrolls'] += 1
            self.sess_used_by_cat['scrolls'] += 1
        if 'you eat' in msg or 'you consume' in msg or 'you devour' in msg:
            self.ep_used_by_cat['comestibles'] += 1
            self.sess_used_by_cat['comestibles'] += 1
        if 'you zap' in msg:
            self.ep_used_by_cat['wands'] += 1
            self.sess_used_by_cat['wands'] += 1

    def episode_summary(self) -> dict:
        """エピソード集計をフラットな辞書で返す（infoに載せる用）。"""
        out = {}
        # カテゴリ
        for cat in CATEGORIES:
            acq = self.ep_acq_by_cat.get(cat, 0)
            used = self.ep_used_by_cat.get(cat, 0)
            out[f'cat_{cat}_acquired'] = int(acq)
            out[f'cat_{cat}_used'] = int(used)
        # アイテム（上位のみ）
        top_items = (self.ep_acq_by_item + self.ep_used_by_item).most_common(10)
        for i, (name, _) in enumerate(top_items):
            sk = _safe_key(name, 30)
            out[f'item_{i}_{sk}_acquired'] = int(self.ep_acq_by_item.get(name, 0))
            out[f'item_{i}_{sk}_used'] = int(self.ep_used_by_item.get(name, 0))
        return out

    def on_episode_end(self, meta: dict | None = None):
        if meta:
            self.episodes_meta.append(meta)
        self.reset_episode()

    def save_session_stats(self, save_dir: str) -> tuple[str, str]:
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'{self.experiment_name}_{self.mode}_{ts}'

        # JSON
        json_path = os.path.join(save_dir, f'{base}.json')
        payload = {
            'experiment': self.experiment_name,
            'mode': self.mode,
            'timestamp': ts,
            'session': {
                'acquired_by_cat': dict(self.sess_acq_by_cat),
                'used_by_cat': dict(self.sess_used_by_cat),
                'acquired_by_item': dict(self.sess_acq_by_item),
                'used_by_item': dict(self.sess_used_by_item),
            },
            'episodes': self.episodes_meta,
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # CSV（カテゴリとアイテム2本）
        csv_path = os.path.join(save_dir, f'{base}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['type', 'name', 'acquired', 'used'])
            for cat in sorted(CATEGORIES):
                w.writerow(['category', cat, self.sess_acq_by_cat.get(cat, 0), self.sess_used_by_cat.get(cat, 0)])
            # 上位アイテムのみ（多すぎ防止）
            merged = (self.sess_acq_by_item + self.sess_used_by_item).most_common(200)
            for name, _ in merged:
                w.writerow(['item', name, self.sess_acq_by_item.get(name, 0), self.sess_used_by_item.get(name, 0)])

        return csv_path, json_path