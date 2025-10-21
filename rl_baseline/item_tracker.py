import os
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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

    def __init__(self, experiment_name: str, mode: str = 'train', enable_detailed_logging: bool = False):
        self.experiment_name = experiment_name
        self.mode = mode  # 'train' or 'eval'
        self.enable_detailed_logging = enable_detailed_logging

        # 前回スナップショット（現在インベントリ）
        self.prev_items: Counter[str] = Counter()
        self.prev_cats: Counter[str] = Counter()

        # アイテム別行動記録（新機能）
        # 構造: {item_name: {'acquired': int, 'actions': Counter({'drink': 2, 'drop': 4, ...})}}
        self.item_actions: Dict[str, Dict] = defaultdict(lambda: {'acquired': 0, 'actions': Counter()})
        
        # カテゴリ別行動記録
        self.category_actions: Dict[str, Dict] = defaultdict(lambda: {'acquired': 0, 'actions': Counter()})

        # 詳細ログ用（オプション）
        self.detailed_logs: List[Dict] = [] if enable_detailed_logging else None

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

        # 行動マッピング（NetHackのアクション番号から行動名へ）
        self.action_mapping = self._init_action_mapping()
        
        # 現在のステップ情報（詳細ログ用）
        self.current_step = 0
        self.last_action = None
        self.last_message = ""

    def _init_action_mapping(self) -> Dict[int, str]:
        """NetHackのアクション番号から行動名へのマッピング"""
        # NetHackの主要アクション（nle.nethack.actionsから）
        return {
            0: 'move_n', 1: 'move_ne', 2: 'move_e', 3: 'move_se',
            4: 'move_s', 5: 'move_sw', 6: 'move_w', 7: 'move_nw',
            8: 'move_up', 9: 'move_down', 10: 'move_wait',
            11: 'pickup', 12: 'kick', 13: 'eat', 14: 'search',
            15: 'quaff', 16: 'read', 17: 'invoke', 18: 'offer',
            19: 'pray', 20: 'drop', 21: 'apply', 22: 'throw',
            23: 'wield', 24: 'wear', 25: 'takeoff', 26: 'puton',
            27: 'remove', 28: 'engrave', 29: 'teleport',
            30: 'help', 31: 'look', 32: 'inventory', 33: 'discoveries',
            34: 'whatdoes', 35: 'history', 36: 'conduct', 37: 'options',
            38: 'explore', 39: 'autopickup', 40: 'save', 41: 'quit',
            42: 'redraw', 43: 'swap_weapons', 44: 'travel', 45: 'rush',
            46: 'zap', 47: 'open', 48: 'close', 49: 'fight',
            50: 'force', 51: 'jump', 52: 'monster', 53: 'name',
            54: 'pay', 55: 'sit', 56: 'turn', 57: 'two_weapon',
            58: 'untrap', 59: 'version', 60: 'wipe', 61: 'enhance'
        }


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

        # 新機能のリセット
        self.item_actions.clear()
        self.category_actions.clear()
        if self.detailed_logs is not None:
            self.detailed_logs.clear()

    ## ここから大きく変更
    def update_from_obs(self, obs: dict, action: Optional[int] = None, msg: str = ""):
        """観察から統計を更新（行動情報付き）"""
        self.current_step += 1
        self.last_action = action
        self.last_message = msg
        
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

        # アイテム取得の検出
        acquired_items = cur_items - self.prev_items
        for item_name, count in acquired_items.items():
            if count > 0:
                self.item_actions[item_name]['acquired'] += count
                # 後方互換性
                self.ep_acq_by_item[item_name] += count
                self.sess_acq_by_item[item_name] += count

        # アイテム消失の検出と行動の推定
        lost_items = self.prev_items - cur_items
        for item_name, count in lost_items.items():
            if count > 0:
                action_name = self._determine_action_for_item(item_name, action, msg)
                self.item_actions[item_name]['actions'][action_name] += count
                
                # 詳細ログの記録
                if self.enable_detailed_logging:
                    self._log_detailed_action(item_name, action_name, count, obs)
                
                # 後方互換性
                self.ep_used_by_item[item_name] += count
                self.sess_used_by_item[item_name] += count

        # カテゴリレベルでも同様の処理
        acquired_cats = cur_cats - self.prev_cats
        for cat_name, count in acquired_cats.items():
            if count > 0:
                self.category_actions[cat_name]['acquired'] += count
                self.ep_acq_by_cat[cat_name] += count
                self.sess_acq_by_cat[cat_name] += count

        lost_cats = self.prev_cats - cur_cats
        for cat_name, count in lost_cats.items():
            if count > 0:
                action_name = self._determine_action_for_category(cat_name, action, msg)
                self.category_actions[cat_name]['actions'][action_name] += count
                self.ep_used_by_cat[cat_name] += count
                self.sess_used_by_cat[cat_name] += count

        # スナップショット更新
        self.prev_items = cur_items
        self.prev_cats = cur_cats

    def _determine_action_for_item(self, item_name: str, action: Optional[int], msg: str) -> str:
        """アイテムに対する行動を推定"""
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode('utf-8', errors='ignore')
        msg_lower = msg.lower()
        item_lower = item_name.lower()

        # メッセージベースの判定（最優先）
        if 'you drink' in msg_lower or 'you quaff' in msg_lower:
            return 'drink'
        elif 'you eat' in msg_lower or 'you consume' in msg_lower:
            return 'eat'
        elif 'you read' in msg_lower:
            return 'read'
        elif 'you zap' in msg_lower:
            return 'zap'
        elif 'you wear' in msg_lower or 'you put on' in msg_lower:
            return 'equip'
        elif 'you wield' in msg_lower:
            return 'wield'
        elif 'you throw' in msg_lower or 'you hurl' in msg_lower:
            return 'throw'
        elif 'you drop' in msg_lower:
            return 'drop'
        elif 'you apply' in msg_lower:
            return 'apply'
        elif 'you offer' in msg_lower:
            return 'offer'
        elif 'cursed' in msg_lower or 'blessed' in msg_lower or 'uncursed' in msg_lower:
            return 'identify_check'  # 祝福状態確認
        
        # アクション番号ベースの判定
        if action is not None:
            action_name = self.action_mapping.get(action, f'action_{action}')
            if action_name in ['drop', 'throw', 'eat', 'quaff', 'read', 'zap', 'wear', 'wield', 'apply']:
                return action_name
        
        # アイテム種類による推定
        if 'potion' in item_lower:
            return 'drink'
        elif 'scroll' in item_lower:
            return 'read'
        elif 'food' in item_lower or 'corpse' in item_lower or 'apple' in item_lower:
            return 'eat'
        elif 'wand' in item_lower:
            return 'zap'
        elif any(armor in item_lower for armor in ['cloak', 'armor', 'robe', 'gloves', 'boots']):
            return 'equip'
        elif any(weapon in item_lower for weapon in ['sword', 'dagger', 'bow', 'spear']):
            return 'wield'
        
        return 'unknown'

    def _determine_action_for_category(self, cat_name: str, action: Optional[int], msg: str) -> str:
        """カテゴリに対する行動を推定"""
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode('utf-8', errors='ignore')
        msg_lower = msg.lower()

        # カテゴリ特有の行動パターン
        category_actions = {
            'potions': {'drink', 'quaff'},
            'scrolls': {'read'},
            'comestibles': {'eat', 'consume'},
            'wands': {'zap'},
            'armor': {'wear', 'equip'},
            'weapons': {'wield'}
        }

        if cat_name in category_actions:
            for action_word in category_actions[cat_name]:
                if action_word in msg_lower:
                    return action_word

        # 汎用的な判定
        if 'you drop' in msg_lower:
            return 'drop'
        elif 'you throw' in msg_lower:
            return 'throw'
        
        if action is not None:
            return self.action_mapping.get(action, f'action_{action}')
        
        return 'unknown'

    def _log_detailed_action(self, item_name: str, action_name: str, count: int, obs: dict):
        """詳細ログの記録（オプション機能）"""
        if self.detailed_logs is None:
            return
        
        log_entry = {
            'step': self.current_step,
            'item': item_name,
            'action': action_name,
            'count': count,
            'message': self.last_message,
            'timestep': int(obs.get('blstats', [0]*30)[20]) if 'blstats' in obs else 0,
            'depth': int(obs.get('blstats', [0]*30)[12]) if 'blstats' in obs else 0,
            'hp': int(obs.get('blstats', [0]*30)[10]) if 'blstats' in obs else 0,
        }
        self.detailed_logs.append(log_entry)

    def get_item_action_stats(self) -> Dict[str, Dict]:
        """アイテム別行動統計を取得"""
        return dict(self.item_actions)

    def get_category_action_stats(self) -> Dict[str, Dict]:
        """カテゴリ別行動統計を取得"""
        return dict(self.category_actions)

    def print_action_summary(self, top_items: int = 20):
        """行動統計のサマリーを表示"""
        print(f"\n=== Item Action Summary ({self.experiment_name}) ===")
        
        # アイテム別統計（取得数順）
        sorted_items = sorted(
            self.item_actions.items(),
            key=lambda x: x[1]['acquired'],
            reverse=True
        )
        
        print(f"\nTop {top_items} Items by Acquisition:")
        for i, (item_name, stats) in enumerate(sorted_items[:top_items]):
            print(f"{i+1}. {item_name} ... acq: {stats['acquired']}")
            for action, count in stats['actions'].most_common():
                print(f"   - {action}: {count}")
        
        # カテゴリ別統計
        print(f"\nCategory Action Summary:")
        for cat_name, stats in sorted(self.category_actions.items()):
            if stats['acquired'] > 0:
                print(f"{cat_name} ... acq: {stats['acquired']}")
                for action, count in stats['actions'].most_common():
                    print(f"   - {action}: {count}")

    def save_action_stats(self, save_dir: str) -> Tuple[str, str, Optional[str]]:
        """行動統計を保存"""
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'{self.experiment_name}_{self.mode}_{ts}'

        # JSON - 行動統計
        json_path = os.path.join(save_dir, f'{base}_actions.json')
        payload = {
            'experiment': self.experiment_name,
            'mode': self.mode,
            'timestamp': ts,
            'item_actions': {k: {'acquired': v['acquired'], 'actions': dict(v['actions'])} 
                           for k, v in self.item_actions.items()},
            'category_actions': {k: {'acquired': v['acquired'], 'actions': dict(v['actions'])} 
                               for k, v in self.category_actions.items()},
            'episodes': self.episodes_meta,
        }
        
        if self.detailed_logs:
            payload['detailed_logs'] = self.detailed_logs
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # 詳細ログの保存（オプション）
        detailed_log_path = None
        if self.enable_detailed_logging and self.detailed_logs:
            detailed_log_path = os.path.join(save_dir, f'{base}_detailed_logs.json')
            with open(detailed_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.detailed_logs, f, indent=2, ensure_ascii=False)

        # return csv_path, json_path, detailed_log_path
        return json_path, json_path, detailed_log_path

    # 使わないが残す
    def update_usage_from_message(self, msg: str | bytes):
        # """任意: メッセージから使用を補強（検出できた場合のみ加算）。"""
        # if isinstance(msg, (bytes, bytearray)):
        #     msg = msg.decode('utf-8', errors='ignore').lower()
        # else:
        #     msg = str(msg).lower()

        # # ざっくりカテゴリのみ
        # if 'you drink' in msg or 'you quaff' in msg:
        #     self.ep_used_by_cat['potions'] += 1
        #     self.sess_used_by_cat['potions'] += 1
        # if 'you read' in msg and 'scroll' in msg:
        #     self.ep_used_by_cat['scrolls'] += 1
        #     self.sess_used_by_cat['scrolls'] += 1
        # if 'you eat' in msg or 'you consume' in msg or 'you devour' in msg:
        #     self.ep_used_by_cat['comestibles'] += 1
        #     self.sess_used_by_cat['comestibles'] += 1
        # if 'you zap' in msg:
        #     self.ep_used_by_cat['wands'] += 1
        #     self.sess_used_by_cat['wands'] += 1
        pass

    def episode_summary(self) -> dict:
        """エピソード集計をフラットな辞書で返す（infoに載せる用）。"""
        out = {}
        # カテゴリ
        for cat in CATEGORIES:
            acq = self.ep_acq_by_cat.get(cat, 0)
            used = self.ep_used_by_cat.get(cat, 0)
            out[f'cat_{cat}_acquired'] = int(acq)
            out[f'cat_{cat}_used'] = int(used)
        # # アイテム（上位のみ）
        # top_items = (self.ep_acq_by_item + self.ep_used_by_item).most_common(10)
        # for i, (name, _) in enumerate(top_items):
        #     sk = _safe_key(name, 30)
        #     out[f'item_{i}_{sk}_acquired'] = int(self.ep_acq_by_item.get(name, 0))
        #     out[f'item_{i}_{sk}_used'] = int(self.ep_used_by_item.get(name, 0))
        # 新機能：行動統計の追加
        for cat_name, stats in self.category_actions.items():
            out[f'cat_{cat_name}_total_actions'] = sum(stats['actions'].values())
        
        return out

    def on_episode_end(self, meta: dict | None = None):
        if meta:
            self.episodes_meta.append(meta)
        self.reset_episode()

    def save_session_stats(self, save_dir: str) -> tuple[str, str]:
        # os.makedirs(save_dir, exist_ok=True)
        # ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # base = f'{self.experiment_name}_{self.mode}_{ts}'

        # # JSON
        # json_path = os.path.join(save_dir, f'{base}.json')
        # payload = {
        #     'experiment': self.experiment_name,
        #     'mode': self.mode,
        #     'timestamp': ts,
        #     'session': {
        #         'acquired_by_cat': dict(self.sess_acq_by_cat),
        #         'used_by_cat': dict(self.sess_used_by_cat),
        #         'acquired_by_item': dict(self.sess_acq_by_item),
        #         'used_by_item': dict(self.sess_used_by_item),
        #     },
        #     'episodes': self.episodes_meta,
        # }
        # with open(json_path, 'w', encoding='utf-8') as f:
        #     json.dump(payload, f, indent=2, ensure_ascii=False)

        # # CSV（カテゴリとアイテム2本）
        # csv_path = os.path.join(save_dir, f'{base}.csv')
        # with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        #     w = csv.writer(f)
        #     w.writerow(['type', 'name', 'acquired', 'used'])
        #     for cat in sorted(CATEGORIES):
        #         w.writerow(['category', cat, self.sess_acq_by_cat.get(cat, 0), self.sess_used_by_cat.get(cat, 0)])
        #     # 上位アイテムのみ（多すぎ防止）
        #     merged = (self.sess_acq_by_item + self.sess_used_by_item).most_common(200)
        #     for name, _ in merged:
        #         w.writerow(['item', name, self.sess_acq_by_item.get(name, 0), self.sess_used_by_item.get(name, 0)])
        # csv_path, json_path, _ = self.save_action_stats(save_dir)
        json_path, _, detailed_log_path = self.save_action_stats(save_dir)

        return json_path, json_path