import numpy as np
import csv
import json
import os
from collections import defaultdict, Counter
from datetime import datetime

class ItemTracker:
    """
    NetHackのアイテム取得・使用統計を記録するクラス
    """
    
    # NetHackのアイテムクラス定義（NetHack Learning Environment準拠）
    ITEM_CLASSES = {
        0: 'unknown',      # 空スロット
        1: 'weapons',
        2: 'armor', 
        3: 'rings',
        4: 'amulets',
        5: 'tools',
        6: 'comestibles',  # 食べ物
        7: 'potions',      # ポーション
        8: 'scrolls',      # 巻物
        9: 'spellbooks',   # 呪文書
        10: 'wands',       # 魔法の杖
        11: 'coins',       # コイン
        12: 'gems',        # 宝石
        13: 'boulders'     # 岩など
    }
    
    def __init__(self, experiment_name="default", is_training=True):
        self.experiment_name = experiment_name
        self.is_training = is_training
        self.mode = "train" if is_training else "eval"
        
        # 統計データの初期化
        self.reset_episode_stats()
        self.reset_session_stats()
        
        # 前回の観測を保存
        self.prev_inventory = {}
        self.prev_inventory_detailed = {}  # 詳細インベントリの初期化
        
    def reset_episode_stats(self):
        """エピソード開始時に呼び出される"""
        # カテゴリ統計をリセット
        self.episode_acquired = defaultdict(int)
        self.episode_used = defaultdict(int)
        
        # 詳細統計をリセット  
        self.episode_acquired_detailed = defaultdict(int)
        self.episode_used_detailed = defaultdict(int)
        
        # メッセージ履歴をクリア
        self.message_history = []
        
    def reset_session_stats(self):
        """セッション統計をリセット"""
        self.session_acquired = defaultdict(int)     # セッション全体の大分類取得数
        self.session_used = defaultdict(int)         # セッション全体の大分類使用数
        
        # 詳細アイテム名のセッション統計
        self.session_acquired_detailed = defaultdict(int)  # セッション全体の詳細名取得数
        self.session_used_detailed = defaultdict(int)      # セッション全体の詳細名使用数
        
        # トレーニング・評価別統計（カテゴリレベル）
        self.train_acquired = defaultdict(int)
        self.train_used = defaultdict(int)
        self.eval_acquired = defaultdict(int)
        self.eval_used = defaultdict(int)
        
        # トレーニング・評価別統計（詳細レベル）
        self.train_acquired_detailed = defaultdict(int)
        self.train_used_detailed = defaultdict(int)
        self.eval_acquired_detailed = defaultdict(int)
        self.eval_used_detailed = defaultdict(int)

        self.episodes_data = []                      # 各エピソードのデータ
        
    def update_inventory(self, observation):
        """インベントリ情報を更新し、変化を検出する"""
        
        # カテゴリベースの更新
        inv_oclasses = observation.get('inv_oclasses', np.array([]))
        current_inventory = {}
        for cls in inv_oclasses:
            if cls > 0:  # 0は空のスロット
                category = self.ITEM_CLASSES.get(cls, f"unknown_{cls}")
                current_inventory[category] = current_inventory.get(category, 0) + 1

        # 詳細アイテムベースの更新
        inv_strs = observation.get('inv_strs', [])
        current_inventory_detailed = {}
        
        for item_str in inv_strs:
            if isinstance(item_str, bytes):
                item_str = item_str.decode('utf-8', errors='ignore')
            item_str = str(item_str).strip()
            
            if item_str and item_str != '\x00' and len(item_str) > 1:
                # NetHackのインベントリ形式: "a - potion of healing" を処理
                if ' - ' in item_str:
                    item_name = item_str.split(' - ', 1)[1].strip()
                else:
                    # 先頭の文字（通常は記号）を除去してアイテム名を取得
                    item_name = item_str[1:].strip() if len(item_str) > 1 else item_str
                
                if item_name and item_name != 'unknown':
                    current_inventory_detailed[item_name] = current_inventory_detailed.get(item_name, 0) + 1

        # 変化を検出（カテゴリレベル）
        for category, count in current_inventory.items():
            prev_count = self.prev_inventory.get(category, 0)
            if count > prev_count:
                acquired = count - prev_count
                self.episode_acquired[category] += acquired
                self.session_acquired[category] += acquired
                if self.mode == 'train':
                    self.train_acquired[category] += acquired
                else:
                    self.eval_acquired[category] += acquired

        # 変化を検出（詳細レベル）
        for item_name, count in current_inventory_detailed.items():
            prev_count = self.prev_inventory_detailed.get(item_name, 0)
            if count > prev_count:
                acquired = count - prev_count
                self.episode_acquired_detailed[item_name] += acquired
                self.session_acquired_detailed[item_name] += acquired
                if self.mode == 'train':
                    self.train_acquired_detailed[item_name] += acquired
                else:
                    self.eval_acquired_detailed[item_name] += acquired

        # 現在のインベントリを保存
        self.prev_inventory = current_inventory.copy()
        self.prev_inventory_detailed = current_inventory_detailed.copy()
        
    def update_from_message(self, message):
        """
        ゲームメッセージからアイテム使用情報を更新
        
        Args:
            message: ゲームメッセージ（bytes or str）
        """
        if isinstance(message, bytes):
            msg_str = message.decode('utf-8', errors='ignore').lower()
        elif isinstance(message, str):
            msg_str = message.lower()
        else:
            return
        
        self.message_history.append(msg_str)
        
        # カテゴリレベルの使用検出
        category_usage = self._extract_usage_from_message(msg_str)
        for category, count in category_usage.items():
            self.episode_used[category] += count
            self.session_used[category] += count
            if self.mode == 'train':
                self.train_used[category] += count
            else:
                self.eval_used[category] += count
        
        # 詳細レベルの使用検出
        detailed_usage = self._extract_detailed_usage_from_message(msg_str)
        for item_name, count in detailed_usage.items():
            self.episode_used_detailed[item_name] += count
            self.session_used_detailed[item_name] += count
            if self.mode == 'train':
                self.train_used_detailed[item_name] += count
            else:
                self.eval_used_detailed[item_name] += count

    def _extract_usage_from_message(self, msg_str):
        """カテゴリレベルでのアイテム使用を検出"""
        usage = {}
        
        if 'you drink' in msg_str or 'you quaff' in msg_str:
            usage['potions'] = usage.get('potions', 0) + 1
            
        if 'you read' in msg_str and 'scroll' in msg_str:
            usage['scrolls'] = usage.get('scrolls', 0) + 1
            
        if 'you eat' in msg_str or 'you consume' in msg_str:
            usage['comestibles'] = usage.get('comestibles', 0) + 1
            
        if 'you zap' in msg_str:
            usage['wands'] = usage.get('wands', 0) + 1
            
        return usage

    def _extract_detailed_usage_from_message(self, msg_str):
        """
        メッセージから詳細なアイテム名を抽出
        
        Args:
            msg_str: メッセージ文字列（小文字）
        
        Returns:
            dict: {item_name: count} の辞書
        """
        usage = {}
        
        try:
            import re
            
            # 各種アイテム使用パターンを定義
            patterns = [
                # ポーション系
                (r'you (?:drink|quaff) (?:a |an |the )?(.+?)(?:\.|!|$)', 'potion'),
                # 巻物系  
                (r'you read (?:a |an |the )?(scroll of .+?)(?:\.|!|$)', 'scroll'),
                # 食物系
                (r'you eat (?:a |an |the )?(.+?)(?:\.|!|$)', 'food'),
                (r'you (?:consume|devour) (?:a |an |the )?(.+?)(?:\.|!|$)', 'food'),
                # 魔法の杖系
                (r'you zap (?:a |an |the )?(.+?)(?:\.|!|$)', 'wand')
            ]
            
            for pattern, item_type in patterns:
                match = re.search(pattern, msg_str)
                if match:
                    item_detail = match.group(1).strip()
                    if item_detail:
                        # 不要な前置詞を除去
                        clean_name = item_detail.replace('of ', '').replace('the ', '')
                        clean_name = re.sub(r'^(a |an |the )', '', clean_name)
                        
                        if clean_name:
                            usage[item_detail] = usage.get(item_detail, 0) + 1
                        break
                        
        except Exception:
            # 抽出に失敗した場合は何もしない
            pass
            
        return usage

    def get_episode_stats(self):
        """エピソード終了時の統計を取得"""
        stats = {}
        
        # カテゴリレベル統計
        for category in self.ITEM_CLASSES.values():
            if category == 'unknown':
                continue
                
            acquired = self.episode_acquired.get(category, 0)
            used = self.episode_used.get(category, 0)
            usage_rate = (used / acquired * 100) if acquired > 0 else 0
            
            stats[f'{category}_acquired'] = acquired
            stats[f'{category}_used'] = used
            stats[f'{category}_usage_rate'] = usage_rate
        
        # 詳細レベル統計（上位5アイテムのみ）
        sorted_acquired = sorted(self.episode_acquired_detailed.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for i, (item_name, count) in enumerate(sorted_acquired):
            safe_name = self._make_safe_key(item_name)
            stats[f'top_item_{i}_{safe_name}_acquired'] = count
            
            used_count = self.episode_used_detailed.get(item_name, 0)
            usage_rate = (used_count / count * 100) if count > 0 else 0
            stats[f'top_item_{i}_{safe_name}_usage_rate'] = usage_rate
        
        return stats

    def _make_safe_key(self, item_name):
        """TensorBoard用の安全なキー名を作成"""
        import re
        # 特殊文字を除去し、アンダースコアに置換
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', item_name.lower())
        # 連続するアンダースコアを1つに
        safe_name = re.sub(r'_+', '_', safe_name)
        # 前後のアンダースコアを除去
        safe_name = safe_name.strip('_')
        # 長さ制限
        return safe_name[:20]

    def end_episode(self, episode_length=0, episode_reward=0):
        """
        エピソード終了時の処理
        
        Args:
            episode_length: エピソードの長さ
            episode_reward: エピソードの報酬
        """
        # エピソードデータを記録
        episode_data = {
            'timestamp': datetime.now().isoformat(),
            'episode_length': episode_length,
            'episode_reward': episode_reward,
            'acquired': dict(self.episode_acquired),
            'used': dict(self.episode_used),
            'final_inventory': dict(self.prev_inventory),
            # 詳細統計も保存
            'acquired_detailed': dict(self.episode_acquired_detailed),
            'used_detailed': dict(self.episode_used_detailed),
            'final_inventory_detailed': dict(self.prev_inventory_detailed)
        }
        self.episodes_data.append(episode_data)
        
        # エピソード統計をリセット
        self.reset_episode_stats()
        
    def get_usage_stats(self, detailed=False):
        """
        使用率統計を計算
        
        Args:
            detailed: True の場合詳細名統計、False の場合大分類統計を返す
        """
        if detailed:
            acquired_dict = self.session_acquired_detailed
            used_dict = self.session_used_detailed
        else:
            acquired_dict = self.session_acquired
            used_dict = self.session_used
            
        stats = {}
        for item_name in set(list(acquired_dict.keys()) + list(used_dict.keys())):
            acquired = acquired_dict.get(item_name, 0)
            used = used_dict.get(item_name, 0)
            usage_rate = (used / acquired * 100) if acquired > 0 else 0
            
            stats[item_name] = {
                'acquired': acquired,
                'used': used,
                'usage_rate': usage_rate
            }
            
        return stats
        
    def save_stats(self, save_dir):
        """
        統計をファイルに保存
        
        Args:
            save_dir: 保存ディレクトリ
        """
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{self.experiment_name}_{self.mode}_{timestamp}"
        
        # CSVファイルに保存（大分類）
        csv_path = os.path.join(save_dir, f"{filename_base}_items.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['item_type', 'acquired', 'used', 'usage_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            stats = self.get_usage_stats(detailed=False)
            for item_name, item_stats in stats.items():
                writer.writerow({
                    'item_type': item_name,
                    'acquired': item_stats['acquired'],
                    'used': item_stats['used'],
                    'usage_rate': round(item_stats['usage_rate'], 2)
                })
        
        # CSVファイルに保存（詳細）
        csv_detailed_path = os.path.join(save_dir, f"{filename_base}_items_detailed.csv")
        with open(csv_detailed_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['item_name', 'acquired', 'used', 'usage_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            stats_detailed = self.get_usage_stats(detailed=True)
            for item_name, item_stats in stats_detailed.items():
                writer.writerow({
                    'item_name': item_name,
                    'acquired': item_stats['acquired'],
                    'used': item_stats['used'],
                    'usage_rate': round(item_stats['usage_rate'], 2)
                })
        
        # NumPy形式で保存
        stats_summary = self.get_usage_stats(detailed=False)
        stats_detailed = self.get_usage_stats(detailed=True)
        combined_stats = {
            'summary': stats_summary,
            'detailed': stats_detailed
        }
        
        if stats_summary or stats_detailed:
            npy_path = os.path.join(save_dir, f"{filename_base}_items.npy")
            np.save(npy_path, combined_stats)
        
        # JSON形式で詳細データを保存
        json_path = os.path.join(save_dir, f"{filename_base}_detailed.json")
        detailed_data = {
            'experiment_name': self.experiment_name,
            'mode': self.mode,
            'timestamp': timestamp,
            'summary': self.get_usage_stats(detailed=False),
            'detailed': self.get_usage_stats(detailed=True),
            'episodes': self.episodes_data
        }
        
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(detailed_data, jsonfile, indent=2, ensure_ascii=False)
            
        print(f"Item statistics saved:")
        print(f"  CSV (summary): {csv_path}")
        print(f"  CSV (detailed): {csv_detailed_path}")
        print(f"  NPY: {npy_path}")
        print(f"  JSON: {json_path}")
        
        return csv_path, npy_path, json_path
        
    def print_summary(self, show_detailed=True):
        """
        統計サマリーをコンソールに出力
        
        Args:
            show_detailed: 詳細統計も表示するかどうか
        """
        stats = self.get_usage_stats(detailed=False)
        
        print(f"\n=== Item Statistics Summary ({self.mode}) ===")
        print(f"Experiment: {self.experiment_name}")
        print(f"Total Episodes: {len(self.episodes_data)}")
        print()
        
        if stats:
            print("=== Category Summary ===")
            print(f"{'Item Type':<15} {'Acquired':<10} {'Used':<10} {'Usage Rate':<12}")
            print("-" * 50)
            for item_name, item_stats in sorted(stats.items()):
                print(f"{item_name:<15} {item_stats['acquired']:<10} {item_stats['used']:<10} {item_stats['usage_rate']:<12.1f}%")
            
            if show_detailed:
                detailed_stats = self.get_usage_stats(detailed=True)
                if detailed_stats:
                    print(f"\n=== Detailed Items (Top 10 by Usage Rate) ===")
                    print(f"{'Item Name':<40} {'Acquired':<10} {'Used':<10} {'Rate':<12}")
                    print("-" * 75)
                    
                    # 使用率でソートしてTop 10を表示
                    sorted_detailed = sorted(detailed_stats.items(), 
                                           key=lambda x: x[1]['usage_rate'], reverse=True)[:10]
                    
                    for item_name, item_stats in sorted_detailed:
                        # アイテム名を40文字に制限
                        display_name = item_name[:39] if len(item_name) > 39 else item_name
                        
                        print(f"{display_name:<40} {item_stats['acquired']:<10} {item_stats['used']:<10} {item_stats['usage_rate']:<12.1f}%")
        else:
            print("No item data recorded.")
