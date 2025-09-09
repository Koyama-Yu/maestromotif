#!/usr/bin/env python3
"""
アイテム統計機能のテスト・使用例

使用方法:
1. トレーニング実行時: 自動的にアイテム統計が記録される
2. 手動でファイルを読み込んで分析:
   python test_item_tracker.py --load path/to/item_stats/
"""

import argparse
import os
import sys
import numpy as np
import json
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_baseline.item_tracker import ItemTracker


def test_item_tracker():
    """ItemTrackerの基本機能テスト"""
    print("=== ItemTracker Basic Test ===")
    
    # テスト用のトラッカー作成
    tracker = ItemTracker(experiment_name="test_experiment", is_training=True)
    
    # 模擬観測データでテスト
    mock_obs = {
        'inv_oclasses': np.array([7, 8, 9, 0, 0]),  # potions, scrolls, spellbooks
        'inv_letters': np.array([ord('a'), ord('b'), ord('c'), 0, 0])
    }
    
    # アイテム更新
    tracker.update_inventory(mock_obs)
    
    # メッセージからの使用判定テスト
    tracker.update_from_message(b"You drink a potion of healing.")
    tracker.update_from_message(b"You read a scroll of identify.")
    
    # エピソード終了
    tracker.end_episode(episode_length=1000, episode_reward=500)
    
    # 統計表示
    tracker.print_summary()
    
    # ファイル保存
    save_dir = "test_item_stats"
    csv_path, npy_path, json_path = tracker.save_stats(save_dir)
    
    print(f"Test files saved:")
    print(f"  CSV: {csv_path}")
    print(f"  NPY: {npy_path}")
    print(f"  JSON: {json_path}")
    
    return save_dir


def analyze_item_stats(stats_dir):
    """保存されたアイテム統計を分析"""
    print(f"\n=== Analyzing Item Stats in {stats_dir} ===")
    
    if not os.path.exists(stats_dir):
        print(f"Directory {stats_dir} does not exist!")
        return
    
    # ディレクトリ内のファイルを探索
    csv_files = []
    json_files = []
    npy_files = []
    
    for filename in os.listdir(stats_dir):
        if filename.endswith('_items.csv'):
            csv_files.append(filename)
        elif filename.endswith('_detailed.json'):
            json_files.append(filename)
        elif filename.endswith('_items.npy'):
            npy_files.append(filename)
    
    print(f"Found {len(csv_files)} CSV files, {len(json_files)} JSON files, {len(npy_files)} NPY files")
    
    # CSV ファイル分析
    for csv_file in csv_files:
        csv_path = os.path.join(stats_dir, csv_file)
        print(f"\n--- Analyzing {csv_file} ---")
        
        try:
            df = pd.read_csv(csv_path)
            print("Item Usage Statistics:")
            print(df.to_string(index=False))
            
            # 使用率の高いアイテムTOP 5
            top_used = df.nlargest(5, 'usage_rate')
            print(f"\nTop 5 Most Used Items:")
            for _, row in top_used.iterrows():
                print(f"  {row['item_type']}: {row['usage_rate']:.1f}% ({row['used']}/{row['acquired']})")
                
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # JSON ファイル分析
    for json_file in json_files:
        json_path = os.path.join(stats_dir, json_file)
        print(f"\n--- Analyzing {json_file} ---")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Experiment: {data['experiment_name']}")
            print(f"Mode: {data['mode']}")
            print(f"Timestamp: {data['timestamp']}")
            print(f"Total Episodes: {len(data['episodes'])}")
            
            if data['episodes']:
                total_reward = sum(ep['episode_reward'] for ep in data['episodes'])
                avg_reward = total_reward / len(data['episodes'])
                avg_length = sum(ep['episode_length'] for ep in data['episodes']) / len(data['episodes'])
                
                print(f"Average Episode Reward: {avg_reward:.2f}")
                print(f"Average Episode Length: {avg_length:.1f}")
                
        except Exception as e:
            print(f"Error reading {json_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Item Tracker Test and Analysis Tool')
    parser.add_argument('--test', action='store_true', help='Run basic functionality test')
    parser.add_argument('--load', type=str, help='Load and analyze statistics from directory')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test files')
    
    args = parser.parse_args()
    
    if args.test:
        save_dir = test_item_tracker()
        
        # 作成したファイルを分析
        if os.path.exists(save_dir):
            analyze_item_stats(save_dir)
            
        if args.cleanup:
            import shutil
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                print(f"\nCleaned up test directory: {save_dir}")
    
    elif args.load:
        analyze_item_stats(args.load)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
