#!/usr/bin/env python3
"""
JSONファイルのエピソード数とxlogfileの行数を比較して、
全workerのエピソードが正しく収集されているか検証する
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict

def count_json_episodes(json_dir):
    """JSONファイルから総エピソード数とworker別の内訳を取得"""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
    total_episodes = 0
    worker_episodes = defaultdict(lambda: defaultdict(int))  # worker_idx -> env_idx -> episodes
    file_info = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            worker_idx = data.get('worker_idx', 'unknown')
            env_idx = data.get('env_idx', 'unknown')
            episodes = data.get('total_episodes', 0)
            
            total_episodes += episodes
            worker_episodes[worker_idx][env_idx] = episodes
            
            file_info.append({
                'file': os.path.basename(json_file),
                'worker': worker_idx,
                'env': env_idx,
                'episodes': episodes
            })
            
        except Exception as e:
            print(f"❌ Failed to read {json_file}: {e}")
    
    return total_episodes, worker_episodes, file_info

def count_xlogfile_lines(xlogfile_dir):
    """xlogfileから総行数とファイル別の内訳を取得"""
    # nle.*.xlogfile と xlogfile* の両方を検索
    patterns = [
        os.path.join(xlogfile_dir, "nle.*.xlogfile"),
        os.path.join(xlogfile_dir, "**/nle.*.xlogfile"),
        os.path.join(xlogfile_dir, "xlogfile*"),
        os.path.join(xlogfile_dir, "**/xlogfile*"),
    ]
    
    xlogfiles = []
    for pattern in patterns:
        xlogfiles.extend(glob.glob(pattern, recursive=True))
    
    # 重複を除去
    xlogfiles = list(set(xlogfiles))
    
    if not xlogfiles:
        print(f"⚠️  Warning: No xlogfile found in {xlogfile_dir}")
        print(f"    Searched patterns: nle.*.xlogfile, xlogfile*")
    
    total_lines = 0
    file_lines = []
    
    for xlogfile in sorted(xlogfiles):
        try:
            with open(xlogfile, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f if line.strip())
            
            total_lines += lines
            file_lines.append({
                'file': os.path.relpath(xlogfile, xlogfile_dir),
                'lines': lines
            })
            
        except Exception as e:
            print(f"❌ Failed to read {xlogfile}: {e}")
    
    return total_lines, file_lines

def print_comparison_report(json_dir, xlogfile_dir):
    """比較レポートを出力"""
    print("=" * 80)
    print("アイテム統計エピソード数 vs xlogfile行数 比較レポート")
    print("=" * 80)
    print()
    
    # JSONファイルの集計
    print("📊 JSONファイルのエピソード数:")
    print("-" * 80)
    total_json_episodes, worker_episodes, file_info = count_json_episodes(json_dir)
    
    # Worker別の内訳
    for worker_idx in sorted(worker_episodes.keys()):
        envs = worker_episodes[worker_idx]
        worker_total = sum(envs.values())
        print(f"  Worker {worker_idx}: {worker_total} episodes")
        for env_idx in sorted(envs.keys()):
            print(f"    - env {env_idx}: {envs[env_idx]} episodes")
    
    print(f"\n  📁 Total JSON files: {len(file_info)}")
    print(f"  📈 Total episodes in JSON: {total_json_episodes}")
    print()
    
    # xlogfileの集計
    print("📝 xlogfileの行数:")
    print("-" * 80)
    total_xlog_lines, file_lines = count_xlogfile_lines(xlogfile_dir)
    
    for file_info_xlog in file_lines:
        print(f"  {file_info_xlog['file']}: {file_info_xlog['lines']} lines")
    
    print(f"\n  📁 Total xlogfiles: {len(file_lines)}")
    print(f"  📈 Total lines in xlogfile: {total_xlog_lines}")
    print()
    
    # 比較結果
    print("🔍 比較結果:")
    print("-" * 80)
    difference = total_json_episodes - total_xlog_lines
    coverage_rate = (total_json_episodes / total_xlog_lines * 100) if total_xlog_lines > 0 else 0
    
    print(f"  JSON episodes:     {total_json_episodes}")
    print(f"  xlogfile lines:    {total_xlog_lines}")
    print(f"  Difference:        {difference:+d}")
    print(f"  Coverage rate:     {coverage_rate:.2f}%")
    print()
    
    # 判定
    if difference == 0:
        print("  ✅ PERFECT: 全てのエピソードが収集されています")
    elif difference > 0:
        print(f"  ⚠️  WARNING: JSONに {difference} エピソード多く記録されています（重複の可能性）")
    else:
        print(f"  ❌ ERROR: JSONに {abs(difference)} エピソード不足しています")
        missing_workers = []
        # 不足しているworkerを特定
        expected_workers = len(file_lines)  # xlogfileの数 = worker数の想定
        actual_workers = len(worker_episodes)
        if actual_workers < expected_workers:
            print(f"     期待されるworker数: {expected_workers}, 実際: {actual_workers}")
    
    print()
    print("=" * 80)

def main():
    """メイン関数"""
    # デフォルトのパス
    json_dir = "./train_dir/skill_policy/debug2/item_stats/final"
    xlogfile_dir = "./train_dir/skill_policy/debug2/ttyrecs/A"
    
    # パスの存在確認
    if not os.path.exists(json_dir):
        print(f"❌ Error: JSON directory not found: {json_dir}")
        return
    
    if not os.path.exists(xlogfile_dir):
        print(f"❌ Error: xlogfile directory not found: {xlogfile_dir}")
        return
    
    # レポート出力
    print_comparison_report(json_dir, xlogfile_dir)

if __name__ == "__main__":
    main()