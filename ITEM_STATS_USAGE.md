# アイテム統計機能の使用方法

## 概要

この機能は、NetHackエージェントのアイテム取得・使用統計を自動的に記録し、分析用データを出力します。

## 機能

### 記録される統計情報
- **アイテム取得数**: エピソード中に取得したアイテムの種類別数量
- **アイテム使用数**: 使用・消費したアイテムの種類別数量  
- **使用率**: (使用数 / 取得数) × 100
- **エピソード詳細**: 各エピソードの長さ、報酬、アイテム利用状況

### 対応アイテムカテゴリ
- `weapons`: 武器
- `armor`: 防具
- `rings`: 指輪
- `amulets`: お守り
- `tools`: 道具
- `comestibles`: 食べ物
- `potions`: ポーション
- `scrolls`: 巻物
- `spellbooks`: 呪文書
- `wands`: 魔法の杖
- `coins`: コイン
- `gems`: 宝石

## 出力ファイル形式

### 1. CSV形式 (`*_items.csv`)
```csv
item_type,acquired,used,usage_rate
potions,15,12,80.0
scrolls,8,5,62.5
comestibles,20,18,90.0
```

### 2. NumPy形式 (`*_items.npy`)
- 辞書形式で統計データを保存
- Python での分析に最適

### 3. JSON形式 (`*_detailed.json`)
```json
{
  "experiment_name": "experiment_name",
  "mode": "train",
  "timestamp": "20250828_123456",
  "summary": {...},
  "episodes": [...]
}
```

## 使用方法

### 1. 自動記録（推奨）
トレーニング実行時、統計は自動的に記録されます：

```bash
python scripts/main.py --experiment_name my_experiment --train_for_env_steps 1000000
```

統計ファイルは以下に保存されます：
```
train_dir/my_experiment/item_stats/
├── my_experiment_train_YYYYMMDD_HHMMSS_items.csv
├── my_experiment_train_YYYYMMDD_HHMMSS_items.npy
└── my_experiment_train_YYYYMMDD_HHMMSS_detailed.json
```

### 2. 評価時の記録
評価実行時も自動的に記録され、train/evalで分けて保存されます：

```bash
python scripts/main.py --experiment_name my_experiment --eval_target altar
```

### 3. 統計分析
保存された統計ファイルを分析するには：

```bash
python test_item_tracker.py --load train_dir/my_experiment/item_stats/
```

## コードレベルでの利用

### 手動での統計取得
```python
from rl_baseline.obs_wrappers import ModifierWrapper

# 環境のラッパーから統計取得
wrapper = env  # ModifierWrapper インスタンス
item_stats = wrapper.get_item_summary()

# 統計の手動保存
wrapper.save_item_statistics(save_dir="custom_stats_dir")
```

### 統計データの読み込み
```python
import numpy as np
import json

# NumPy形式
stats = np.load("path/to/stats_items.npy", allow_pickle=True).item()

# JSON形式
with open("path/to/stats_detailed.json", 'r') as f:
    detailed_stats = json.load(f)
```

## 分析例

### Pythonでの分析例
```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv("path/to/stats_items.csv")

# 使用率の可視化
df.plot(x='item_type', y='usage_rate', kind='bar')
plt.title('Item Usage Rate by Type')
plt.ylabel('Usage Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 最も使用頻度の高いアイテム
top_used = df.nlargest(5, 'usage_rate')
print("Top 5 Most Used Items:")
print(top_used[['item_type', 'usage_rate']])
```

## 注意事項

1. **パフォーマンス**: アイテム追跡は軽量ですが、大量の統計を記録する場合はメモリ使用量に注意してください。

2. **ファイル容量**: 長期間のトレーニングでは、詳細なエピソードデータが大きくなる場合があります。

3. **アイテム検出精度**: インベントリの変化とメッセージ解析の両方で使用・取得を検出していますが、一部のケースで見落としがある可能性があります。

## トラブルシューティング

### よくある問題

**Q: 統計ファイルが生成されない**
A: `ModifierWrapper`が環境ラッパーチェーンに含まれているか確認してください。

**Q: 一部のアイテムが記録されない**
A: NetHackのアイテムクラス番号が予期したものと異なる場合があります。`ItemTracker.ITEM_CLASSES`を確認してください。

**Q: メモリ使用量が多い**
A: `ItemTracker.reset_session_stats()`を定期的に呼び出して統計をクリアするか、エピソード詳細の記録を無効にしてください。

## カスタマイズ

アイテム分類や検出ロジックをカスタマイズする場合は、`rl_baseline/item_tracker.py`を編集してください。特に：

- `ITEM_CLASSES`: アイテムクラスの定義
- `update_from_message()`: メッセージベースの検出ロジック
- `update_inventory()`: インベントリベースの検出ロジック
