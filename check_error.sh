#!/bin/bash

# エラーログの保存ディレクトリ
LOG_DIR="errorlog"
DATE_DIR="$(date '+%Y-%m%d')"
DATE_TIME="$(date '+%H_%M_%S')"  # ":" を "_" に変更

# 今日のエラーログフォルダを作成
mkdir -p "$LOG_DIR/$DATE_DIR"

# その日生成されたファイルの番号を取得
LOG_COUNT=$(find "$LOG_DIR/$DATE_DIR" -maxdepth 1 -type f -name '[0-9]*-*' 2>/dev/null | wc -l)
LOG_COUNT=$((LOG_COUNT + 1))

# エラーログのファイル名
LOG_FILE="$LOG_DIR/$DATE_DIR/$LOG_COUNT-$DATE_TIME.log"

# コマンドの実行
"$@" 2> "$LOG_FILE"
EXIT_CODE=$?

# エラーが発生した場合の処理
if [[ -s "$LOG_FILE" || $EXIT_CODE -ne 0 ]]; then
    echo "Error Occured! Check the log: $LOG_FILE"
    ln -sf "../$LOG_FILE" "$LOG_DIR/latest"  # 最新のエラーログを指すように更新
else
    rm -f "$LOG_FILE"  # エラーがなかった場合はログを削除
fi
