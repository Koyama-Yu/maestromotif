import pickle
import csv

# .pkl ファイルを開く
with open("./og_dataset/descender.pkl", "rb") as f:
    data = pickle.load(f)

# データの型を確認
print(type(data))
# CSVファイルに書き込む
with open("./og_dataset_csv/descender.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    # リストの各要素を行として書き込む
    for row in data:
        writer.writerow(row)