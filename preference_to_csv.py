import numpy as np
import os

input_directory = 'preference'
output_directory = 'preference_csv'

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 入力ディレクトリ内のすべての.npyファイルを処理
for filename in os.listdir(input_directory):
    if filename.endswith('.npy'):
        input_filepath = os.path.join(input_directory, filename)
        data = np.load(input_filepath)
        
        # CSVファイルとして保存
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        csv_filepath = os.path.join(output_directory, csv_filename)
        np.savetxt(csv_filepath, data, delimiter=',')
        
        print(f"File {filename} has been converted to {csv_filename} and saved in {output_directory}.")