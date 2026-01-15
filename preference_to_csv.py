import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs='*', default=None,
                    help='Specific .npy files to convert (relative to input_directory if not absolute).')
parser.add_argument('--input_directory', default='preference',
                    help='Directory containing .npy files.')
parser.add_argument('--output_directory', default='preference_csv',
                    help='Directory to write .csv files.')
flags = parser.parse_args()

input_directory = flags.input_directory
output_directory = flags.output_directory

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

if flags.input:
    filenames = []
    for name in flags.input:
        if os.path.isabs(name):
            filenames.append(name)
        else:
            filenames.append(os.path.join(input_directory, name))
else:
    filenames = [
        os.path.join(input_directory, filename)
        for filename in os.listdir(input_directory)
        if filename.endswith('.npy')
    ]

for filepath in filenames:
    filename = os.path.basename(filepath)
    data = np.load(filepath)

    # CSVファイルとして保存
    csv_filename = os.path.splitext(filename)[0] + '.csv'
    csv_filepath = os.path.join(output_directory, csv_filename)
    np.savetxt(csv_filepath, data, delimiter=',')

    print(f"File {filename} has been converted to {csv_filename} and saved in {output_directory}.")
