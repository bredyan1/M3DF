import os
import pandas as pd
from sklearn.model_selection import KFold
from itertools import zip_longest

# 读取 CSV 文件，获取文件名列
input_csv_path = "/home/yanyiqun/MCAT/brca.csv"  # 替换为实际的name_gbmlgg.csv路径
data = pd.read_csv(input_csv_path)

# 确保文件名列存在
if 'case_id' not in data.columns:
    raise ValueError("CSV文件中缺少'Matching_Folder_Name'列，请确保列名为'Matching_Folder_Name'")

# 获取文件名列表
files = data['case_id'].unique()

# 初始化五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 遍历每个折
for i, (train_index, val_index) in enumerate(kf.split(files), 1):
    train_files = files[train_index]
    val_files = files[val_index]
    
    # 使用 zip_longest 来确保两个列表的长度匹配
    fold_data = pd.DataFrame(list(zip_longest(train_files, val_files, fillvalue=None)), columns=['train', 'val'])
    
    # 保存到CSV文件中
    output_csv_path = f"/home/yanyiqun/MCAT/splits/5fold/tcga_brca/splits_{i}.csv"
    fold_data.to_csv(output_csv_path, index=False)

    print(f"保存第{i}折的训练集和验证集到：{output_csv_path}")
