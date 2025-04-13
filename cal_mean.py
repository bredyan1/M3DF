import os
import pandas as pd

# 指定要遍历的文件夹路径
root_folder_path = '/home/yanyiqun/MCAT/results/5fold'

# 使用os.walk递归遍历文件夹及其子文件夹
for dirpath, dirnames, filenames in os.walk(root_folder_path):
    # 检查子文件夹中是否包含summary_latest.csv文件
    if 'summary_latest.csv' in filenames:
        file_path = os.path.join(dirpath, 'summary_latest.csv')
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 计算统计量
        mean_val = df['val_cindex'].mean()
        std_val = df['val_cindex'].std()
        
        # 格式化字符串（保留3位小数）
        formatted_result = f"{mean_val:.3f}±{std_val:.3f}"
        
        # 构造新的文件名
        new_filename = f'summary_{formatted_result}.csv'
        new_file_path = os.path.join(dirpath, new_filename)
        
        # 保存新的CSV文件
        df.to_csv(new_file_path, index=False)
        
        print(f"文件已保存为 {new_filename} 在 {dirpath}")