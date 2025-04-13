import os
import csv
import shutil

# 设置CSV文件A的路径
csv_a_path = '/home/yanyiqun/MCAT/matching_case_ids_gbm.csv'

# 设置文件夹B的路径
folder_b_path = '/mnt/sdb/yanyiqun/TCGA-GBM_mri/TCIA_TCGA-GBM_09-16-2015/TCGA-GBM'

# 设置目标文件夹C的路径
folder_c_path = '/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_gbmlgg/MRI_gbm/MRI'  # 需要替换为实际路径

# 读取CSV文件A，获取matching_case_id列的内容
matching_case_ids = set()
with open(csv_a_path, newline='', encoding='utf-8') as csvfile:
    csvreader = csv.DictReader(csvfile)  # 假设CSV文件有标题行
    for row in csvreader:
        matching_case_ids.add(row['matching_case_id'])

# 初始化计数器
count = 0

# 确保目标文件夹C存在
if not os.path.exists(folder_c_path):
    os.makedirs(folder_c_path)

# 遍历文件夹B中的所有子文件夹
for subfolder in os.scandir(folder_b_path):
    if subfolder.is_dir() and subfolder.name in matching_case_ids:
        found = False
        # 如果子文件夹名称在matching_case_ids中，遍历它的二级子文件夹
        for subfolder1 in os.scandir(subfolder.path):
            if found: break  # 跳出二级子文件夹循环
            for subfolder2 in os.scandir(subfolder1.path):
                if 'T1' in subfolder2.name or 't1' in subfolder2.name:
                    count += 1
                    found = True  # 标记为找到，跳出所有循环
                    # 复制找到的子文件夹到文件夹C，并重命名
                    target_path = os.path.join(folder_c_path, subfolder.name)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)  # 如果目标路径不存在，创建它
                    
                    # 使用shutil.copytree复制文件夹
                    shutil.copytree(subfolder2.path, os.path.join(target_path, subfolder2.name))
                    break  # 跳出最内层循环

# 输出计数
print(f"Number of matched second-level subfolders with 'T1' or 't1': {count}")