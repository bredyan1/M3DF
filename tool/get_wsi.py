import os
import shutil
import pandas as pd

def get_prefixes_from_csv(csv_file):
    """
    从 name.csv 文件中读取所有前12个字符
    """
    df = pd.read_csv(csv_file)
    # 假设列名为 'FilePrefix'，根据实际列名调整
    prefixes = set(df['SubfolderName'].tolist())  
    return prefixes

def copy_matching_svs_files(folder_A, folder_B, prefixes):
    """
    遍历文件夹A中的所有子文件夹，找到匹配的 .svs 文件并复制到文件夹B
    """
    if not os.path.exists(folder_B):
        os.makedirs(folder_B)  # 如果文件夹B不存在，创建它
    
    # 遍历文件夹A及其所有子文件夹
    for dirpath, dirnames, filenames in os.walk(folder_A):
        for filename in filenames:
            if filename.endswith('.svs'):
                file_prefix = filename[:12]  # 获取文件的前12个字符
                if file_prefix in prefixes:
                    # 如果前12个字符匹配，则复制文件到文件夹B
                    source_file = os.path.join(dirpath, filename)
                    destination_file = os.path.join(folder_B, filename)
                    shutil.copy(source_file, destination_file)
                    print(f"复制了文件: {filename}")

# 指定文件路径
csv_file = '/home/yanyiqun/MCAT/name.csv'  # name.csv 文件路径
folder_A = '/mnt/sdb/yanyiqun/TCGA-GBM_WSI'  # 文件夹A路径
folder_B = '/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_gbmlgg/WSIs_gbm'  # 文件夹B路径

# 从 CSV 文件中获取前12个字符
prefixes = get_prefixes_from_csv(csv_file)

# 复制匹配的 .svs 文件到文件夹B
copy_matching_svs_files(folder_A, folder_B, prefixes)

print('匹配的 .svs 文件已复制到文件夹B。')
