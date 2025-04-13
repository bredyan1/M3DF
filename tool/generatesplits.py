# import os
# import pandas as pd
# from sklearn.model_selection import KFold

# # 步骤1：提取文件夹名称
# def get_folder_names(path):
#     # 提取所有.pt文件的文件名
#     return [filename[:12] for filename in os.listdir(path) if filename.endswith('.pt')]

# # 步骤2：五折交叉验证划分
# def split_folders(folders, n_splits=5):
#     kf = KFold(n_splits=n_splits, shuffle=True)
#     splits = []
#     for train_index, val_index in kf.split(folders):
#         train_folders = [folders[i] for i in train_index]
#         val_folders = [folders[i] for i in val_index]
#         splits.append((train_folders, val_folders))
#     return splits

# # 步骤3：保存到CSV文件
# def save_splits_to_csv(splits, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     for i, (train, val) in enumerate(splits):
#         df = pd.DataFrame({'train': pd.Series(train), 'val': pd.Series(val)})
#         output_path = os.path.join(output_folder, f'splits_{i}.csv')
#         df.to_csv(output_path, index=False)

# # 主程序
# if __name__ == "__main__":
#     A_folder_path = '/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_gbmlgg/tcga_gbmlgg_mri_features_3d_resnet50'  # 替换为实际的A文件夹路径
#     C_folder_path = '/home/yanyiqun/MCAT/splits/5fold/tcga_gbmlgg'  # 替换为实际的C文件夹路径
#     B = get_folder_names(A_folder_path)
#     splits = split_folders(B)
#     save_splits_to_csv(splits, C_folder_path)
import os
import pandas as pd
from sklearn.model_selection import KFold

# 步骤1：从CSV文件中读取文件名
def get_selected_files_from_csv(csv_file, column_name):
    # 从CSV文件中读取指定列的文件名或前12个字符
    df = pd.read_csv(csv_file)
    return df[column_name].tolist()  # 假设列名为column_name，返回列表

# 步骤2：五折交叉验证划分
def split_folders(folders, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    splits = []
    for train_index, val_index in kf.split(folders):
        train_folders = [folders[i] for i in train_index]
        val_folders = [folders[i] for i in val_index]
        splits.append((train_folders, val_folders))
    return splits

# 步骤3：保存到CSV文件
def save_splits_to_csv(splits, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, (train, val) in enumerate(splits):
        df = pd.DataFrame({'train': pd.Series(train), 'val': pd.Series(val)})
        output_path = os.path.join(output_folder, f'splits_{i}.csv')
        df.to_csv(output_path, index=False)

# 主程序
if __name__ == "__main__":
    # CSV文件路径，假设它包含了我们需要的文件名（或前12个字符），列名为'FileName'
    csv_file_path = '/home/yanyiqun/MCAT/name3.csv'  # 替换为你的CSV文件路径
    column_name = 'CommonValues'  # 替换为实际的列名，包含文件名或前12个字符
    
    A_folder_path = '/mnt/sdb/yanyiqun/TCGA_ROOT_DIR/tcga_gbmlgg/tcga_gbmlgg_mri_features_3d_resnet50'  # 替换为实际的A文件夹路径
    C_folder_path = '/home/yanyiqun/MCAT/splits/5fold/tcga_gbmlgg'  # 替换为实际的C文件夹路径
    
    # 从CSV文件中读取选择的文件名（或前12个字符）
    selected_files = get_selected_files_from_csv(csv_file_path, column_name)
    # 提取选定文件名的前12个字符，并且确保它们存在于A文件夹中
    selected_folders = [filename[:12] for filename in selected_files if filename[:12] in os.listdir(A_folder_path)]
    
    # 执行五折交叉验证
    splits = split_folders(selected_files)
    
    # 保存分割结果到CSV文件
    save_splits_to_csv(splits, C_folder_path)
