import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 1. 数据加载与预处理（修改重点在此部分）
# ----------------------
def load_data(excel_path, csv_folder):
    """加载并合并Excel经济数据和CSV企业数量数据"""
    # 读取Excel文件（所有年份工作表）
    xls = pd.ExcelFile(excel_path)
    grp_dfs = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df['年份'] = int(sheet_name)  # 假设工作表名称为年份
        grp_dfs.append(df)
    grp_data = pd.concat(grp_dfs, ignore_index=True)

    # 读取CSV企业数据（关键修改：仅保留城市和净增加列）
    enterprise_dfs = []
    for file in os.listdir(csv_folder):
        if file.endswith('.csv'):
            year = int(file.split('_')[1].split('.')[0])
            # 使用usecols参数指定需要读取的列
            df = pd.read_csv(os.path.join(csv_folder, file), usecols=['城市', '注销'])
            df['年份'] = year
            enterprise_dfs.append(df)
    enterprise_data = pd.concat(enterprise_dfs, ignore_index=True)

    # 合并数据（此时enterprise_data仅含城市、净增加、年份三列）
    merged_data = pd.merge(grp_data, enterprise_data, on=['城市', '年份'], how='inner')
    return merged_data.sort_values(['城市', '年份'])

# 文件路径设置（需修改为实际路径）
excel_path = "../../城市科技经济指数/城市影响因素.xlsx"
csv_folder = "../../城市企业统计/净增加"
data = load_data(excel_path, csv_folder)

# ----------------------
# 2. 探索性数据分析(EDA)
# ----------------------
print("数据维度:", data.shape)
print("\n前5行数据:")
print(data.head())
print("\n缺失值统计:")
print(data.isnull().sum())

# ----------------------
# 3. 保存合并后的数据到CSV文件
# ----------------------
output_file = "../数据文件/合并后的城市经济与企业数据1.csv"
data.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n数据已保存至：{output_file}")