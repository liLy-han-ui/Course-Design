import pandas as pd
import os

# ========================
# 配置参数
# ========================
forecast_file = "../预测结果/各城市企业数量预测_2024_2030.csv"  # 预测数据文件
historical_folder = "../../城市企业统计/全部企业/"  # 历史数据文件夹
output_file = "../预测结果/combined_city_data_2000_2030.csv"  # 输出文件名

# ========================
# 1. 读取预测数据
# ========================
forecast_df = pd.read_csv(forecast_file)
forecast_cities = forecast_df['城市'].unique().tolist()

# ========================
# 2. 加载所有年份的历史数据（2000 - 2023）
# ========================
all_historical_data = {}

for year in range(2000, 2024):
    file_path = os.path.join(historical_folder, f"全部企业_{year}.csv")

    if not os.path.exists(file_path):
        print(f"警告：找不到 {file_path}")
        continue

    df_year = pd.read_csv(file_path)

    for _, row in df_year.iterrows():
        city = row['城市']
        count = row['全部企业']

        if city not in all_historical_data:
            all_historical_data[city] = {}

        all_historical_data[city][year] = count

# ========================
# 3. 合并历史数据和预测数据（只处理在预测列表中的城市）
# ========================
combined_data = []

for city in forecast_cities:
    row = {"城市": city}

    # 添加历史数据 (2000 - 2023)
    for year in range(2000, 2024):
        row[year] = all_historical_data.get(city, {}).get(year, None)

    # 添加预测数据 (2024 - 2030)
    forecast_row = forecast_df[forecast_df['城市'] == city].iloc[0].to_dict()
    for year in range(2024, 2031):
        col_name = str(year)
        if col_name in forecast_row:
            row[int(col_name)] = forecast_row[col_name]
        else:
            row[int(col_name)] = None

    combined_data.append(row)

# ========================
# 4. 构建 DataFrame 并导出为 CSV
# ========================
combined_df = pd.DataFrame(combined_data)
combined_df = combined_df[["城市"] + list(range(2000, 2031))]  # 排序列顺序

combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"✅ 数据已成功导出至：{output_file}")