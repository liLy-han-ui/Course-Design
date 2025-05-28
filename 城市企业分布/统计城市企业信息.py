import pandas as pd
import os

# 输入输出路径
input_file = "../原始数据/modified_data.csv"  # 输入文件路径
output_dir = "../城市企业统计"  # 输出根目录

# 创建输出目录
os.makedirs(f"{output_dir}/全部企业", exist_ok=True)
os.makedirs(f"{output_dir}/净增加", exist_ok=True)

# 读取 CSV 文件
df = pd.read_csv(input_file, header=0, on_bad_lines='skip')

# 数据预处理：替换 '-' 为 NaN 并删除缺省值行
df_clean = df.replace('-', pd.NA).dropna(how='any').copy()

# 转换日期列到 datetime 类型，并提取年份
df_clean['成立日期'] = pd.to_datetime(df_clean['成立日期'], errors='coerce')
df_clean['核准日期'] = pd.to_datetime(df_clean['核准日期'], errors='coerce')

# 提取成立年份
df_clean['成立年份'] = df_clean['成立日期'].dt.year

# 设置注销状态集合
status_valid = ["在业", "正常", "存续", "其他", "迁出", "核准设立", "存续/在业"]

# 提取注销年份
df_clean['注销年份'] = df_clean.apply(
    lambda row: row['核准日期'].year if (row['登记状态'] not in status_valid and pd.notnull(row['核准日期'])) else None,
    axis=1
)

# 定义时间跨度变量
time_span = 1  # 时间跨度，单位为年

# 定义起始年份和结束年份
start_year = 2000
end_year = 2025

# 根据时间跨度生成要统计的年份和时间范围
years_to_check = list(range(start_year, end_year + 1, time_span))
time_ranges = [(y, y + time_span - 1) for y in range(start_year, end_year + 1, time_span)]

# 统计每个指定年份的全部企业数量
for year_to_check in years_to_check:
    all_valid_companies_df = df_clean[
        (df_clean['成立年份'] <= year_to_check) &
        ((df_clean['注销年份'].isna()) | (df_clean['注销年份'] > year_to_check))
    ]
    all_city_count = all_valid_companies_df.groupby("所属城市", as_index=False).size().rename(columns={"size": "全部企业"})
    all_city_count.rename(columns={"所属城市": "城市"}, inplace=True)

    all_output_file = f"{output_dir}/全部企业/全部企业_{year_to_check}.csv"
    all_city_count.to_csv(all_output_file, index=False)
    print(f"统计完成，已将 {year_to_check} 的全部企业结果保存到 {all_output_file}")

# 统计每段时间内的净增加企业数量
for start_year_range, end_year_range in time_ranges:
    new_companies_df = df_clean[
        (df_clean['成立年份'] >= start_year_range) &
        (df_clean['成立年份'] <= end_year_range)
    ]
    closed_companies_df = df_clean[
        df_clean['注销年份'].notna() &
        (df_clean['注销年份'] >= start_year_range) &
        (df_clean['注销年份'] <= end_year_range)
    ]

    # 新增公司统计
    new_city_count = new_companies_df.groupby("所属城市", as_index=False).size().rename(columns={"size": "新增"})

    # 注销公司统计
    closed_city_count = closed_companies_df.groupby("所属城市", as_index=False).size().rename(columns={"size": "注销"})

    # 合并新增和注销
    result_df = pd.merge(new_city_count, closed_city_count, on="所属城市", how="outer").fillna(0)

    # 计算净增加
    result_df["净增加"] = result_df["新增"].astype(int) - result_df["注销"].astype(int)

    # 修改列名为“城市”
    result_df.rename(columns={"所属城市": "城市"}, inplace=True)

    # 保存到CSV
    net_output_file = f"{output_dir}/净增加/净增加企业_{start_year_range}.csv"
    result_df.to_csv(net_output_file, index=False)
    print(f"统计完成，已将 {start_year_range}-{end_year_range} 的净增加结果保存到 {net_output_file}")