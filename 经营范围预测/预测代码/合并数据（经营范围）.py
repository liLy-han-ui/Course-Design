import pandas as pd
from collections import defaultdict

# 1. 加载统计结果文件（2000 年数据）
stat_df = pd.read_csv("../../企业经营范围/企业统计结果.csv")
# 提取需要的字段：年份、领域、累计存活企业数量
stat_df = stat_df[['年份', '领域', '累计存活企业数量']]
all_years = list(range(2000, 2031))
# 2. 加载预测文件（2025~2030 年）
pred_df = pd.read_csv("../预测结果/综合预测结果.csv")
# 只取预测值（忽略下限和上限）
pred_df = pred_df[['领域', '年份', '预测值']]

# 重命名列以便后续处理
pred_df.rename(columns={'预测值': '累计存活企业数量'}, inplace=True)

# 确保只合并出现在预测结果中的领域
all_domains = sorted(set(pred_df['领域']))

# 构建最终 DataFrame
full_data = []

for domain in all_domains:
    # 初始化每年数据为空
    yearly_data = {year: None for year in all_years}

    # 填充统计结果中的已有数据
    stat_data = stat_df[(stat_df['领域'] == domain)]
    for _, row in stat_data.iterrows():
        if row['年份'] in yearly_data:
            yearly_data[row['年份']] = row['累计存活企业数量']

    # 填充预测数据
    pred_data = pred_df[(pred_df['领域'] == domain)]
    for _, row in pred_data.iterrows():
        if row['年份'] in yearly_data:
            yearly_data[row['年份']] = row['累计存活企业数量']


    # 填充缺失值：向前填充（即使用上一年的数据）
    last_known_value = 0  # 默认从零开始
    for year in sorted(yearly_data):
        if yearly_data[year] is not None:
            last_known_value = int(yearly_data[year])
        else:
            yearly_data[year] = last_known_value  # 填充为上一年的值

    # 转换为列表加入 full_data
    for year in yearly_data:
        full_data.append({
            "年份": year,
            "领域": domain,
            "累计存活企业数量": yearly_data[year]
        })

# 转为 DataFrame
full_df = pd.DataFrame(full_data, columns=["年份", "领域", "累计存活企业数量"])

# 排序
full_df.sort_values(by=["领域", "年份"], inplace=True)
full_df.reset_index(drop=True, inplace=True)

# 保存到新文件
full_df.to_csv("../预测结果/企业技术领域发展数据_2000_2030.csv", index=False)

print("✅ 数据已成功合并并保存为：企业技术领域发展数据_2000_2030.csv")