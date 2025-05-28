import pandas as pd

# 读取两个CSV文件，请替换为您自己的文件路径
df1 = pd.read_csv("../数据文件/合并后的城市经济与企业数据1.csv", encoding='utf-8')   # 注销数据文件
df2 = pd.read_csv("../../城市企业数量预测/数据文件/合并后的城市经济与企业数据.csv", encoding='utf-8')  # 全部企业数据文件

# 显示前几行数据查看结构
print("注销数据前5行：")
print(df1.head())
print("\n全部企业数据前5行：")
print(df2.head())

# 合并两个数据集：按 '城市' 和 '年份' 进行左连接（left join）
merged_df = pd.merge(
    df1,
    df2[['城市', '年份', '全部企业']],
    on=['城市', '年份'],
    how='left'
)

# 检查是否合并成功
print("\n合并后的数据前5行：")
print(merged_df.head())

# 添加“注销率”列：注销 / 全部企业（注意除零处理）
merged_df['注销率'] = merged_df['注销'] / merged_df['全部企业']


final_columns = [
    '城市', 'GRP（单位：亿元）', 'AI百度指数', '政策数量',
        '科学技术支出（单位：万元）',
    '教育支出（单位：万元）', '年份',
    '全部企业', '注销', '注销率'
]
# 生成最终DataFrame
final_df = merged_df[final_columns]

# 保存结果到新的CSV文件
final_df.to_csv("../数据文件/合并后的企业流失数据.csv", index=False, encoding='utf-8-sig')

print("\n✅ 数据合并完成，已保存为：合并后的企业流失数据.csv")