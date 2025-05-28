import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from linearmodels import PanelOLS
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据（路径需确认）
df = pd.read_csv("../数据文件/合并后的企业流失数据.csv")

# ----------- 列名统一 -----------
df = df.rename(columns={
    '科学技术支出（单位：万元）': '科技支出',
    '教育支出（单位：万元）': '教育支出',
    '注销': '注销'
})

# ----------- 缺失值处理 -----------
df = df.sort_values(by=['城市', '年份'])
df = df.groupby('城市').apply(lambda x: x.ffill()).reset_index(drop=True)

# ----------- 异常值处理（Z-score）-----------
z_scores = df.groupby('城市')[['科技支出', '教育支出', '注销']].transform(
    lambda x: (x - x.mean()) / x.std()
)
df = df[(z_scores.abs() < 3).all(axis=1)]

# ----------- 标准化（按年份分组）-----------
scaler = StandardScaler()
for col in ['科技支出', '教育支出', '注销']:
    df[f'{col}_z'] = df.groupby('城市')[col].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )

# ----------- 构建面板索引 -----------
df = df.set_index(['城市', '年份'])

# ----------- 可视化示例 -----------
sample_city = df.index.get_level_values('城市').unique()[0]
df_sample = df.xs(sample_city).reset_index()

plt.figure(figsize=(10, 5))
plt.plot(df_sample['年份'], df_sample['教育支出_z'], label='教育支出')
plt.plot(df_sample['年份'], df_sample['科技支出_z'], label='科技支出')
plt.plot(df_sample['年份'], df_sample['注销_z'], label='注销')
plt.title(f'{sample_city} 教育/科技支出与注销企业趋势')
plt.legend()
plt.show()

# ----------- 相关性热力图 -----------
corr = df[['科技支出', '教育支出', '注销']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('变量间相关性')
plt.show()

# ----------- 面板回归模型 -----------
model = PanelOLS(
    dependent=df['注销'],
    exog=df[['科技支出', '教育支出', 'GRP（单位：亿元）', "全部企业", '政策数量']],
    entity_effects=True,
    time_effects=True
)
results = model.fit()
print(results.summary)

# 提取残差
df['残差'] = results.resids

# ----------- 风险评分模型（按年份+城市）-----------
df_reset = df.reset_index()

# 按年份+城市标准化后的指标
risk_by_year_city = df_reset.groupby(['年份', '城市'])[['教育支出_z', '科技支出_z', '注销_z']].mean().reset_index()

# 计算风险评分（公式可根据业务调整）
risk_by_year_city['风险评分'] = (
    -(0.1 * risk_by_year_city['教育支出_z'])
    -(0.1 * risk_by_year_city['科技支出_z'])
    +(0.8 * risk_by_year_city['注销_z'])
)

# 合并回原始数据框
df = pd.merge(df_reset, risk_by_year_city[['年份', '城市', '风险评分']], on=['年份', '城市'], how='left').set_index(['城市', '年份'])

# ----------- 动态预警分析（新规则）-----------
df_reset = df.reset_index()

# 添加增长率列
df_reset['风险评分_growth'] = df_reset.groupby('城市')['风险评分'].pct_change()
df_reset['注销_growth'] = df_reset.groupby('城市')['注销'].pct_change()

# 设置阈值
score_threshold = 1.9

# 判断是否满足各条规则
above_threshold = df_reset['风险评分'] > score_threshold
rule1 = (df_reset['风险评分_growth'] > 0) & (df_reset['风险评分_growth'].shift(1) > 0)  # 连续两年增长
rule2 = df_reset['注销_growth'] > 1  # 注销增长率超过阈值
rule3 = (above_threshold & above_threshold.shift(1)) | (~above_threshold.shift(1).fillna(True)) & above_threshold  # 首次或连续两年高于阈值

# 最终判断：取每个城市最新一年的记录作为结果
dynamic_alert_data = df_reset.groupby('城市').apply(
    lambda g: pd.Series({
        '预警等级': (
            '红色预警' if ((rule1[g.index[-1]] if len(g) > 0 else False) and rule3[g.index[-1]] if len(g) > 0 else False) else
            '橙色预警' if (rule2[g.index[-1]] if len(g) > 0 else False or rule3[g.index[-1]] if len(g) > 0 else False) else
            '黄色预警' if (rule1[g.index[-1]] if len(g) > 0 else False) else
            '无预警'
        ),
        '风险评分增长率': df_reset.loc[g.index[-1], '风险评分_growth'],
        '注销增长率': df_reset.loc[g.index[-1], '注销_growth'],
        '最新风险评分': df_reset.loc[g.index[-1], '风险评分']
    })
).reset_index()

# 将动态预警结果合并到 final_risk
trend_df = dynamic_alert_data.set_index('城市')[['预警等级', '风险评分增长率', '注销增长率', '最新风险评分']]
trend_df = trend_df.rename(columns={'预警等级': '动态预警'})

# 输出动态预警结果
print("动态预警结果：\n", trend_df['动态预警'].value_counts())

# ----------- 聚类分析 -----------
# 使用所有年份的平均风险指标
risk_df = df_reset.groupby('城市')[['教育支出_z', '科技支出_z', '注销_z', '风险评分']].mean()

X = risk_df[['教育支出_z', '科技支出_z', '注销_z']].fillna(0)

# 确定最佳聚类数
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss, marker='o')
plt.title('肘部法则确定最佳聚类数')
plt.xlabel('聚类数')
plt.ylabel('WCSS')
plt.show()

# 选择 n_clusters=3
kmeans = KMeans(n_clusters=3, random_state=42)
risk_df['聚类类别'] = kmeans.fit_predict(X)

# 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='教育支出_z', y='科技支出_z',
    hue='聚类类别', size='注销_z',
    data=risk_df.reset_index(),
    palette='viridis',
    sizes=(20, 200)
)
plt.title('城市风险聚类分布（教育支出 vs 科技支出）')
plt.xlabel('标准化教育支出')
plt.ylabel('标准化科技支出')
plt.show()

# ----------- 最终输出 -----------
final_risk = risk_df.join(trend_df).join(df.groupby('城市')['残差'].mean())

# 根据风险评分划分风险等级
conditions = [
    (final_risk['最新风险评分'] >1.9),
    (final_risk['最新风险评分'] > 1.3) & (final_risk['最新风险评分'] <= 1.9),
    (final_risk['最新风险评分'] <= 1.3)
]
choices = ['高风险', '中风险', '低风险']

final_risk['风险等级'] = np.select(conditions, choices, default='低风险')

# 导出结果
final_risk.sort_values('风险等级', ascending=False).to_csv('../预测结果/城市风险评级结果.csv')
print("分析完成！结果已保存至 ../预测结果/城市风险评级结果.csv")