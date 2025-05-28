import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ======================
# 数据读取与预处理
# ======================
data = pd.read_csv("../数据文件/合并后的城市经济与企业数据.csv")

def filter_by_year_range(df, start_year, end_year):
    return df[(df['年份'] >= start_year) & (df['年份'] <= end_year)]

start_year = 2015
end_year = 2023
data = filter_by_year_range(data, start_year, end_year)

# 创建特征（如滞后项、增长率等）
def create_features(df):
    df = df.sort_values(['城市', '年份'])
    df['企业数量_Lag1'] = df.groupby('城市')['全部企业'].shift(1)
    df['GRP增长率'] = df.groupby('城市')['GRP（单位：亿元）'].pct_change()
    df['累计政策'] = df.groupby('城市')['政策数量'].cumsum()
    return df.dropna()

data = create_features(data)

# ======================
# 特征选择 + 按年份分组归一化（Min-Max Normalization）
# ======================
features_for_clustering = ['GRP（单位：亿元）', 'AI百度指数','政策数量',
                           '科学技术支出（单位：万元）', '教育支出（单位：万元）']

# 定义一个函数：对每一组（每年）进行 Min-Max 归一化
def min_max_normalize(group, features):
    group = group.copy()
    for col in features:
        min_val = group[col].min()
        max_val = group[col].max()
        if max_val != min_val:
            group[col] = (group[col] - min_val) / (max_val - min_val)
        else:
            group[col] = 0  # 避免除以0
    return group

# 按“年份”分组并应用 Min-Max 归一化
normalized_data = data.groupby('年份').apply(lambda g: min_max_normalize(g, features_for_clustering))

# 重置索引以便后续操作
normalized_data.reset_index(drop=True, inplace=True)

# ======================
# 聚类分析
# ======================
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
normalized_data['cluster'] = kmeans.fit_predict(normalized_data[features_for_clustering])

# ======================
# 可视化聚类结果（PCA降维后展示）
# ======================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_data[features_for_clustering])
pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = normalized_data['cluster']

plt.figure(figsize=(10, 7))
for cluster in range(n_clusters):
    subset = pca_df[pca_df['Cluster'] == cluster]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Cluster {cluster}')

plt.title("Clusters Visualization using PCA after Min-Max Normalization by Year")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()

# ======================
# 查看每个组别包含哪些城市
# ======================
city_to_cluster_dict = normalized_data.set_index('城市')['cluster'].to_dict()

for cluster_id in range(n_clusters):
    cities_in_cluster = normalized_data[normalized_data['cluster'] == cluster_id]['城市'].unique()
    print(f"Cluster {cluster_id} 包含以下城市：")
    for city in cities_in_cluster:
        print(f"- {city}")