from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 设置路径
model_save_dir = '../预测模型'
result_save_dir = '../预测结果'

# 创建目录（如果不存在）
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(result_save_dir, exist_ok=True)

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
features_for_clustering = ['GRP（单位：亿元）', 'AI百度指数', '全部企业',
                           '科学技术支出（单位：万元）', '教育支出（单位：万元）']

def min_max_normalize(group, features):
    group = group.copy()
    for col in features:
        min_val = group[col].min()
        max_val = group[col].max()
        if max_val != min_val:
            group[col] = (group[col] - min_val) / (max_val - min_val)
        else:
            group[col] = 0
    return group

normalized_data = data.groupby('年份').apply(lambda g: min_max_normalize(g, features_for_clustering))
normalized_data.reset_index(drop=True, inplace=True)

# ======================
# 聚类分析
# ======================
n_clusters = 6
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

# ======================
# 分析 Cluster 内部趋势和波动率
# ======================
def analyze_cluster_trend_and_volatility(cluster_data):
    trend_strengths = []
    volatilities = []

    for city, group in cluster_data.groupby('城市'):
        y = group['全部企业'].values
        growth = pd.Series(y).pct_change().dropna()
        if len(growth) < 2:
            continue
        trend_strength = abs(growth.mean()) * len(growth)
        volatility = growth.std()
        trend_strengths.append(trend_strength)
        volatilities.append(volatility)

    avg_trend = np.mean(trend_strengths) if trend_strengths else 0
    avg_volatility = np.mean(volatilities) if volatilities else 0

    return {
        'trend': avg_trend,
        'volatility': avg_volatility
    }

cluster_profiles = {}
for cluster_id in range(n_clusters):
    mask = normalized_data['cluster'] == cluster_id
    cluster_data = normalized_data[mask]
    profile = analyze_cluster_trend_and_volatility(cluster_data)
    cluster_profiles[cluster_id] = profile

# ======================
# 模型类型映射
# ======================
model_mapping = {}

for cluster_id, profile in cluster_profiles.items():
    trend = profile['trend']
    volatility = profile['volatility']

    if trend > 0.5 and volatility > 0.3:
        model_type = 'exponential_smoothing'
    elif trend < 0.2 and volatility < 0.1:
        model_type = 'linear_regression'
    else:
        model_type = 'holt'

    model_mapping[cluster_id] = model_type

# ======================
# 模型训练并保存 + 测试评估
# ======================
trained_models = {}  # 存放所有 cluster 的训练模型
evaluation_results = []  # 存放评估结果

# 划分训练集和测试集（例如最后两年作为测试集）
train_data = normalized_data[normalized_data['年份'] < end_year - 1]
test_data = normalized_data[normalized_data['年份'] >= end_year - 1]

for cluster_id in range(n_clusters):
    train_mask = train_data['cluster'] == cluster_id
    test_mask = test_data['cluster'] == cluster_id
    train_cluster = train_data[train_mask]
    test_cluster = test_data[test_mask]

    # 提取训练数据
    all_y_train = []
    for city, group in train_cluster.groupby('城市'):
        y_train = group['全部企业'].values
        all_y_train.extend(y_train)

    X_train = np.arange(len(all_y_train)).reshape(-1, 1)
    y_train = np.array(all_y_train)

    # 训练模型
    model_type = model_mapping[cluster_id]

    if model_type == 'exponential_smoothing':
        model = ExponentialSmoothing(y_train, trend='add', seasonal=None).fit()
    elif model_type == 'linear_regression':
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        model = lr
    elif model_type == 'holt':
        model = Holt(y_train).fit()
    else:
        sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
        model = sarima

    trained_models[cluster_id] = model

    # 保存模型
    model_file = os.path.join(model_save_dir, f"cluster_{cluster_id}_model.pkl")
    joblib.dump(model, model_file)

    # 测试模型
    mae_list = []
    rmse_list = []

    for city, group in test_cluster.groupby('城市'):
        y_test = group['全部企业'].values
        if len(y_test) < 2:
            continue

        if model_type == 'exponential_smoothing':
            model_test = ExponentialSmoothing(y_test, trend='add', seasonal=None).fit()
            forecast = model_test.forecast(steps=1)
        elif model_type == 'linear_regression':
            X_test = np.arange(len(y_test)).reshape(-1, 1)
            model_test = LinearRegression().fit(X_test, y_test)
            forecast = model_test.predict(np.array([[len(y_test)]]))
        elif model_type == 'holt':
            model_test = Holt(y_test).fit()
            forecast = model_test.forecast(steps=1)
        else:
            model_test = SARIMAX(y_test, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
            forecast = model_test.get_forecast(steps=1).predicted_mean.values

        true_value = y_test[-1]
        pred_value = forecast[0]

        mae = abs(true_value - pred_value)
        rmse = (true_value - pred_value) ** 2

        mae_list.append(mae)
        rmse_list.append(rmse)

    if mae_list:
        avg_mae = np.mean(mae_list)
        avg_rmse = np.sqrt(np.mean(rmse_list))
        evaluation_results.append({
            'Cluster': cluster_id,
            'Model Type': model_type,
            'MAE': avg_mae,
            'RMSE': avg_rmse
        })

# 输出评估结果
print("\n=== 模型评估结果 ===")
eval_df = pd.DataFrame(evaluation_results)
print(eval_df.to_string(index=False))

# 保存评估结果
eval_df.to_csv(os.path.join(result_save_dir, "模型评估结果.csv"), index=False, encoding='utf-8-sig')

# ======================
# 预测函数（无输出、无绘图）
# ======================
def evaluate_and_predict(city_name, data_all, future_years=7):
    city_data = data_all[data_all['城市'] == city_name].sort_values('年份')
    y_city = city_data['全部企业'].values

    city_cluster = city_to_cluster_dict.get(city_name)
    if city_cluster is None:
        raise ValueError(f"城市 {city_name} 未找到对应的聚类编号")

    model_type = model_mapping[city_cluster]
    model = trained_models[city_cluster]

    if model_type == 'exponential_smoothing':
        model = ExponentialSmoothing(y_city, trend='add', seasonal=None).fit()
        forecast = model.forecast(steps=future_years)
    elif model_type == 'linear_regression':
        X_city = np.arange(len(y_city)).reshape(-1, 1)
        model.fit(X_city, y_city)
        forecast = model.predict(np.arange(len(y_city), len(y_city) + future_years).reshape(-1, 1))
    elif model_type == 'holt':
        model = Holt(y_city).fit()
        forecast = model.forecast(steps=future_years)
    else:
        model = SARIMAX(y_city, order=(1, 1, 1), seasonal_order=(0, 1, 0, 12)).fit(disp=False)
        forecast = model.get_forecast(steps=future_years).predicted_mean.values

    forecast = np.round(forecast).astype(int)

    last_year = city_data['年份'].max()
    pred_years = [last_year + i + 1 for i in range(forecast.shape[0])]
    return dict(zip(pred_years, forecast))

# ======================
# 遍历所有城市进行预测 + 保存到 CSV
# ======================
all_cities = data['城市'].unique()
future_years = 7  # 预测 2024 - 2030
all_predictions = []

for city_name in all_cities:
    try:
        prediction = evaluate_and_predict(
            city_name=city_name,
            data_all=data,
            future_years=future_years
        )
        row = {'城市': city_name}
        for year in range(2024, 2024 + future_years):
            row[str(year)] = prediction.get(year, None)
        all_predictions.append(row)
    except Exception as e:
        print(f"预测失败: {city_name}, 错误: {e}")
        continue

# 保存预测结果
csv_file_path = os.path.join(result_save_dir, "各城市企业数量预测_2024_2030.csv")
output_df = pd.DataFrame(all_predictions)
output_df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 预测完成，已保存至: {csv_file_path}")