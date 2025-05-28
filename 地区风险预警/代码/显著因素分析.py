import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# 设置中文显示和图形样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_and_preprocess(filepath):
    """数据加载与预处理（修复分组警告）"""
    df = pd.read_csv(filepath)

    # 数据清洗（添加group_keys=False修复警告）
    df = df.sort_values(['城市', '年份'])
    df = df.groupby('城市', group_keys=False).apply(lambda x: x.ffill().bfill())

    # 创建强度指标（避免除零错误）
    df['GRP（单位：亿元）'] = df['GRP（单位：亿元）'].replace(0, np.nan)
    df['科技支出强度'] = df['科学技术支出（单位：万元）']/df['GRP（单位：亿元）']
    df['教育支出强度'] = df['教育支出（单位：万元）']/df['GRP（单位：亿元）']

    # 创建滞后变量（优化缺失值处理）
    lag_years = 0
    df = df.sort_values(['城市', '年份'])
    for col in ['科学技术支出（单位：万元）', '政策数量']:
        df[f'{col}_lag{lag_years}'] = df.groupby('城市')[col].shift(lag_years)
    df = df.dropna(subset=[f'科学技术支出（单位：万元）_lag{lag_years}',
                           f'政策数量_lag{lag_years}'])

    # 缩尾处理（分城市处理）
    numeric_cols = ['科技支出强度', '教育支出强度', 'AI百度指数', 'GRP（单位：亿元）']
    df[numeric_cols] = df.groupby('城市', group_keys=False)[numeric_cols].transform(
        lambda x: x.clip(x.quantile(0.05), x.quantile(0.95))
    )

    return df.set_index(['城市', '年份'])


def run_panel_regression(df):
    """执行面板回归分析（优化变量选择）"""
    # 准备变量（移除不显著变量）
    exog_vars = [
        'AI百度指数',
        'GRP（单位：亿元）',
        '政策数量_lag0',
        '科技支出强度',
        '教育支出强度'
    ]

    # 确保数据对齐
    valid_index = df[exog_vars].dropna().index
    exog = df.loc[valid_index, exog_vars]
    dep_var = df.loc[valid_index, '注销']

    # 固定效应模型（简化模型）
    model = PanelOLS(
        dependent=dep_var,
        exog=exog,
        entity_effects=True,
        time_effects=True
    )

    return model.fit(cov_type='clustered', cluster_entity=True)


def diagnose_model(results, exog_df):
    """模型诊断（修复维度问题）"""
    # 多重共线性检查
    vif_data = pd.DataFrame()
    vif_data["Variable"] = exog_df.columns
    vif_data["VIF"] = [variance_inflation_factor(exog_df.values, i)
                       for i in range(exog_df.shape[1])]

    # 残差分析（修复维度问题）
    residuals = results.resids
    predicted = results.predict()

    # 转换为numpy数组并展平
    residuals_flat = np.asarray(residuals).flatten()
    predicted_flat = np.asarray(predicted).flatten()

    # 确保长度一致
    min_length = min(len(residuals_flat), len(predicted_flat))
    residuals_flat = residuals_flat[:min_length]
    predicted_flat = predicted_flat[:min_length]

    plt.figure(figsize=(12, 5))

    # 残差分布图
    plt.subplot(1, 2, 1)
    sns.histplot(residuals_flat, kde=True, bins=30)
    plt.title('残差分布')

    # 拟合值 vs 残差
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=predicted_flat, y=residuals_flat, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('拟合值 vs 残差')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.tight_layout()

    return vif_data


def identify_key_factors(results, threshold=0.1):
    """识别关键因素（优化输出格式）"""
    params = results.params
    pvalues = results.pvalues

    # 创建结果表格
    result_df = pd.DataFrame({
        '系数': params,
        'P值': pvalues,
        '显著性': pvalues.apply(lambda x: '***' if x < 0.01 else '**' if x < 0.05 else '*' if x < 0.1 else '')
    })

    # 筛选显著变量
    significant_vars = result_df[result_df['P值'] < threshold].index.tolist()

    return result_df, significant_vars


# -------------------- 执行主程序 --------------------
if __name__ == "__main__":
    # 1. 数据加载与预处理
    df = load_and_preprocess("../数据文件/合并后的企业流失数据.csv")

    # 2. 运行面板回归
    results = run_panel_regression(df)
    print("========== 回归结果汇总 ==========")
    print(results.summary)

    # 3. 模型诊断（使用原始数据框替代model.exog）
    exog_df = df[results.model.exog.vars]
    vif_table = diagnose_model(results, exog_df)
    print("\n========== 多重共线性检查 ==========")
    print(vif_table)

    # 4. 识别关键因素
    result_table, key_vars = identify_key_factors(results)
    print("\n========== 回归系数详情 ==========")
    print(result_table)

    print("\n========== 关键影响因素（p<0.1） ==========")
    print(result_table.loc[key_vars])