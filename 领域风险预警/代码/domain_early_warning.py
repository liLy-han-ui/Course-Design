import pandas as pd
import numpy as np


# 1. 数据加载与预处理
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['年份'] = df['年份'].astype(int)
    historical = df[df['年份'] <= 2024]
    forecast = df[df['年份'] >= 2025]
    return df, historical, forecast


# 2. 计算增长率
def calculate_growth_rate(group):
    group = group.sort_values('年份')
    group['增长率'] = group['累计存活企业数量'].pct_change().fillna(0)
    group['增长率'] = group['增长率'].replace([np.inf, -np.inf], 0)
    return group


# 3. 动态波动率阈值计算
def calculate_dynamic_volatility_threshold(hist_growth_rates):
    sigma_hist = hist_growth_rates.std()
    q1, q3 = np.percentile(hist_growth_rates, [25, 75])
    iqr = q3 - q1
    return sigma_hist + 1.5 * iqr  # 动态波动率阈值


# 4. 风险评分函数（返回每个指标的得分）
def calculate_risk_score_details(row, forecast_period='短期'):
    growth_rates = np.array(row['预测期增长率'])  # 确保是 NumPy 数组
    avg_growth = row['平均增长率']
    volatility = row['波动率']
    dynamic_vol_threshold = row['动态波动阈值']
    deviation = row['下降幅度']

    # 初始化每个指标的得分
    scores = {
        '负增长年数 ≥ 50%': 0,
        '连续负增长年数': 0,
        '平均增长率是否为负': 0,
        '下降幅度 > 30%': 0,
        '波动率 > 动态波动阈值': 0
    }

    # 1. 负增长年数 ≥ 总预测期的50%
    negative_years = sum(growth_rates < 0)
    total_years = len(growth_rates)
    if negative_years / total_years >= 0.5:
        scores['负增长年数 ≥ 50%'] = 1

    # 2. 连续负增长年数（短期：≥2年；长期：≥3年）
    consecutive_years = 0
    for rate in growth_rates:
        if rate < 0:
            consecutive_years += 1
            if (forecast_period == '短期' and consecutive_years >= 2) or (
                    forecast_period == '长期' and consecutive_years >= 3):
                scores['连续负增长年数'] = 1
                break
        else:
            consecutive_years = 0

    # 3. 平均增长率是否为负
    if avg_growth < 0:
        scores['平均增长率是否为负'] = 1

    # 4. 下降幅度 > 30%
    if deviation > 0.3:
        scores['下降幅度 > 30%'] = 1

    # 5. 波动率 > 动态波动阈值
    if volatility > dynamic_vol_threshold:
        scores['波动率 > 动态波动阈值'] = 1

    return scores  # 返回每个指标的得分（1/0）


# 5. 风险等级判定
def assess_risk_level(score):
    if score >= 4:
        return '高风险'
    elif score >= 2:
        return '中风险'
    else:
        return '低风险'


# 6. 计算风险评分表（包含中间计算结果）
def calculate_risk_scores_with_details(df, historical_df, forecast_df, forecast_period='短期'):
    results = []

    for domain, group in df.groupby('领域'):
        # 获取历史数据
        hist_group = historical_df[historical_df['领域'] == domain]
        hist_growth = hist_group[hist_group['年份'] >= 2019]['增长率'].values
        weighted_hist_growth = np.average(hist_growth, weights=np.arange(1, len(hist_growth) + 1))
        hist_volatility = hist_growth.std()

        # 获取预测期增长率
        fc_group = forecast_df[forecast_df['领域'] == domain]
        fc_growth_rates = fc_group['增长率'].values
        if forecast_period == '短期':
            fc_growth_rates = fc_growth_rates[:3]  # 取前3年（2025-2027）
        else:
            fc_growth_rates = fc_growth_rates  # 全部6年（2025-2030）

        # 计算指标
        avg_growth = fc_growth_rates.mean()
        volatility = fc_growth_rates.std()
        deviation = (avg_growth - weighted_hist_growth) / (abs(weighted_hist_growth) + 1e-8)
        dynamic_vol_threshold = calculate_dynamic_volatility_threshold(hist_growth)

        # 计算每个指标的得分
        score_details = calculate_risk_score_details({
            '预测期增长率': fc_growth_rates,
            '平均增长率': avg_growth,
            '波动率': volatility,
            '动态波动阈值': dynamic_vol_threshold,
            '下降幅度': abs(deviation)
        }, forecast_period=forecast_period)

        # 保存结果（包含中间计算结果）
        result = {
            '领域': domain,
            '平均增长率': avg_growth,
            '波动率': volatility,
            '动态波动阈值': dynamic_vol_threshold,
            '下降幅度': abs(deviation),
            '历史基准增长率': weighted_hist_growth
        }
        result.update(score_details)  # 添加每个指标的得分
        results.append(result)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    # 计算总分和风险等级
    results_df['风险评分'] = results_df[
        ['负增长年数 ≥ 50%', '连续负增长年数', '平均增长率是否为负', '下降幅度 > 30%', '波动率 > 动态波动阈值']].sum(
        axis=1)
    results_df['风险等级'] = results_df['风险评分'].apply(assess_risk_level)

    return results_df


# 7. 主流程执行
if __name__ == "__main__":
    # 加载数据
    df, _, _ = load_data("../../经营范围预测/预测结果/企业技术领域发展数据_2000_2030.csv")

    # 预处理：计算增长率
    df_grouped = df.groupby('领域').apply(calculate_growth_rate).reset_index(drop=True)
    historical = df_grouped[df_grouped['年份'] <= 2024]
    forecast = df_grouped[df_grouped['年份'] >= 2025]

    # 短期风险评分（2025-2027）
    short_risk_df = calculate_risk_scores_with_details(df, historical, forecast, forecast_period='短期')
    short_risk_df = short_risk_df.rename(columns={
        col: f'短期_{col}' for col in short_risk_df.columns if col not in ['领域']
    })

    # 长期风险评分（2025-2030）
    long_risk_df = calculate_risk_scores_with_details(df, historical, forecast, forecast_period='长期')
    long_risk_df = long_risk_df.rename(columns={
        col: f'长期_{col}' for col in long_risk_df.columns if col not in ['领域']
    })

    # 合并短期和长期评分（用于对比表）
    comparison_df = pd.merge(
        short_risk_df[
            ['领域', '短期_负增长年数 ≥ 50%', '短期_连续负增长年数', '短期_平均增长率是否为负', '短期_下降幅度 > 30%',
             '短期_波动率 > 动态波动阈值', '短期_风险评分', '短期_风险等级']],
        long_risk_df[
            ['领域', '长期_负增长年数 ≥ 50%', '长期_连续负增长年数', '长期_平均增长率是否为负', '长期_下降幅度 > 30%',
             '长期_波动率 > 动态波动阈值', '长期_风险评分', '长期_风险等级']],
        on='领域', how='outer'
    )

    # 保存结果到Excel
    output_file = '../结果/风险评分结果_优化版.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        # 1. 短期风险评分表（包含中间计算结果）
        short_risk_df.to_excel(writer, sheet_name='短期风险评分', index=False)

        # 2. 长期风险评分表（包含中间计算结果）
        long_risk_df.to_excel(writer, sheet_name='长期风险评分', index=False)

        # 3. 对比表（仅包含打分、总分和风险等级）
        comparison_df.to_excel(writer, sheet_name='短期vs长期对比', index=False)

    print(f"结果已保存到 '{output_file}' 的多个工作表中。")
