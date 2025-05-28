import pandas as pd
import re

# 定义汇率转换表（与原代码完全一致）
exchange_rates = {
    "万人民币": 1.0,
    "万美元": 7.0,
    "万港元": 0.9,
    "万港币": 0.9,
    "万澳门元": 0.85,
    "万欧元": 7.8,
    "万澳大利亚元": 4.8,
    "万瑞典克朗": 0.66,
    "万日元": 0.05,
    "万英镑": 9.0,
    "万加拿大元": 5.3,
    "万新加坡元": 5.2
}

# 读取数据（保持原始读取方式）
df = pd.read_csv(r"../原始数据/data.csv", encoding='utf-8')


# 严格等效的注册资本转换函数（保持原逻辑）
def convert_capital(capital_str):
    try:
        # 完全相同的正则匹配逻辑
        match = re.match(r"^([\d.]+)(万[\u4e00-\u9fff]+)", str(capital_str))
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).strip()  # 与原处理保持一致

        # 完全相同的汇率查询逻辑
        rate = exchange_rates.get(unit)
        if rate is None:
            return None

        return value * rate
    except (ValueError, AttributeError):
        return None


# 逐行应用转换（保持原apply逻辑）
df["注册资本_万人民币"] = df["注册资本"].apply(convert_capital)

# 完全相同的过滤逻辑（保持dropna条件）
df = df.dropna(subset=["注册资本_万人民币"])

# 严格等效的日期处理逻辑
df["成立日期"] = pd.to_datetime(
    df["成立日期"],
    format="%Y-%m-%d",
    errors='coerce'  # 与Spark的to_date行为一致
).dt.date  # 保持日期类型一致性

# 完全相同的年份提取逻辑
df["成立年份"] = pd.to_datetime(df["成立日期"]).dt.year

# 严格等效的年份过滤条件（2000-2024）
df = df[
    (df["成立年份"] >= 2000) &
    (df["成立年份"] <= 2024) &
    (df["成立日期"].notnull())  # 保持与Spark过滤空日期一致
    ]

# 完全相同的分组聚合逻辑
capital_sum = df.groupby(
    "成立年份",
    as_index=False
)["注册资本_万人民币"].sum(numeric_only=True)

# 严格保持输出格式一致
capital_sum.columns = ["成立年份", "总注册资本_万人民币"]

# 保存结果（保持完全相同的输出格式）
capital_sum.to_csv("../其余信息统计结果/capital_sum_data.csv", index=False, encoding="utf-8")
