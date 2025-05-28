from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_date, udf
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd

# 初始化Spark
spark = SparkSession.builder \
    .appName("InvestmentDataProcessor") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# 定义包含原始中文列名的Schema
schema = StructType([
    StructField("公司名称", StringType(), True),
    StructField("登记状态", StringType(), True),
    StructField("企业规模", StringType(), True),
    StructField("注册资本", StringType(), True),
    StructField("成立日期", StringType(), True),
    StructField("核准日期", StringType(), True),
    StructField("所属省份", StringType(), True),
    StructField("所属城市", StringType(), True),
    StructField("所属区县", StringType(), True),
    StructField("公司类型", StringType(), True),
    StructField("国标行业门类", StringType(), True),
    StructField("国标行业大类", StringType(), True),
    StructField("经营范围", StringType(), True)
])

# 读取CSV并缓存
df_spark = spark.read.csv(
    r"../原始数据/data.csv",
    header=True,
    schema=schema,
    enforceSchema=True,
    multiLine=True
).repartition(200).cache()

# 重命名关键列
df_spark = df_spark \
    .withColumnRenamed("登记状态", "status") \
    .withColumnRenamed("公司类型", "company_type") \
    .withColumnRenamed("成立日期", "establish_date") \
    .withColumnRenamed("核准日期", "approval_date")

# 分类UDF
@udf(returnType=StringType())
def classify_investment_type(company_type):
    if any(k in company_type for k in [
        "国有独资", "国有控股", "全民所有制", "国有经营单位", "国有分公司",
        "国有与境内合资", "国有独资分支机构", "事业单位营业", "集体所有制"
    ]):
        return "国有企业"

    elif any(k in company_type for k in [
        "外商", "外国", "外资", "中外合资", "中外合作",
        "外商投资企业", "外商合资", "外商独资", "外商投资、非独资"
    ]):
        return "外商投资企业"

    elif any(k in company_type for k in [
        "港澳台", "台港澳", "港、澳、台", "港澳台与境内",
        "台港澳与境内", "港澳台法人独资", "港澳台自然人独资",
        "港澳台投资、非独资", "港澳台合资", "港澳台与外国投资者合资"
    ]):
        return "港澳台投资企业"

    elif "法人商事主体" in company_type:
        if any(k in company_type for k in ["国有", "外商", "港澳台"]):
            return "国有企业" if "国有" in company_type else (
                "外商投资企业" if "外商" in company_type else "港澳台投资企业"
            )
        else:
            return "民营企业"

    else:
        return "民营企业"

# 应用分类UDF
df_spark = df_spark.withColumn(
    "investment_type",
    classify_investment_type(col("company_type"))
).cache()

# 日期处理（合并为单次转换）
df_spark = df_spark.select(
    "*",
    year(to_date(col("establish_date"), "yyyy-MM-dd")).alias("establish_year"),
    year(to_date(col("approval_date"), "yyyy-MM-dd")).alias("approval_year")
).cache()

# 使用Spark SQL优化聚合计算
df_spark.createOrReplaceTempView("companies")

base_alive = spark.sql("""
    SELECT investment_type, COUNT(*) AS base_count
    FROM companies
    WHERE establish_year < 2000
    GROUP BY investment_type
""").toPandas()

base_dead = spark.sql("""
    SELECT investment_type, COUNT(*) AS dead_count
    FROM companies
    WHERE status IN ('注销', '吊销', '责令关闭', '撤销', '已歇业') 
    AND approval_year < 2000
    GROUP BY investment_type
""").toPandas()

base_df = (
    pd.merge(base_alive, base_dead, on="investment_type", how="outer")
    .fillna(0)
    .assign(alive_2000=lambda x: x.base_count - x.dead_count)
)

# 合并年度数据计算（减少转换次数）
annual_new = spark.sql("""
    SELECT establish_year AS year, investment_type, COUNT(*) AS new
    FROM companies
    GROUP BY establish_year, investment_type
""").toPandas()

annual_death = spark.sql("""
    SELECT approval_year AS year, investment_type, COUNT(*) AS death
    FROM companies
    WHERE status IN ('注销', '吊销', '责令关闭', '撤销', '已歇业') 
    AND approval_year IS NOT NULL
    GROUP BY approval_year, investment_type
""").toPandas()

# 构建完整时间-类型索引（使用Pandas向量化操作）
full_years = pd.DataFrame({"year": range(2000, 2025)})
investment_types = ["国有企业", "外商投资企业", "港澳台投资企业", "民营企业"]
full_index = full_years.assign(key=1).merge(
    pd.DataFrame({"investment_type": investment_types}).assign(key=1),
    on='key'
).drop('key', axis=1)

# 合并数据
result = (
    full_index
    .merge(annual_new, on=["year", "investment_type"], how="left")
    .merge(annual_death, on=["year", "investment_type"], how="left")
    .fillna(0)
)

# 转换数据类型
result = result.assign(
    new=lambda x: x.new.astype(int),
    death=lambda x: x.death.astype(int),
    net_growth=lambda x: x.new - x.death
)

# 计算累计存活（使用Pandas向量化）
base_dict = dict(zip(base_df.investment_type, base_df.alive_2000))
for inv_type in investment_types:
    mask = result.investment_type == inv_type
    result.loc[mask, 'cumulative'] = base_dict.get(inv_type, 0) + result[mask].groupby(
        'investment_type').cumsum().net_growth

# 计算占比
for metric in ["new", "death", "net_growth", "cumulative"]:
    total = result.groupby("year")[metric].transform("sum")
    result[f"{metric}_pct"] = (result[metric] / total * 100).round(1)

# 保存处理结果
result.to_csv("../其余信息统计结果/investment_data.csv", index=False, encoding="utf-8")

print("数据处理完成，已保存为 investment_data.csv")
