from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_date
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import numpy as np

# ---------- 1. 初始化Spark（关键优化点）----------
spark = SparkSession.builder \
    .appName("DataProcessor") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

# ---------- 2. 数据加载与预处理（重大改进）----------
# 定义明确schema（避免类型推断消耗内存）
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

# 直接使用Spark读取CSV（避免Pandas中转）
df_spark = spark.read.csv(
    r"../原始数据/data.csv",
    header=True,  # 自动读取列名
    schema=schema,  # 使用预定义schema
    enforceSchema=True,  # 强制类型校验
    multiLine=True
).repartition(200)  # 增加分区数

# 重命名列（Spark原生操作更高效）
df_spark = df_spark.withColumnRenamed("成立日期", "establish_date") \
                   .withColumnRenamed("核准日期", "approval_date") \
                   .withColumnRenamed("登记状态", "status")

# ---------- 3. 日期处理（保持原逻辑）----------
df_spark = df_spark.withColumn("establish_year", year(to_date(col("establish_date"), "yyyy-MM-dd"))) \
                   .withColumn("approval_year", year(to_date(col("approval_date"), "yyyy-MM-dd")))

# ---------- 4. 核心指标计算（优化缓存策略）----------
# 持久化关键数据
df_spark.cache()

base = (
    df_spark.filter("establish_year < 2000").count()
    - df_spark.filter("status IN ('注销','吊销','责令关闭','撤销','已歇业') AND approval_year < 2000").count()
)

# 优化聚合操作
annual_new = df_spark.groupBy("establish_year").count() \
                     .withColumnRenamed("count", "new") \
                     .toPandas()

annual_death = df_spark.filter(
    col("status").isin(['注销', '吊销', '责令关闭', '撤销', '已歇业']) & col("approval_year").isNotNull()
).groupBy("approval_year").count() \
 .withColumnRenamed("approval_year", "year") \
 .withColumnRenamed("count", "death") \
 .toPandas()

# 释放缓存
df_spark.unpersist()

full_years = pd.DataFrame({"year": range(2000, 2025)})
result = (
    full_years
    .merge(annual_new, left_on="year", right_on="establish_year", how="left")
    .merge(annual_death, on="year", how="left")
    .fillna(0)
    .assign(
        new=lambda x: x['new'].astype(int),
        death=lambda x: x['death'].astype(int),
        net_growth=lambda x: x['new'] - x['death'],
        total=lambda x: base + x['net_growth'].cumsum()
    )
)

for col in ["new", "death", "net_growth", "total"]:
    result[f"{col}_growth%"] = result[col].pct_change().fillna(0).replace([np.inf, -np.inf], 0) * 100

result["prev_total"] = result["total"].shift(1, fill_value=base)
average = (result["prev_total"] + result["total"]) / 2
result["birth_rate%"] = (result["new"] / average) * 100
result["death_rate%"] = (result["death"] / average) * 100
result["natural_growth%"] = result["birth_rate%"] - result["death_rate%"]
result.loc[0, ["birth_rate%", "death_rate%", "natural_growth%"]] = 0

# 保存处理结果到CSV
result.to_csv("../其余信息统计结果/processed_data.csv", index=False, encoding="utf-8")
print("数据处理完成，已保存为 processed_data.csv")
