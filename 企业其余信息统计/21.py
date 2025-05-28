from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_date
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd

# 初始化Spark（优化配置）
spark = SparkSession.builder \
    .appName("DataProcessor") \
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

# 读取数据（保持中文列名）
df_spark = spark.read.csv(
    r"../原始数据/data.csv",
    header=True,
    schema=schema,
    enforceSchema=True,
    multiLine=True
).repartition(200)

# 重命名关键列（保持后续Pandas逻辑兼容）
df_spark = df_spark \
    .withColumnRenamed("登记状态", "status") \
    .withColumnRenamed("企业规模", "company_size") \
    .withColumnRenamed("成立日期", "establish_date") \
    .withColumnRenamed("核准日期", "approval_date")

df_spark = (
    df_spark
    .withColumn("establish_year", year(to_date(col("establish_date"), "yyyy-MM-dd")))
    .withColumn("approval_year", year(to_date(col("approval_date"), "yyyy-MM-dd")))
)

# 定义规模顺序
size_order = ["微型", "小型", "中型", "大型"]

# 计算2000年前存活企业规模分布
base_alive = (
    df_spark.filter("establish_year < 2000")
    .groupBy("company_size")
    .count()
    .withColumnRenamed("count", "base_count")
    .toPandas()
)

base_dead = (
    df_spark.filter("status IN ('注销', '吊销', '责令关闭', '撤销', '已歇业') AND approval_year < 2000")
    .groupBy("company_size")
    .count()
    .withColumnRenamed("count", "dead_count")
    .toPandas()
)

base_df = (
    base_alive.merge(base_dead, on="company_size", how="outer")
    .fillna(0)
    .assign(alive_2000=lambda x: x.base_count - x.dead_count)
)

# 计算年度新增/死亡企业规模分布
annual_new = (
    df_spark.groupBy("establish_year", "company_size")
    .count()
    .withColumnRenamed("count", "new")
    .toPandas()
)

annual_death = (
    df_spark.filter(
        col("status").isin(['注销', '吊销', '责令关闭', '撤销', '已歇业']) &
        col("approval_year").isNotNull()
    )
    .groupBy("approval_year", "company_size")
    .count()
    .withColumnRenamed("approval_year", "year")
    .withColumnRenamed("count", "death")
    .toPandas()
)

# 构建完整时间-规模索引
full_years = pd.DataFrame({"year": range(2000, 2025)})
full_sizes = pd.DataFrame({"company_size": size_order})
full_index = full_years.assign(key=1).merge(full_sizes.assign(key=1), on='key').drop('key', axis=1)

# 合并所有数据
result = (
    full_index
    .merge(annual_new, left_on=["year", "company_size"],
           right_on=["establish_year", "company_size"], how="left")
    .merge(annual_death, on=["year", "company_size"], how="left")
    .fillna(0)
    .assign(new=lambda x: x.new.astype(int),
            death=lambda x: x.death.astype(int),
            net_growth=lambda x: x.new - x.death)
)

# 计算累计存活
base_dict = dict(zip(base_df.company_size, base_df.alive_2000))
for size in size_order:
    mask = result.company_size == size
    result.loc[mask, 'cumulative'] = base_dict.get(size, 0) + result[mask].net_growth.cumsum()

# 计算占比
for metric in ["new", "death", "net_growth", "cumulative"]:
    total = result.groupby("year")[metric].transform("sum")
    result[f"{metric}_pct"] = (result[metric] / total * 100).round(1)

# 保存处理结果
result.to_csv("../其余信息统计结果/scale_data.csv", index=False, encoding="utf-8")
print("数据处理完成，已保存为 scale_data.csv")
