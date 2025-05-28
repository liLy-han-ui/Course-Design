import pandas as pd
from itertools import product

# 读取数据（直接使用Pandas）
df = pd.read_csv(r"../原始数据/data.csv", encoding='utf-8')

# 列重命名（保持与原映射完全一致）
column_mapping = {
    "成立日期": "establish_date",
    "核准日期": "approval_date",
    "登记状态": "status",
    "国标行业门类": "industry_sector",
    "国标行业大类": "industry_class"
}
df.rename(columns=column_mapping, inplace=True)

# 行业分类映射（与原数据完全一致）
sector_mapping = {
    "农、林、牧、渔业": (1, "第一产业"),
    "采矿业": (2, "第二产业"),
    "制造业": (2, "第二产业"),
    "电力、热力、燃气及水生产和供应业": (2, "第二产业"),
    "建筑业": (2, "第二产业"),
    "批发和零售业": (3, "第三产业"),
    "交通运输、仓储和邮政业": (3, "第三产业"),
    "住宿和餐饮业": (3, "第三产业"),
    "信息传输、软件和信息技术服务业": (3, "第三产业"),
    "金融业": (3, "第三产业"),
    "房地产业": (3, "第三产业"),
    "租赁和商务服务业": (3, "第三产业"),
    "科学研究和技术服务业": (3, "第三产业"),
    "水利、环境和公共设施管理业": (3, "第三产业"),
    "居民服务、修理和其他服务业": (3, "第三产业"),
    "教育": (3, "第三产业"),
    "卫生和社会工作": (3, "第三产业"),
    "文化、体育和娱乐业": (3, "第三产业")
}

# 添加行业分类字段（向量化实现）
df["sector_id"] = df["industry_sector"].map(lambda x: sector_mapping.get(x, (4, "其他"))[0])
df["display_label"] = df["sector_id"].astype(str) + "-" + df["industry_sector"] + "-" + df["industry_class"]

# 日期处理（向量化转换）
df["establish_date"] = pd.to_datetime(df["establish_date"], errors='coerce')
df["approval_date"] = pd.to_datetime(df["approval_date"], errors='coerce')

# 提取年份（处理无效日期）
df["establish_year"] = df["establish_date"].dt.year
df["approval_year"] = df["approval_date"].dt.year

# 基础数据处理
base_dead_mask = (
    df["status"].isin(['注销', '吊销', '责令关闭', '撤销', '已歇业']) &
    df["approval_year"].notnull() &
    (df["approval_year"] < 2000)
)

base_alive = df[df["establish_year"] < 2000].groupby("display_label").size().reset_index(name='base_count')
base_dead = df[base_dead_mask].groupby("display_label").size().reset_index(name='dead_count')

base = pd.merge(base_alive, base_dead, on="display_label", how="outer").fillna(0)
base["alive_2000"] = (base["base_count"] - base["dead_count"]).astype(int)

# 每年新增企业（重命名年份列）
annual_new = df.groupby(["establish_year", "display_label"]).size().reset_index(name='new')
annual_new.rename(columns={"establish_year": "year"}, inplace=True)

# 每年死亡企业（重命名年份列）
death_mask = df["status"].isin(['注销', '吊销', '责令关闭', '撤销', '已歇业'])
annual_death = df[death_mask].groupby(["approval_year", "display_label"]).size().reset_index(name='death')
annual_death.rename(columns={"approval_year": "year"}, inplace=True)

# 生成完整年份-行业组合
full_years = pd.DataFrame({"year": range(2000, 2025)})
unique_labels = df["display_label"].unique()

full_index = pd.DataFrame(
    list(product(full_years["year"], unique_labels)),
    columns=["year", "display_label"]
)

# 合并所有数据（确保类型一致）
result = pd.merge(
    full_index.astype({"year": int}),
    annual_new.astype({"year": int}),
    on=["year", "display_label"],
    how="left"
)
result = pd.merge(
    result,
    annual_death.astype({"year": int}),
    on=["year", "display_label"],
    how="left"
)
result.fillna(0, inplace=True)

# 数据类型转换
result["new"] = result["new"].astype(int)
result["death"] = result["death"].astype(int)
result["net_growth"] = result["new"] - result["death"]

# 计算累计存活（向量化实现）
base_dict = dict(zip(base["display_label"], base["alive_2000"]))
result["cumulative"] = result.groupby("display_label")["net_growth"].cumsum().add(
    result["display_label"].map(base_dict).fillna(0)
).astype(int)

# 保存结果
result.to_csv("../其余信息统计结果/industry_heatmap_data.csv", index=False, encoding="utf-8")
print("行业热力图数据处理完成，已保存为 industry_heatmap_data.csv")
