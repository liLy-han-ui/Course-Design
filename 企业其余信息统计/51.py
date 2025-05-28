import pandas as pd
import numpy as np

# 读取数据（直接使用Pandas）
df_pd = pd.read_csv(r"../原始数据/data.csv", encoding='utf-8')

# 日期转换（向量化操作）
df_pd["成立日期"] = pd.to_datetime(df_pd["成立日期"], format="%Y-%m-%d", errors='coerce')
df_pd["核准日期"] = pd.to_datetime(df_pd["核准日期"], format="%Y-%m-%d", errors='coerce')

# 定义状态分类
death_states = {'注销', '吊销', '责令关闭', '撤销', '已歇业'}
alive_states = {'存续', '在业', '存续/在业', '迁出', '其它'}

# 向量化计算生存周期（替代apply）
mask_valid_start = df_pd["成立日期"] <= pd.Timestamp("2024-12-31")

# 死亡企业条件
mask_death = (
    df_pd["登记状态"].isin(death_states) &
    df_pd["核准日期"].notnull() &
    (df_pd["核准日期"] <= pd.Timestamp("2024-12-31"))
)
death_end = df_pd.loc[mask_death, "核准日期"]

# 存活企业条件
mask_alive = (
    (df_pd["登记状态"].isin(alive_states) & mask_valid_start) |
    (df_pd["登记状态"].isin(death_states) &
     df_pd["核准日期"].notnull() &
     (df_pd["核准日期"] > pd.Timestamp("2024-12-31")))
)
alive_end = pd.Timestamp("2024-12-31")

# 向量化计算
df_pd["生存周期（月）"] = np.where(
    mask_death,
    (death_end - df_pd["成立日期"]).dt.days / 30.44,
    np.where(
        mask_alive,
        (alive_end - df_pd["成立日期"]).dt.days / 30.44,
        np.nan
    )
)

# 过滤无效数据
df_pd = df_pd.dropna(subset=["生存周期（月）"])

# 分组统计（使用向量化操作）
bins = np.arange(0, df_pd["生存周期（月）"].max() + 1, 6)
labels = [f"{i}-{i+5}月" for i in bins[:-1]]

# 分箱处理
death_cycle = df_pd.loc[
    df_pd["登记状态"].isin(death_states) &
    (df_pd["核准日期"] <= pd.Timestamp("2024-12-31")),
    "生存周期（月）"
]

alive_cycle = df_pd.loc[
    (df_pd["登记状态"].isin(alive_states) & mask_valid_start) |
    (df_pd["登记状态"].isin(death_states) &
     df_pd["核准日期"].notnull() &
     (df_pd["核准日期"] > pd.Timestamp("2024-12-31"))),
    "生存周期（月）"
]

# 分箱统计
death_counts = pd.cut(death_cycle, bins=bins, labels=labels, right=False).value_counts().sort_index()
alive_counts = pd.cut(alive_cycle, bins=bins, labels=labels, right=False).value_counts().sort_index()

# 构建结果表
survival_data = pd.DataFrame({
    "生存周期区间": labels,
    "已死亡企业数量": death_counts.reindex(labels, fill_value=0),
    "存活企业数量": alive_counts.reindex(labels, fill_value=0)
}).reset_index(drop=True)

# 保存结果
survival_data.to_csv("../其余信息统计结果/survival_cycle_data.csv", index=False, encoding="utf-8")
