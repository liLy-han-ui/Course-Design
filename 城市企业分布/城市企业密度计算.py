import os
import pandas as pd

# 定义输入和输出路径
input_directory = "../城市企业统计/全部企业"  # 原始数据目录
city_stats_file = "../城市信息/city_statistics.csv"  # 城市面积信息文件
output_directory = "../城市企业密度"  # 输出目录

# 创建输出目录
os.makedirs(output_directory, exist_ok=True)

# 读取城市面积信息
city_stats_path = city_stats_file
city_data = pd.read_csv(city_stats_path)

# 确保城市名称列一致（去掉可能的空格）
city_data["城市"] = city_data["城市"].str.strip()

# 遍历每五年的统计文件
for file_name in os.listdir(input_directory):
    if file_name.endswith(".csv"):
        try:
            # 打印调试信息
            print(f"正在处理文件: {file_name}")

            # 检查文件名是否包含下划线
            if "_" not in file_name:
                raise ValueError("文件名中缺少下划线")

            # 分割文件名
            parts = file_name.split("_")
            print(f"文件名分割结果: {parts}")  # 打印分割结果

            # 提取年份信息
            year_part = parts[-1]  # 获取最后一个部分，如 "2000.csv"
            print(f"提取的年份部分: {year_part}")  # 打印年份部分

            year = int(year_part.split(".")[0])  # 提取年份数字
            print(f"解析出的年份: {year}")  # 打印解析出的年份

            input_file_path = os.path.join(input_directory, file_name)

            # 读取企业统计数据
            enterprise_data = pd.read_csv(input_file_path)
            enterprise_data["城市"] = enterprise_data["城市"].str.strip()

            # 合并数据，添加面积信息
            merged_data = pd.merge(enterprise_data, city_data, on="城市", how="left")

            # 对面积进行单位换算（乘以 0.0001）
            merged_data["面积"] = merged_data["面积"] * 0.0001

            # 过滤掉面积为 0 的城市
            merged_data = merged_data[merged_data["面积"] > 0]

            # 计算企业密度
            merged_data["企业密度"] = merged_data["全部企业"] / merged_data["面积"]

            # 只保留需要的列：城市、全部企业、企业密度
            result_data = merged_data[["城市", "全部企业", "企业密度"]]

            # 保存结果到输出目录
            output_file = os.path.join(output_directory, f"{year}_density.csv")
            result_data.to_csv(output_file, index=False, encoding="utf-8-sig")

            print(f"成功处理文件: {file_name}, 年份: {year}")
        except (IndexError, ValueError) as e:
            print(f"文件名格式错误，跳过文件：{file_name}，错误原因：{e}")
            continue

print("企业密度计算完成，结果已存储到目录:", output_directory)