import pandas as pd

# 输入文件路径
input_file = "../原始数据/data.csv"  # 输入的CSV文件路径
output_file = "../原始数据/filtered_data.csv"  # 输出的过滤后的CSV文件路径

def filter_records(input_file, output_file):
    """
    过滤输入CSV文件中的记录，仅保留所属城市为“自治区直辖县级行政区划”、“省直辖县级行政区划”、“海南省”、“贵州省”、“四川省”的记录。
    将结果保存到新的CSV文件中。
    """
    # 读取输入的CSV文件
    df = pd.read_csv(input_file, delimiter=',')

    # 定义需要筛选的城市列表
    target_cities = ["自治区直辖县级行政区划", "省直辖县级行政区划", "海南省", "贵州省", "四川省"]

    # 筛选出符合条件的记录
    filtered_df = df[df['所属城市'].isin(target_cities)]

    # 将筛选后的结果保存到新的CSV文件
    filtered_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已将符合条件的记录保存到 {output_file}")

if __name__ == "__main__":
    filter_records(input_file, output_file)