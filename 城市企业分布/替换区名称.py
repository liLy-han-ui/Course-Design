import pandas as pd

# 输入输出文件路径
input_file = "../原始数据/data.csv"  # 输入的CSV文件路径
output_file = "../原始数据/modified_data.csv"  # 输出的修改后的CSV文件路径

def replace_city_names(input_file, output_file):
    """
    替换输入CSV文件中指定城市的名称，并将结果保存到新的CSV文件。
    对于“自治区直辖县级行政区划”和“省直辖县级行政区划”，将其替换为对应的“所属区县”值。
    """
    # 读取输入的CSV文件
    df = pd.read_csv(input_file)

    # 定义替换规则
    replacement_rules = {
        "河北省": "雄安新区",
        "山西省": "山西转型综合改革示范区",
        "江苏省": "中国（江苏）自由贸易试验区苏州片区",
        "辽宁省": "辽宁省沈抚示范区",
        "海南省": "海南省洋浦经济开发区",
        "贵州省": "贵州省贵安新区",
        "四川省": "中国（四川）自由贸易试验区",
        "江西省": "江西省赣江新区直管区"
    }

    # 使用replace方法替换“所属城市”列中的值
    df['所属城市'] = df['所属城市'].replace(replacement_rules)

    # 处理“自治区直辖县级行政区划”和“省直辖县级行政区划”
    special_cases = ["自治区直辖县级行政区划", "省直辖县级行政区划"]
    for case in special_cases:
        # 将符合条件的“所属城市”替换为对应的“所属区县”值
        df.loc[df['所属城市'] == case, '所属城市'] = df.loc[df['所属城市'] == case, '所属区县']

    # 将修改后的DataFrame保存到新的CSV文件
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已将修改后的记录保存到 {output_file}")

if __name__ == "__main__":
    replace_city_names(input_file, output_file)