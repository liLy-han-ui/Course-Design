from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import chardet


def detect_file_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        detector = chardet.universaldetector.UniversalDetector()
        for line in f:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


def load_and_preprocess(filepath):
    """增强版数据加载函数（自动处理编码）"""
    # 支持的编码尝试列表（按优先级排序）
    encodings_to_try = [
        'utf-8-sig',  # 处理BOM头
        'gb18030',  # 中文扩展编码
        'utf-8',
        'gbk',
        'latin1'
    ]

    # 自动检测编码
    try:
        detected_encoding = detect_file_encoding(filepath)
        encodings_to_try.insert(0, detected_encoding)
    except Exception as e:
        print(f"编码自动检测失败：{str(e)}，将使用备选编码列表")

    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"成功用 {encoding.upper()} 编码加载数据")

            # 验证中文列名
            required_columns = ['城市', 'GRP（单位：亿元）', 'AI百度指数', '政策数量',
                                '科学技术支出（单位：万元）', '教育支出（单位：万元）',
                                '年份', '全部企业']
            if all(col in df.columns for col in required_columns):
                # 类型转换
                df = df.astype({
                    'GRP（单位：亿元）': float,
                    'AI百度指数': int,
                    '政策数量': int,
                    '科学技术支出（单位：万元）': float,
                    '教育支出（单位：万元）': float,
                    '年份': int,
                    '全部企业': int
                })
                # 可选重命名
                df = df.rename(columns={
                    '城市': 'City',
                    'GRP（单位：亿元）': 'GRP',
                    'AI百度指数': 'AI_Index',
                    '政策数量': 'Policy_Count',
                    '科学技术支出（单位：万元）': 'Science_Expenditure',
                    '教育支出（单位：万元）': 'Education_Expenditure',
                    '年份': 'Year',
                    '全部企业': 'Enterprises'
                })
                return df
            else:
                print(f"编码 {encoding} 加载成功，但列名不匹配")
                continue

        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码时发生错误：{str(e)}")
            continue

    print("所有编码尝试均失败")
    return None


def check_exponential(dataframe, city):
    """指数规律检验（增强版）"""
    if dataframe is None:
        return

    try:
        city_data = dataframe[dataframe['City'] == city].sort_values('Year')
        if len(city_data) < 2:
            print(f"{city} 数据不足，需要至少2年数据")
            return

        # 处理可能的零值
        if (city_data['Enterprises'] <= 0).any():
            min_val = city_data['Enterprises'].min()
            replacement = 1 if min_val == 0 else abs(min_val) + 1
            print(f"{city} 包含非正值（最小值={min_val}），已自动调整为{replacement}")
            city_data['Enterprises'] = city_data['Enterprises'].replace(
                to_replace=min_val,
                value=replacement
            )

        X = city_data[['Year']]
        y = np.log(city_data['Enterprises'])

        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))

        print(f"\n{city} 分析结果：")
        print(f"▪ 数据年限：{len(city_data)}年（{city_data['Year'].min()}-{city_data['Year'].max()}）")
        print(f"▪ 年均增长率：{model.coef_[0]:.2%}")
        print(f"▪ 初始基准值：{np.exp(model.intercept_):.0f} 家企业")
        print(f"▪ 模型解释度（R²）：{r2:.4f}")
        print("→ " + ("符合指数增长特征" if r2 > 0.85 else "未呈现显著指数增长"))

    except Exception as e:
        print(f"{city} 分析失败：{str(e)}")


if __name__ == "__main__":
    file_path = '../数据文件/合并后的城市经济与企业数据.csv'

    # 首次加载尝试
    df = load_and_preprocess(file_path)

    # 备选加载方案
    if df is None:
        print("\n尝试备选加载方案...")
        try:
            # 使用系统默认编码
            df = pd.read_csv(file_path)
            df.columns = [col.strip() for col in df.columns]  # 去除列名空格
            print("使用系统默认编码加载成功")
        except Exception as e:
            print(f"最终加载失败：{str(e)}")
            exit()

    # 数据校验
    if 'Enterprises' not in df.columns:
        print("关键列缺失，请检查数据文件")
        exit()

    print("\n数据样例：")
    print(df[['City', 'Year', 'Enterprises']].head())

    # 执行分析
    print("\n开始企业增长模式分析：")
    for city in df['City'].unique():
        check_exponential(df, city)