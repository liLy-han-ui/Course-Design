import re
import pandas as pd
from collections import defaultdict
import jieba
from tqdm import tqdm  # 导入 tqdm 库

# 定义人工智能领域关键词分类
ai_field_keywords = {
    "深度学习": {"深度学习", "机器学习", "神经网络"},
    "人工智能硬件开发": {"硬件开发", "机器人开发", "可穿戴设备开发"},
    "机器人": {"机器人", "自动化"},
    "软件开发": {"软件开发", "智能算法", "人工智能软件开发"},
    "自然语言处理": {"自然语言处理", "语音识别", "文本分类", "情感分析", "词嵌入", "语义分析", "问答系统", "机器翻译"},
    "数据服务": {"数据采集", "数据标注", "数据分析"},
    "云计算": {"云计算", "云存储", "云服务"},
    "大数据分析": {"大数据分析", "数据挖掘", "数据管理与分析平台"},
    "大数据模型": {"大语言模型", "语言模型"},
    "计算机视觉": {"计算机视觉", "计算机幻觉"},
    "物联网": {"物联网", "智能家居", "智能硬件"},
    "智能系统（智能+）": {"智能设备", "移动终端设备", "智能系统", "智能产品", "自动驾驶", "智能驾驶", "智能医疗", "智慧医疗"},
    "半导体": {"半导体"}
}

# 定义一个函数来分割经营范围
def split_business_scope(scope_text):
    pattern = r"[，,:：。；.;\s]+"
    segments = re.split(pattern, scope_text)
    return [segment.strip() for segment in segments if segment.strip()]

# 定义一个函数来判断是否包含某个领域的关键词
def count_ai_fields(segment, field_keywords):
    words = jieba.lcut(segment)  # 使用 jieba 分词
    matched_fields = set()
    for word in words:
        for field, keywords in field_keywords.items():
            if word in keywords:
                matched_fields.add(field)
    return matched_fields

# 按年份统计企业数量（增量更新方式）
def analyze_company_data(input_file):
    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 数据清洗
    df.replace('-', pd.NA, inplace=True)
    df.dropna(inplace=True)

    # 清理日期列
    df['成立日期'] = pd.to_datetime(df['成立日期'], errors='coerce')
    df['核准日期'] = df['核准日期'].fillna('').astype(str)

    # 定义一个函数来解析注销年份并打印异常日期
    def parse_year_with_debug(date_str, row_id):
        try:
            date = pd.to_datetime(date_str)
            return date.year
        except Exception as e:
            print(f"忽略异常日期记录 ID: {row_id}, 核准日期: {date_str}, 错误: {e}")
            return None

    # 提取成立年份和注销年份
    df['成立年份'] = df['成立日期'].dt.year
    df['注销年份'] = df.apply(
        lambda row: parse_year_with_debug(row['核准日期'], row.name)
                    if row['登记状态'] in ['注销', '吊销', '停业', '撤销', '已歇业', '责令关闭'] else None,
        axis=1
    )

    # 过滤掉异常日期的记录
    df = df[df['注销年份'].notnull() | (df['登记状态'].isin(['存续', '在营', '开业']))]

    # 初始化统计结果
    year_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    cumulative_alive = defaultdict(int)  # 累计存活企业数
    start_year = 2000
    end_year = 2025

    # 增量更新统计
    with tqdm(total=(end_year - start_year + 1), desc="统计每年企业数量") as pbar:
        for year_to_check in range(start_year, end_year + 1):
            # 新增企业：统计在该年成立的企业
            new_companies_df = df[df['成立年份'] == year_to_check]
            for _, row in new_companies_df.iterrows():
                business_scope = row['经营范围']
                segments = split_business_scope(business_scope)
                matched_fields = set()
                for segment in segments:
                    matched_fields.update(count_ai_fields(segment, ai_field_keywords))
                for field in matched_fields:
                    year_stats[year_to_check][field]['new'] += 1
                    cumulative_alive[field] += 1

            # 注销企业：统计在该年注销的企业
            closed_companies_df = df[df['注销年份'] == year_to_check]
            for _, row in closed_companies_df.iterrows():
                business_scope = row['经营范围']
                segments = split_business_scope(business_scope)
                matched_fields = set()
                for segment in segments:
                    matched_fields.update(count_ai_fields(segment, ai_field_keywords))
                for field in matched_fields:
                    year_stats[year_to_check][field]['dead'] += 1
                    cumulative_alive[field] -= 1

            # 计算净增长企业和累计存活企业
            for field in ai_field_keywords.keys():
                year_stats[year_to_check][field]['net_growth'] = (
                    year_stats[year_to_check][field]['new'] - year_stats[year_to_check][field]['dead']
                )
                year_stats[year_to_check][field]['cumulative_alive'] = cumulative_alive[field]

            pbar.update(1)  # 更新进度条

    return year_stats

# 保存统计结果到CSV文件
def save_statistics_to_csv(year_stats, output_file):
    rows = []
    for year, fields in sorted(year_stats.items()):
        for field, stats in fields.items():
            rows.append({
                "年份": year,
                "领域": field,
                "新增企业数量": stats['new'],
                "注销企业数量": stats['dead'],
                "净增长企业数量": stats['net_growth'],
                "累计存活企业数量": stats['cumulative_alive']
            })
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(output_file, index=False, encoding='utf-8')

# 主程序入口
if __name__ == "__main__":
    input_file = "../原始数据/data.csv"  # 输入文件路径
    output_file_stats = "../企业经营范围/企业统计结果.csv"  # 输出文件路径
    print("start")
    # 分析企业数据
    year_stats = analyze_company_data(input_file)

    # 保存统计结果到CSV文件
    save_statistics_to_csv(year_stats, output_file_stats)

    print("处理完成！统计结果已保存到", output_file_stats)