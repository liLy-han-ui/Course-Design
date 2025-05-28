import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
import time
import re

# 输入文件路径
input_file = "../城市企业统计/全部企业/全部企业_2025.csv"  # 包含城市的CSV文件
output_file = "../城市信息/city_statistics.csv"  # 输出的城市统计文件

# 默认的User-Agent列表（避免被反爬虫机制屏蔽）
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",  # Safari User-Agent
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"
]


def get_wikipedia_page(city_name):
    """
    根据城市名生成对应的维基百科页面URL。
    """
    return f"https://zh.wikipedia.org/wiki/{city_name.replace(' ', '_')}"


def extract_area_from_html(content):
    """
    从HTML内容中提取面积信息。
    """
    soup = BeautifulSoup(content, "html.parser")
    area_data = "未找到面积信息"

    # 查找Infobox表格
    infobox = soup.find("table", class_="infobox geography vcard")
    if infobox:
        # 查找<th>标签，其文本包含“总计”
        area_th = infobox.find("th", string=re.compile(r"总计"))
        if area_th:
            # 找到<th>标签后，查找其后的<td>标签
            area_td = area_th.find_next_sibling("td", class_="infobox-data")
            if area_td:
                area_text = area_td.get_text(separator=" ", strip=True)
                # 使用正则表达式匹配“xxx平方公里”或“xxx平方千米”
                match = re.search(r"([\d,\.]+)\s*(平方公里|平方千米)", area_text)
                if match:
                    area_data = match.group(1).replace(",", "")  # 去掉千分位逗号
                else:
                    print(f"未匹配到面积数值: {area_text}")
            else:
                print("未找到与面积相关的<td>标签")
        else:
            # 如果未找到<th>标签，尝试查找<td>标签中的面积信息
            area_td = infobox.find("td", string=re.compile(r"[\d,\.]+\s*(平方公里|平方千米)"))
            if area_td:
                area_text = area_td.get_text(separator=" ", strip=True)
                match = re.search(r"([\d,\.]+)\s*(平方公里|平方千米)", area_text)
                if match:
                    area_data = match.group(1).replace(",", "")
                else:
                    print(f"未匹配到面积数值: {area_text}")
            else:
                print("未找到面积相关的<th>或<td>标签")
    else:
        print("未找到Infobox表格")

    print(f"提取到的面积: {area_data}")
    return area_data


def extract_info_from_wikipedia(url):
    """
    从维基百科页面中提取面积信息。
    """
    max_retries = 3  # 最大重试次数
    for attempt in range(max_retries):
        try:
            user_agent = random.choice(USER_AGENTS)
            headers = {"user-agent": user_agent}
            print(f"正在使用 User-Agent: {user_agent}")  # 调试输出
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return extract_area_from_html(response.content)
            else:
                print(f"请求失败，状态码: {response.status_code}")
                return "请求失败"
        except Exception as e:
            print(f"访问维基百科页面失败: {e}")
            if attempt < max_retries - 1:
                print(f"第 {attempt + 1} 次重试...")
                time.sleep(2)  # 等待一段时间后重试
            else:
                print("达到最大重试次数，放弃请求。")
                return "请求失败"


def main():
    # 读取输入的CSV文件
    city_data = pd.read_csv(input_file)

    # 如果输出文件已存在，读取已处理过的城市名称
    processed_cities = set()
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        processed_cities = set(existing_data["城市"].unique())

    results = []

    for _, row in city_data.iterrows():
        city_name = row["城市"]


        # 如果城市已经处理过，则跳过
        if city_name in processed_cities:
            print(f"\n城市 {city_name} 已经处理过，跳过...")
            continue

        print(f"\n正在处理城市: {city_name}")

        # 获取维基百科页面URL
        wiki_url = get_wikipedia_page(city_name)
        print(f"访问URL: {wiki_url}")  # 调试输出

        # 提取面积信息
        area = extract_info_from_wikipedia(wiki_url)

        # 打印当前城市的结果
        print(f"{city_name}: 面积={area}")

        # 将结果添加到列表中
        results.append({
            "城市": city_name,

            "面积": area
        })

        # 增加延迟以避免触发反爬虫机制
        time.sleep(1)

    # 将新结果追加到现有的CSV文件
    output_df = pd.DataFrame(results)
    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        combined_data = pd.concat([existing_data, output_df], ignore_index=True)
        combined_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"城市面积统计已保存到 {output_file}")


if __name__ == "__main__":
    main()