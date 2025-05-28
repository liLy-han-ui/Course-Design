import pandas as pd
import requests
import time
import plotly.express as px
import os

# 缓存文件路径
cache_file = "../城市信息/geocode_cache.csv"


# 加载缓存
def load_cache(cache_file):
    """加载地理编码缓存"""
    if os.path.exists(cache_file):
        cache_df = pd.read_csv(cache_file)
        return {row['城市']: {"经度": row['经度'], "纬度": row['纬度']} for _, row in cache_df.iterrows()}
    return {}


# 保存缓存
def save_cache(cache_data, cache_file):
    """保存地理编码缓存"""
    pd.DataFrame.from_dict(cache_data, orient="index").reset_index().rename(columns={"index": "城市"}).to_csv(
        cache_file, index=False)


# 调用高德地图 API 获取坐标
def get_coordinates_from_api(city, api_key):
    """通过高德地图 API 获取城市的经纬度"""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&key={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") == "1" and data.get("info") == "OK":
            location = data["geocodes"][0]["location"].split(",")
            return float(location[0]), float(location[1])
    except Exception as e:
        print(f"调用高德地图 API 失败: {e}")
    return None


# 带缓存的地理编码函数
def get_coordinates_cached(city, api_key, cache):
    """带缓存的地理编码函数"""
    if city in cache:
        return cache[city]["经度"], cache[city]["纬度"]

    # 调用高德地图 API 获取坐标
    coordinates = get_coordinates_from_api(city, api_key)
    if coordinates:
        cache[city] = {"经度": coordinates[0], "纬度": coordinates[1]}
        save_cache(cache, cache_file)  # 更新缓存文件
        return coordinates

    # 如果 API 调用失败，返回 None
    return None, None


# 主程序
if __name__ == "__main__":
    # 设置高德地图 API Key
    amap_api_key ="40ffc143b862f76151b44d6529bda460"

    # 加载缓存
    cache = load_cache(cache_file)

    # 加载之前生成的城市统计数据
    input_file = "城市企业统计.csv"
    city_count_pd = pd.read_csv(input_file)

    # 初始化经度和纬度列
    city_count_pd['经度'] = None
    city_count_pd['纬度'] = None

    # 分批处理数据以避免超出 API 配额限制
    batch_size = 50
    for i in range(0, len(city_count_pd), batch_size):
        batch = city_count_pd.iloc[i:i + batch_size].copy()  # 创建副本以避免 SettingWithCopyWarning
        batch.loc[:, ['经度', '纬度']] = batch['所属城市'].apply(
            lambda city: pd.Series(get_coordinates_cached(city, amap_api_key, cache))
        )

        # 合并结果到主 DataFrame
        city_count_pd.update(batch)  # 更新主 DataFrame
        time.sleep(1)  # 每批次之间暂停 1 秒

    # 删除无效坐标数据
    valid_cities = city_count_pd.dropna(subset=['经度', '纬度'])
    if valid_cities.empty:
        print("没有有效的城市坐标数据可用于生成热力图")
    else:
        print(f"有效城市记录数: {len(valid_cities)}/{len(city_count_pd)}")

        # 检查列名
        print(valid_cities.columns)

        # 可视化部分：生成热力图
        fig = px.density_mapbox(
            valid_cities,
            lat='纬度',
            lon='经度',
            z='count',
            radius=10,
            zoom=3.5,
            center={"lat": 35.8617, "lon": 104.1954},
            title="2025年仍存在的各城市企业数量热力图",
            hover_name="所属城市",
            hover_data={"count": True},
            mapbox_style="carto-positron"
        )
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        fig.show()
        fig.write_html("企业分布热力图.html")