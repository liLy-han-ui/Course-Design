import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
data = pd.read_csv("../数据文件/合并后的城市经济与企业数据1.csv")

# 确保年份是整数类型
data['年份'] = data['年份'].astype(int)

# 获取所有不同的城市名称
cities = data['城市'].unique()

# 创建一个新的图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 定义一个函数用于更新曲线的可见性
def update_visibility(label):
    for line in lines:
        if line.get_label() == label:
            line.set_visible(not line.get_visible())
    plt.legend()
    plt.draw()

# 绘制每个城市的曲线
lines = []
for city in cities:
    city_data = data[data['城市'] == city]
    line, = ax.plot(city_data['年份'], city_data['注销'], marker='o', label=city)
    lines.append(line)

# 添加图例，并使图例响应点击事件
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
fig.canvas.mpl_connect('pick_event', lambda event: update_visibility(event.artist.get_label()) if isinstance(event.artist, plt.Line2D) else None)

# 设置标题和标签
plt.title('不同城市的企业注销数量变化趋势')
plt.xlabel('年份')
plt.ylabel('企业数量')

# 调整布局以防止标签被裁剪
plt.tight_layout()

# 显示图形
plt.show()