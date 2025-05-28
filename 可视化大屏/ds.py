from collections import defaultdict
import plotly.express as px
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pyecharts.charts import Map
from pyecharts import options as opts
import tempfile


# =====================
# 页面配置
# =====================
st.set_page_config(
    page_title="AI企业智能分析平台",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.big-title {
    font-size: 32px !important;
    font-weight: 700;
    color: #2B5F91;
    margin-bottom: 20px;
}
.metric-card {
    background: linear-gradient(135deg, #f0f2f6 0%, #e0e7ff 100%);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #2B5F91;
}
.metric-label {
    font-size: 18px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

global start_year,end_year,time_span
start_year=2000
end_year=2025
time_span=1
# =====================
# 数据加载与预处理
# =====================
@st.cache_data(ttl=3600)
def load_all_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path1 = os.path.join(current_dir, "..", "其余信息统计结果", "processed_data.csv")
    data_path2 = os.path.join(current_dir, "..", "其余信息统计结果", "scale_data.csv")
    data_path3 = os.path.join(current_dir, "..", "其余信息统计结果", "investment_data.csv")
    data_path4 = os.path.join(current_dir, "..", "其余信息统计结果", "industry_heatmap_data.csv")
    data_path5 = os.path.join(current_dir, "..", "其余信息统计结果", "survival_cycle_data.csv")
    data_path6 = os.path.join(current_dir, "..", "其余信息统计结果", "capital_sum_data.csv")
    trend_df = pd.read_csv(data_path1)
    scale_df = pd.read_csv(data_path2)
    invest_df = pd.read_csv(data_path3)
    heatmap_df = pd.read_csv(data_path4)
    survival_df = pd.read_csv(data_path5)
    capital_df = pd.read_csv(data_path6)

    for df in [trend_df, scale_df, invest_df, heatmap_df, survival_df, capital_df]:
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df.sort_values('year', inplace=True)

    return trend_df, scale_df, invest_df, heatmap_df, survival_df, capital_df

trend_df, scale_df, invest_df, heatmap_df, survival_df, capital_df = load_all_data()
latest_year = trend_df['year'].max()
latest_data = trend_df[trend_df['year'] == latest_year].iloc[0]
# =====================
# 数据加载函数
# =====================
def load_data(year, metric):
    """根据年份和指标加载数据"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        if metric == "全部企业":
        
            file_path = os.path.join(current_dir, "..", "城市企业统计", "全部企业", f"全部企业_{year}.csv")
            data = pd.read_csv(file_path)
            return data['城市'].tolist(), data['全部企业'].fillna(0).tolist()
        elif metric == "企业密度":
            file_path = os.path.join(current_dir, "..", "城市企业密度", f"{year}_density.csv")
            data = pd.read_csv(file_path)
            return data['城市'].tolist(), data['企业密度'].fillna(0).tolist()
        elif metric in ["新增", "注销", "净增加"]:
            file_path = os.path.join(current_dir, "..", "城市企业统计", "净增加", f"净增加企业_{year}.csv")
            data = pd.read_csv(file_path)
            return data['城市'].tolist(), data[metric].fillna(0).tolist()
    except FileNotFoundError:
        st.warning(f"文件未找到：{file_path}")
        return [], []
    except Exception as e:
        st.error(f"加载数据时发生错误：{e}")
        return [], []

# 加载数据函数
@st.cache_data
def load_data1():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    # 构建目标文件的绝对路径
    excel_file_path = os.path.join(current_dir, "..", "领域风险预警", "结果", "风险评分结果_优化版.xlsx")
    # 使用绝对路径读取Excel文件的不同Sheet
    short_long = pd.read_excel(excel_file_path, sheet_name="短期vs长期对比", header=0)
    short = pd.read_excel(excel_file_path, sheet_name="短期风险评分", header=0)
    long = pd.read_excel(excel_file_path, sheet_name="长期风险评分", header=0)
    # 清理列名中的换行符和空格
    short_long.columns = short_long.columns.str.replace(r'\s+', '', regex=True)
    short.columns = short.columns.str.replace(r'\s+', '', regex=True)
    long.columns = long.columns.str.replace(r'\s+', '', regex=True)

    # 合并数据
    combined = pd.merge(
        short[["领域", "短期_平均增长率", "短期_波动率", "短期_风险评分"]],
        long[["领域", "长期_平均增长率", "长期_波动率", "长期_风险评分"]],
        on="领域",
        how="inner"
    )

    return short_long, short, long, combined
# =====================
# 定义颜色分段规则
# =====================
def get_color_pieces(metric):
    """根据指标返回对应的颜色分段规则"""
    color_schemes = {
        "全部企业": [
            {"min": 10000, "color": "#8A0808"},
            {"min": 9001, "max": 9999, "color": "#940707"},
            {"min": 7001, "max": 9000, "color": "#A00606"},
            {"min": 5001, "max": 7000, "color": "#AD0505"},
            {"min": 4001, "max": 5000, "color": "#B40404"},
            {"min": 3001, "max": 4000, "color": "#C80303"},
            {"min": 1001, "max": 3000, "color": "#E30202"},
            {"min": 301, "max": 1000, "color": "#DF0101"},
            {"min": 0, "max": 300, "color": "#F6CECE"}
        ],
        "企业密度": [
            {"min": 40000, "color": "#08519c"},
            {"min": 10000, "max": 49999, "color": "#1363b3"},
            {"min": 5001, "max": 9999, "color": "#2a77c6"},
            {"min": 3001, "max": 5000, "color": "#4292c6"},
            {"min": 1001, "max": 3000, "color": "#6baed6"},
            {"min": 100, "max":1000 , "color": "#9ecae1"},
            {"min": 51, "max": 100, "color": "#c6dbef"},
            {"min": 11, "max": 50, "color": "#e6f3ff"},
            {"min": 0, "max": 10, "color": "#deebf7"}
        ],
        "新增": [
            {"min": 10000, "color": "#006837"},
            {"min": 5001, "max": 9999, "color": "#007843"},
            {"min": 4001, "max": 5000, "color": "#1b8a4c"},
            {"min": 2001, "max": 4000, "color": "#31a354"},
            {"min": 1001, "max": 2000, "color": "#52b965"},
            {"min": 601, "max": 1000, "color": "#78c679"},
            {"min": 401, "max": 600, "color": "#a1d99b"},
            {"min": 201, "max": 400, "color": "#c2e699"},
            {"min": 0, "max": 200, "color": "#e5f5e0"}
        ],
        "消亡": [
            {"min": 2000, "color": "#8B0000"},
            {"min": 1601, "max": 1999, "color": "#a01414"},
            {"min": 1401, "max": 1600, "color": "#b22222"},
            {"min": 1001, "max": 1400, "color": "#c43030"},
            {"min": 801, "max": 1000, "color": "#cd5c5c"},
            {"min": 401, "max": 800, "color": "#d97777"},
            {"min": 201, "max":400 , "color": "#f08080"},
            {"min": 101, "max": 200, "color": "#f7bcbc"},
            {"min": 0, "max": 100, "color": "#ffe6e6"}
        ],
        "净增加": [
            {"min": 10000, "color": "#00008B"},
            {"min": 8001, "max": 9999, "color": "#00009b"},
            {"min": 7001, "max": 8000, "color": "#0000cd"},
            {"min": 5001, "max": 7000, "color": "#0000e3"},
            {"min": 3501, "max": 5000, "color": "#1E90FF"},
            {"min": 2001, "max": 3500, "color": "#4da6ff"},
            {"min": 1001, "max": 2000, "color": "#87CEFA"},
            {"min": 501, "max": 1000, "color": "#b3e2ff"},
            {"min": 0, "max": 500, "color": "#e6f7ff"},
            {"max": -1, "color": "#8A0808"},
        ]
    }
    return color_schemes.get(metric, [])
def create_map(cities, values, title, pieces):
    """创建 Pyecharts 中国市级地图"""
    map_chart = (
        Map(init_opts=opts.InitOpts(bg_color="white", width="100%", height="1100px"))
        .add(
            series_name=title,
            data_pair=list(zip(cities, values)),
            maptype="china-cities",
            is_map_symbol_show=False,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=pieces,
                pos_left="5%",
                pos_top="middle",
                item_width=25,
                item_height=25,

            ),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    return map_chart

# 数据加载函数


def load_historical_data(year):
    try:
        return pd.read_csv(f"../城市企业统计/全部企业/全部企业_{year}.csv")
    except FileNotFoundError:
        return None

# 主程序逻辑

# =====================
# 侧边导航栏
# =====================
with st.sidebar:
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建目标文件的绝对路径
    image_path = os.path.join(current_dir, "AI.png")
    
    # 读取并显示图片
    st.image(image_path, width=268)

    st.markdown('<div style="font-size:24px;color:#2B5F91;">AI Enterprise Analytics</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    # 选择模块类型
    module_mode = st.radio("选择模块",
                           ["现状透视", "前瞻洞察","危机预防"],
                           captions=["企业数据探索性分析", "未来趋势预测","风险预警单元"])
    st.markdown("---")
    if module_mode == "现状透视":
        analysis_mode = st.radio("现状透视",
                                 ["总体趋势洞察",
                                  "企业规模分析",
                                  "投资类型分析",
                                  "行业热力图分析",
                                  "生存周期分析",
                                  "注册资本分析",
                                  "互联网技术领域分析",
                                  "地区分布分析"],
                                 captions=["企业数量变化分析",
                                           "企业规模分布分析",
                                           "投资主体构成分析",
                                           "行业热度全景分析",
                                           "生存周期分布分析",
                                           "年新增企业注册资本规模分析",
                                           "年存活企业互联网技术领域分布分析",
                                           "企业分布时空分析"]
                                 )
    elif module_mode == "前瞻洞察":
        forecast_mode = st.radio("前瞻洞察",
                                 ["地区企业预测",
                                  "技术领域预测"],
                                 captions=["地方企业发展蓝图",
                                           "行业技术前景展望"])
    elif module_mode == "危机预防":
        risk_avert_mode = st.radio("危机预防",
                                 ["企业流失风险城市分析",
                                  "领域风险预测"],
                                 captions=["基于历史数据的城市风险评分",
                                           "关键技术领域风险预测"])

    st.markdown("---")
    st.write(f"数据更新时间：{latest_year}")
if module_mode == "现状透视":
    # =====================
    # 总体趋势分析模块
    # =====================
    if analysis_mode == "总体趋势洞察":
        st.markdown('<div class="big-title">AI企业生态趋势全景</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">2024年AI企业</h3>', unsafe_allow_html=True)
        # 顶部指标卡
        col1, col2, col3, col4 = st.columns(4)
        metrics = {
            "累计总量": latest_data['total'],
            "年度新增": latest_data['new'],
            "年度消亡": latest_data['death'],
            "年度净增长": latest_data['net_growth']
        }

        for idx, (label, value) in enumerate(metrics.items()):
            with [col1, col2, col3, col4][idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="margin: 4rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">趋势动态分析</h3>', unsafe_allow_html=True)

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            metric_options = {
                "新增企业": {"col": "new", "color": "#2B5F91"},
                "消亡企业": {"col": "death", "color": "#E74C3C"},
                "净增长": {"col": "net_growth", "color": "#F1C40F"},
                "企业总量": {"col": "total", "color": "#2ECC71"}
            }
            selected_metric = st.selectbox("选择分析维度", list(metric_options.keys()))
            config = metric_options[selected_metric]

        # 创建组合图表
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 主Y轴 - 数量
        fig.add_trace(go.Bar(
            x=trend_df["year"],
            y=trend_df[config["col"]],
            name=f"{selected_metric}数量",
            marker_color=config["color"],
            opacity=0.8,
            hovertemplate="年份: %{x}<br>数量: %{y:,.0f}<extra></extra>"
        ), secondary_y=False)

        # 次Y轴 - 增速
        growth_col = f"{config['col']}_growth%"
        if growth_col in trend_df.columns:
            fig.add_trace(go.Scatter(
                x=trend_df["year"],
                y=trend_df[growth_col],
                name=f"{selected_metric}增速",
                line=dict(color=config["color"], width=3, dash="dot"),
                hovertemplate="年份: %{x}<br>增速: %{y:.1f}%<extra></extra>"
            ), secondary_y=True)

        fig.update_layout(
            title=f"{selected_metric}趋势分析（{trend_df['year'].min()} - {latest_year}）",
            template="plotly_dark",
            hovermode='x',
            xaxis_title="年份",
            yaxis=dict(title="企业数量", showgrid=False),
            yaxis2=dict(title="增速 (%)", overlaying="y", side="right",
                        showgrid=False, range=[-100, 200]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI企业自然变动分析
        with st.container():
            st.markdown('<h3 class="section-title">AI企业自然变动分析</h3>', unsafe_allow_html=True)
            rate_fig = go.Figure()

            rate_config = {
                "birth_rate%": {"name": "新生率", "color": "#2ECC71"},
                "death_rate%": {"name": "消亡率", "color": "#E74C3C"},
                "natural_growth%": {"name": "自然增长率", "color": "#3498DB"}
            }

            # 生命周期率分析模块的修正代码
            for rate_col, config in rate_config.items():
                rate_fig.add_trace(go.Scatter(
                    x=trend_df["year"],
                    y=trend_df[rate_col],
                    name=config["name"],  # 保留图例名称
                    line=dict(color=config["color"], width=3),
                    mode="lines+markers",
                    hovertemplate=
                    "<b>%{x}年</b><br>"
                    "%{text}: %{y:.1f}%<extra></extra>",
                    text=[config["name"]] * len(trend_df)
                ))

            rate_fig.add_hline(y=0, line_color="yellow", line_width=2, opacity=0.5)
            rate_fig.update_layout(
                template="plotly_dark",
                yaxis_title="百分比 (%)",
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                hovermode='x',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(rate_fig, use_container_width=True)

    # =====================
    # 企业规模分析模块
    # =====================
    elif analysis_mode == "企业规模分析":
        st.markdown('<div class="big-title">企业规模多维透视</div>', unsafe_allow_html=True)

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            metric_options = {
                "新增企业": "new",
                "消亡企业": "death",
                "净增长": "net_growth",
                "总量": "cumulative"
            }
            selected_metric = st.selectbox("选择分析维度", list(metric_options.keys()))
            display_mode = st.radio(
                "选择显示模式",
                ["柱状图+趋势线", "仅柱状图", "仅趋势线"],
                horizontal=True,
            )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        COLOR_MAP = {
            "微型": {"fill": "#a1c4fd", "line": "#1a73e8"},
            "小型": {"fill": "#a7ffeb", "line": "#12b28d"},
            "中型": {"fill": "#ffd7a6", "line": "#ff6d00"},
            "大型": {"fill": "#e0a8ff", "line": "#9c27b0"}
        }
        SIZE_ORDER = ["微型", "小型", "中型", "大型"]

        for size in SIZE_ORDER:
            subset = scale_df[scale_df.company_size == size]
            metric_col = metric_options[selected_metric]

            if display_mode in ["柱状图+趋势线", "仅柱状图"]:
                fig.add_trace(go.Bar(
                    x=subset.year,
                    y=subset[metric_col],
                    name=f"{size}企业",
                    marker=dict(color=COLOR_MAP[size]["fill"]),
                    customdata=subset[[f"{metric_col}_pct"]].values,
                    hovertemplate=(
                        f"<b>{size}企业</b><br>"
                        "年份: %{x}<br>"
                        "数量: %{y:,.0f}<br>"
                        "占比: %{customdata[0]:.1f}%"
                        "<extra></extra>"
                    ),
                    legendrank=SIZE_ORDER.index(size)
                ), secondary_y=False)

            if display_mode in ["柱状图+趋势线", "仅趋势线"]:
                fig.add_trace(go.Scatter(
                    x=subset.year,
                    y=subset[f"{metric_col}_pct"],
                    name=f"{size}占比",
                    line=dict(color=COLOR_MAP[size]["line"], width=2, dash="dot"),
                    hovertemplate=(
                        f"<b>{size}企业</b><br>"
                        "年份: %{x}<br>"
                        "占比: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                    legendrank=SIZE_ORDER.index(size) + 4
                ), secondary_y=True)

        layout_config = {
            "title": f"{selected_metric}规模分布（{scale_df['year'].min()} - {scale_df['year'].max()}）",
            "template": "plotly_dark",
            "hovermode": "closest",
            "xaxis_title": "年份",
            "yaxis": dict(title="企业数量", showgrid=False),
            "yaxis2": dict(title="占比 (%)", overlaying="y", side="right",
                           showgrid=False, range=[0, 100]),
            "barmode": "group",
            "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        }

        if display_mode == "仅趋势线":
            layout_config["yaxis"]["showticklabels"] = False
            layout_config["yaxis"]["title"] = ""

        fig.update_layout(**layout_config)
        st.plotly_chart(fig, use_container_width=True)

    # =====================
    # 投资类型分析模块
    # =====================
    elif analysis_mode == "投资类型分析":
        st.markdown('<div class="big-title">投资主体深度分析</div>', unsafe_allow_html=True)

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            metric_options = {
                "新增企业": "new",
                "消亡企业": "death",
                "净增长": "net_growth",
                "总量": "cumulative"
            }
            selected_metric = st.selectbox("选择分析维度", list(metric_options.keys()))
            display_mode = st.radio(
                "选择显示模式",
                ["柱状图+趋势线", "仅柱状图", "仅趋势线"],
                horizontal=True,
            )

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        COLOR_MAP = {
            "国有企业": {"fill": "rgba(255,69,0,0.6)", "line": "rgb(255,69,0)"},
            "外商投资企业": {"fill": "rgba(102,178,255,0.6)", "line": "rgb(0,102,204)"},
            "港澳台投资企业": {"fill": "rgba(55,83,109,0.6)", "line": "rgb(33,50,65)"},
            "民营企业": {"fill": "rgba(255,193,7,0.6)", "line": "rgb(255,152,0)"}
        }
        INVESTMENT_ORDER = ["国有企业", "外商投资企业", "港澳台投资企业", "民营企业"]

        for inv_type in INVESTMENT_ORDER:
            subset = invest_df[invest_df.investment_type == inv_type]
            metric_col = metric_options[selected_metric]

            if display_mode in ["柱状图+趋势线", "仅柱状图"]:
                fig.add_trace(go.Bar(
                    x=subset.year,
                    y=subset[metric_col],
                    name=inv_type,
                    marker=dict(color=COLOR_MAP[inv_type]["fill"]),
                    customdata=subset[f"{metric_col}_pct"],
                    hovertemplate=(
                        f"<b>{inv_type}</b><br>"
                        "年份: %{x}<br>"
                        "数量: %{y:,.0f}<br>"
                        "占比: %{customdata:.1f}%"
                        "<extra></extra>"
                    ),
                    legendrank=INVESTMENT_ORDER.index(inv_type)
                ), secondary_y=False)

            if display_mode in ["柱状图+趋势线", "仅趋势线"]:
                fig.add_trace(go.Scatter(
                    x=subset.year,
                    y=subset[f"{metric_col}_pct"],
                    name=f"{inv_type}占比",
                    line=dict(color=COLOR_MAP[inv_type]["line"], width=2, dash="dot"),
                    hovertemplate=(
                        f"<b>{inv_type}占比</b><br>"
                        "年份: %{x}<br>"
                        "百分比: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                    legendrank=INVESTMENT_ORDER.index(inv_type) + 4
                ), secondary_y=True)

        layout_config = {
            "title": f"{selected_metric}投资类型分布（{invest_df['year'].min()} - {latest_year}）",
            "template": "plotly_dark",
            "hovermode": "closest",
            "xaxis_title": "年份",
            "yaxis": dict(title="企业数量", showgrid=False),
            "yaxis2": dict(title="占比 (%)", overlaying="y", side="right",
                           showgrid=False, range=[0, 100]),
            "barmode": "group",
            "legend": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        }

        if display_mode == "仅趋势线":
            layout_config["yaxis"]["showticklabels"] = False
            layout_config["yaxis"]["title"] = ""

        fig.update_layout(**layout_config)
        st.plotly_chart(fig, use_container_width=True)

    # =====================
    # 经营范围热力图模块
    # =====================
    elif analysis_mode == "互联网技术领域分析":
        st.markdown('<div class="big-title">企业技术领域统计</div>', unsafe_allow_html=True)

        # 数据加载
        @st.cache_data(ttl=3600)
        def load_business_scope_data():
                        # 获取当前脚本所在目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 构建目标文件的绝对路径
            file_path = os.path.join(current_dir, "..", "企业经营范围", "企业统计结果.csv")
            business_scope_df = pd.read_csv(file_path) # 确保 'data.csv' 文件路径正确
            return business_scope_df

        business_scope_df = load_business_scope_data()

        # 选择要可视化的指标
        metric_options = {
            "新增企业": "新增企业数量",
            "消亡企业": "注销企业数量",
            "净增长企业": "净增长企业数量",
            "累计存活企业": "累计存活企业数量"
        }
        selected_metric = st.selectbox("选择分析维度", list(metric_options.keys()))
        metric = metric_options[selected_metric]

        # 创建堆叠柱状图
        fig = go.Figure()

        for field in business_scope_df['领域'].unique():
            field_data = business_scope_df[business_scope_df['领域'] == field]
            fig.add_trace(go.Bar(
                x=field_data['年份'],       # X轴为年份
                y=field_data[metric],      # Y轴为选定的度量标准
                name=field,                # 每个领域的名称
                text=field_data[metric],   # 显示数值
                textposition='inside',     # 数值显示在柱子内部
            ))

        # 更新布局
        fig.update_layout(
            title=f"{selected_metric}堆叠柱状图",  # 图表标题
            xaxis_title="年份",                  # X轴标题
            yaxis_title="企业数量",              # Y轴标题
            barmode='stack',                    # 堆叠模式
            legend_title="领域"                 # 图例标题
        )

        # 显示图形
        st.plotly_chart(fig, use_container_width=True)
    # =====================
    # 行业热力图分析模块
    # =====================
    elif analysis_mode == "行业热力图分析":
        st.markdown('<div class="big-title">行业热度全景分析</div>', unsafe_allow_html=True)

        # 定义颜色分段规则
        COLOR_SCHEMES2 = {
            "新增": [
                {"min": 0, "max": 1, "color": "#ffffff"},  # 白色
                {"min": 2, "max": 99, "color": "#f7fcf5"},  # 浅绿色
                {"min": 100, "max": 499, "color": "#e5f5e0"},  # 更深一点的浅绿色
                {"min": 500, "max": 1999, "color": "#c7e9c0"},  # 浅绿色
                {"min": 2000, "max": 4999, "color": "#a1d99b"},  # 绿色
                {"min": 5000, "max": 9999, "color": "#74c476"},  # 深绿色
                {"min": 10000, "max": 19999, "color": "#41ab5d"},  # 更深的绿色
                {"min": 20000, "max": 49999, "color": "#238b45"},  # 非常深的绿色
                {"min": 50000, "max": 99999, "color": "#006d2c"},  # 几乎黑色的绿色
                {"min": 100000, "color": "#00441b"}  # 黑绿色
            ],
            "消亡": [
                {"min": 0, "max": 1, "color": "#ffffff"},  # 白色
                {"min": 2, "max": 99, "color": "#fff5f0"},  # 浅红色
                {"min": 100, "max": 499, "color": "#fee0d2"},  # 更深一点的浅红色
                {"min": 500, "max": 999, "color": "#fcbba1"},  # 浅红色
                {"min": 1000, "max": 1999, "color": "#fc9272"},  # 红色
                {"min": 2000, "max": 4999, "color": "#fb6a4a"},  # 深红色
                {"min": 5000, "max": 9999, "color": "#ef3b2c"},  # 更深的红色
                {"min": 10000, "max": 19999, "color": "#cb181d"},  # 非常深的红色
                {"min": 20000, "max": 29999, "color": "#a50f15"},  # 几乎黑色的红色
                {"min": 30000, "color": "#67000d"}  # 黑红色
            ],
            "净增长": [
                {"min": -1000, "max": -1, "color": "#ffcccc"},  # 浅粉色（负值）
                {"min": 0, "max": 1, "color": "#ffffff"},  # 白色
                {"min": 2, "max": 99, "color": "#f7fbff"},  # 浅蓝色
                {"min": 100, "max": 499, "color": "#deebf7"},  # 更深一点的浅蓝色
                {"min": 500, "max": 999, "color": "#c6dbef"},  # 浅蓝色
                {"min": 1000, "max": 1999, "color": "#9ecae1"},  # 蓝色
                {"min": 2000, "max": 4999, "color": "#6baed6"},  # 深蓝色
                {"min": 5000, "max": 9999, "color": "#4292c6"},  # 更深的蓝色
                {"min": 10000, "max": 19999, "color": "#2171b5"},  # 非常深的蓝色
                {"min": 20000, "max": 49999, "color": "#08519c"},  # 几乎黑色的蓝色
                {"min": 50000, "max": 79999, "color": "#08306b"},  # 黑蓝色
                {"min": 80000, "color": "#00008B"}  # 深蓝色
            ],
            "总量": [
                {"min": 0, "max": 1, "color": "#ffffff"},  # 白色
                {"min": 2, "max": 99, "color": "#f7fbff"},  # 浅蓝色
                {"min": 100, "max": 499, "color": "#deebf7"},  # 更深一点的浅蓝色
                {"min": 500, "max": 1999, "color": "#c6dbef"},  # 浅蓝色
                {"min": 2000, "max": 4999, "color": "#9ecae1"},  # 蓝色
                {"min": 5000, "max": 9999, "color": "#6baed6"},  # 深蓝色
                {"min": 10000, "max": 19999, "color": "#4292c6"},  # 更深的蓝色
                {"min": 20000, "max": 49999, "color": "#2171b5"},  # 非常深的蓝色
                {"min": 50000, "max": 99999, "color": "#08519c"},  # 几乎黑色的蓝色
                {"min": 100000, "max": 199999, "color": "#08306b"},  # 黑蓝色
                {"min": 200000, "max": 299999, "color": "#00008B"},  # 深蓝色
                {"min": 300000, "max": 399999, "color": "#0000CD"},  # 更深的蓝色
                {"min": 400000, "color": "#00008B"}  # 最深蓝色
            ]
        }

        # 修改指标映射（主要修改点2）
        METRIC_OPTIONS = {
            "新增企业": {"col": "new", "title": "新增"},
            "消亡企业": {"col": "death", "title": "消亡"},
            "净增长": {"col": "net_growth", "title": "净增长"},
            "总量": {"col": "cumulative", "title": "总量"}
        }

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            selected_metric = st.selectbox(
                "选择分析指标",
                options=list(METRIC_OPTIONS.keys()),
                index=0
            )
            # 获取对应颜色方案（主要修改点3）
            selected_title = METRIC_OPTIONS[selected_metric]["title"]
            color_scheme = COLOR_SCHEMES2.get(selected_title, [])


        # 重构create_heatmap函数（主要修改点4）
        def create_heatmap(df, metric_col, color_scheme):
            heatmap_data = df.pivot_table(
                index="display_label",
                columns="year",
                values=metric_col,
                fill_value=0
            )

            # 计算全局 zmin 和 zmax
            zmin = min(piece["min"] for piece in color_scheme if "min" in piece)
            zmax = max(piece.get("max", zmin) for piece in color_scheme)

            # 生成 colorscale
            colorscale = []
            for piece in color_scheme:
                # 处理起始值和结束值
                start = (piece.get("min", zmin) - zmin) / (zmax - zmin)
                end = (piece.get("max", zmax) - zmin) / (zmax - zmin)
                start = max(0.0, min(start, 1.0))  # 限制在 [0,1]
                end = max(0.0, min(end, 1.0))
                colorscale.append([start, piece["color"]])
                colorscale.append([end, piece["color"]])

            # 去重并排序（修复关键点）
            unique_colorscale = []
            seen = set()
            for item in colorscale:
                key = (round(item[0], 5), item[1])
                if key not in seen:
                    seen.add(key)
                    unique_colorscale.append(item)
            unique_colorscale.sort(key=lambda x: x[0])

            # 确保首尾为 0.0 和 1.0
            if unique_colorscale:
                if unique_colorscale[0][0] > 0.0:
                    unique_colorscale.insert(0, [0.0, unique_colorscale[0][1]])
                if unique_colorscale[-1][0] < 1.0:
                    unique_colorscale.append([1.0, unique_colorscale[-1][1]])
            else:
                unique_colorscale = [[0.0, "#ffffff"], [1.0, "#000000"]]  # 默认颜色

            # 转换为 Plotly 格式（确保是列表的列表）
            final_colorscale = []
            for item in unique_colorscale:
                final_colorscale.append([float(item[0]), str(item[1])])

            # 创建热力图
            fig = go.Figure(go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale=final_colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title="企业数量", tickformat=",d"),
                hoverongaps=False
            ))

            fig.update_layout(
                title=f"{selected_metric}行业热力图",
                template="plotly_dark",
                height=800
            )
            return fig


        # 使用新参数调用函数（主要修改点5）
        heatmap_fig = create_heatmap(
            heatmap_df,
            METRIC_OPTIONS[selected_metric]["col"],
            color_scheme
        )

        st.plotly_chart(heatmap_fig, use_container_width=True)

    # =====================
    # 生存周期分析模块
    # =====================
    elif analysis_mode == "生存周期分析":
        st.markdown('<div class="big-title">企业生存周期深度分析</div>', unsafe_allow_html=True)

        # 绘制堆叠面积图
        fig = go.Figure()

        # 已死亡企业
        fig.add_trace(go.Scatter(
            x=survival_df["生存周期区间"],
            y=survival_df["已死亡企业数量"],
            mode="lines",
            stackgroup="one",
            name="消亡企业",
            line=dict(color="rgb(255, 69, 0)", width=2),
            fillcolor="rgba(255, 69, 0, 0.5)"
        ))

        # 存活企业
        fig.add_trace(go.Scatter(
            x=survival_df["生存周期区间"],
            y=survival_df["存活企业数量"],
            mode="lines",
            stackgroup="one",
            name="存活企业",
            line=dict(color="rgb(0, 128, 0)", width=2),
            fillcolor="rgba(0, 128, 0, 0.5)"
        ))

        fig.update_layout(
            title="企业生存周期分布（月）",
            xaxis_title="生存周期区间",
            yaxis_title="企业数量",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================
    # 地区分布分析
    # =====================
    elif analysis_mode == "地区分布分析":
        st.markdown('<div class="big-title">城市企业地理分布热力图</div>', unsafe_allow_html=True)

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            selected_year = st.slider(
                "选择年份",
                min_value=start_year,
                max_value=end_year,
                value=start_year
            )
            metric_options = {
                "全部企业": "全部企业",
                "企业密度": "企业密度",
                "新增": "新增",
                "消亡": "注销",  # 显示“消亡”，实际用“注销”
                "净增加": "净增加"
            }
            
            selected_display = st.selectbox(
                "选择指标",
                options=list(metric_options.keys())
            )
            
            # 获取实际列名
            selected_metric = metric_options[selected_display]

        # 加载数据
        cities, values = load_data(selected_year, selected_metric)
        if not cities or not values:
            st.error("无法加载指定年份和指标的数据，请检查输入参数或数据文件。")
        else:
            title = f"{selected_display}_{selected_year}"

            # 获取对应指标的颜色分段规则
            pieces = get_color_pieces(selected_metric)

            # 创建地图
            map_chart = create_map(cities, values, title, pieces)

            # 保存为临时 HTML 文件
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                html_file_path = tmp_file.name
            map_chart.render(html_file_path)

            # 读取 HTML 内容并嵌入 Streamlit
            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            # 显示热力图
            try:
                st.components.html(html_content, height=800, scrolling=True)
            except AttributeError:
                st.components.v1.html(html_content, height=800, scrolling=True)

            # 删除临时文件
            os.unlink(html_file_path)
    # 注册资本分析模块
    # =====================
    else:
        st.markdown('<div class="big-title">企业注册资本规模分析</div>', unsafe_allow_html=True)

        # 绘制折线图
        fig_capital = go.Figure()

        fig_capital.add_trace(go.Scatter(
            x=capital_df["成立年份"],
            y=capital_df["总注册资本_万人民币"],
            mode="lines+markers",
            name="注册资本总额",
            line=dict(color="#4CAF50", width=3),
            marker=dict(size=8)
        ))

        fig_capital.update_layout(
            title="2000-2024年新增AI企业注册资本总额变化趋势（单位：万人民币）",
            xaxis_title="年份",
            yaxis_title="注册资本总额（万人民币）",
            template="plotly_dark",
            hovermode="x unified",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis=dict(tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_capital, use_container_width=True)
    # =====================
    # 前瞻洞察 - 地区企业预测模块
    # =====================
elif module_mode == "前瞻洞察":
    if forecast_mode == "地区企业预测":
        st.markdown('<div class="big-title">地区企业数量预测趋势分析</div>', unsafe_allow_html=True)
        st.markdown('展示2000年至2030年各城市企业数量的历史与预测变化趋势')

        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建目标文件的绝对路径
        file_path = os.path.join(current_dir, "..", "城市企业数量预测", "预测结果", "combined_city_data_2000_2030.csv")
        
        # 读取数据
        forecast_df = pd.read_csv(file_path)

        # 2. 构建城市-省份映射（示例）
        city_to_province = {
            "北京市": "北京市",
            "上海市": "上海市",
            "天津市": "天津市",
            "重庆市": "重庆市",
            "哈尔滨市": "黑龙江省",
            "兰州市": "甘肃省",
            "昆明市": "云南省",
            "贵阳市": "贵州省",
            "海口市": "海南省",
            "台北市": "台湾省",
            "呼和浩特市": "内蒙古自治区",
            "乌鲁木齐市": "新疆维吾尔自治区",
            "拉萨市": "西藏自治区",
            "七台河市": "黑龙江省",
            "三亚市": "海南省",
            "三明市": "福建省",
            "三门峡市": "河南省",
            "东莞市": "广东省",
            "东营市": "山东省",
            "中卫市": "宁夏回族自治区",
            "中山市": "广东省",
            "临汾市": "山西省",
            "临沂市": "山东省",
            "临沧市": "云南省",
            "丹东市": "辽宁省",
            "丽水市": "浙江省",
            "丽江市": "云南省",
            "乌兰察布市": "内蒙古自治区",
            "乌海市": "内蒙古自治区",
            "乐山市": "四川省",
            "九江市": "江西省",
            "云浮市": "广东省",
            "亳州市": "安徽省",
            "伊春市": "黑龙江省",
            "佛山市": "广东省",
            "佳木斯市": "黑龙江省",
            "保定市": "河北省",
            "保山市": "云南省",
            "信阳市": "河南省",
            "儋州市": "海南省",
            "克拉玛依市": "新疆维吾尔自治区",
            "六安市": "安徽省",
            "六盘水市": "贵州省",
            "内江市": "四川省",
            "包头市": "内蒙古自治区",
            "北海市": "广西壮族自治区",
            "十堰市": "湖北省",
            "南京市": "江苏省",
            "南充市": "四川省",
            "南宁市": "广西壮族自治区",
            "南平市": "福建省",
            "南昌市": "江西省",
            "南通市": "江苏省",
            "南阳市": "河南省",
            "厦门市": "福建省",
            "双鸭山市": "黑龙江省",
            "台州市": "浙江省",
            "合肥市": "安徽省",
            "吉安市": "江西省",
            "吉林市": "吉林省",
            "吕梁市": "山西省",
            "吴忠市": "宁夏回族自治区",
            "周口市": "河南省",
            "呼伦贝尔市": "内蒙古自治区",
            "咸宁市": "湖北省",
            "咸阳市": "陕西省",
            "唐山市": "河北省",
            "商丘市": "河南省",
            "商洛市": "陕西省",
            "嘉兴市": "浙江省",
            "嘉峪关市": "甘肃省",
            "四平市": "吉林省",
            "固原市": "宁夏回族自治区",
            "大同市": "山西省",
            "大庆市": "黑龙江省",
            "大连市": "辽宁省",
            "天水市": "甘肃省",
            "太原市": "山西省",
            "威海市": "山东省",
            "娄底市": "湖南省",
            "孝感市": "湖北省",
            "宁德市": "福建省",
            "宁波市": "浙江省",
            "安庆市": "安徽省",
            "安康市": "陕西省",
            "安阳市": "河南省",
            "安顺市": "贵州省",
            "定西市": "甘肃省",
            "宜宾市": "四川省",
            "宜昌市": "湖北省",
            "宜春市": "江西省",
            "宝鸡市": "陕西省",
            "宣城市": "安徽省",
            "宿州市": "安徽省",
            "宿迁市": "江苏省",
            "岳阳市": "湖南省",
            "崇左市": "广西壮族自治区",
            "巴中市": "四川省",
            "巴彦淖尔市": "内蒙古自治区",
            "常州市": "江苏省",
            "常德市": "湖南省",
            "平凉市": "甘肃省",
            "平顶山市": "河南省",
            "广元市": "四川省",
            "广安市": "四川省",
            "广州市": "广东省",
            "庆阳市": "甘肃省",
            "廊坊市": "河北省",
            "延安市": "陕西省",
            "开封市": "河南省",
            "张家口市": "河北省",
            "张家界市": "湖南省",
            "张掖市": "甘肃省",
            "徐州市": "江苏省",
            "德州市": "山东省",
            "德阳市": "四川省",
            "忻州市": "山西省",
            "怀化市": "湖南省",
            "惠州市": "广东省",
            "成都市": "四川省",
            "扬州市": "江苏省",
            "承德市": "河北省",
            "抚州市": "江西省",
            "抚顺市": "辽宁省",
            "揭阳市": "广东省",
            "攀枝花市": "四川省",
            "新乡市": "河南省",
            "新余市": "江西省",
            "无锡市": "江苏省",
            "日照市": "山东省",
            "昭通市": "云南省",
            "晋中市": "山西省",
            "晋城市": "山西省",
            "普洱市": "云南省",
            "景德镇市": "江西省",
            "曲靖市": "云南省",
            "朔州市": "山西省",
            "朝阳市": "辽宁省",
            "本溪市": "辽宁省",
            "来宾市": "广西壮族自治区",
            "杭州市": "浙江省",
            "松原市": "吉林省",
            "枣庄市": "山东省",
            "柳州市": "广西壮族自治区",
            "株洲市": "湖南省",
            "桂林市": "广西壮族自治区",
            "梅州市": "广东省",
            "梧州市": "广西壮族自治区",
            "榆林市": "陕西省",
            "武威市": "甘肃省",
            "武汉市": "湖北省",
            "毕节市": "贵州省",
            "永州市": "湖南省",
            "汉中市": "陕西省",
            "汕头市": "广东省",
            "汕尾市": "广东省",
            "江门市": "广东省",
            "池州市": "安徽省",
            "沈阳市": "辽宁省",
            "沧州市": "河北省",
            "河池市": "广西壮族自治区",
            "河源市": "广东省",
            "泉州市": "福建省",
            "泰安市": "山东省",
            "泰州市": "江苏省",
            "泸州市": "四川省",
            "洛阳市": "河南省",
            "济南市": "山东省",
            "济宁市": "山东省",
            "海东市": "青海省",
            "深圳市": "广东省",
            "清远市": "广东省",
            "温州市": "浙江省",
            "渭南市": "陕西省",
            "湖州市": "浙江省",
            "湘潭市": "湖南省",
            "湛江市": "广东省",
            "滁州市": "安徽省",
            "滨州市": "山东省",
            "漯河市": "河南省",
            "漳州市": "福建省",
            "潍坊市": "山东省",
            "潮州市": "广东省",
            "濮阳市": "河南省",
            "烟台市": "山东省",
            "焦作市": "河南省",
            "牡丹江市": "黑龙江省",
            "玉林市": "广西壮族自治区",
            "玉溪市": "云南省",
            "珠海市": "广东省",
            "白城市": "吉林省",
            "白山市": "吉林省",
            "白银市": "甘肃省",
            "百色市": "广西壮族自治区",
            "益阳市": "湖南省",
            "盐城市": "江苏省",
            "盘锦市": "辽宁省",
            "眉山市": "四川省",
            "石嘴山市": "宁夏回族自治区",
            "石家庄市": "河北省",
            "福州市": "福建省",
            "秦皇岛市": "河北省",
            "绍兴市": "浙江省",
            "绥化市": "黑龙江省",
            "绵阳市": "四川省",
            "聊城市": "山东省",
            "上饶市":"江西省",
            "淄博市":"山东省",
            "淮北市":"安徽省",
            "淮南市":"安徽省",
            "淮安市":"江苏省",
            "肇庆市": "广东省",
            "自贡市": "四川省",
            "舟山市": "浙江省",
            "芜湖市": "安徽省",
            "苏州市": "江苏省",
            "茂名市": "广东省",
            "荆州市": "湖北省",
            "荆门市": "湖北省",
            "莆田市": "福建省",
            "菏泽市": "山东省",
            "萍乡市": "江西省",
            "营口市": "辽宁省",
            "葫芦岛市": "辽宁省",
            "蚌埠市": "安徽省",
            "衡水市": "河北省",
            "衡阳市": "湖南省",
            "衢州市": "浙江省",
            "襄阳市": "湖北省",
            "西宁市": "青海省",
            "西安市": "陕西省",
            "许昌市": "河南省",
            "贵港市": "广西壮族自治区",
            "贺州市": "广西壮族自治区",
            "资阳市": "四川省",
            "赣州市": "江西省",
            "赤峰市": "内蒙古自治区",
            "辽源市": "吉林省",
            "辽阳市": "辽宁省",
            "达州市": "四川省",
            "运城市": "山西省",
            "连云港市": "江苏省",
            "通化市": "吉林省",
            "通辽市": "内蒙古自治区",
            "遂宁市": "四川省",
            "遵义市": "贵州省",
            "邢台市": "河北省",
            "邯郸市": "河北省",
            "邵阳市": "湖南省",
            "郑州市": "河南省",
            "郴州市": "湖南省",
            "鄂尔多斯市": "内蒙古自治区",
            "鄂州市": "湖北省",
            "酒泉市": "甘肃省",
            "金华市": "浙江省",
            "金昌市": "甘肃省",
            "钦州市": "广西壮族自治区",
            "铁岭市": "辽宁省",
            "铜仁市": "贵州省",
            "铜川市": "陕西省",
            "铜陵市": "安徽省",
            "银川市": "宁夏回族自治区",
            "锦州市": "辽宁省",
            "镇江市": "江苏省",
            "长春市": "吉林省",
            "长沙市": "湖南省",
            "长治市": "山西省",
            "阜新市": "辽宁省",
            "阜阳市": "安徽省",
            "防城港市": "广西壮族自治区",
            "阳江市": "广东省",
            "阳泉市": "山西省",
            "陇南市": "甘肃省",
            "随州市": "湖北省",
            "雅安市": "四川省",
            "青岛市": "山东省",
            "鞍山市": "辽宁省",
            "韶关市": "广东省",
            "马鞍山市": "安徽省",
            "驻马店市": "河南省",
            "鸡西市": "黑龙江省",
            "鹤壁市": "河南省",
            "鹤岗市": "黑龙江省",
            "鹰潭市": "江西省",
            "黄冈市": "湖北省",
            "黄山市": "安徽省",
            "黄石市": "湖北省",
            "黑河市": "黑龙江省",
            "齐齐哈尔市": "黑龙江省",
            "龙岩市": "福建省"
        }

        cities = sorted(set(forecast_df['城市']))

        # 按省份分类
        province_to_cities = defaultdict(list)
        for city in cities:
            province = city_to_province.get(city, '未知')
            province_to_cities[province].append(city)

        provinces = sorted(province_to_cities.keys())

        # 用户选择面板
        selected_provinces = st.multiselect("选择省份", options=provinces, key="province_multiselect")

        if not selected_provinces:
            st.warning("请选择至少一个省份。")
        else:
            # 根据选择的省份筛选城市
            available_cities = [city for province in selected_provinces for city in province_to_cities[province]]

            selected_cities = st.multiselect(
                "选择城市进行对比分析",
                options=sorted(available_cities),
                default=[],
                key="city_multiselect"
            )

            if not selected_cities:
                st.warning("请至少选择一个城市进行分析。")
            else:
                # 提取每个城市的完整序列（2000~2030）
                all_years = list(range(2000, 2031))
                combined_data = {}

                for city in selected_cities:
                    row = forecast_df[forecast_df['城市'] == city]
                    if not row.empty:
                        values = []
                        for year in map(str, all_years):
                            val = row[year].values[0] if year in row.columns else None
                            values.append(val)
                        combined_data[city] = values

                trend_df = pd.DataFrame(combined_data, index=all_years)
                trend_df.index.name = "年份"

                # 时间范围筛选器
                min_year, max_year = min(all_years), max(all_years)
                start_year, end_year = st.slider("选择显示的时间范围", min_value=min_year, max_value=max_year,
                                                 value=(min_year, max_year), key="time_slider")

                # 筛选数据
                filtered_df = trend_df.loc[start_year:end_year]

                # 绘制折线图
                fig = go.Figure()
                # 添加一条竖线：表示真实值与预测值的分界点（2025）
                fig.add_vline(
                    x=2023,
                    line_width=2,
                    line_dash="dash",  # 虚线样式
                    line_color="red",
                    annotation_text="真实值 / 预测值分界线",  # 可选标注文本
                    annotation_position="top right"
                )
                for city in selected_cities:
                    valid_series = filtered_df[city].dropna()
                    fig.add_trace(go.Scatter(
                        x=valid_series.index.astype(int),
                        y=valid_series.values,
                        mode='lines+markers',
                        name=city,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))

                fig.update_layout(
                    title=f"选定城市的企业数量趋势（{start_year} - {end_year}）",
                    xaxis_title="年份",
                    yaxis_title="企业数量",
                    template="plotly_dark",
                    hovermode="x unified",
                    width=800,  # 调整宽度
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
                    margin=dict(t=80, b=40, l=50, r=50)
                )

                st.plotly_chart(fig, use_container_width=True)

                # 显示表格（可选）
                with st.expander("查看原始数据表格"):
                    st.dataframe(filtered_df.style.format("{:.0f}").highlight_max(axis=0))

    elif forecast_mode=="技术领域预测":
        st.markdown('<div class="big-title">行业技术前景展望</div>', unsafe_allow_html=True)
 
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建目标文件的绝对路径
        file_path = os.path.join(current_dir, "..", "经营范围预测", "预测结果", "企业技术领域发展数据_2000_2030.csv")
        
        # 读取数据
        forecast_df = pd.read_csv(file_path)
        # 用户选择面板 - 图表类型选择
        chart_type = st.selectbox("选择图表类型", options=["热力图", "折线图"], key="chart_type_select")

        # 假设 forecast_df 已经加载完毕
        if chart_type == "热力图":
            # 热力图数据准备
            heatmap_data = forecast_df.pivot(index='领域', columns='年份', values='累计存活企业数量')

            # 创建一个自定义的色阶，确保有足够的分段来区分不同的数据级别
            color_scale = [
                {"min": 0, "max": 1, "color": "#ffffff"},  # 白色
                {"min": 2, "max": 75, "color": "#fff0f5"},  # 极浅玫瑰红
                {"min": 76, "max": 200, "color": "#fff7f3"},  # 更浅玫瑰红
                {"min": 201, "max": 500, "color": "#ffe4e1"},  # 浅玫瑰红
                {"min": 601, "max": 1000, "color": "#ffd1d9"},  # 稍深一点的浅玫瑰红
                {"min": 1001, "max": 3000, "color": "#fde0dd"},  # 浅粉红
                {"min": 3001, "max": 4000, "color": "#fcc5c0"},  # 淡粉红
                {"min": 4001, "max": 5000, "color": "#fca6b2"},  # 稍深淡粉红
                {"min": 5001, "max": 6000, "color": "#fa9fb5"},  # 中粉红
                {"min": 6001, "max": 8000, "color": "#f78faa"},  # 较深中粉红
                {"min": 8001, "max": 10000, "color": "#f768a1"},  # 中等深度的粉红
                {"min": 10001, "max": 20000, "color": "#f45a8d"},  # 稍深中粉红
                {"min": 20001, "max": 40000, "color": "#dd3497"},  # 深粉红
                {"min": 40001, "max": 60000, "color": "#d01c8b"},  # 更深粉红
                {"min": 60001, "max": 80000, "color": "#b8007f"},  # 更深粉红
                {"min": 80001, "max": 100000, "color": "#ae017e"},  # 更深的紫红
                {"min": 100001, "max": 200000, "color": "#9e0168"},  # 更深紫红
                {"min": 200001, "max": 400000, "color": "#8e0152"},  # 更深紫红
                {"min": 400001, "max": 1000000, "color": "#7e013e"},  # 更深紫红
                {"min": 1000001, "max": 1500000, "color": "#49006a"},  # 更深紫红
                {"min": 1500001, "max": 2000000, "color": "#2A0045"},  # 更深紫红
                {"min": 2000001, "color": "#1D0033"}  # 深紫红（接近黑色）
            ]
            # 根据颜色区间生成颜色比例尺
            # 指定要在 colorbar 上显示的数值
            custom_tickvals = [0,  200000,400000,800000,1000000,1500000,2000000]
            custom_ticktext = [str(val) for val in custom_tickvals]  # 或者可以自定义为 ['0', '1K', '8K', ...]

            # 构建颜色比例尺（colorscale）逻辑保持不变
            colorscale = []

            zmin = heatmap_data.min().min()
            zmax = heatmap_data.max().max()

            for i, scale in enumerate(color_scale):
                if 'max' in scale:
                    current_max = scale['max']
                else:
                    current_max = zmax

                ratio_min = (scale.get('min', zmin) - zmin) / (zmax - zmin)
                ratio_max = (current_max - zmin) / (zmax - zmin)

                colorscale.append([ratio_min, scale['color']])
                colorscale.append([ratio_max, scale['color']])

            # 去重并排序（修复关键点）
            unique_colorscale = []
            seen = set()
            for item in colorscale:
                key = (round(item[0], 5), item[1])
                if key not in seen:
                    seen.add(key)
                    unique_colorscale.append(item)
            unique_colorscale.sort(key=lambda x: x[0])

            # 确保首尾为 0.0 和 1.0
            if unique_colorscale:
                if unique_colorscale[0][0] > 0.0:
                    unique_colorscale.insert(0, [0.0, unique_colorscale[0][1]])
                if unique_colorscale[-1][0] < 1.0:
                    unique_colorscale.append([1.0, unique_colorscale[-1][1]])
            else:
                unique_colorscale = [[0.0, "#ffffff"], [1.0, "#000000"]]

            final_colorscale = [[val, color] for val, color in unique_colorscale]

            # 创建热力图
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="年份", y="领域", color="企业数量"),
                x=heatmap_data.columns,
                y=heatmap_data.index
            )
            # 假设我们想在2025年处添加一条竖线
            year_divider = 2025

            # 设置颜色轴属性
            fig.update_layout(
                coloraxis=dict(
                    colorscale=final_colorscale,
                    colorbar=dict(
                        tickvals=custom_tickvals,  # 使用你指定的数值
                        ticktext=custom_ticktext,  # 对应的文字标签
                        title="企业数量",
                        len=1,  # 控制长度（相对高度）
                        thickness=30,  # 控制宽度（像素）
                        xpad=10,
                        ypad=10
                    )
                ),
                title="人工智能企业技术领域热力图",
                xaxis_title="年份",
                yaxis_title="领域",  # 调整图表宽度
                height=600  # 调整图表高度
            )

            # 显示热力图
            st.plotly_chart(fig)


        elif chart_type == "折线图":
            # 折线图数据准备
            line_data = forecast_df.sort_values(by=['领域', '年份'])
            fig = px.line(line_data,
                x='年份',
                y='累计存活企业数量',
                color='领域',
                title="人工智能企业技术领域趋势图",
                labels={'年份': '年份', '累计存活企业数量': '企业数量'})
            # 添加一条竖线：表示真实值与预测值的分界点（2025）
            fig.add_vline(
                x=2025,
                line_width=2,
                line_dash="dash",  # 虚线样式
                line_color="red",
                annotation_text="真实值 / 预测值分界线",  # 可选标注文本
                annotation_position="top right"
                )
            # 调整图表样式
            fig.update_layout(
                xaxis_title="年份",
                yaxis_title="企业数量",
                height=600,
                legend_title_text='领域'
            )
            # 显示折线图
            st.plotly_chart(fig)
else:
    if risk_avert_mode == "企业流失风险城市分析":
        st.markdown('<div class="big-title">企业流失风险城市全景分析</div>', unsafe_allow_html=True)

        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建目标文件的绝对路径
        file_path = os.path.join(current_dir, "..", "地区风险预警", "预测结果", "城市风险评级结果.csv")
        # 读取数据
        df_risk = pd.read_csv(file_path)

        with st.expander("图表控制面板（点击隐藏）", expanded=True):
            selected_metric = st.selectbox(
                "选择显示指标",
                ["风险等级","动态预警"]
            )
        if selected_metric=="风险等级":
            # 风险等级映射为数值（用于颜色渲染）
            risk_level_mapping = {
                "高风险": 3,
                "中风险": 2,
                "低风险": 1
            }

            df_risk["风险等级_数值"] = df_risk["风险等级"].map(risk_level_mapping)

            # 准备数据
            cities = df_risk["城市"].tolist()
            values = df_risk["风险等级_数值"].astype(int).tolist()

            # 设置 visualmap 分段规则（pieces），按你指定的顺序显示图例
            pieces = [
                {"value": "3", "color": "#B40404", "label": "高风险"},
                {"value": "2", "color": "#DF0101", "label": "中风险"},
                {"value": "1", "color": "#F6CECE", "label": "低风险"}
            ]

            # 创建地图
            map_chart = create_map(cities, values, title="城市综合风险等级分布", pieces=pieces)

            # 临时保存并渲染 HTML 到 Streamlit
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                html_file_path = tmp_file.name
            map_chart.render(html_file_path)

            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            try:
                st.components.html(html_content, height=800, scrolling=True)
            except AttributeError:
                st.components.v1.html(html_content, height=800, scrolling=True)
            os.unlink(html_file_path)
        else:
            # 风险等级映射为数值（用于颜色渲染）
            risk_level_mapping = {
                "红色预警": 3,
                "橙色预警": 2,
                "黄色预警": 1,
                "无预警": 0
            }

            df_risk["动态预警_数值"] = df_risk["动态预警"].map(risk_level_mapping)

            # 准备数据
            cities = df_risk["城市"].tolist()
            values = df_risk["动态预警_数值"].astype(int).tolist()

            # 设置 visualmap 分段规则（pieces），按你指定的顺序显示图例
            pieces = [
                {"value": "3", "color": "red", "label": "红色预警：\n注销企业显著增加，\n风险评分达到高风险阈值"},
                {"value": "2", "color": "orange", "label": "橙色预警：\n任意指标存在异常"},
                {"value": "1", "color": "yellow", "label": "黄色预警：\n风险评分恶化\n但评分未突破阈值"},
                {"value": "0", "color": "white", "label": "无预警：\n正常且健康状态"}
            ]

            # 创建地图
            map_chart = create_map(cities, values, title="城市动态预警信息", pieces=pieces)

            # 临时保存并渲染 HTML 到 Streamlit
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
                html_file_path = tmp_file.name
            map_chart.render(html_file_path)

            with open(html_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            try:
                st.components.html(html_content, height=800, scrolling=True)
            except AttributeError:
                st.components.v1.html(html_content, height=800, scrolling=True)
            os.unlink(html_file_path)
    elif risk_avert_mode=="领域风险预测":
        st.markdown('<div class="big-title">关键技术领域风险预测</div>', unsafe_allow_html=True)
        # 加载数据
        short_long_df, short_df, long_df, combined_df = load_data1()

        # === 风险等级对比表格（低饱和度颜色）===

        st.subheader("短期 vs 长期风险等级")
        # 定义低饱和度颜色映射
        risk_color_map = {
            "低风险": "#d4edda",  # 浅蓝绿色
            "中风险": "#fff3cd",  # 米黄色
            "高风险": "#f8d7da"  # 浅粉红色
        }
        # 创建 DataFrame 并应用背景颜色
        risk_level_df = short_long_df[["领域", "短期_风险等级", "长期_风险等级"]]
        # 使用 Styler 添加背景颜色
        def color_risk(val):
            return f"background-color: {risk_color_map.get(val, '#ffffff')}"
        # 直接显示带颜色的表格
        st.dataframe(
            risk_level_df.style.applymap(color_risk),
            use_container_width=True
        )
        st.markdown('<div style="margin: 2rem 0;"></div>', unsafe_allow_html=True)

        # === 风险指标热力图（低饱和度配色）===
        st.subheader("关键风险指标分布（0=未触发，1=触发）")
        # 定义短期和长期指标
        short_metrics = [
            "短期_负增长年数≥50%",
            "短期_连续负增长年数",
            "短期_平均增长率是否为负",
            "短期_下降幅度>30%",
            "短期_波动率>动态波动阈值"
        ]
        long_metrics = [
            "长期_负增长年数≥50%",
            "长期_连续负增长年数",
            "长期_平均增长率是否为负",
            "长期_下降幅度>30%",
            "长期_波动率>动态波动阈值"
        ]
        # 分开选择短期和长期指标
        col1, col2 = st.columns(2)
        with col1:
            selected_short = st.multiselect(
                "选择短期指标",
                options=short_metrics,
                default=["短期_下降幅度>30%", "短期_波动率>动态波动阈值"]
            )
        with col2:
            selected_long = st.multiselect(
                "选择长期指标",
                options=long_metrics,
                default=["长期_下降幅度>30%", "长期_波动率>动态波动阈值"]
            )

        # 合并选中的指标
        selected_metrics = selected_short + selected_long

        if selected_metrics:
            # 将领域设为索引以显示名称而非数字
            data = short_long_df.set_index("领域")[selected_metrics].T

            # 绘制热力图（使用低饱和度颜色渐变）
            fig = px.imshow(
                data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=[
                    "#c3e6cb",  # 浅绿色
                    "#f5c6cb"  # 浅红色
                ],
                title="风险指标触发情况"
            )

            # 显示领域名称而非数字索引
            fig.update_yaxes(tickmode='array', tickvals=data.index, ticktext=data.index)

            fig.update_layout(
                xaxis_title="领域",
                yaxis_title="指标",
                width=800,
                height=400,
                font=dict(color="#333333"),  # 深灰色文字
                title_font=dict(color="#333333")
            )

            st.plotly_chart(fig, use_container_width=True)


# =====================
# 页脚信息
# =====================
st.markdown("---")
st.caption("数据来源：天眼查企业数据 | 分析工具：Python/Streamlit/Plotly | 开发者：大数据综合课程设计-第七组")

