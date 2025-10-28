#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyecharts可视化脚本 - 共享单车数据分析Dashboard
基于Spark分析结果生成交互式可视化页面
"""

import pandas as pd
import json
import os
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie, HeatMap, Grid, Page, Radar, Scatter
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType

# 项目路径
PROJECT_ROOT = os.path.expanduser('~/bike-sharing-analysis')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
WEB_DIR = os.path.join(PROJECT_ROOT, 'web')
os.makedirs(WEB_DIR, exist_ok=True)

print("="*70)
print("Pyecharts可视化 - 共享单车数据分析Dashboard")
print("="*70)

# ==================== 1. 加载分析结果 ====================
print("\n[1/7] 加载分析结果...")

hourly_df = pd.read_csv(os.path.join(RESULTS_DIR, 'hourly_analysis.csv'))
season_df = pd.read_csv(os.path.join(RESULTS_DIR, 'season_analysis.csv'))
weather_df = pd.read_csv(os.path.join(RESULTS_DIR, 'weather_analysis.csv'))
zone_df = pd.read_csv(os.path.join(RESULTS_DIR, 'zone_analysis.csv'))
od_matrix_df = pd.read_csv(os.path.join(RESULTS_DIR, 'od_matrix.csv'))
bike_type_df = pd.read_csv(os.path.join(RESULTS_DIR, 'bike_type_analysis.csv'))

with open(os.path.join(RESULTS_DIR, 'overall_stats.json'), 'r') as f:
    overall_stats = json.load(f)

print("✓ 数据加载完成")

# ==================== 2. 小时趋势图 ====================
print("\n[2/7] 生成小时趋势图...")

def create_hourly_chart():
    hours = hourly_df['hr'].tolist()
    order_counts = hourly_df['order_count'].tolist()
    avg_distances = hourly_df['avg_distance'].tolist()
    avg_fees = hourly_df['avg_fee'].tolist()
    
    line = (
        Line(init_opts=opts.InitOpts(width="1400px", height="500px", theme=ThemeType.LIGHT))
        .add_xaxis([f"{h:02d}:00" for h in hours])
        .add_yaxis(
            "订单量",
            order_counts,
            is_smooth=True,
            symbol_size=8,
            linestyle_opts=opts.LineStyleOpts(width=3, color="#5470c6"),
            itemstyle_opts=opts.ItemStyleOpts(color="#5470c6"),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"),
                    opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average", name="平均值")]
            ),
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="平均距离(km)",
                type_="value",
                min_=2.5,
                max_=3.5,
                position="right",
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="24小时订单量趋势",
                subtitle="识别早晚高峰时段",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_weight="bold")
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            xaxis_opts=opts.AxisOpts(
                name="时段",
                type_="category",
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(
                name="订单量",
                type_="value",
                axislabel_opts=opts.LabelOpts(formatter="{value}")
            ),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
        )
    )
    
    line2 = (
        Line()
        .add_xaxis([f"{h:02d}:00" for h in hours])
        .add_yaxis(
            "平均距离",
            [round(d, 2) for d in avg_distances],
            yaxis_index=1,
            is_smooth=True,
            symbol_size=6,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#91cc75", type_="dashed"),
            itemstyle_opts=opts.ItemStyleOpts(color="#91cc75"),
        )
    )
    
    line.overlap(line2)
    return line

hourly_chart = create_hourly_chart()
print("✓ 小时趋势图生成完成")

# ==================== 3. 季节 & 天气对比图 ====================
print("\n[3/7] 生成季节和天气对比图...")

def create_season_weather_chart():
    # 季节柱状图
    season_bar = (
        Bar(init_opts=opts.InitOpts(width="580px", height="400px"))
        .add_xaxis(season_df['season_name'].tolist())
        .add_yaxis(
            "订单量",
            season_df['order_count'].tolist(),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode("""
                    new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {offset: 0, color: '#83bff6'},
                        {offset: 0.5, color: '#188df0'},
                        {offset: 1, color: '#188df0'}
                    ])
                """)
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="季节需求分布", pos_left="center"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),
            yaxis_opts=opts.AxisOpts(name="订单量"),
        )
    )
    
    # 天气饼图
    weather_pie = (
        Pie(init_opts=opts.InitOpts(width="580px", height="400px"))
        .add(
            "",
            [list(z) for z in zip(weather_df['weather_type'].tolist(), 
                                   weather_df['order_count'].tolist())],
            radius=["30%", "75%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(formatter="{b}: {c}\n({d}%)"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="天气影响分布", pos_left="center"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
    )
    
    # 合并到Grid
    grid = (
        Grid(init_opts=opts.InitOpts(width="1200px", height="450px"))
        .add(season_bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%", pos_bottom="15%"))
        .add(weather_pie, grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%"))
    )
    
    return grid

season_weather_chart = create_season_weather_chart()
print("✓ 季节和天气对比图生成完成")

# ==================== 4. 区域热力图 ====================
print("\n[4/7] 生成区域热力图...")

def create_zone_heatmap():
    # 准备OD矩阵数据
    zones = sorted(zone_df['start_zone'].unique())
    
    # 创建完整的OD矩阵
    od_pivot = od_matrix_df.pivot_table(
        index='start_zone', 
        columns='end_zone', 
        values='flow_count', 
        fill_value=0
    )
    
    # 确保所有区域都存在
    for zone in zones:
        if zone not in od_pivot.index:
            od_pivot.loc[zone] = 0
        if zone not in od_pivot.columns:
            od_pivot[zone] = 0
    
    od_pivot = od_pivot.loc[zones, zones]
    
    # 转换为Pyecharts需要的格式
    data = []
    for i, start_zone in enumerate(zones):
        for j, end_zone in enumerate(zones):
            value = int(od_pivot.loc[start_zone, end_zone])
            data.append([j, i, value])
    
    heatmap = (
        HeatMap(init_opts=opts.InitOpts(width="1200px", height="500px"))
        .add_xaxis([z.replace('_', ' ') for z in zones])
        .add_yaxis(
            "起点区域",
            [z.replace('_', ' ') for z in zones],
            data,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="起点-终点流量热力图 (OD Matrix)",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=18)
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=int(od_pivot.max().max()),
                is_calculable=True,
                orient="horizontal",
                pos_left="center",
                pos_bottom="5%",
            ),
            xaxis_opts=opts.AxisOpts(
                name="终点区域",
                axislabel_opts=opts.LabelOpts(rotate=45)
            ),
            yaxis_opts=opts.AxisOpts(name="起点区域"),
        )
    )
    
    return heatmap

zone_heatmap = create_zone_heatmap()
print("✓ 区域热力图生成完成")

# ==================== 5. 区域热度排名 ====================
print("\n[5/7] 生成区域热度排名...")

def create_zone_ranking():
    zones = [z.replace('_', ' ') for z in zone_df['start_zone'].tolist()]
    counts = zone_df['order_count'].tolist()
    
    bar = (
        Bar(init_opts=opts.InitOpts(width="1200px", height="450px"))
        .add_xaxis(zones)
        .add_yaxis(
            "订单量",
            counts,
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode("""
                    new echarts.graphic.LinearGradient(0, 1, 0, 0, [
                        {offset: 0, color: '#2378f7'},
                        {offset: 0.7, color: '#83bff6'},
                        {offset: 1, color: '#83bff6'}
                    ])
                """)
            ),
            label_opts=opts.LabelOpts(position="top", formatter="{c}"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="区域热度排名",
                subtitle="各区域订单起点统计",
                pos_left="center"
            ),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(rotate=30)
            ),
            yaxis_opts=opts.AxisOpts(name="订单量"),
        )
    )
    
    return bar

zone_ranking = create_zone_ranking()
print("✓ 区域热度排名生成完成")

# ==================== 6. 单车类型对比雷达图 ====================
print("\n[6/7] 生成单车类型对比图...")

def create_bike_comparison():
    # 准备雷达图数据
    schema = [
        ("订单量", 70000),
        ("平均时长(秒)", 1000),
        ("平均距离(km)", 4),
        ("平均费用($)", 5),
    ]
    
    data = []
    for _, row in bike_type_df.iterrows():
        data.append(
            [
                row['order_count'],
                row['avg_duration'],
                row['avg_distance'],
                row['avg_fee']
            ]
        )
    
    radar = (
        Radar(init_opts=opts.InitOpts(width="1200px", height="500px"))
        .add_schema(
            schema=schema,
            shape="circle",
        )
        .add(
            "普通车",
            [data[0]],
            linestyle_opts=opts.LineStyleOpts(color="#5470c6", width=2),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#5470c6"),
        )
        .add(
            "助力车",
            [data[1]],
            linestyle_opts=opts.LineStyleOpts(color="#91cc75", width=2),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.3, color="#91cc75"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="单车类型对比分析",
                pos_left="center"
            ),
            legend_opts=opts.LegendOpts(pos_top="8%"),
        )
    )
    
    return radar

bike_comparison = create_bike_comparison()
print("✓ 单车类型对比图生成完成")

# ==================== 7. 整体统计卡片 ====================
print("\n[7/7] 生成整体统计信息...")

def create_stats_html():
    html = f"""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 32px;">
            🚴 共享单车数据分析Dashboard
        </h1>
        <p style="color: white; text-align: center; margin-top: 10px; font-size: 16px;">
            基于华盛顿特区Capital Bikeshare 100,000条订单数据
        </p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 10px;">总订单数</div>
            <div style="color: white; font-size: 36px; font-weight: bold;">{overall_stats['total_orders']:,}</div>
        </div>
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 10px;">总收入</div>
            <div style="color: white; font-size: 36px; font-weight: bold;">${overall_stats['total_revenue']:,.0f}</div>
        </div>
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 10px;">平均骑行时长</div>
            <div style="color: white; font-size: 36px; font-weight: bold;">{overall_stats['avg_duration_minutes']:.1f} min</div>
        </div>
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 10px;">平均骑行距离</div>
            <div style="color: white; font-size: 36px; font-weight: bold;">{overall_stats['avg_distance_km']:.2f} km</div>
        </div>
    </div>
    """
    return html

stats_html = create_stats_html()

# ==================== 8. 组合成完整页面 ====================
print("\n[8/8] 生成完整Dashboard...")

page = Page(layout=Page.SimplePageLayout)
page.page_title = "共享单车数据分析Dashboard"

# 添加所有图表
page.add(
    hourly_chart,
    season_weather_chart,
    zone_ranking,
    zone_heatmap,
    bike_comparison,
)

# 保存页面
dashboard_path = os.path.join(WEB_DIR, 'analysis_dashboard.html')
page.render(dashboard_path)

# 在HTML头部插入统计卡片
with open(dashboard_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 在body标签后插入统计卡片
content = content.replace('<body>', f'<body>\n{stats_html}')

with open(dashboard_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Dashboard已保存: {dashboard_path}")

print("\n" + "="*70)
print("✅ 可视化Dashboard生成完成！")
print("="*70)
print(f"\n📊 在浏览器中查看:")
print(f"  file://{dashboard_path}")
print(f"\n或在WSL中运行:")
print(f"  explorer.exe {dashboard_path}")
print("\n包含的可视化:")
print("  1. 24小时订单量趋势图")
print("  2. 季节需求分布 + 天气影响")
print("  3. 区域热度排名")
print("  4. 起点-终点流量热力图")
print("  5. 单车类型对比雷达图")
print("  6. 整体统计卡片")
print("="*70)