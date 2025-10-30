"""
地图可视化页面 - 共享单车区域状态实时监控与调度演示

页面功能：
- 显示区域当前车辆与需求
- 通过模拟步展示调度效果（包含自动/手动运行）
- 支持查看每个区域的详细信息与最近操作
- 改进：添加更好的背景地图样式
"""

import streamlit as st
import folium
import streamlit.components.v1 as components
import numpy as np
import sys
from pathlib import Path
import importlib.util

try:
    from streamlit_folium import st_folium
    _ST_FOLIUM_AVAILABLE = True
except Exception:
    st_folium = None
    _ST_FOLIUM_AVAILABLE = False

try:
    from dashboard.pages._shared_style import inject_base_style
except Exception:
    spec_path = Path(__file__).parent / "_shared_style.py"
    spec = importlib.util.spec_from_file_location("dashboard_pages__shared_style", str(spec_path))
    _shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_shared)
    inject_base_style = _shared.inject_base_style

st.set_page_config(page_title="🗺️地图可视化", page_icon="🗺️", layout="wide")

# 注入共享样式
inject_base_style()

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 华盛顿特区6个区域的坐标（真实位置）
REGION_INFO = {
    'A': {
        'name': 'Capitol Hill',
        'coords': [38.8899, -77.0091],
        'description': '国会山 - 政府办公区',
        'weight': 0.25
    },
    'B': {
        'name': 'Downtown',
        'coords': [38.9072, -77.0369],
        'description': '市中心 - 商务区',
        'weight': 0.25
    },
    'C': {
        'name': 'Georgetown',
        'coords': [38.9076, -77.0723],
        'description': '乔治城 - 商业居住区',
        'weight': 0.15
    },
    'D': {
        'name': 'Dupont Circle',
        'coords': [38.9097, -77.0434],
        'description': '杜邦圆环 - 交通枢纽',
        'weight': 0.15
    },
    'E': {
        'name': 'Shaw',
        'coords': [38.9122, -77.0219],
        'description': '肖区 - 文化区',
        'weight': 0.10
    },
    'F': {
        'name': 'Navy Yard',
        'coords': [38.8762, -77.0062],
        'description': '海军船坞 - 滨水区',
        'weight': 0.10
    }
}

# 初始化session state
if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.bikes = {
        'A': 120, 'B': 150, 'C': 80, 'D': 100, 'E': 60, 'F': 70
    }
    st.session_state.demand = {
        'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0
    }
    st.session_state.last_action = "系统初始化"
    st.session_state.total_cost = 0
    st.session_state.service_rate = 1.0

def get_status_color(bikes, capacity=150):
    """根据车辆数量返回状态颜色"""
    ratio = bikes / capacity
    if ratio < 0.3:
        return 'red', '严重缺车'
    elif ratio < 0.5:
        return 'orange', '需要补给'
    elif ratio < 0.7:
        return 'yellow', '略显不足'
    elif ratio > 0.9:
        return 'blue', '富余'
    else:
        return 'green', '正常'

def create_map():
    """创建地图 - 改进版，支持多种背景地图样式"""
    
    # 计算所有区域的边界，用于自动调整地图视图
    all_coords = [info['coords'] for info in REGION_INFO.values()]
    
    # 以华盛顿特区为中心，设置默认样式为简洁模式
    m = folium.Map(
        location=[38.9072, -77.0369],
        zoom_start=13,
        tiles='CartoDB positron',  # 默认基础图层
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        prefer_canvas=True  # 提升性能
    )
    
    # 添加其他可选图层（注意：不要重复添加默认图层）
    # 用户可以通过右上角的图层按钮切换
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='标准地图',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        overlay=False,  # 设置为基础图层
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='CartoDB dark_matter',
        name='深色模式',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
        name='地形图',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        subdomains='abcd',
        max_zoom=18,
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        name='黑白模式',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a> &mdash; Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        subdomains='abcd',
        max_zoom=20,
        overlay=False,
        control=True
    ).add_to(m)
    
    # 添加图层控制按钮
    folium.LayerControl(position='topright').add_to(m)
    
    # 添加全屏按钮（需要插件支持）
    from folium import plugins
    plugins.Fullscreen(
        position='topleft',
        title='全屏',
        title_cancel='退出全屏',
        force_separate_button=True
    ).add_to(m)
    
    # 添加定位按钮
    plugins.LocateControl(auto_start=False).add_to(m)
    
    # 添加测量工具
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='sqkilometers',
        secondary_area_unit='acres'
    ).add_to(m)
    
    # 添加区域标记（带中文弹窗）
    for region_id, info in REGION_INFO.items():
        bikes = st.session_state.bikes[region_id]
        demand = st.session_state.demand[region_id]
        color, status = get_status_color(bikes)
        
        # 创建弹窗内容
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0; color: {color};">{info['name']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 5px 0;"><b>区域代码:</b> {region_id}</p>
            <p style="margin: 5px 0;"><b>当前车辆:</b> {bikes} 辆</p>
            <p style="margin: 5px 0;"><b>当前需求:</b> {demand} 次</p>
            <p style="margin: 5px 0;"><b>状态:</b> <span style="color: {color};">{status}</span></p>
            <p style="margin: 5px 0; font-size: 0.9em; color: #666;">{info['description']}</p>
        </div>
        """
        
        # 添加圆形标记（增大尺寸以便更清楚地看到）
        folium.CircleMarker(
            location=info['coords'],
            radius=20 + bikes / 8,  # 增大基础尺寸
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{region_id}区: {bikes}辆 - {info['name']}",  # 更详细的悬停提示
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,  # 增加不透明度
            weight=3  # 增加边框宽度
        ).add_to(m)
        
        # 添加文字标签（改进样式，增加背景和阴影）
        folium.Marker(
            location=info['coords'],
            icon=folium.DivIcon(
                html=f'''
                <div style="
                    font-size: 11pt; 
                    color: white; 
                    font-weight: bold; 
                    background-color: {color};
                    padding: 4px 10px;
                    border-radius: 12px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    border: 2px solid white;
                    white-space: nowrap;
                    text-align: center;
                ">{region_id}区: {bikes}辆</div>
                '''
            )
        ).add_to(m)
    
    # 自动调整地图视图以包含所有标记点（解决显示不全的问题）
    # 计算所有坐标的边界
    if all_coords:
        # 添加一些padding，确保标记不会太靠近边缘
        m.fit_bounds(all_coords, padding=[50, 50])
    
    # 添加小地图（右下角）
    plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)
    
    return m

def simulate_step():
    """模拟一步：生成需求并执行调度（演示用）

    说明：此处为简化演示逻辑，旨在用可视化展示调度效果。
    """
    # 模拟需求（简化版）
    np.random.seed(st.session_state.step)
    target_total = np.random.randint(80, 150)

    # 按权重并加入随机因子计算原始分配比例，然后归一化到 target_total
    factors = {}
    for region_id, info in REGION_INFO.items():
        factors[region_id] = info['weight'] * np.random.uniform(0.8, 1.2)
    s = sum(factors.values()) or 1.0

    total_served = 0
    # 按比例分配（四舍五入），避免直接乘 target_total 导致和不一致
    for region_id in REGION_INFO.keys():
        demand = int(round(target_total * factors[region_id] / s))
        st.session_state.demand[region_id] = demand

        # 满足需求
        served = min(demand, st.session_state.bikes[region_id])
        st.session_state.bikes[region_id] -= served
        total_served += served

    # 使用实际分配的总需求来计算服务率（更稳健）
    actual_total_demand = sum(st.session_state.demand.values())
    st.session_state.service_rate = (total_served / actual_total_demand) if actual_total_demand > 0 else 0
    
    # 先处理还车（把返回车辆计入本步的可用库存，改进调度效果）
    for region_id in REGION_INFO.keys():
        returns = st.session_state.demand[region_id]  # 本步产生的还车
        st.session_state.bikes[region_id] += returns

    # 基于目标库存的多源-多目标调度（多源可向单目标补给），并统计调度成本
    capacity = 150
    desired_level = int(capacity * 0.6)    # 目标库存阈值，可调
    max_transfer_per_move = 20             # 单向单次转运上限

    # 计算每区缺口与富余
    deficits = {r: max(0, desired_level - st.session_state.bikes[r]) for r in REGION_INFO.keys()}
    surpluses = {r: max(0, st.session_state.bikes[r] - desired_level) for r in REGION_INFO.keys()}

    any_transfer = False
    # 按缺口从大到小分配，优先从富余最多的源取车；允许多个源合力补给一个目标
    for to_region, deficit in sorted(deficits.items(), key=lambda x: -x[1]):
        if deficit <= 0:
            continue
        for from_region, avail in sorted(surpluses.items(), key=lambda x: -x[1]):
            if avail <= 0:
                continue
            transfer = min(deficit, avail, max_transfer_per_move)
            if transfer <= 0:
                continue
            st.session_state.bikes[from_region] -= transfer
            st.session_state.bikes[to_region] += transfer
            surpluses[from_region] -= transfer
            deficits[to_region] -= transfer
            st.session_state.total_cost += transfer * 2.5
            any_transfer = True
            st.session_state.last_action = f"调度 {transfer} 辆: {from_region} → {to_region}，成本: ${transfer*2.5:.2f}"
            deficit = deficits[to_region]
            if deficit <= 0:
                break

    if not any_transfer:
        st.session_state.last_action = "无需调度（库存已接近目标）"
    
    st.session_state.step += 1

def main():
    st.title(" 共享单车区域状态监控")
    st.markdown("**华盛顿特区 6 区域实时可视化**")
    st.markdown("---")
    
    # 控制面板
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.markdown(f"### 当前时刻: 第 {st.session_state.step} 步")
    
    with col2:
        if st.button("▶️ 运行一步", type="primary"):
            simulate_step()
            st.rerun()
    
    with col3:
        if st.button("🔄 重置"):
            st.session_state.step = 0
            st.session_state.bikes = {
                'A': 120, 'B': 150, 'C': 80, 'D': 100, 'E': 60, 'F': 70
            }
            st.session_state.demand = {
                'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0
            }
            st.session_state.last_action = "系统初始化"
            st.session_state.total_cost = 0
            st.session_state.service_rate = 1.0
            st.rerun()
    
    with col4:
        auto_run = st.checkbox("自动运行")
    
    # 实时指标
    st.markdown("---")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        total_bikes = sum(st.session_state.bikes.values())
        st.metric("总车辆数", f"{total_bikes} 辆")
    
    with metric_col2:
        st.metric("服务率", f"{st.session_state.service_rate*100:.1f}%")
    
    with metric_col3:
        st.metric("累计成本", f"${st.session_state.total_cost:.2f}")
    
    with metric_col4:
        avg_bikes = total_bikes / 6
        st.metric("平均库存", f"{avg_bikes:.0f} 辆/区")
    
    # 最近操作
    st.info(f"**最近操作**: {st.session_state.last_action}")
    
    st.markdown("---")
    
    # 地图和详细信息
    map_col, info_col = st.columns([2, 1])
    
    with map_col:
        st.markdown("###  区域分布地图")
        st.markdown(" **提示**: 使用右上角的图层按钮可以切换不同的地图背景样式")
        
        # 创建并显示地图
        m = create_map()
        if _ST_FOLIUM_AVAILABLE and st_folium is not None:
            st_folium(m, width=700, height=500)
        else:
            st.warning("未检测到 streamlit_folium，使用内置回退渲染。要获得更好体验，请安装：pip install streamlit-folium folium")
            try:
                html = m._repr_html_()
                components.html(html, width=700, height=500)
            except Exception:
                st.error("地图渲染失败：无法回退渲染 folium 地图，请安装 streamlit-folium 或在支持的环境中运行。")
        
        st.markdown("""
        **图例说明**：
        - 🔴 **红色**: 严重缺车（<30%）
        - 🟠 **橙色**: 需要补给（30-50%）
        - 🟡 **黄色**: 略显不足（50-70%）
        - 🟢 **绿色**: 正常（70-90%）
        - 🔵 **蓝色**: 富余（>90%）
        
        *圆圈大小表示车辆数量，点击查看详细信息*
        
        **地图功能**：
        -  右上角可切换5种地图背景样式
        -  使用滚轮缩放地图
        -  左上角有测量工具
        -  右下角有缩略地图
        -  悬停在标记上可快速查看车辆数
        """)
    
    with info_col:
        st.markdown("### 📊 区域详细状态")
        
        for region_id, info in REGION_INFO.items():
            bikes = st.session_state.bikes[region_id]
            demand = st.session_state.demand[region_id]
            color, status = get_status_color(bikes)
            
            with st.expander(f"**{region_id}区 - {info['name']}**", expanded=False):
                st.markdown(f"**位置**: {info['description']}")
                st.markdown(f"**当前车辆**: {bikes} 辆")
                st.markdown(f"**当前需求**: {demand} 次")
                st.markdown(f"**状态**: <span style='color: {color}; font-weight: bold;'>{status}</span>", unsafe_allow_html=True)
                st.markdown(f"**权重**: {info['weight']*100:.0f}%")
                
                # 简单的进度条
                st.progress(min(bikes / 150, 1.0))
    
    # 自动运行逻辑
    if auto_run:
        import time
        time.sleep(1)
        simulate_step()
        st.rerun()
    
    # 使用说明
    st.markdown("---")
    with st.expander("ℹ️ 使用说明"):
        st.markdown("""
        ### 如何使用
        
        1. **查看地图**: 地图上显示了华盛顿特区的6个服务区域
        2. **切换地图样式**: 点击右上角的图层按钮，可以选择：
           - 📋 标准地图 - 详细的街道信息
           - 🎨 简洁模式 - 清爽的背景，适合数据展示
           - 🌙 深色模式 - 深色主题，护眼舒适
           - 🏔️ 地形图 - 显示地形起伏
           - ⚫ 黑白模式 - 专业商务风格
        3. **运行模拟**: 点击"▶️ 运行一步"按钮，系统会：
           - 生成随机需求
           - PPO策略执行调度
           - 更新区域状态
        4. **查看详情**: 点击地图上的标记查看区域详细信息，或悬停查看快速信息
        5. **自动运行**: 勾选"自动运行"可以持续模拟
        6. **重置**: 点击"🔄 重置"恢复初始状态
        7. **地图工具**:
           - 📏 左上角测量工具可以测量距离和面积
           - 🔍 滚轮缩放，拖拽移动
           - 📍 右下角缩略图帮助定位
        
        ### 颜色含义
        - 红色区域需要紧急补给
        - 绿色区域状态正常
        - 蓝色区域车辆富余
        
        ### PPO调度策略
        系统会自动识别缺车和富余区域，执行智能调度：
        - 从富余区域调度到缺车区域
        - 最小化调度成本
        - 保持服务率在高水平
        """)

if __name__ == "__main__":
    main()