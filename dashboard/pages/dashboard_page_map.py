"""
地图可视化页面 - 共享单车区域状态实时监控与调度演示

页面功能：
- 显示区域当前车辆与需求
- 通过模拟步展示调度效果（包含自动/手动运行）
- 支持查看每个区域的详细信息与最近操作
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

st.set_page_config(page_title="地图可视化", page_icon="🗺️", layout="wide")

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
    """创建地图"""
    # 以华盛顿特区为中心
    m = folium.Map(
        location=[38.9072, -77.0369],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
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
        
        # 添加圆形标记
        folium.CircleMarker(
            location=info['coords'],
            radius=15 + bikes / 10,  # 大小反映车辆数量
            popup=folium.Popup(popup_html, max_width=250),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
        
    # 添加文字标签（显示区域代码与车辆数）
        folium.Marker(
            location=info['coords'],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 12pt; color: black; font-weight: bold;">{region_id}: {bikes}辆</div>'
            )
        ).add_to(m)
    
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
    st.title("🗺️ 共享单车区域状态监控")
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
        st.markdown("### 📍 区域分布地图")
        # 创建并显示地图（如果没有 streamlit_folium，则回退使用 components.html 渲染）
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
        2. **运行模拟**: 点击"▶️ 运行一步"按钮，系统会：
           - 生成随机需求
           - PPO策略执行调度
           - 更新区域状态
        3. **查看详情**: 点击地图上的标记查看区域详细信息
        4. **自动运行**: 勾选"自动运行"可以持续模拟
        5. **重置**: 点击"🔄 重置"恢复初始状态
        
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