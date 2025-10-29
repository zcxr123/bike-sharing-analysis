"""
共享单车智能调度系统 - Dashboard主页
Day 10 - 项目概览
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="共享单车RL调度系统",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .insight-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# 侧边栏（中文化）
st.sidebar.title("🚲 导航")
st.sidebar.markdown("---")
st.sidebar.info("""
**共享单车智能调度系统（简述）**

基于强化学习（PPO）的成本优化与智能调度平台。

核心成果示例：
- 成本降低：约 76%
- ROI 提升：约 4.3 倍
- 年度效益（示例）：约 $283K
- 服务率：接近 98%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 使用与页面导航

左侧为页面导航（点击进入对应页面）：
- 📊 项目概览
- 🗺️ 地图可视化
- 📈 策略对比
- 🔍 决策分析
- 💰 ROI 计算器

快速链接：
- GitHub 仓库（需联网）
- 技术报告（本地/远程）
""")

# 加载数据
@st.cache_data
def load_summary():
    """加载汇总数据"""
    try:
        with open('data/summary.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return {}

@st.cache_data
def load_comparison():
    """加载对比数据"""
    try:
        return pd.read_csv('data/comparison.csv')
    except:
        return pd.DataFrame()

# 主页内容
def main():
    # 标题
    st.markdown('<div class="main-header">基于Spark的共享单车数据分析平台与智能调度系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于强化学习的成本优化方案</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 加载数据
    summary = load_summary()
    core_metrics = summary.get('core_metrics', {})
    
    # 核心成果指标
    st.markdown("### 🎯 核心成果")
    st.markdown("")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cost_reduction = core_metrics.get('cost_reduction_pct', 76.0)
        cost_saving = core_metrics.get('cost_reduction_abs', 1652)
        st.metric(
            "成本降低",
            f"{cost_reduction:.1f}%",
            delta=f"-${cost_saving:.0f}/周",
            delta_color="normal"
        )
    
    with col2:
        roi_improvement = core_metrics.get('roi_improvement', 4.3)
        roi_day8 = core_metrics.get('roi_day8', 244.2)
        st.metric(
            "ROI提升",
            f"{roi_improvement:.1f}倍",
            delta=f"达到{roi_day8:.1f}",
            delta_color="normal"
        )
    
    with col3:
        annual_benefit = core_metrics.get('annual_benefit', 283660)
        st.metric(
            "年度效益",
            f"${annual_benefit/1000:.0f}K",
            delta="单城市",
            delta_color="normal"
        )
    
    with col4:
        service_rate = core_metrics.get('service_rate_day8', 98.12)
        st.metric(
            "服务率",
            f"{service_rate:.1f}%",
            delta="最优平衡",
            delta_color="normal"
        )
    
    st.markdown("")
    st.markdown("---")
    
    # 项目进度
    st.markdown("### 📅 项目进度")
    st.markdown("")
    
    progress_col1, progress_col2 = st.columns([3, 1])
    
    with progress_col1:
        # 进度条
        milestones = [
            ("M1: 数据与分析", 100),
            ("M2: 调度模拟器", 100),
            ("M3: RL训练", 100),
            ("M4: 项目收尾", 50)
        ]
        
        for milestone, progress in milestones:
            st.progress(progress / 100, text=f"{milestone} - {progress}%")
    
    with progress_col2:
        st.metric("总体进度", "88%", delta="Day 10/12")
    
    st.markdown("")
    st.markdown("---")
    
    # 关键洞察
    st.markdown("### 🧠 核心发现")
    st.markdown("")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        with st.expander("**💡 高频低成本策略**", expanded=True):
            st.markdown("""
            PPO发现了一个反直觉的创新策略：
            
            - **调度频率**: 基线的18倍
            - **总成本**: 仅基线的14%
            - **单次成本**: $0.08（极低）
            
            **为什么？**
            - 智能路径选择，避开高成本路径
            - 小额度高频次，灵活响应需求
            - 提前布局，避免紧急调度
            
            这是AI自己学会的策略！
            """)
        
        with st.expander("**🎯 98%的经济学智慧**"):
            st.markdown("""
            PPO没有追求100%服务率，而是自动停在98%：
            
            - **原因**: 最后2%需要4倍成本
            - **结果**: 边际收益递减
            - **智慧**: 自动找到最优平衡点
            
            这体现了AI的经济学智慧！
            """)
    
    with insight_col2:
        with st.expander("**⏰ 预测性调度策略**", expanded=True):
            st.markdown("""
            PPO学会了在需求高峰**前**调度：
            
            - **高峰时段**: 15-17点、22-23点
            - **低谷时段**: 0-4点（减少调度）
            - **策略**: 未雨绸缪，提前布局
            
            不是被动响应，而是主动预防！
            """)
        
        with st.expander("**📈 规模效应体现**"):
            st.markdown("""
            高需求期成本反而更低：
            
            - **高需求期**: $1.62/步
            - **低需求期**: $1.72/步
            - **原因**: 批量服务摊薄成本
            
            这是经济学的规模效应！
            """)
    
    st.markdown("")
    st.markdown("---")
    
    # 技术路线
    st.markdown("### 🛠️ 技术架构")
    st.markdown("")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **🧠 核心算法**
        - Proximal Policy Optimization (PPO)
        - 成本感知奖励函数
        - 超参数调优
        """)
    
    with tech_col2:
        st.markdown("""
        **🔧 技术栈**
        - Stable-Baselines3
        - OpenAI Gym
        - Python 3.10+
        - Streamlit Dashboard
        """)
    
    with tech_col3:
        st.markdown("""
        **📊 评估方法**
        - 多场景测试
        - 基线对比
        - 决策可解释性分析
        - ROI量化评估
        """)
    
    st.markdown("")
    st.markdown("---")
    
    # 快速导航
    st.markdown("### 🚀 快速导航")
    st.markdown("")
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        st.success("""
        **🗺️ 地图可视化**
        
        实时监控6个区域状态，运行调度模拟演示
        
        👉 点击左侧导航栏
        """)
    
    with nav_col2:
        st.info("""
        **📈 策略对比**
        
        查看Day 7、Day 8和基线策略的详细对比分析
        
        👉 点击左侧导航栏
        """)
    
    with nav_col3:
        st.info("""
        **🔍 决策分析**
        
        深入理解PPO的决策机制和高频低成本策略
        
        👉 点击左侧导航栏
        """)
    
    with nav_col4:
        st.info("""
        **💰 ROI计算器**
        
        计算不同规模下的经济效益和投资回报
        
        👉 点击左侧导航栏
        """)
    
    st.markdown("")
    st.markdown("---")
    
    # 数据预览
    if not summary:
        st.warning("⚠️ 未找到汇总数据，请先运行 `python3 scripts/day10_prepare_data.py`")
    else:
        with st.expander("📊 数据统计概览"):
            df = load_comparison()
            if not df.empty:
                st.markdown(f"**策略数**: {df['model'].nunique()}")
                st.markdown(f"**场景数**: {df['scenario'].nunique()}")
                st.markdown(f"**总评估轮数**: {len(df)}")
                
                st.markdown("**策略列表**:")
                for model in df['model'].unique():
                    st.markdown(f"  - {model}")
    
    # 页脚
    st.markdown("")
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>共享单车智能调度系统 v1.0 | 基于强化学习的成本优化方案</p>
        <p>© 2025 | Day 10 Dashboard</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()