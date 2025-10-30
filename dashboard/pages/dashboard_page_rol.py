"""
ROI计算器页面 - 经济效益评估
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
try:
    from dashboard.pages._shared_style import inject_base_style
except Exception:
    import importlib.util
    from pathlib import Path
    spec_path = Path(__file__).parent / "_shared_style.py"
    spec = importlib.util.spec_from_file_location("dashboard_pages__shared_style", str(spec_path))
    _shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_shared)
    inject_base_style = _shared.inject_base_style

st.set_page_config(page_title=" ROI计算器", layout="wide")

# 注入共享样式
inject_base_style()

# 支持从 dashboard 根 data 目录加载（更稳健）
DATA_DIR = Path(__file__).parent.parent / "data"

def calculate_roi(n_cities, weekly_demand, cost_reduction_rate=0.76, profit_increase_rate=0.03):
    """
    计算ROI
    
    基于Day 8的实际数据：
    - 成本降低: 76% ($2,172 → $520，节省$1,652)
    - 利润提升: 约3% ($123,197 → $127,045，增加$3,848)
    """
    # 基础数据（单城市周数据）
    base_cost_saving = 1652  # Day 7 → Day 8成本节省
    base_profit_increase = 3848  # Day 7 → Day 8利润增加
    
    # 根据需求量调整（简化线性模型）
    demand_factor = weekly_demand / 2000  # 基准需求2000
    
    # 计算效益
    weekly_cost_saving = base_cost_saving * n_cities * demand_factor
    weekly_profit_increase = base_profit_increase * n_cities * demand_factor
    weekly_total_benefit = weekly_cost_saving + weekly_profit_increase
    
    # 年度和多年效益
    annual_benefit = weekly_total_benefit * 52
    three_year_benefit = annual_benefit * 3
    five_year_benefit = annual_benefit * 5
    
    # ROI计算（假设实施成本）
    implementation_cost = n_cities * 50000  # 假设每城市5万实施成本
    roi = (annual_benefit - implementation_cost) / implementation_cost if implementation_cost > 0 else 0
    payback_period = implementation_cost / weekly_total_benefit if weekly_total_benefit > 0 else 0
    
    return {
        'weekly_cost_saving': weekly_cost_saving,
        'weekly_profit_increase': weekly_profit_increase,
        'weekly_total_benefit': weekly_total_benefit,
        'annual_benefit': annual_benefit,
        'three_year_benefit': three_year_benefit,
        'five_year_benefit': five_year_benefit,
        'implementation_cost': implementation_cost,
        'roi': roi,
        'payback_period_weeks': payback_period
    }


def main():
    st.title(" ROI计算器")
    st.markdown("**经济效益评估工具**")
    st.markdown("---")
    
    # 参数设置
    st.markdown("###  参数设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_cities = st.slider(
            "城市数量",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            help="部署该系统的城市数量"
        )
    
    with col2:
        weekly_demand = st.slider(
            "周需求量（单城市）",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100,
            help="每个城市的周需求量"
        )
    
    # 计算
    results = calculate_roi(n_cities, weekly_demand)
    
    st.markdown("---")
    
    # 核心结果展示
    st.markdown("###  经济效益分析")
    st.markdown("")
    
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        st.metric(
            "周效益",
            f"${results['weekly_total_benefit']:,.0f}",
            delta=f"{n_cities}个城市"
        )
    
    with result_col2:
        st.metric(
            "年效益",
            f"${results['annual_benefit']/1000:.0f}K",
            delta="52周计算"
        )
    
    with result_col3:
        st.metric(
            "5年效益",
            f"${results['five_year_benefit']/1000000:.1f}M",
            delta="长期价值"
        )
    
    with result_col4:
        st.metric(
            "投资回报率",
            f"{results['roi']*100:.0f}%",
            delta=f"{results['payback_period_weeks']:.1f}周回本"
        )
    
    st.markdown("")
    st.markdown("---")
    
    # 详细拆解
    st.markdown("###  效益拆解")
    
    detail_col1, detail_col2 = st.columns(2)
    
    with detail_col1:
        st.markdown("#### 成本节省")
        st.info(f"""
        **周成本节省**: ${results['weekly_cost_saving']:,.0f}
        - 单城市: ${results['weekly_cost_saving']/n_cities:,.0f}
        - 来源: v1.0 → v2.0 成本降低76%
        
        **年成本节省**: ${results['weekly_cost_saving']*52:,.0f}
        """)
        
        st.markdown("#### 利润增加")
        st.success(f"""
        **周利润增加**: ${results['weekly_profit_increase']:,.0f}
        - 单城市: ${results['weekly_profit_increase']/n_cities:,.0f}
        - 来源: 更高效的服务带来更多收益
        
        **年利润增加**: ${results['weekly_profit_increase']*52:,.0f}
        """)
    
    with detail_col2:
        # 饼图 - 效益构成
        fig = go.Figure(data=[go.Pie(
            labels=['成本节省', '利润增加'],
            values=[results['weekly_cost_saving'], results['weekly_profit_increase']],
            hole=.3
        )])
        fig.update_layout(
            title="周效益构成",
            height=300
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # 敏感性分析
    st.markdown("###  敏感性分析")
    
    tab1, tab2 = st.tabs(["城市数量影响", "需求量影响"])
    
    with tab1:
        # 城市数量敏感性
        cities_range = range(1, 101, 5)
        benefits = [calculate_roi(c, weekly_demand)['annual_benefit'] for c in cities_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(cities_range),
            y=benefits,
            mode='lines+markers',
            name='年度效益',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # 添加当前位置标记
        current_benefit = results['annual_benefit']
        fig.add_trace(go.Scatter(
            x=[n_cities],
            y=[current_benefit],
            mode='markers',
            name='当前配置',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title='城市数量 vs 年度效益',
            xaxis_title='城市数量',
            yaxis_title='年度效益 ($)',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
        
        # 规模化潜力
        st.info(f"""
        **规模化潜力分析**：
        - 10个城市: ${calculate_roi(10, weekly_demand)['annual_benefit']:,.0f}
        - 25个城市: ${calculate_roi(25, weekly_demand)['annual_benefit']:,.0f}
        - 50个城市: ${calculate_roi(50, weekly_demand)['annual_benefit']:,.0f}
        - 100个城市: ${calculate_roi(100, weekly_demand)['annual_benefit']:,.0f}
        
         **洞察**: 效益随城市数量**线性增长**，规模化优势明显
        """)
    
    with tab2:
        # 需求量敏感性
        demand_range = range(500, 5001, 250)
        benefits_by_demand = [calculate_roi(n_cities, d)['annual_benefit'] for d in demand_range]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(demand_range),
            y=benefits_by_demand,
            mode='lines+markers',
            name='年度效益',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))
        
        # 当前位置
        fig.add_trace(go.Scatter(
            x=[weekly_demand],
            y=[current_benefit],
            mode='markers',
            name='当前配置',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title='需求量 vs 年度效益',
            xaxis_title='周需求量',
            yaxis_title='年度效益 ($)',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # 回本期分析
    st.markdown("###  回本期分析")
    
    payback_col1, payback_col2 = st.columns([2, 1])
    
    with payback_col1:
        # 累计效益曲线
        weeks = list(range(0, 53))
        cumulative_benefit = [results['weekly_total_benefit'] * w - results['implementation_cost'] for w in weeks]
        
        fig = go.Figure()
        
        # 累计效益线
        fig.add_trace(go.Scatter(
            x=weeks,
            y=cumulative_benefit,
            mode='lines',
            name='累计净效益',
            fill='tozeroy',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 盈亏平衡线
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="盈亏平衡")
        
        # 回本点标记
        if results['payback_period_weeks'] < 52:
            fig.add_vline(
                x=results['payback_period_weeks'],
                line_dash="dot",
                line_color="green",
                annotation_text=f"回本点: {results['payback_period_weeks']:.1f}周"
            )
        
        fig.update_layout(
            title='累计净效益曲线（第一年）',
            xaxis_title='周数',
            yaxis_title='累计净效益 ($)',
            height=400
        )
        st.plotly_chart(fig, width='stretch')
    
    with payback_col2:
        st.markdown("#### 投资分析")
        
        st.metric("实施成本", f"${results['implementation_cost']:,.0f}")
        st.metric("回本周期", f"{results['payback_period_weeks']:.1f}周")
        st.metric("年ROI", f"{results['roi']*100:.0f}%")
        
        if results['payback_period_weeks'] < 13:
            st.success(" 优秀：3个月内回本")
        elif results['payback_period_weeks'] < 26:
            st.info(" 良好：6个月内回本")
        else:
            st.warning(" 需要更长回本期")
    
    st.markdown("---")
    
    # 多年效益展望
    st.markdown("###  多年效益展望")
    
    years = [1, 2, 3, 4, 5]
    cumulative_benefits = [
        results['annual_benefit'] * y - results['implementation_cost']
        for y in years
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'第{y}年' for y in years],
        y=cumulative_benefits,
        text=[f'${b/1000:.0f}K' for b in cumulative_benefits],
        textposition='auto',
        marker_color='#2ca02c'
    ))
    fig.update_layout(
        title='多年累计净效益',
        yaxis_title='累计净效益 ($)',
        height=400
    )
    st.plotly_chart(fig, width='stretch')
    
    st.info(f"""
    **5年总效益**: ${results['five_year_benefit']:,.0f}
    
    **年均效益**: ${results['annual_benefit']:,.0f}
    
    **5年ROI**: {(results['five_year_benefit'] - results['implementation_cost']) / results['implementation_cost'] * 100:.0f}%
    """)
    
    st.markdown("---")
    
    # 下载报告
    st.markdown("###  下载分析报告")
    
    report_data = {
        '参数': ['城市数量', '周需求量', '实施成本'],
        '值': [n_cities, weekly_demand, f"${results['implementation_cost']:,.0f}"],
        '': ['', '', ''],
        '效益指标': ['周效益', '年效益', '5年效益'],
        '金额': [
            f"${results['weekly_total_benefit']:,.0f}",
            f"${results['annual_benefit']:,.0f}",
            f"${results['five_year_benefit']:,.0f}"
        ],
        ' ': ['', '', ''],
        '投资回报': ['ROI', '回本周期', '5年ROI'],
        '数值': [
            f"{results['roi']*100:.0f}%",
            f"{results['payback_period_weeks']:.1f}周",
            f"{(results['five_year_benefit'] - results['implementation_cost']) / results['implementation_cost'] * 100:.0f}%"
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.download_button(
        label=" 下载ROI分析报告 (CSV)",
        data=csv,
        file_name=f"roi_analysis_{n_cities}cities.csv",
        mime="text/csv"
    )
    
    # 假设条件说明
    with st.expander(" 计算假设说明"):
        st.markdown("""
        **基础数据来源**：
        - 成本节省：基于v1.0 → v2.0的76%降低（$2,172 → $520，节省$1,652/周）
        - 利润增加：基于v1.0 → v2.0的3%提升（$123,197 → $127,045，增加$3,848/周）
        
        **假设条件**：
        - 实施成本：假设每城市$50,000（一次性）
        - 效益线性增长：随城市数量和需求量线性扩展
        - 稳定运营：假设系统稳定运行，效益持续产生
        
        **注意事项**：
        - 实际效益可能因城市特点、季节变化等因素有所差异
        - 实施成本包括系统部署、人员培训等
        - 建议根据实际情况调整参数
        """)

if __name__ == "__main__":
    main()