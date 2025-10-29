"""
策略对比分析页面
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
try:
    from dashboard.pages._shared_style import inject_base_style
except Exception:
    # 回退：从同目录动态加载模块，支持直接在 dashboard 目录下运行
    import importlib.util
    from pathlib import Path
    spec_path = Path(__file__).parent / "_shared_style.py"
    spec = importlib.util.spec_from_file_location("dashboard_pages__shared_style", str(spec_path))
    _shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_shared)
    inject_base_style = _shared.inject_base_style

st.set_page_config(page_title="策略对比", page_icon="📈", layout="wide")

# 注入共享样式
inject_base_style()

# 加载数据（更稳健的相对路径）
DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_data
def load_data():
    try:
        p = DATA_DIR / 'comparison.csv'
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def main():
    st.markdown('<div class="page-header">📈 策略性能对比</div>', unsafe_allow_html=True)
    st.markdown("---")
    # 快速导航按钮
    coln = st.columns([1,1,6])[0]
    if st.button("🏠 返回主页"):
        st.experimental_set_query_params(page='')
    st.markdown("\n")
    
    # 加载数据
    df = load_data()
    
    if df.empty:
        st.error("❌ 未找到对比数据，请先运行数据准备脚本")
        st.code("python3 scripts/day10_prepare_data.py")
        return
    
    # 控制面板
    st.markdown("### 🎛️ 控制面板")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 策略选择
        available_models = df['model'].unique().tolist()
        default_models = [m for m in available_models if 'Day8' in m or 'Proportional' in m]
        selected_models = st.multiselect(
            "选择对比策略",
            options=available_models,
            default=default_models[:2] if len(default_models) >= 2 else available_models[:2]
        )
    
    with col2:
        # 场景选择
        available_scenarios = ['全部'] + df['scenario'].unique().tolist()
        selected_scenario = st.selectbox(
            "选择场景",
            options=available_scenarios
        )
    
    with col3:
        # 指标选择
        metric_options = {
            '服务率': 'service_rate',
            '净利润': 'net_profit',
            '调度成本': 'total_cost'
        }
        selected_metric_name = st.selectbox(
            "选择指标",
            options=list(metric_options.keys())
        )
        selected_metric = metric_options[selected_metric_name]
    
    # 筛选数据
    filtered_df = df[df['model'].isin(selected_models)]
    if selected_scenario != '全部':
        filtered_df = filtered_df[filtered_df['scenario'] == selected_scenario]
    
    if filtered_df.empty:
        st.warning("⚠️ 没有数据，请调整筛选条件")
        return
    
    st.markdown("---")
    
    # 对比图表
    st.markdown("### 📊 对比图表")
    
    tab1, tab2, tab3 = st.tabs(["柱状图对比", "箱线图分析", "散点图关系"])
    
    with tab1:
        # 柱状图
        avg_data = filtered_df.groupby('model')[selected_metric].mean().reset_index()
        
        fig = px.bar(
            avg_data,
            x='model',
            y=selected_metric,
            color='model',
            title=f'{selected_metric_name}对比（平均值）'
        )
        
        # 添加数值标签
        for i, row in avg_data.iterrows():
            fig.add_annotation(
                x=row['model'],
                y=row[selected_metric],
                text=f"{row[selected_metric]:.2f}",
                showarrow=False,
                yshift=10
            )

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        # 箱线图
        fig = px.box(
            filtered_df,
            x='model',
            y=selected_metric,
            color='model',
            title=f'{selected_metric_name}分布'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        # 散点图 - 成本 vs 服务率
        fig = px.scatter(
            filtered_df,
            x='total_cost',
            y='service_rate',
            color='model',
            size='net_profit',
            hover_data=['scenario', 'episode'],
            title='成本-服务率权衡分析'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # 详细统计
    st.markdown("### 📋 详细统计")
    
    summary = filtered_df.groupby('model').agg({
        'service_rate': ['mean', 'std', 'min', 'max'],
        'net_profit': ['mean', 'std', 'min', 'max'],
        'total_cost': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
    summary = summary.reset_index()
    
    st.dataframe(summary, width='stretch')
    
    # 下载按钮
    csv = summary.to_csv(index=False)
    st.download_button(
        label="📥 下载统计数据",
        data=csv,
        file_name="strategy_comparison_summary.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # 关键洞察
    st.markdown("### 💡 关键洞察")
    
    if len(selected_models) >= 2:
        model1, model2 = selected_models[0], selected_models[1]
        data1 = filtered_df[filtered_df['model'] == model1]
        data2 = filtered_df[filtered_df['model'] == model2]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sr_diff = (data2['service_rate'].mean() - data1['service_rate'].mean()) * 100
            st.metric(
                "服务率差异",
                f"{abs(sr_diff):.2f}%",
                delta=f"{'+' if sr_diff > 0 else ''}{sr_diff:.2f}%",
                delta_color="normal" if sr_diff > 0 else "inverse"
            )
        
        with col2:
            cost_diff = data2['total_cost'].mean() - data1['total_cost'].mean()
            cost_pct = (cost_diff / data1['total_cost'].mean()) * 100 if data1['total_cost'].mean() > 0 else 0
            st.metric(
                "成本差异",
                f"${abs(cost_diff):.0f}",
                delta=f"{'+' if cost_diff > 0 else ''}{cost_pct:.1f}%",
                delta_color="inverse" if cost_diff > 0 else "normal"
            )
        
        with col3:
            profit_diff = data2['net_profit'].mean() - data1['net_profit'].mean()
            profit_pct = (profit_diff / data1['net_profit'].mean()) * 100 if data1['net_profit'].mean() > 0 else 0
            st.metric(
                "利润差异",
                f"${abs(profit_diff):.0f}",
                delta=f"{'+' if profit_diff > 0 else ''}{profit_pct:.1f}%",
                delta_color="normal" if profit_diff > 0 else "inverse"
            )
        
        st.info(f"""
        **对比结论**：
        
        {model2} 相比 {model1}：
        - 服务率 {'提高' if sr_diff > 0 else '降低'} {abs(sr_diff):.2f}%
        - 成本 {'增加' if cost_diff > 0 else '降低'} ${abs(cost_diff):.0f} ({abs(cost_pct):.1f}%)
        - 利润 {'增加' if profit_diff > 0 else '降低'} ${abs(profit_diff):.0f} ({abs(profit_pct):.1f}%)
        """)
    
    # 原始数据查看
    with st.expander("📄 查看原始数据"):
        st.dataframe(filtered_df, width='stretch')

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 下载原始数据",
            data=csv,
            file_name="filtered_comparison_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()