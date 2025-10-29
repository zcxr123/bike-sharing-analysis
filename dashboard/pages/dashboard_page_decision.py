"""
决策分析页面 - 理解PPO的决策机制
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

st.set_page_config(page_title="决策分析", page_icon="🔍", layout="wide")

# 注入共享样式
inject_base_style()

# 加载数据
@st.cache_data
def load_decision_data():
    try:
        return pd.read_csv('data/decisions.csv')
    except Exception:
        return pd.DataFrame()


def main():
    st.title("🔍 决策分析")
    st.markdown("**深入理解 PPO 的决策机制**")
    st.markdown("---")

    # 加载数据
    df = load_decision_data()

    if df.empty:
        st.warning(
            """
            ⚠️ 未找到决策数据

            可能原因：尚未运行 Day 9 的决策分析或数据文件缺失。

            解决方法：
            1) 请先完成 Day 9 的决策生成步骤；
            2) 或运行数据准备脚本：`python3 scripts/day10_prepare_data.py`；

            提示：如果没有 Day 9 的决策数据，本页面将以示例说明方式展示应有内容。
            """
        )

        # 显示示例说明
        st.markdown("---")
        st.markdown("### 💡 本页面将展示的内容")

        col1, col2 = st.columns(2)

        with col1:
            st.info(
                """
                **🎯 决策模式分析**
                - 调度频率分析
                - 调度时段分布
                - 成本时间模式
                - 高峰/低谷对比
                """
            )

            st.info(
                """
                **📍 空间分布分析**
                - 热点区域识别
                - 调度路径可视化
                - 区域调度频率
                - 空间效率分析
                """
            )

        with col2:
            st.info(
                """
                **💰 成本结构分析**
                - 单次调度成本
                - 成本分布统计
                - 距离-成本关系
                - 效率优化机会
                """
            )

            st.info(
                """
                **🧠 策略洞察**
                - 高频低成本策略
                - 预测性调度
                - 批量服务效应
                - 策略可解释性洞察
                """
            )

        return

    # 如果有数据，显示分析
    st.markdown("### 📊 决策概览")

    # 核心指标
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)

    with mcol1:
        st.metric("总决策数", f"{len(df):,}")

    with mcol2:
        avg_cost = df['rebalance_cost'].mean() if 'rebalance_cost' in df.columns else 0
        st.metric("平均单次成本", f"${avg_cost:.2f}")

    with mcol3:
        avg_moves = df['num_moves'].mean() if 'num_moves' in df.columns else 0
        st.metric("平均调度次数", f"{avg_moves:.1f}")

    with mcol4:
        total_served = df['total_served'].sum() if 'total_served' in df.columns else 0
        st.metric("总服务需求", f"{total_served:,}")

    st.markdown("---")

    # 时间模式分析
    st.markdown("### ⏰ 时间模式分析")

    if 'hour' in df.columns and 'rebalance_cost' in df.columns:
        tab1, tab2 = st.tabs(["调度频率", "成本分布"])

        with tab1:
            # 每小时调度频率
            hourly_freq = df.groupby('hour').size().reset_index(name='count')

            fig = px.bar(
                hourly_freq,
                x='hour',
                y='count',
                title='每小时调度频率',
                labels={'hour': '小时', 'count': '调度次数'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

            # 高峰时段识别
            top_hours = hourly_freq.nlargest(5, 'count')
            st.info(
                f"""
                **高峰调度时段**：
                {', '.join([f"{int(h)}:00" for h in top_hours['hour'].values])}

                这些时段通常对应通勤高峰或需求旺盛期
                """
            )

        with tab2:
            # 每小时成本分布
            hourly_cost = df.groupby('hour')['rebalance_cost'].agg(['sum', 'mean']).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_cost['hour'],
                y=hourly_cost['sum'],
                name='总成本',
                yaxis='y',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Scatter(
                x=hourly_cost['hour'],
                y=hourly_cost['mean'],
                name='平均成本',
                yaxis='y2',
                marker_color='red',
                mode='lines+markers'
            ))

            fig.update_layout(
                title='每小时成本分析',
                xaxis_title='小时',
                yaxis_title='总成本 ($)',
                yaxis2=dict(
                    title='平均成本 ($)',
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("⚠️ 时间数据不完整，无法显示时间模式分析")

    st.markdown("---")

    # 成本分析
    st.markdown("### 💰 成本分析")

    if 'rebalance_cost' in df.columns:
        ccol1, ccol2 = st.columns(2)

        with ccol1:
            # 成本分布直方图
            fig = px.histogram(
                df,
                x='rebalance_cost',
                nbins=30,
                title='单次调度成本分布'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

        with ccol2:
            # 成本箱线图
            fig = px.box(
                df,
                y='rebalance_cost',
                title='成本分布统计'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

        # 成本统计
        cost_stats = df['rebalance_cost'].describe()
        st.dataframe(cost_stats.to_frame().T, width='stretch')

    st.markdown("---")

    # 调度效率分析
    st.markdown("### 📈 调度效率分析")

    if 'num_moves' in df.columns and 'rebalance_cost' in df.columns:
        # 调度次数 vs 成本
        fig = px.scatter(
            df,
            x='num_moves',
            y='rebalance_cost',
            title='调度次数 vs 成本关系',
            opacity=0.5,
            trendline='ols'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

        st.info(
            """
            **洞察**：
            - 如果散点图显示较低的相关性，说明 PPO 学会了优化调度路径
            - 相同次数的调度，成本差异大，说明路径选择很重要
            - 趋势线的斜率反映了平均单次调度的边际成本
            """
        )

    # 原始数据查看
    with st.expander("📄 查看原始决策数据"):
        st.dataframe(df.head(100), width='stretch')

        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 下载完整决策数据",
            data=csv,
            file_name="decision_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()