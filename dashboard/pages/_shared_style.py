import streamlit as st

STYLE = """
<style>
    :root { --primary: #1f77b4; --accent: #2ca02c; --muted: #6c757d; }
    .main-header { font-size: 2.4rem; font-weight: 700; color: var(--primary); margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: var(--muted); margin-bottom: 1rem; }
    .page-header { font-size: 1.6rem; font-weight:700; color: var(--primary); }
    .metric-card { background:#f8f9fb; padding:10px; border-radius:8px; }
    .insight-box { background:#f0f7ff; padding:8px; border-left:4px solid var(--primary); border-radius:6px; }
    .subtle { color: var(--muted); }
    .small { font-size:0.9rem; color:var(--muted); }
</style>
"""


def inject_base_style():
    """注入共享 CSS 样式（在页面顶部调用）"""
    st.markdown(STYLE, unsafe_allow_html=True)
