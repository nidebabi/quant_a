import streamlit as st

from ui.backtest_page import render_backtest_page
from ui.dashboard_page import render_dashboard_page
from ui.decision_page import render_decision_page
from ui.factor_select_page import render_factor_select_page
from ui.hot_news_page import render_hot_news_page
from ui.placeholder_page import render_placeholder_page
from ui.report_review_page import render_report_review_page
from ui.settings_page import render_settings_page
from ui.sidebar import render_sidebar
from ui.styles import load_global_styles


st.set_page_config(
    page_title="StockAgent A-Share AI",
    page_icon="📈",
    layout="wide",
)

load_global_styles()
page = render_sidebar()

if page == "项目总览":
    render_dashboard_page()
elif page == "策略训练":
    render_backtest_page()
elif page == "候选筛选":
    render_factor_select_page()
elif page == "次日预测":
    render_decision_page()
elif page == "市场情报":
    render_hot_news_page()
elif page == "交付报告":
    render_report_review_page()
elif page == "设置":
    render_settings_page()
else:
    render_placeholder_page(page)
