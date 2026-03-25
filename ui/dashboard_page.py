from __future__ import annotations

import pandas as pd
import streamlit as st

from services.ai_service import ai_available
from services.data_service import clear_data_caches, get_data_status, get_market_overview_snapshot


def _pct_text(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2%}"


def _stale_text(days: int | None) -> str:
    if days is None:
        return "-"
    if days <= 0:
        return "已是今天"
    return f"落后 {days} 天"


def render_dashboard_page() -> None:
    head1, head2 = st.columns([1, 1])
    with head1:
        st.markdown('<div class="page-title">项目总览</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-desc">这里区分本地训练缓存和远程市场接口状态。如果当前日期晚于本地特征日期，说明训练样本还没有更新到最新交易日。</div>',
            unsafe_allow_html=True,
        )
    with head2:
        action_left, action_right = st.columns(2)
        if action_left.button("刷新远程状态", use_container_width=True):
            clear_data_caches()
            st.rerun()
        if action_right.button("刷新页面缓存", use_container_width=True):
            clear_data_caches()
            st.rerun()

    status = get_data_status(force_remote_refresh=False)
    snapshot = get_market_overview_snapshot()
    artifacts = st.session_state.get("model_artifacts")
    latest_predictions = st.session_state.get("candidate_predictions", pd.DataFrame())
    latest_intelligence = st.session_state.get("market_intelligence", {})

    if status.feature_stale_days and status.feature_stale_days > 0:
        st.warning(
            f"今天是 {status.today_date}，但本地训练特征最后日期是 {status.latest_feature_date}。"
            f"这说明本地量化训练缓存目前还落后 {status.feature_stale_days} 天。"
        )

    row1 = st.columns(4)
    row1[0].metric("本地特征数据", "就绪" if status.features_ready else "缺失")
    row1[1].metric("本地标签数据", "就绪" if status.labels_ready else "缺失")
    row1[2].metric("远程 AkShare 接口", "已接入" if status.akshare_ready else "未安装")
    row1[3].metric("AI 摘要", "已启用" if ai_available() else "未配置")

    row2 = st.columns(4)
    row2[0].metric("今天日期", status.today_date)
    row2[1].metric("本地特征日期", status.latest_feature_date or "-")
    row2[2].metric("特征时效", _stale_text(status.feature_stale_days))
    row2[3].metric("远程股票数量", status.remote_stock_count if status.remote_stock_count is not None else "-")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("数据状态")
        st.write(
            "页面里的训练、候选和预测优先使用本地 parquet。"
            "如果本地最后日期晚于远程接口状态，就说明本地训练样本还没有重建到最新交易日。"
        )
        st.caption(
            f"本地特征行数: {snapshot.get('features_rows', 0)} | "
            f"本地标签行数: {snapshot.get('labels_rows', 0)} | "
            f"远程状态检查时间: {status.remote_checked_at or '-'}"
        )
        if artifacts is not None:
            st.success(artifacts.train_summary)
        else:
            st.info("还没有训练结果，先去“策略训练”页运行一次模型训练。")

    with right:
        st.subheader("关键摘要")
        st.metric("样本股票数", snapshot.get("stock_count", "-"))
        st.metric("平均换手率", "-" if snapshot.get("avg_turnover") is None else f"{snapshot.get('avg_turnover'):.2f}")
        st.metric("平均成交额(亿)", "-" if snapshot.get("avg_amount_yi") is None else f"{snapshot.get('avg_amount_yi'):.2f}")
        st.metric("平均次日收益", _pct_text(snapshot.get("avg_next_day_return")))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("最新预测候选")
        if isinstance(latest_predictions, pd.DataFrame) and not latest_predictions.empty:
            st.dataframe(latest_predictions.head(10), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无候选结果。")

    with col2:
        st.subheader("最新市场情报")
        if isinstance(latest_intelligence, dict) and latest_intelligence.get("news_df") is not None:
            news_df = latest_intelligence.get("news_df", pd.DataFrame())
            if isinstance(news_df, pd.DataFrame) and not news_df.empty:
                preview = news_df[["platform", "title", "direction"]].head(10).copy()
                st.dataframe(preview, use_container_width=True, hide_index=True)
            else:
                st.caption("暂无市场情报。")
        else:
            st.caption("暂无市场情报。")
