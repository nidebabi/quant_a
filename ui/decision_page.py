from __future__ import annotations

import pandas as pd
import streamlit as st

from services.data_service import (
    fetch_stock_daily_history,
    fetch_stock_intraday_history,
    fetch_stock_news,
    get_data_status,
)
from services.model_service import score_candidates_with_models


def _pct(series: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(series[col], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "-")


def render_decision_page() -> None:
    st.markdown('<div class="page-title">次日预测</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">盘后基于候选股给出次日高开、开盘后走强、收盘上涨和触达目标位的概率。</div>',
        unsafe_allow_html=True,
    )

    artifacts = st.session_state.get("model_artifacts")
    status = get_data_status()

    if artifacts is None:
        st.warning("请先去“策略训练”页训练模型。")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        trade_date = st.text_input("交易日", value=status.latest_feature_date)
    with c2:
        top_n = st.number_input("预测标的数量", min_value=5, max_value=50, value=20, step=5)
    with c3:
        sector_filter = st.text_input("板块过滤", value="全部")

    if st.button("生成次日预测", type="primary"):
        prediction_df = score_candidates_with_models(
            artifacts,
            trade_date=trade_date,
            top_n=int(top_n),
            sector_filter=sector_filter,
        )
        st.session_state["candidate_predictions"] = prediction_df

    prediction_df = st.session_state.get("candidate_predictions", pd.DataFrame())
    if not isinstance(prediction_df, pd.DataFrame) or prediction_df.empty:
        st.info("暂无预测结果。")
        return

    display_df = prediction_df.copy()
    for col in ["gap_up_prob", "intraday_up_prob", "close_up_prob", "next_touch_tp_prob", "综合AI评分"]:
        if col in display_df.columns:
            display_df[col] = _pct(display_df, col)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    symbol = st.text_input("查看个股明细", value=str(prediction_df.iloc[0]["code"]))
    if not symbol:
        return

    left, right = st.columns(2)
    with left:
        daily_df, daily_source = fetch_stock_daily_history(symbol, start_date="20240101", end_date="20260331")
        st.subheader("K线数据")
        st.caption(f"来源: {daily_source}")
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            st.dataframe(daily_df.tail(30), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无 K 线数据。")

    with right:
        minute_df, minute_source = fetch_stock_intraday_history(symbol)
        st.subheader("分时数据")
        st.caption(f"来源: {minute_source}")
        if isinstance(minute_df, pd.DataFrame) and not minute_df.empty:
            st.dataframe(minute_df.head(120), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无分时数据。")

    news_df, news_source = fetch_stock_news(symbol)
    st.subheader("个股新闻")
    st.caption(f"来源: {news_source}")
    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        st.dataframe(news_df.head(20), use_container_width=True, hide_index=True)
    else:
        st.caption("暂无个股新闻。")
