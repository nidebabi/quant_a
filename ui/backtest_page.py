from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from services.model_service import run_prediction_backtest, train_quant_models


def render_backtest_page() -> None:
    st.markdown('<div class="page-title">策略训练</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">用本地A股历史样本训练次日预测模型，输出成功率、留出集回测、特征重要性和规则总结。</div>',
        unsafe_allow_html=True,
    )

    train_col, note_col = st.columns([1, 2])
    with train_col:
        run_train = st.button("开始训练", type="primary", use_container_width=True)
    with note_col:
        st.caption("训练基于本地 `features_all.parquet` 与 `labels_all.parquet`。若要接实时数据，可后续用 AkShare 接口刷新。")

    if run_train:
        with st.spinner("正在训练模型并生成留出集回测结果..."):
            artifacts = train_quant_models()
            st.session_state["model_artifacts"] = artifacts
            st.session_state["training_backtest"] = run_prediction_backtest(artifacts.holdout_predictions, top_n=10)

    artifacts = st.session_state.get("model_artifacts")
    backtest = st.session_state.get("training_backtest", {})

    if artifacts is None:
        st.info("点击上方按钮开始训练。")
        return

    top_row = st.columns(4)
    metrics_df = artifacts.metrics_df if isinstance(artifacts.metrics_df, pd.DataFrame) else pd.DataFrame()
    top_row[0].metric("模型数量", len(artifacts.models))
    top_row[1].metric("特征数量", len(artifacts.features))
    top_row[2].metric("留出集样本", len(artifacts.holdout_predictions))
    top_row[3].metric("平均准确率", "-" if metrics_df.empty else f"{metrics_df['accuracy'].mean():.2%}")

    st.success(artifacts.train_summary)

    tab1, tab2, tab3 = st.tabs(["训练指标", "留出集回测", "特征重要性"])

    with tab1:
        if metrics_df.empty:
            st.caption("暂无训练指标。")
        else:
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab2:
        daily_df = backtest.get("daily_df", pd.DataFrame())
        st.info(backtest.get("summary", "暂无回测摘要"))
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            fig = px.line(daily_df, x="date", y="equity", markers=True, title="留出集策略净值")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(daily_df, use_container_width=True, hide_index=True)

    with tab3:
        importance_df = artifacts.feature_importance_df
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            selected_target = st.selectbox("选择目标", importance_df["target"].dropna().unique().tolist())
            view_df = importance_df[importance_df["target"] == selected_target].head(15).copy()
            fig = px.bar(view_df, x="importance", y="feature", orientation="h", title=f"{selected_target} Top15 特征")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view_df, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无特征重要性结果。")
