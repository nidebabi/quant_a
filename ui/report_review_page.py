from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from services.data_service import REPORTS_DIR, get_data_status


def _safe_markdown_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "无"
    return df.head(10).to_markdown(index=False)


def render_report_review_page() -> None:
    st.markdown('<div class="page-title">交付报告</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">这里把训练结果、候选预测和市场情报汇总成一份可直接交付的项目报告。</div>',
        unsafe_allow_html=True,
    )

    artifacts = st.session_state.get("model_artifacts")
    predictions = st.session_state.get("candidate_predictions", pd.DataFrame())
    intelligence = st.session_state.get("market_intelligence", {})
    status = get_data_status()

    if st.button("生成交付报告", type="primary"):
        report_path = Path(REPORTS_DIR) / "delivery_report.md"
        content = f"""# A股AI量化项目交付报告

## 数据状态
- 最新特征日期: {status.latest_feature_date or "-"}
- 最新标签日期: {status.latest_label_date or "-"}
- AkShare接口: {"已接入" if status.akshare_ready else "未安装"}

## 训练总结
{artifacts.train_summary if artifacts is not None else "尚未训练模型"}

## 训练指标
{_safe_markdown_df(artifacts.metrics_df if artifacts is not None else pd.DataFrame())}

## 最新候选预测
{_safe_markdown_df(predictions)}

## 市场情报摘要
{intelligence.get("summary", "尚未生成市场情报")}

## 板块映射
{_safe_markdown_df(intelligence.get("impact_df", pd.DataFrame()) if isinstance(intelligence, dict) else pd.DataFrame())}
"""
        report_path.write_text(content, encoding="utf-8")
        st.session_state["delivery_report_path"] = str(report_path)
        st.success(f"报告已生成: {report_path}")

    report_path = st.session_state.get("delivery_report_path")
    if report_path:
        st.code(report_path)

    st.subheader("交付概览")
    st.write("这个项目当前已经具备本地训练、候选筛选、次日预测、热点情报和报告导出能力。")

    if artifacts is not None:
        st.write(artifacts.train_summary)

    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        st.dataframe(predictions.head(10), use_container_width=True, hide_index=True)
