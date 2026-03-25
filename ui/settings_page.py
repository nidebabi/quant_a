from __future__ import annotations

import streamlit as st

from services.data_service import clear_data_caches, get_data_status
from services.update_service import DEFAULT_SAFE_LIMIT, update_market_data


def render_settings_page() -> None:
    st.markdown('<div class="page-title">设置</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">这里负责本地数据增量更新与缓存管理。当前更新器默认走保守模式：串行执行、小样本优先、不会默认全市场刷新。</div>',
        unsafe_allow_html=True,
    )

    status = get_data_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("当前日期", status.today_date)
    c2.metric("本地特征日期", status.latest_feature_date or "-")
    c3.metric("落后天数", status.feature_stale_days if status.feature_stale_days is not None else "-")

    left, right = st.columns(2)
    with left:
        sample_limit = st.number_input(
            "小样本更新股票数",
            min_value=1,
            max_value=50,
            value=DEFAULT_SAFE_LIMIT,
            step=1,
            help="默认只更新前几只股票做安全增量，不会直接扫全市场。",
        )
        if st.button("执行小样本增量更新", type="primary", use_container_width=True):
            with st.spinner("正在按保守模式执行增量更新并重建特征/标签..."):
                result = update_market_data(limit=int(sample_limit), max_workers=1)
                st.session_state["latest_update_result"] = result
                clear_data_caches()
                st.success("增量更新完成。")
                st.rerun()

    with right:
        st.info("当前更新路径：东方财富分钟线聚合日线。")
        if st.button("清理本地缓存", use_container_width=True):
            clear_data_caches()
            st.success("缓存已清理。")
            st.rerun()

    result = st.session_state.get("latest_update_result")
    if result:
        st.subheader("最近一次更新结果")
        st.write(
            f"检查文件: {result.checked_files} | 更新文件: {result.updated_files} | "
            f"失败文件: {result.failed_files} | 更新前日期: {result.latest_local_date_before} | "
            f"更新后日期: {result.latest_local_date_after}"
        )
        st.caption(f"特征行数: {result.feature_rows} | 标签行数: {result.label_rows}")
        st.caption(f"更新路径: {result.source_path}")
        if result.errors:
            with st.expander("失败明细", expanded=False):
                st.code("\n".join(result.errors[:30]))
