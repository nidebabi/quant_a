from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

from services.intel_service import build_market_intelligence


PLATFORM_THEMES = {
    "财联社": "market-card-red",
    "东方财富": "market-card-blue",
    "新浪财经": "market-card-sky",
    "同花顺": "market-card-indigo",
    "金十数据": "market-card-gold",
}


def _inject_market_page_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #eef2f7 !important;
        }

        section.main .block-container {
            background: #ffffff !important;
            border-radius: 24px;
            padding: 28px 28px 36px 28px !important;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        }

        section.main .block-container,
        section.main .block-container *:not(section):not(button) {
            color: #111827 !important;
        }

        .market-page-title {
            font-size: 40px;
            font-weight: 900;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .market-page-desc {
            font-size: 14px;
            line-height: 1.8;
            color: #475569;
            margin-bottom: 18px;
        }

        .market-page-note {
            font-size: 12px;
            color: #64748b;
        }

        .market-board-title {
            margin: 12px 0 14px 0;
            font-size: 16px;
            font-weight: 800;
            color: #111827;
        }

        .market-card-flag, .market-detail-flag {
            display: none;
        }

        div[data-testid="stVerticalBlock"]:has(.market-card-flag) {
            border-radius: 20px;
            padding: 14px;
            border: 1px solid rgba(15, 23, 42, 0.06);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
            min-height: 420px;
        }

        div[data-testid="stVerticalBlock"]:has(.market-card-red) { background: linear-gradient(180deg, #ffe3e7 0%, #fff7f8 100%); }
        div[data-testid="stVerticalBlock"]:has(.market-card-blue) { background: linear-gradient(180deg, #dbeeff 0%, #f6fbff 100%); }
        div[data-testid="stVerticalBlock"]:has(.market-card-sky) { background: linear-gradient(180deg, #dff7ff 0%, #f7fcff 100%); }
        div[data-testid="stVerticalBlock"]:has(.market-card-indigo) { background: linear-gradient(180deg, #e5e7ff 0%, #f8f9ff 100%); }
        div[data-testid="stVerticalBlock"]:has(.market-card-gold) { background: linear-gradient(180deg, #fff0c7 0%, #fffaf0 100%); }

        div[data-testid="stVerticalBlock"]:has(.market-detail-flag) {
            background: #ffffff;
            border-radius: 22px;
            padding: 18px 22px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
            margin-bottom: 12px;
        }

        .market-card-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .market-card-title {
            font-size: 18px;
            font-weight: 900;
            color: #0f172a;
        }

        .market-card-source {
            font-size: 11px;
            color: #6b7280;
            font-weight: 600;
        }

        div[data-testid="stVerticalBlock"]:has(.market-card-flag) div[data-testid="stButton"] > button {
            min-height: 38px;
            margin-bottom: 8px;
            border-radius: 12px !important;
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: none !important;
            text-align: left;
            justify-content: flex-start;
            font-size: 13px;
            font-weight: 700;
        }

        div[data-testid="stVerticalBlock"]:has(.market-card-flag) div[data-testid="stButton"] > button:hover {
            background: #f8fafc !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }

        div[data-testid="stVerticalBlock"]:has(.market-card-flag) div[data-testid="stButton"] > button[kind="primary"] {
            background: #eef6ff !important;
            color: #0f172a !important;
            border-color: #93c5fd !important;
        }

        .market-back-wrap div[data-testid="stButton"] > button,
        .market-toolbar-wrap div[data-testid="stButton"] > button {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #dbe3ef !important;
            box-shadow: none !important;
            min-height: 40px;
        }

        .market-toolbar-wrap div[data-testid="stButton"] > button[kind="primary"] {
            background: #111827 !important;
            color: #ffffff !important;
            border-color: #111827 !important;
        }

        .market-status {
            font-size: 12px;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _need_refresh() -> bool:
    last_refresh = st.session_state.get("market_intelligence_refreshed_at")
    if last_refresh is None:
        return True
    return datetime.now() - last_refresh > timedelta(minutes=2)


def _load_realtime_intelligence(force: bool = False) -> dict:
    if force or "market_intelligence" not in st.session_state or _need_refresh():
        st.session_state["market_intelligence"] = build_market_intelligence(limit=10, force_refresh=True)
        st.session_state["market_intelligence_refreshed_at"] = datetime.now()
    return st.session_state.get("market_intelligence", {})


def _open_detail(platform: str, item: dict) -> None:
    st.session_state["selected_intel_platform"] = platform
    st.session_state["selected_intel_item"] = item
    st.session_state["market_intel_mode"] = "detail"
    st.rerun()


def _back_to_board() -> None:
    st.session_state["market_intel_mode"] = "board"
    st.rerun()


def _render_board_card(platform: str, board: dict) -> None:
    theme = PLATFORM_THEMES.get(platform, "market-card-blue")
    items = board.get("items", [])

    with st.container():
        st.markdown(f'<div class="market-card-flag {theme}"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="market-card-head">
                <div class="market-card-title">{platform}</div>
                <div class="market-card-source">{board.get("source", "-")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for item in items[:8]:
            label = f"{item['rank']}. {item['title'][:34]}{'...' if len(item['title']) > 34 else ''}"
            if st.button(label, key=f"intel_rank_{platform}_{item['rank']}", use_container_width=True):
                _open_detail(platform, item)


def _render_detail_page() -> None:
    platform = st.session_state.get("selected_intel_platform", "")
    item = st.session_state.get("selected_intel_item", {})
    sectors = "、".join(item.get("mapped_sectors", [])) if item.get("mapped_sectors") else "暂未明确板块"

    back_col, meta_col = st.columns([1, 4], gap="small")
    with back_col:
        st.markdown('<div class="market-back-wrap">', unsafe_allow_html=True)
        if st.button("返回热榜", use_container_width=True):
            _back_to_board()
        st.markdown("</div>", unsafe_allow_html=True)
    with meta_col:
        st.markdown(f'<div class="market-status">详情来源：{platform}</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="market-detail-flag"></div>', unsafe_allow_html=True)
        st.markdown(f"## {item.get('title', '')}")
        st.caption(f"{item.get('published_at', '')} | {item.get('direction', '中性')}")
        st.write(item.get("analysis", ""))
        content = str(item.get("content", "")).strip()
        if content and content != "nan":
            st.write(content)
        st.caption(f"映射板块：{sectors}")
        st.markdown(f"[查看原文]({item.get('link', '#')})")


def _render_board_page(boards: dict) -> None:
    st.markdown('<div class="market-board-title">平台热榜</div>', unsafe_allow_html=True)
    platforms = list(boards.keys())
    for i in range(0, len(platforms), 3):
        cols = st.columns(3, gap="medium")
        for col, platform in zip(cols, platforms[i:i + 3]):
            with col:
                _render_board_card(platform, boards.get(platform, {}))


def render_hot_news_page() -> None:
    _inject_market_page_styles()

    st.markdown('<div class="market-page-title">市场情报</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="market-page-desc">五个平台分别做成独立热榜卡片，卡片里只展示该平台热榜标题；点击任意标题后，跳转到专门的详情页查看原文、板块映射和利好利空总结。</div>',
        unsafe_allow_html=True,
    )

    toolbar_left, toolbar_right = st.columns([1, 3], gap="small")
    with toolbar_left:
        st.markdown('<div class="market-toolbar-wrap">', unsafe_allow_html=True)
        refresh = st.button("刷新热榜", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with toolbar_right:
        refreshed_at = st.session_state.get("market_intelligence_refreshed_at")
        refreshed_text = refreshed_at.strftime("%H:%M:%S") if refreshed_at else "未刷新"
        st.markdown(
            f'<div class="market-status">实时平台：财联社、东方财富、新浪财经、同花顺、金十数据 | 最近刷新：{refreshed_text}</div>',
            unsafe_allow_html=True,
        )

    intelligence = _load_realtime_intelligence(force=refresh)
    boards = intelligence.get("boards", {})
    st.markdown(f'<div class="market-page-note">数据状态：{intelligence.get("source", "-")}</div>', unsafe_allow_html=True)

    mode = st.session_state.get("market_intel_mode", "board")
    if mode == "detail" and st.session_state.get("selected_intel_item"):
        _render_detail_page()
    else:
        _render_board_page(boards)
