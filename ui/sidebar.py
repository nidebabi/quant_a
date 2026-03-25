import streamlit as st

from ui.menu import MENU_ITEMS


def render_sidebar() -> str:
    if "current_page" not in st.session_state:
        st.session_state.current_page = MENU_ITEMS[0]

    with st.sidebar:
        st.markdown('<div class="sidebar-shell">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-logo">StockAgent</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-logo-accent">A-Share AI Ops</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-subtitle">中国A股量化研究、训练回测、次日预测、热点情报的一体化决策台。</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-group-label">Workspace</div>', unsafe_allow_html=True)

        for item in MENU_ITEMS:
            active = st.session_state.current_page == item
            button_label = f"● {item}" if active else item
            if st.button(
                button_label,
                key=f"menu_{item}",
                type="primary" if active else "secondary",
                use_container_width=True,
            ):
                st.session_state.current_page = item

        st.markdown(
            '<div class="sidebar-footnote">建议顺序：先看数据日期，再训练模型，然后筛候选、看次日预测，最后结合市场情报做盘后决策。</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    return st.session_state.current_page
