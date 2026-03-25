from __future__ import annotations

import os

import streamlit as st

from services.ai_service import ai_available, get_ai_config
from services.data_service import get_data_status


def render_placeholder_page(page_name: str) -> None:
    status = get_data_status()
    ai_cfg = get_ai_config()

    st.markdown(f'<div class="page-title">{page_name}</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">这里放运行环境、接口状态和配置说明。</div>', unsafe_allow_html=True)

    st.subheader("环境状态")
    c1, c2, c3 = st.columns(3)
    c1.metric("特征数据", "就绪" if status.features_ready else "缺失")
    c2.metric("标签数据", "就绪" if status.labels_ready else "缺失")
    c3.metric("AI配置", "已配置" if ai_available() else "未配置")

    st.subheader("建议环境变量")
    st.code(
        "\n".join(
            [
                "OPENAI_API_KEY=你的密钥",
                f"OPENAI_BASE_URL={ai_cfg['base_url']}",
                f"OPENAI_MODEL={ai_cfg['model']}",
            ]
        )
    )

    st.subheader("说明")
    st.write(
        "如果你后续要切换到自己的模型平台，只需要保持 OpenAI 兼容接口即可。"
        "数据侧优先使用本地历史数据，远程行情和资讯接口由 AkShare 统一承接。"
    )
