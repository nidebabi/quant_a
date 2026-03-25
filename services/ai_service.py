from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


def get_ai_config() -> Dict[str, str]:
    return {
        "api_key": os.getenv("OPENAI_API_KEY", "").strip(),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
    }


def ai_available() -> bool:
    cfg = get_ai_config()
    return bool(cfg["api_key"])


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Optional[str]:
    cfg = get_ai_config()
    if not cfg["api_key"]:
        return None

    url = f"{cfg['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    except Exception:
        return None
    return None


def summarize_with_ai(title: str, payload: Dict[str, Any], fallback: str) -> str:
    summary = chat_completion(
        system_prompt=(
            "你是中国A股量化研究助手。输出要面向交易后复盘场景，"
            "强调概率、风险、行业传导与可执行建议。禁止夸大确定性。"
        ),
        user_prompt=f"{title}\n\n请基于以下结构化数据生成简洁中文摘要：\n{json.dumps(payload, ensure_ascii=False)}",
    )
    return summary or fallback


def summarize_news_items(items: List[Dict[str, Any]], fallback: str) -> str:
    summary = chat_completion(
        system_prompt=(
            "你是中国A股盘后情报分析师。请把事件总结为：核心事件、受影响板块、"
            "可能利多或利空的链条、明日开盘关注点。保持审慎。"
        ),
        user_prompt=json.dumps(items[:20], ensure_ascii=False),
        max_tokens=700,
    )
    return summary or fallback
