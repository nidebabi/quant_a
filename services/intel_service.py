from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd

from services.data_service import _call_akshare


SECTOR_KEYWORDS = {
    "人工智能": ["ai", "算力", "大模型", "数据中心", "机器人", "智能体"],
    "半导体": ["芯片", "半导体", "晶圆", "存储", "光刻", "封测"],
    "新能源": ["光伏", "锂电", "储能", "风电", "新能源车", "电池"],
    "军工": ["军工", "国防", "导弹", "航天", "卫星", "战机"],
    "石油化工": ["原油", "油价", "天然气", "石化", "化工", "中东"],
    "黄金有色": ["黄金", "白银", "铜", "铝", "稀土", "有色"],
    "医药": ["医药", "创新药", "器械", "集采", "医保"],
    "金融地产": ["地产", "银行", "保险", "证券", "降准", "降息", "按揭"],
    "消费出口": ["消费", "白酒", "出口", "关税", "航运", "外贸"],
}

POSITIVE_KEYWORDS = ["增长", "上涨", "签约", "交付", "突破", "支持", "合作", "投产", "降息", "降准", "回购"]
NEGATIVE_KEYWORDS = ["下跌", "冲突", "制裁", "紧张", "亏损", "风险", "关闭", "停火失败", "供应缺口", "关税", "裁员"]

PLATFORM_SOURCES: Dict[str, Tuple[str, Dict[str, Any], str]] = {
    "财联社": ("stock_info_global_cls", {}, "https://www.cls.cn/telegraph"),
    "东方财富": ("stock_info_global_em", {}, "https://finance.eastmoney.com/"),
    "新浪财经": ("stock_info_global_sina", {}, "https://finance.sina.com.cn/"),
    "同花顺": ("stock_info_global_ths", {}, "https://news.10jqka.com.cn/realtimenews.html"),
    "金十数据": ("stock_js_weibo_report", {"time_period": "CNHOUR12"}, "https://datacenter.jin10.com/market"),
}


def _match_sectors(text: str) -> List[str]:
    lowered = str(text).lower()
    hits: List[str] = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            hits.append(sector)
    return hits


def _infer_direction(text: str) -> str:
    lowered = str(text).lower()
    pos = sum(1 for key in POSITIVE_KEYWORDS if key in lowered)
    neg = sum(1 for key in NEGATIVE_KEYWORDS if key in lowered)
    if pos > neg:
        return "利好"
    if neg > pos:
        return "利空"
    return "中性"


def _clean_title(raw: str) -> str:
    text = str(raw).replace("\n", " ").strip()
    return " ".join(text.split())


def _title_from_content(content: str, limit: int = 34) -> str:
    text = _clean_title(content)
    if not text:
        return ""
    return text[:limit]


def _build_analysis(title: str, content: str, sectors: List[str], direction: str) -> str:
    core = _clean_title(content)
    if len(core) > 120:
        core = f"{core[:120]}..."
    sector_text = "、".join(sectors) if sectors else "暂未明确板块"
    return f"{direction}；映射板块：{sector_text}；摘要：{core or title}"


def _normalize_cls(df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    out = df.rename(columns={"标题": "title", "内容": "content", "发布日期": "date", "发布时间": "time"}).copy()
    out["published_at"] = out["date"].astype(str) + " " + out["time"].astype(str)
    out["link"] = default_link
    return out[["title", "content", "published_at", "link"]]


def _normalize_em(df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    out = df.rename(columns={"标题": "title", "摘要": "content", "发布时间": "published_at", "链接": "link"}).copy()
    out["link"] = out["link"].fillna(default_link).replace({"": default_link})
    return out[["title", "content", "published_at", "link"]]


def _normalize_sina(df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    out = df.rename(columns={"时间": "published_at", "内容": "content"}).copy()
    out["content"] = out["content"].astype(str).fillna("")
    out["title"] = out["content"].map(_title_from_content)
    out["link"] = default_link
    return out[["title", "content", "published_at", "link"]]


def _normalize_ths(df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    out = df.rename(columns={"标题": "title", "内容": "content", "发布时间": "published_at", "链接": "link"}).copy()
    out["link"] = out["link"].fillna(default_link).replace({"": default_link})
    return out[["title", "content", "published_at", "link"]]


def _normalize_jin10(df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    out = df.rename(columns={"name": "title", "rate": "rate"}).copy()
    out["title"] = out["title"].astype(str).fillna("")
    out["content"] = out["rate"].astype(str).map(lambda x: f"热股榜涨跌幅：{x}")
    out["published_at"] = "实时热股榜"
    out["link"] = default_link
    return out[["title", "content", "published_at", "link"]]


def _normalize_platform_df(platform: str, df: pd.DataFrame, default_link: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "content", "published_at", "link"])

    if platform == "财联社":
        out = _normalize_cls(df, default_link)
    elif platform == "东方财富":
        out = _normalize_em(df, default_link)
    elif platform == "新浪财经":
        out = _normalize_sina(df, default_link)
    elif platform == "同花顺":
        out = _normalize_ths(df, default_link)
    elif platform == "金十数据":
        out = _normalize_jin10(df, default_link)
    else:
        out = pd.DataFrame(columns=["title", "content", "published_at", "link"])

    out["title"] = out["title"].astype(str).map(_clean_title)
    out["content"] = out["content"].astype(str).map(_clean_title)
    out["published_at"] = out["published_at"].astype(str).fillna("")
    out["link"] = out["link"].astype(str).fillna(default_link).replace({"": default_link, "nan": default_link})

    # 过滤掉标题为空、纯时间格式或明显占位的脏数据
    out = out[out["title"].str.len() > 1].copy()
    out = out[~out["title"].str.fullmatch(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")].copy()
    out = out[~out["title"].str.contains(r"热榜\s*\d+$", regex=True)].copy()
    return out[["title", "content", "published_at", "link"]].reset_index(drop=True)


def _fetch_platform_board(
    platform: str,
    func_name: str,
    kwargs: Dict[str, Any],
    default_link: str,
    limit: int,
) -> Dict[str, Any]:
    df, source = _call_akshare(func_name, **kwargs)
    normalized = _normalize_platform_df(platform, df, default_link).head(limit).copy()

    items: List[Dict[str, Any]] = []
    for idx, row in normalized.iterrows():
        title = str(row["title"]).strip()
        content = str(row["content"]).strip()
        full_text = f"{title} {content}"
        sectors = _match_sectors(full_text)
        direction = _infer_direction(full_text)
        items.append(
            {
                "rank": idx + 1,
                "platform": platform,
                "title": title,
                "content": content,
                "published_at": str(row["published_at"]),
                "link": str(row["link"]),
                "direction": direction,
                "mapped_sectors": sectors,
                "analysis": _build_analysis(title, content, sectors, direction),
            }
        )

    return {"platform": platform, "source": source, "items": items}


@lru_cache(maxsize=4)
def _build_market_intelligence_cached(limit: int) -> Dict[str, Any]:
    boards: Dict[str, Dict[str, Any]] = {}
    all_items: List[Dict[str, Any]] = []
    for platform, (func_name, kwargs, default_link) in PLATFORM_SOURCES.items():
        board = _fetch_platform_board(platform, func_name, kwargs, default_link, limit)
        boards[platform] = board
        all_items.extend(board["items"])
    return {
        "boards": boards,
        "news_df": pd.DataFrame(all_items),
        "source": " | ".join([f"{name}:{boards[name].get('source', '-')}" for name in boards]),
    }


def build_market_intelligence(limit: int = 10, force_refresh: bool = False) -> Dict[str, Any]:
    if force_refresh:
        _build_market_intelligence_cached.cache_clear()
    return _build_market_intelligence_cached(limit)
