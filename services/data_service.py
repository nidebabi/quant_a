from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = DATA_DIR / "cache"
FEATURE_PATH = DATA_DIR / "features" / "features_all.parquet"
LABEL_PATH = DATA_DIR / "labels" / "labels_all.parquet"
RAW_DIR = DATA_DIR / "raw"

for path in [CACHE_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


try:
    import akshare as ak  # type: ignore
except Exception:
    ak = None


@dataclass
class DataStatus:
    features_ready: bool
    labels_ready: bool
    akshare_ready: bool
    latest_feature_date: str
    latest_label_date: str
    today_date: str
    feature_stale_days: Optional[int]
    label_stale_days: Optional[int]
    remote_stock_count: Optional[int]
    remote_checked_at: str


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _file_stamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@lru_cache(maxsize=4)
def _read_parquet_cached(path_str: str, stamp: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = _safe_to_datetime(df["date"])
    return df


def clear_data_caches() -> None:
    _read_parquet_cached.cache_clear()
    _merged_features_labels_cached.cache_clear()
    _market_overview_snapshot_cached.cache_clear()
    _remote_market_snapshot_cached.cache_clear()


def load_features() -> pd.DataFrame:
    return _read_parquet_cached(str(FEATURE_PATH), _file_stamp(FEATURE_PATH)).copy()


def load_labels() -> pd.DataFrame:
    return _read_parquet_cached(str(LABEL_PATH), _file_stamp(LABEL_PATH)).copy()


@lru_cache(maxsize=2)
def _merged_features_labels_cached(feature_stamp: float, label_stamp: float) -> pd.DataFrame:
    features = _read_parquet_cached(str(FEATURE_PATH), feature_stamp)
    labels = _read_parquet_cached(str(LABEL_PATH), label_stamp)
    if features.empty or labels.empty:
        return pd.DataFrame()

    merged = features.merge(labels, on=["date", "code", "name"], how="inner")
    merged = merged.sort_values(["date", "code"]).reset_index(drop=True)
    merged["gap_up_flag"] = (merged["next_open"] > merged["close"]).astype(int)
    merged["intraday_up_flag"] = (merged["next_close"] > merged["next_open"]).astype(int)
    merged["close_up_flag"] = (merged["next_close"] > merged["close"]).astype(int)
    merged["next_day_return"] = np.where(merged["close"] > 0, merged["next_close"] / merged["close"] - 1, np.nan)
    merged["next_open_return"] = np.where(merged["close"] > 0, merged["next_open"] / merged["close"] - 1, np.nan)
    return merged


def merge_features_and_labels() -> pd.DataFrame:
    return _merged_features_labels_cached(_file_stamp(FEATURE_PATH), _file_stamp(LABEL_PATH)).copy()


def get_latest_trade_date(features: Optional[pd.DataFrame] = None) -> Optional[pd.Timestamp]:
    feature_df = features if features is not None else load_features()
    if feature_df.empty or "date" not in feature_df.columns:
        return None
    return pd.to_datetime(feature_df["date"]).max()


def get_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked_prefixes = ("next_",)
    blocked = {
        "gap_up_flag",
        "intraday_up_flag",
        "close_up_flag",
        "next_day_return",
        "next_open_return",
        "next_red_close_flag",
        "next_touch_tp_flag",
        "tradable_flag",
    }
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in blocked or col.startswith(blocked_prefixes) or col == "date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def _normalize_candidate_frame(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    if df.empty:
        return df

    if "sector_name" not in df.columns:
        df["sector_name"] = "未知板块"

    for col in ["amount", "turnover", "ret_3", "ret_10", "close_loc", "vol_ratio", "upper_shadow_ratio"]:
        if col not in df.columns:
            df[col] = np.nan

    score = (
        pd.to_numeric(df["close_loc"], errors="coerce").fillna(0) * 25
        + pd.to_numeric(df["vol_ratio"], errors="coerce").fillna(0) * 15
        + pd.to_numeric(df["ret_3"], errors="coerce").fillna(0) * 100
        + pd.to_numeric(df["turnover"], errors="coerce").fillna(0) * 2
        + pd.to_numeric(df["amount"], errors="coerce").fillna(0) / 1e8 * 0.4
        - pd.to_numeric(df["upper_shadow_ratio"], errors="coerce").fillna(0) * 8
    )
    df["factor_score"] = score
    return df


def get_candidate_pool(trade_date: Optional[str] = None, top_n: int = 30, sector_filter: str = "全部") -> pd.DataFrame:
    features = load_features()
    if features.empty:
        return pd.DataFrame()

    df = _normalize_candidate_frame(features)
    if trade_date:
        dt = pd.to_datetime(trade_date, errors="coerce")
        if pd.notna(dt):
            df = df[df["date"] == dt].copy()
    else:
        latest_dt = get_latest_trade_date(df)
        if latest_dt is not None:
            df = df[df["date"] == latest_dt].copy()

    trend_mask = df["trend_flag"] == 1 if "trend_flag" in df.columns else pd.Series(True, index=df.index)
    df = df[trend_mask].copy()
    if sector_filter and sector_filter != "全部" and "sector_name" in df.columns:
        df = df[df["sector_name"].astype(str) == str(sector_filter)].copy()

    df = df[
        (pd.to_numeric(df["amount"], errors="coerce") >= 2e8)
        & (pd.to_numeric(df["turnover"], errors="coerce") >= 1)
        & (pd.to_numeric(df["ret_3"], errors="coerce") >= -0.02)
        & (pd.to_numeric(df["close_loc"], errors="coerce") >= 0.55)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("factor_score", ascending=False).head(top_n).copy()
    return pd.DataFrame(
        {
            "交易日": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"),
            "代码": df["code"].astype(str),
            "名称": df["name"].astype(str),
            "板块": df["sector_name"].astype(str),
            "因子评分": pd.to_numeric(df["factor_score"], errors="coerce").round(2),
            "成交额(亿)": (pd.to_numeric(df["amount"], errors="coerce") / 1e8).round(2),
            "换手率": pd.to_numeric(df["turnover"], errors="coerce").round(2),
            "近3日涨幅": pd.to_numeric(df["ret_3"], errors="coerce").round(4),
            "近10日涨幅": pd.to_numeric(df["ret_10"], errors="coerce").round(4),
            "收盘位置": pd.to_numeric(df["close_loc"], errors="coerce").round(4),
            "量比": pd.to_numeric(df["vol_ratio"], errors="coerce").round(2),
        }
    )


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    if not text:
        return text
    if text.startswith(("sh", "sz", "bj")):
        return text
    if text.startswith(("6", "5")):
        return f"sh{text}"
    if text.startswith(("0", "3", "2")):
        return f"sz{text}"
    return text


def _call_akshare(func_name: str, **kwargs: Any) -> Tuple[pd.DataFrame, str]:
    if ak is None:
        return pd.DataFrame(), "AkShare 未安装"
    func = getattr(ak, func_name, None)
    if func is None:
        return pd.DataFrame(), f"AkShare 缺少接口 {func_name}"
    try:
        df = func(**kwargs)
        if isinstance(df, pd.DataFrame):
            return df, "AkShare"
        return pd.DataFrame(), f"{func_name} 返回值不是 DataFrame"
    except Exception as exc:
        return pd.DataFrame(), f"{func_name} 调用失败: {exc}"


@lru_cache(maxsize=1)
def _remote_market_snapshot_cached(today_key: str) -> Dict[str, Any]:
    if ak is None:
        return {"remote_stock_count": None, "checked_at": "", "source": "AkShare 未安装"}
    try:
        spot_df = ak.stock_zh_a_spot_em()
        return {
            "remote_stock_count": int(len(spot_df)) if isinstance(spot_df, pd.DataFrame) else None,
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "AkShare:stock_zh_a_spot_em",
        }
    except Exception as exc:
        return {"remote_stock_count": None, "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "source": f"remote failed: {exc}"}


def get_remote_market_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    today_key = date.today().isoformat()
    if force_refresh:
        _remote_market_snapshot_cached.cache_clear()
    return _remote_market_snapshot_cached(today_key)


def get_data_status(force_remote_refresh: bool = False) -> DataStatus:
    features = load_features()
    labels = load_labels()
    latest_feature_date = str(features["date"].max().date()) if not features.empty and "date" in features.columns else ""
    latest_label_date = str(labels["date"].max().date()) if not labels.empty and "date" in labels.columns else ""
    today_date = date.today().isoformat()

    feature_stale_days = None
    label_stale_days = None
    if latest_feature_date:
        feature_stale_days = (date.fromisoformat(today_date) - date.fromisoformat(latest_feature_date)).days
    if latest_label_date:
        label_stale_days = (date.fromisoformat(today_date) - date.fromisoformat(latest_label_date)).days

    remote_info = get_remote_market_snapshot(force_refresh=force_remote_refresh)
    return DataStatus(
        features_ready=not features.empty,
        labels_ready=not labels.empty,
        akshare_ready=ak is not None,
        latest_feature_date=latest_feature_date,
        latest_label_date=latest_label_date,
        today_date=today_date,
        feature_stale_days=feature_stale_days,
        label_stale_days=label_stale_days,
        remote_stock_count=remote_info.get("remote_stock_count"),
        remote_checked_at=remote_info.get("checked_at", ""),
    )


def fetch_stock_daily_history(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> Tuple[pd.DataFrame, str]:
    normalized = _normalize_symbol(symbol)
    local_path = RAW_DIR / f"{normalized[-6:]}.parquet"
    if local_path.exists():
        df = pd.read_parquet(local_path)
        return df, "本地缓存"

    df, source = _call_akshare(
        "stock_zh_a_hist",
        symbol=normalized[-6:],
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )
    return df, source


def fetch_stock_intraday_history(symbol: str) -> Tuple[pd.DataFrame, str]:
    normalized = _normalize_symbol(symbol)
    df, source = _call_akshare("stock_zh_a_hist_min_em", symbol=normalized[-6:], period="1", adjust="")
    return df, source


def fetch_stock_news(symbol: str) -> Tuple[pd.DataFrame, str]:
    normalized = _normalize_symbol(symbol)
    return _call_akshare("stock_news_em", symbol=normalized[-6:])


def fetch_global_news() -> Tuple[pd.DataFrame, str]:
    attempts: Iterable[Tuple[str, Dict[str, Any]]] = [
        ("stock_info_global_cls", {}),
        ("news_economic_baidu", {}),
    ]
    for func_name, kwargs in attempts:
        df, source = _call_akshare(func_name, **kwargs)
        if not df.empty:
            return df, f"{source}:{func_name}"
    return pd.DataFrame(), "未获取到远程新闻"


def fetch_sector_snapshot() -> Tuple[pd.DataFrame, str]:
    attempts: Iterable[Tuple[str, Dict[str, Any]]] = [
        ("stock_board_industry_name_em", {}),
        ("stock_sector_fund_flow_rank", {"indicator": "今日", "sector_type": "行业资金流"}),
    ]
    for func_name, kwargs in attempts:
        df, source = _call_akshare(func_name, **kwargs)
        if not df.empty:
            return df, f"{source}:{func_name}"
    return pd.DataFrame(), "未获取到远程板块数据"


def cache_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cached_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=2)
def _market_overview_snapshot_cached(feature_stamp: float, label_stamp: float) -> Dict[str, Any]:
    features = _read_parquet_cached(str(FEATURE_PATH), feature_stamp)
    labels = _read_parquet_cached(str(LABEL_PATH), label_stamp)
    merged = _merged_features_labels_cached(feature_stamp, label_stamp)

    if features.empty:
        return {
            "latest_trade_date": "",
            "stock_count": 0,
            "avg_turnover": None,
            "avg_amount_yi": None,
            "avg_next_day_return": None,
        }

    latest_dt = pd.to_datetime(features["date"]).max()
    latest_day = features[features["date"] == latest_dt].copy()
    latest_merged = merged[merged["date"] == latest_dt].copy() if not merged.empty else pd.DataFrame()

    avg_amount_yi = None
    if "amount" in latest_day.columns and latest_day["amount"].notna().sum() > 0:
        avg_amount_yi = float(pd.to_numeric(latest_day["amount"], errors="coerce").mean() / 1e8)

    avg_turnover = None
    if "turnover" in latest_day.columns and latest_day["turnover"].notna().sum() > 0:
        avg_turnover = float(pd.to_numeric(latest_day["turnover"], errors="coerce").mean())

    avg_next_day_return = None
    if not latest_merged.empty and latest_merged["next_day_return"].notna().sum() > 0:
        avg_next_day_return = float(latest_merged["next_day_return"].mean())

    return {
        "latest_trade_date": str(latest_dt.date()) if latest_dt is not None else "",
        "stock_count": int(latest_day["code"].nunique()) if "code" in latest_day.columns else len(latest_day),
        "avg_turnover": avg_turnover,
        "avg_amount_yi": avg_amount_yi,
        "avg_next_day_return": avg_next_day_return,
        "features_rows": int(len(features)),
        "labels_rows": int(len(labels)),
    }


def get_market_overview_snapshot() -> Dict[str, Any]:
    return _market_overview_snapshot_cached(_file_stamp(FEATURE_PATH), _file_stamp(LABEL_PATH))
