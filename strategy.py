import numpy as np
import pandas as pd
from settings import (
    MIN_AMOUNT, MIN_TURNOVER, MAX_TURNOVER,
    RET3_MIN, RET3_MAX, RET10_MAX,
    CLOSE_LOC_MIN, VOL_RATIO_MIN, VOL_RATIO_MAX,
    UPPER_SHADOW_MAX
)

SECTOR_CANDIDATE_COLS = [
    "sector_name", "industry_name", "industry", "sector", "board_name", "板块", "行业"
]

def get_default_params():
    return {
        # 原有条件开关
        "use_trend": True,
        "use_amount": True,
        "use_turnover": True,
        "use_ret3": True,
        "use_ret10": True,
        "use_close_loc": True,
        "use_vol_ratio": True,
        "use_upper_shadow": True,
        "use_breakout": True,

        # 原有区间参数
        "amount_min": float(MIN_AMOUNT),
        "amount_max": 1e20,

        "turnover_min": float(MIN_TURNOVER),
        "turnover_max": float(MAX_TURNOVER),

        "ret3_min": float(RET3_MIN),
        "ret3_max": float(RET3_MAX),

        "ret10_min": -100.0,
        "ret10_max": float(RET10_MAX),

        "close_loc_min": float(CLOSE_LOC_MIN),

        "vol_ratio_min": float(VOL_RATIO_MIN),
        "vol_ratio_max": float(VOL_RATIO_MAX),

        "upper_shadow_max": float(UPPER_SHADOW_MAX),

        "breakout_min": -0.02,
        "breakout_max": 0.08,

        # 新增：板块/龙头层
        "use_sector_hot": True,                 # 是否启用强板块过滤
        "sector_rank_max": 2,                  # 板块强度排名前N
        "sector_hot_3d_min": 0.55,             # 板块近3日热度下限
        "sector_breadth_min": 0.35,            # 板块广度下限

        "use_sector_position": True,           # 是否启用板块内地位过滤
        "stock_sector_ret_rank_max": 5,        # 板块内涨幅排名前N
        "stock_sector_amount_rank_max": 3,     # 板块内成交额排名前N
        "require_leader_or_core": True,        # 前排或中军即可
    }

def normalize_params(params=None):
    defaults = get_default_params()
    if params:
        defaults.update(params)
    return defaults

def _in_range(series, min_val=None, max_val=None):
    mask = pd.Series(True, index=series.index)
    if min_val is not None:
        mask &= series >= min_val
    if max_val is not None:
        mask &= series <= max_val
    return mask

def _safe_rank_desc(series):
    return series.rank(pct=True, ascending=False, method="average")

def _safe_rank_asc(series):
    return series.rank(pct=True, ascending=True, method="average")

def detect_sector_col(df: pd.DataFrame):
    for c in SECTOR_CANDIDATE_COLS:
        if c in df.columns:
            return c
    return None

def prepare_sector_features(feature_df: pd.DataFrame):
    """
    给 features 补一层“板块/龙头代理特征”
    优先使用已有的行业/板块列；如果没有，则自动关闭板块层能力
    """
    x = feature_df.copy()
    sector_col = detect_sector_col(x)

    # 没有板块数据：补空列，保证后面代码不报错
    if sector_col is None:
        x["sector_name"] = "无板块数据"
        x["sector_rank"] = np.nan
        x["sector_hot_3d"] = np.nan
        x["sector_breadth"] = np.nan
        x["sector_strength_raw"] = np.nan
        x["stock_sector_ret_rank"] = np.nan
        x["stock_sector_amount_rank"] = np.nan
        x["is_sector_leader"] = 0
        x["is_sector_core"] = 0
        return x, False, None

    x["sector_name"] = x[sector_col].astype(str).fillna("未知板块").replace("nan", "未知板块")

    # 生成日收益代理
    if "preclose" in x.columns:
        x["day_ret"] = np.where(x["preclose"] > 0, x["close"] / x["preclose"] - 1, np.nan)
    elif "open" in x.columns:
        x["day_ret"] = np.where(x["open"] > 0, x["close"] / x["open"] - 1, np.nan)
    else:
        x["day_ret"] = np.nan

    # 板块内“强势广度”代理
    strong_flag = (
        (x["day_ret"].fillna(0) > 0) |
        (x["ret_3"].fillna(0) > 0) |
        (x["close_loc"].fillna(0) >= 0.7)
    ).astype(int)
    x["_sector_positive_flag"] = strong_flag

    # 日度板块聚合
    sector_daily = (
        x.groupby(["date", "sector_name"], dropna=False)
        .agg(
            sector_stock_count=("code", "nunique"),
            sector_amount_sum=("amount", "sum"),
            sector_day_ret_mean=("day_ret", "mean"),
            sector_ret3_mean=("ret_3", "mean"),
            sector_ret10_mean=("ret_10", "mean"),
            sector_breadth=("_sector_positive_flag", "mean"),
            sector_close_loc_mean=("close_loc", "mean"),
            sector_vol_ratio_mean=("vol_ratio", "mean"),
        )
        .reset_index()
    )

    # 每日横截面排名，做成板块强度
    sector_daily["r_day_ret"] = sector_daily.groupby("date")["sector_day_ret_mean"].rank(pct=True, ascending=False)
    sector_daily["r_ret3"] = sector_daily.groupby("date")["sector_ret3_mean"].rank(pct=True, ascending=False)
    sector_daily["r_ret10"] = sector_daily.groupby("date")["sector_ret10_mean"].rank(pct=True, ascending=False)
    sector_daily["r_amount"] = sector_daily.groupby("date")["sector_amount_sum"].rank(pct=True, ascending=False)
    sector_daily["r_breadth"] = sector_daily.groupby("date")["sector_breadth"].rank(pct=True, ascending=False)
    sector_daily["r_close_loc"] = sector_daily.groupby("date")["sector_close_loc_mean"].rank(pct=True, ascending=False)

    sector_daily["sector_strength_raw"] = (
        0.25 * sector_daily["r_day_ret"] +
        0.20 * sector_daily["r_ret3"] +
        0.10 * sector_daily["r_ret10"] +
        0.20 * sector_daily["r_amount"] +
        0.15 * sector_daily["r_breadth"] +
        0.10 * sector_daily["r_close_loc"]
    )

    sector_daily["sector_rank"] = (
        sector_daily.groupby("date")["sector_strength_raw"]
        .rank(method="min", ascending=False)
    )

    sector_daily = sector_daily.sort_values(["sector_name", "date"]).reset_index(drop=True)
    sector_daily["sector_hot_3d"] = (
        sector_daily.groupby("sector_name")["sector_strength_raw"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    x = x.merge(
        sector_daily[[
            "date", "sector_name", "sector_rank", "sector_hot_3d",
            "sector_breadth", "sector_strength_raw", "sector_ret3_mean"
        ]],
        on=["date", "sector_name"],
        how="left"
    )

    # 板块内个股地位
    x["stock_sector_ret_rank"] = (
        x.groupby(["date", "sector_name"])["ret_3"]
        .rank(method="min", ascending=False)
    )
    x["stock_sector_amount_rank"] = (
        x.groupby(["date", "sector_name"])["amount"]
        .rank(method="min", ascending=False)
    )

    x["is_sector_leader"] = (x["stock_sector_ret_rank"] <= 3).astype(int)
    x["is_sector_core"] = (x["stock_sector_amount_rank"] <= 2).astype(int)

    return x.drop(columns=["_sector_positive_flag"], errors="ignore"), True, sector_col

def row_pass_signal(row, params=None) -> bool:
    p = normalize_params(params)

    required_cols = [
        "close", "amount", "turnover", "ret_3", "ret_10",
        "close_loc", "vol_ratio", "upper_shadow_ratio",
        "breakout20", "trend_flag"
    ]
    for c in required_cols:
        if pd.isna(row.get(c, None)):
            return False

    if p["use_trend"] and row["trend_flag"] != 1:
        return False

    if p["use_amount"]:
        if not (p["amount_min"] <= row["amount"] <= p["amount_max"]):
            return False

    if p["use_turnover"]:
        if not (p["turnover_min"] <= row["turnover"] <= p["turnover_max"]):
            return False

    if p["use_ret3"]:
        if not (p["ret3_min"] <= row["ret_3"] <= p["ret3_max"]):
            return False

    if p["use_ret10"]:
        if not (p["ret10_min"] <= row["ret_10"] <= p["ret10_max"]):
            return False

    if p["use_close_loc"]:
        if row["close_loc"] < p["close_loc_min"]:
            return False

    if p["use_vol_ratio"]:
        if not (p["vol_ratio_min"] <= row["vol_ratio"] <= p["vol_ratio_max"]):
            return False

    if p["use_upper_shadow"]:
        if row["upper_shadow_ratio"] > p["upper_shadow_max"]:
            return False

    if p["use_breakout"]:
        if not (p["breakout_min"] <= row["breakout20"] <= p["breakout_max"]):
            return False

    # 板块层过滤
    if p["use_sector_hot"]:
        if pd.isna(row.get("sector_rank", np.nan)):
            return False
        if row["sector_rank"] > p["sector_rank_max"]:
            return False
        if row.get("sector_hot_3d", 0) < p["sector_hot_3d_min"]:
            return False
        if row.get("sector_breadth", 0) < p["sector_breadth_min"]:
            return False

    # 板块内地位过滤
    if p["use_sector_position"]:
        ret_rank_ok = row.get("stock_sector_ret_rank", 999999) <= p["stock_sector_ret_rank_max"]
        amount_rank_ok = row.get("stock_sector_amount_rank", 999999) <= p["stock_sector_amount_rank_max"]

        if p["require_leader_or_core"]:
            if not (ret_rank_ok or amount_rank_ok):
                return False
        else:
            if not ret_rank_ok:
                return False

    return True

def select_candidates(feature_df: pd.DataFrame, params=None) -> pd.DataFrame:
    p = normalize_params(params)
    x = feature_df.copy()

    required_cols = [
        "date", "code", "name", "close", "amount", "turnover",
        "ret_3", "ret_10", "close_loc", "vol_ratio",
        "upper_shadow_ratio", "breakout20", "trend_flag"
    ]
    x = x.dropna(subset=required_cols)

    if p["use_trend"]:
        x = x[x["trend_flag"] == 1]

    if p["use_amount"]:
        x = x[_in_range(x["amount"], p["amount_min"], p["amount_max"])]

    if p["use_turnover"]:
        x = x[_in_range(x["turnover"], p["turnover_min"], p["turnover_max"])]

    if p["use_ret3"]:
        x = x[_in_range(x["ret_3"], p["ret3_min"], p["ret3_max"])]

    if p["use_ret10"]:
        x = x[_in_range(x["ret_10"], p["ret10_min"], p["ret10_max"])]

    if p["use_close_loc"]:
        x = x[x["close_loc"] >= p["close_loc_min"]]

    if p["use_vol_ratio"]:
        x = x[_in_range(x["vol_ratio"], p["vol_ratio_min"], p["vol_ratio_max"])]

    if p["use_upper_shadow"]:
        x = x[x["upper_shadow_ratio"] <= p["upper_shadow_max"]]

    if p["use_breakout"]:
        x = x[_in_range(x["breakout20"], p["breakout_min"], p["breakout_max"])]

    if p["use_sector_hot"] and "sector_rank" in x.columns:
        x = x[
            (x["sector_rank"] <= p["sector_rank_max"]) &
            (x["sector_hot_3d"] >= p["sector_hot_3d_min"]) &
            (x["sector_breadth"] >= p["sector_breadth_min"])
        ]

    if p["use_sector_position"] and "stock_sector_ret_rank" in x.columns:
        ret_rank_ok = x["stock_sector_ret_rank"] <= p["stock_sector_ret_rank_max"]
        amount_rank_ok = x["stock_sector_amount_rank"] <= p["stock_sector_amount_rank_max"]

        if p["require_leader_or_core"]:
            x = x[ret_rank_ok | amount_rank_ok]
        else:
            x = x[ret_rank_ok]

    if x.empty:
        return x

    score_parts = []

    if p["use_close_loc"]:
        score_parts.append(_safe_rank_desc(x["close_loc"]))

    if p["use_vol_ratio"]:
        score_parts.append(_safe_rank_desc(x["vol_ratio"]))

    if p["use_amount"]:
        score_parts.append(_safe_rank_desc(x["amount"]))

    if p["use_ret3"]:
        score_parts.append(_safe_rank_desc(x["ret_3"]))

    if p["use_breakout"]:
        breakout_score_raw = -(x["breakout20"] - 0.01).abs()
        score_parts.append(_safe_rank_desc(breakout_score_raw))

    if p["use_upper_shadow"]:
        shadow_score_raw = -x["upper_shadow_ratio"]
        score_parts.append(_safe_rank_desc(shadow_score_raw))

    # 新增：板块强度分
    if p["use_sector_hot"] and "sector_strength_raw" in x.columns:
        score_parts.append(_safe_rank_desc(x["sector_strength_raw"]))

    # 新增：板块内地位分
    if p["use_sector_position"] and "stock_sector_ret_rank" in x.columns:
        leader_score = (
            0.6 * _safe_rank_asc(x["stock_sector_ret_rank"]) +
            0.4 * _safe_rank_asc(x["stock_sector_amount_rank"])
        )
        score_parts.append(leader_score)

    if score_parts:
        x["score"] = sum(score_parts) / len(score_parts)
    else:
        x["score"] = 1.0

    keep_cols = [
        "date", "code", "name", "sector_name",
        "sector_rank", "sector_hot_3d", "sector_breadth",
        "stock_sector_ret_rank", "stock_sector_amount_rank",
        "is_sector_leader", "is_sector_core",
        "open", "high", "low", "close",
        "amount", "turnover", "ret_3", "ret_10", "close_loc",
        "vol_ratio", "upper_shadow_ratio", "breakout20", "score"
    ]
    keep_cols = [c for c in keep_cols if c in x.columns]

    return x.sort_values("score", ascending=False)[keep_cols].reset_index(drop=True)