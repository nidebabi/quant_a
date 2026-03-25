from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "features" / "features_all.parquet"


@dataclass
class ColumnMap:
    date: Optional[str] = None
    code: Optional[str] = None
    name: Optional[str] = None
    sector: Optional[str] = None
    amount: Optional[str] = None
    turnover: Optional[str] = None
    volume_ratio: Optional[str] = None
    score: Optional[str] = None
    risk: Optional[str] = None
    close_position: Optional[str] = None
    upper_shadow: Optional[str] = None
    breakout_range: Optional[str] = None
    ret_3d: Optional[str] = None
    ret_10d: Optional[str] = None
    sector_rank: Optional[str] = None
    sector_hot: Optional[str] = None
    sector_breadth: Optional[str] = None
    sector_inner_rank: Optional[str] = None
    sector_amount_rank: Optional[str] = None


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return float(value)

    text = _safe_str(value)
    if not text:
        return default
    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value, default=None)
    if number is None:
        return default
    return int(round(number))


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for col in df.columns:
        col_l = str(col).lower()
        for candidate in candidates:
            if candidate.lower() in col_l:
                return col
    return None


def _detect_columns(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        date=_pick_column(df, ["trade_date", "date", "dt", "日期"]),
        code=_pick_column(df, ["stock_code", "code", "ts_code", "symbol", "ticker", "代码"]),
        name=_pick_column(df, ["stock_name", "name", "股票名称", "名称"]),
        sector=_pick_column(df, ["sector_name", "sector", "industry", "concept", "theme", "板块", "概念"]),
        amount=_pick_column(df, ["amount", "成交额", "turnover_amount", "成交金额"]),
        turnover=_pick_column(df, ["turnover_rate", "turnover", "换手率"]),
        volume_ratio=_pick_column(df, ["volume_ratio", "vol_ratio", "量比"]),
        score=_pick_column(df, ["score", "total_score", "rank_score", "综合评分", "分数"]),
        risk=_pick_column(df, ["risk", "pred_risk", "risk_score", "预测风险", "风险"]),
        close_position=_pick_column(df, ["close_position", "收盘位置"]),
        upper_shadow=_pick_column(df, ["upper_shadow", "upper_shadow_ratio", "上影线"]),
        breakout_range=_pick_column(df, ["breakout_range", "突破区间"]),
        ret_3d=_pick_column(df, ["ret_3d", "return_3d", "近3日涨跌幅", "pct_chg_3d"]),
        ret_10d=_pick_column(df, ["ret_10d", "return_10d", "近10日涨跌幅", "pct_chg_10d"]),
        sector_rank=_pick_column(df, ["sector_rank", "板块排名", "sector_top_n"]),
        sector_hot=_pick_column(df, ["sector_hot", "sector_hot_3d", "板块近3日热度", "sector_heat_3d"]),
        sector_breadth=_pick_column(df, ["sector_breadth", "板块广度"]),
        sector_inner_rank=_pick_column(df, ["sector_inner_rank", "sector_stock_rank", "板块内涨幅排名"]),
        sector_amount_rank=_pick_column(df, ["sector_amount_rank", "板块内成交额排名"]),
    )


def _normalize_date_series(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.date


def _normalize_amount_to_yi(series: pd.Series) -> pd.Series:
    s = _to_numeric_series(series)
    valid = s.dropna()
    if valid.empty:
        return s
    q95 = valid.quantile(0.95)

    if q95 > 1_000_000:
        return s / 1e8
    if q95 > 10_000 and q95 <= 1_000_000:
        return s / 1e4
    return s


def _normalize_turnover_to_pct(series: pd.Series) -> pd.Series:
    s = _to_numeric_series(series)
    valid = s.dropna()
    if valid.empty:
        return s
    q95 = valid.quantile(0.95)

    if q95 <= 2.5:
        return s * 100
    return s


def _normalize_ratio_like(series: pd.Series) -> pd.Series:
    return _to_numeric_series(series)


def _standardize_dataframe(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnMap]:
    df = raw_df.copy()
    cols = _detect_columns(df)

    if cols.date:
        df["__date__"] = _normalize_date_series(df[cols.date])
    else:
        df["__date__"] = pd.NaT

    df["__code__"] = df[cols.code].astype(str) if cols.code else ""
    df["__name__"] = df[cols.name].astype(str) if cols.name else ""
    df["__sector__"] = df[cols.sector].astype(str) if cols.sector else "未知板块"

    df["__amount_yi__"] = _normalize_amount_to_yi(df[cols.amount]) if cols.amount else pd.Series([None] * len(df))
    df["__turnover_pct__"] = _normalize_turnover_to_pct(df[cols.turnover]) if cols.turnover else pd.Series([None] * len(df))
    df["__volume_ratio__"] = _normalize_ratio_like(df[cols.volume_ratio]) if cols.volume_ratio else pd.Series([None] * len(df))
    df["__close_position__"] = _normalize_ratio_like(df[cols.close_position]) if cols.close_position else pd.Series([None] * len(df))
    df["__upper_shadow__"] = _normalize_ratio_like(df[cols.upper_shadow]) if cols.upper_shadow else pd.Series([None] * len(df))
    df["__breakout_range__"] = _normalize_ratio_like(df[cols.breakout_range]) if cols.breakout_range else pd.Series([None] * len(df))
    df["__ret_3d__"] = _normalize_ratio_like(df[cols.ret_3d]) if cols.ret_3d else pd.Series([None] * len(df))
    df["__ret_10d__"] = _normalize_ratio_like(df[cols.ret_10d]) if cols.ret_10d else pd.Series([None] * len(df))
    df["__sector_rank__"] = _normalize_ratio_like(df[cols.sector_rank]) if cols.sector_rank else pd.Series([None] * len(df))
    df["__sector_hot__"] = _normalize_ratio_like(df[cols.sector_hot]) if cols.sector_hot else pd.Series([None] * len(df))
    df["__sector_breadth__"] = _normalize_ratio_like(df[cols.sector_breadth]) if cols.sector_breadth else pd.Series([None] * len(df))
    df["__sector_inner_rank__"] = _normalize_ratio_like(df[cols.sector_inner_rank]) if cols.sector_inner_rank else pd.Series([None] * len(df))
    df["__sector_amount_rank__"] = _normalize_ratio_like(df[cols.sector_amount_rank]) if cols.sector_amount_rank else pd.Series([None] * len(df))
    df["__score_raw__"] = _normalize_ratio_like(df[cols.score]) if cols.score else pd.Series([None] * len(df))
    df["__risk_raw__"] = _normalize_ratio_like(df[cols.risk]) if cols.risk else pd.Series([None] * len(df))

    return df, cols


def _safe_quantile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    ranked = s.rank(pct=True, ascending=ascending)
    return ranked.fillna(0.5)


def _safe_abs_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.abs()


def _build_score_and_risk(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    amount_s = pd.to_numeric(out["__amount_yi__"], errors="coerce")
    turnover_s = pd.to_numeric(out["__turnover_pct__"], errors="coerce")
    volume_ratio_s = pd.to_numeric(out["__volume_ratio__"], errors="coerce")
    close_position_s = pd.to_numeric(out["__close_position__"], errors="coerce")
    breakout_range_s = pd.to_numeric(out["__breakout_range__"], errors="coerce")
    ret_3d_s = pd.to_numeric(out["__ret_3d__"], errors="coerce")
    upper_shadow_s = pd.to_numeric(out["__upper_shadow__"], errors="coerce")
    score_raw_s = pd.to_numeric(out["__score_raw__"], errors="coerce")
    risk_raw_s = pd.to_numeric(out["__risk_raw__"], errors="coerce")

    if score_raw_s.notna().sum() > 0:
        out["__score__"] = score_raw_s
    else:
        score = (
            _safe_quantile_rank(amount_s, ascending=True) * 25
            + _safe_quantile_rank(turnover_s, ascending=True) * 20
            + _safe_quantile_rank(volume_ratio_s, ascending=True) * 20
            + _safe_quantile_rank(close_position_s, ascending=True) * 15
            + _safe_quantile_rank(breakout_range_s, ascending=True) * 10
            + _safe_quantile_rank(ret_3d_s, ascending=True) * 10
        )
        out["__score__"] = score.round(2)

    if risk_raw_s.notna().sum() > 0:
        out["__risk__"] = risk_raw_s
    else:
        turnover_risk = _safe_quantile_rank(turnover_s, ascending=True) * 4
        vol_risk = _safe_quantile_rank(volume_ratio_s, ascending=True) * 3
        upper_shadow_risk = _safe_quantile_rank(upper_shadow_s, ascending=True) * 2
        ret_risk = _safe_quantile_rank(_safe_abs_series(ret_3d_s), ascending=True) * 1
        out["__risk__"] = (turnover_risk + vol_risk + upper_shadow_risk + ret_risk).round(2)

    return out


def _field_range(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"min": None, "max": None, "median": None, "non_null_count": 0}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
        "non_null_count": int(s.notna().sum()),
    }


def _append_stage(diag: List[Dict[str, Any]], stage: str, before_count: int, after_count: int, detail: str) -> None:
    diag.append(
        {
            "阶段": stage,
            "过滤前": before_count,
            "过滤后": after_count,
            "减少数量": before_count - after_count,
            "说明": detail,
        }
    )


def _between_filter(df: pd.DataFrame, col: str, low: Optional[float], high: Optional[float]) -> pd.DataFrame:
    out = df.copy()
    s = pd.to_numeric(out[col], errors="coerce")
    mask = pd.Series(True, index=out.index)
    if low is not None:
        mask &= s >= low
    if high is not None:
        mask &= s <= high
    mask &= s.notna()
    return out.loc[mask].copy()


def _load_feature_df() -> pd.DataFrame:
    env_path = os.getenv("FACTOR_FEATURE_PATH", "").strip()
    feature_path = Path(env_path) if env_path else DEFAULT_FEATURE_PATH

    if not feature_path.exists():
        raise FileNotFoundError(f"未找到特征数据文件：{feature_path}")

    df = pd.read_parquet(feature_path)
    if df.empty:
        raise ValueError("特征数据为空，无法进行因子选股。")
    return df


def get_default_factor_params() -> Dict[str, Any]:
    return {
        "candidate_date": None,
        "sector_filter": "全部",
        "top_n": 5,
        "enable_model_threshold": False,
        "min_score": 0.0,
        "max_risk": 10.0,

        "enable_amount": True,
        "amount_min_yi": 2.0,
        "amount_max_yi": 999999.0,

        "enable_turnover": False,
        "turnover_min_pct": 5.0,
        "turnover_max_pct": 30.0,

        "enable_volume_ratio": False,
        "volume_ratio_min": 1.2,
        "volume_ratio_max": 6.0,

        "enable_ret_3d": False,
        "ret_3d_min": 0.02,
        "ret_3d_max": 0.18,

        "enable_ret_10d": False,
        "ret_10d_min": -1.0,
        "ret_10d_max": 0.28,

        "enable_close_position": False,
        "close_position_min": 0.65,

        "enable_upper_shadow": False,
        "upper_shadow_max": 0.35,

        "enable_breakout_range": False,
        "breakout_min": -0.02,
        "breakout_max": 0.08,

        "enable_sector_filter": False,
        "sector_rank_top_n": 2,
        "sector_hot_min": 0.55,
        "sector_breadth_min": 0.35,

        "enable_sector_inner_filter": False,
        "sector_inner_rank_top_n": 5,
        "sector_amount_rank_top_n": 3,
        "allow_mid_sector": False,
    }


def get_default_params() -> Dict[str, Any]:
    return get_default_factor_params()


def build_custom_params_from_ui(
    ui_params: Optional[Dict[str, Any]] = None,
    sector_ready: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    兼容旧版 backtest_page.py 的导入。
    现在允许旧页面传 sector_ready 或其他额外关键字参数，统一忽略/兼容。
    """
    defaults = get_default_factor_params()
    if not isinstance(ui_params, dict):
        ui_params = {}

    params = deepcopy(defaults)

    alias_map = {
        "trade_date": "candidate_date",
        "date": "candidate_date",
        "sector_name": "sector_filter",
        "sector": "sector_filter",
        "output_count": "top_n",
        "candidate_count": "top_n",
        "topn": "top_n",

        "enable_score_risk_filter": "enable_model_threshold",
        "enable_model_filter": "enable_model_threshold",
        "min_total_score": "min_score",
        "min_predict_score": "min_score",
        "max_pred_risk": "max_risk",
        "max_predict_risk": "max_risk",

        "enable_amount_filter": "enable_amount",
        "amount_min": "amount_min_yi",
        "amount_max": "amount_max_yi",

        "enable_turnover_filter": "enable_turnover",
        "turnover_min": "turnover_min_pct",
        "turnover_max": "turnover_max_pct",

        "enable_vol_ratio_filter": "enable_volume_ratio",
        "enable_volume_ratio_filter": "enable_volume_ratio",
        "vol_ratio_min": "volume_ratio_min",
        "vol_ratio_max": "volume_ratio_max",

        "enable_3d_filter": "enable_ret_3d",
        "pct_3d_min": "ret_3d_min",
        "pct_3d_max": "ret_3d_max",

        "enable_10d_filter": "enable_ret_10d",
        "pct_10d_min": "ret_10d_min",
        "pct_10d_max": "ret_10d_max",

        "enable_close_pos_filter": "enable_close_position",
        "close_pos_min": "close_position_min",

        "enable_upper_shadow_filter": "enable_upper_shadow",
        "upper_shadow_max_value": "upper_shadow_max",

        "enable_breakout_filter": "enable_breakout_range",
        "breakout_low": "breakout_min",
        "breakout_high": "breakout_max",

        "enable_sector_boost_filter": "enable_sector_filter",
        "sector_top_n": "sector_rank_top_n",
        "sector_heat_min": "sector_hot_min",
        "sector_width_min": "sector_breadth_min",

        "enable_sector_inner_rank_filter": "enable_sector_inner_filter",
        "sector_stock_rank_top_n": "sector_inner_rank_top_n",
        "sector_turnover_rank_top_n": "sector_amount_rank_top_n",
        "allow_sector_mid": "allow_mid_sector",
    }

    merged_inputs = {}
    merged_inputs.update(ui_params)

    # 兼容旧页面额外传入的 kwargs
    if isinstance(kwargs, dict):
        merged_inputs.update(kwargs)

    # 兼容旧页面可能传的 sector_ready
    if sector_ready is not None:
        merged_inputs["sector_ready"] = sector_ready

    for key, value in merged_inputs.items():
        target_key = alias_map.get(key, key)
        if target_key in params:
            params[target_key] = value

    return params


def get_available_dates() -> List[str]:
    df = _load_feature_df()
    std_df, cols = _standardize_dataframe(df)
    if cols.date is None:
        return []
    dates = std_df["__date__"].dropna().astype(str).unique().tolist()
    dates = sorted(dates)
    return dates


def get_available_sectors(candidate_date: Optional[str] = None) -> List[str]:
    df = _load_feature_df()
    std_df, _ = _standardize_dataframe(df)

    if candidate_date:
        date_obj = pd.to_datetime(candidate_date, errors="coerce")
        if pd.notna(date_obj):
            std_df = std_df[std_df["__date__"] == date_obj.date()].copy()

    sectors = sorted([x for x in std_df["__sector__"].dropna().astype(str).unique().tolist() if x.strip()])
    return ["全部"] + sectors


def run_factor_selection(params: Dict[str, Any]) -> Dict[str, Any]:
    raw_df = _load_feature_df()
    std_df, col_map = _standardize_dataframe(raw_df)
    std_df = _build_score_and_risk(std_df)

    candidate_date = _safe_str(params.get("candidate_date"))
    if not candidate_date:
        available_dates = std_df["__date__"].dropna().astype(str).unique().tolist()
        if not available_dates:
            raise ValueError("数据中没有可用交易日。")
        candidate_date = sorted(available_dates)[-1]

    candidate_dt = pd.to_datetime(candidate_date, errors="coerce")
    if pd.isna(candidate_dt):
        raise ValueError(f"候选日期无效：{candidate_date}")

    diag_rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    initial_count = len(std_df)
    day_df = std_df[std_df["__date__"] == candidate_dt.date()].copy()
    _append_stage(diag_rows, "按日期取样", initial_count, len(day_df), f"候选日期：{candidate_date}")

    if day_df.empty:
        return {
            "created_at": _now_str(),
            "candidate_date": candidate_date,
            "params": deepcopy(params),
            "selected_stocks": [],
            "result_df": pd.DataFrame(),
            "diagnostic_df": pd.DataFrame(diag_rows),
            "field_range_df": pd.DataFrame(),
            "summary_text": f"{candidate_date} 没有样本数据，请检查数据文件是否包含该日期。",
            "notes": ["候选日期在特征数据中不存在，或日期字段未正确映射。"],
            "column_mapping": vars(col_map),
            "base_sample_count": 0,
            "after_sector_sample_count": 0,
        }

    base_range_df = pd.DataFrame(
        [
            {"字段": "成交额(亿)", **_field_range(day_df["__amount_yi__"])},
            {"字段": "换手率(%)", **_field_range(day_df["__turnover_pct__"])},
            {"字段": "量比", **_field_range(day_df["__volume_ratio__"])},
            {"字段": "近3日涨跌幅", **_field_range(day_df["__ret_3d__"])},
            {"字段": "近10日涨跌幅", **_field_range(day_df["__ret_10d__"])},
            {"字段": "收盘位置", **_field_range(day_df["__close_position__"])},
            {"字段": "上影线", **_field_range(day_df["__upper_shadow__"])},
            {"字段": "突破区间", **_field_range(day_df["__breakout_range__"])},
            {"字段": "综合评分", **_field_range(day_df["__score__"])},
            {"字段": "预测风险", **_field_range(day_df["__risk__"])},
        ]
    )

    working_df = day_df.copy()
    sector_filter = _safe_str(params.get("sector_filter"), "全部")
    if sector_filter and sector_filter != "全部":
        before = len(working_df)
        working_df = working_df[working_df["__sector__"] == sector_filter].copy()
        _append_stage(diag_rows, "板块筛选", before, len(working_df), f"板块：{sector_filter}")

    after_sector_count = len(working_df)

    def apply_cond(enable_key: str, stage_name: str, field_col: str, low_key: Optional[str], high_key: Optional[str], label: str):
        nonlocal working_df, notes
        enabled = bool(params.get(enable_key))
        if not enabled:
            return

        if field_col not in working_df.columns:
            notes.append(f"{stage_name} 未执行：缺少字段 {field_col}")
            return

        non_null_count = pd.to_numeric(working_df[field_col], errors="coerce").notna().sum()
        if non_null_count == 0:
            before = len(working_df)
            working_df = working_df.iloc[0:0].copy()
            _append_stage(diag_rows, stage_name, before, 0, f"{label} 字段全为空，无法过滤")
            notes.append(f"{stage_name} 对应字段全为空，请检查字段映射或数据预处理。")
            return

        low = _to_float(params.get(low_key)) if low_key else None
        high = _to_float(params.get(high_key)) if high_key else None

        before = len(working_df)
        working_df = _between_filter(working_df, field_col, low, high)
        detail = f"{label}"
        if low is not None:
            detail += f" 下限={low}"
        if high is not None:
            detail += f" 上限={high}"
        _append_stage(diag_rows, stage_name, before, len(working_df), detail)

    apply_cond("enable_amount", "成交额过滤", "__amount_yi__", "amount_min_yi", "amount_max_yi", "成交额(亿)")
    apply_cond("enable_turnover", "换手率过滤", "__turnover_pct__", "turnover_min_pct", "turnover_max_pct", "换手率(%)")
    apply_cond("enable_volume_ratio", "量比过滤", "__volume_ratio__", "volume_ratio_min", "volume_ratio_max", "量比")
    apply_cond("enable_ret_3d", "近3日涨跌幅过滤", "__ret_3d__", "ret_3d_min", "ret_3d_max", "近3日涨跌幅")
    apply_cond("enable_ret_10d", "近10日涨跌幅过滤", "__ret_10d__", "ret_10d_min", "ret_10d_max", "近10日涨跌幅")

    if bool(params.get("enable_close_position")):
        before = len(working_df)
        if working_df["__close_position__"].notna().sum() == 0:
            working_df = working_df.iloc[0:0].copy()
            _append_stage(diag_rows, "收盘位置过滤", before, 0, "收盘位置字段全为空")
            notes.append("收盘位置字段全为空，请检查字段映射。")
        else:
            low = _to_float(params.get("close_position_min"))
            working_df = _between_filter(working_df, "__close_position__", low, None)
            _append_stage(diag_rows, "收盘位置过滤", before, len(working_df), f"收盘位置下限={low}")

    if bool(params.get("enable_upper_shadow")):
        before = len(working_df)
        if working_df["__upper_shadow__"].notna().sum() == 0:
            working_df = working_df.iloc[0:0].copy()
            _append_stage(diag_rows, "上影线过滤", before, 0, "上影线字段全为空")
            notes.append("上影线字段全为空，请检查字段映射。")
        else:
            high = _to_float(params.get("upper_shadow_max"))
            working_df = _between_filter(working_df, "__upper_shadow__", None, high)
            _append_stage(diag_rows, "上影线过滤", before, len(working_df), f"上影线上限={high}")

    if bool(params.get("enable_breakout_range")):
        before = len(working_df)
        if working_df["__breakout_range__"].notna().sum() == 0:
            working_df = working_df.iloc[0:0].copy()
            _append_stage(diag_rows, "突破区间过滤", before, 0, "突破区间字段全为空")
            notes.append("突破区间字段全为空，请检查字段映射。")
        else:
            low = _to_float(params.get("breakout_min"))
            high = _to_float(params.get("breakout_max"))
            working_df = _between_filter(working_df, "__breakout_range__", low, high)
            _append_stage(diag_rows, "突破区间过滤", before, len(working_df), f"突破区间下限={low} 上限={high}")

    if bool(params.get("enable_sector_filter")):
        before = len(working_df)
        sector_rank_top_n = _to_int(params.get("sector_rank_top_n"), 2)
        sector_hot_min = _to_float(params.get("sector_hot_min"))
        sector_breadth_min = _to_float(params.get("sector_breadth_min"))

        if working_df["__sector_rank__"].notna().sum() == 0 and working_df["__sector_hot__"].notna().sum() == 0 and working_df["__sector_breadth__"].notna().sum() == 0:
            notes.append("板块增强过滤已开启，但缺少相关字段，已跳过。")
            _append_stage(diag_rows, "板块增强过滤", before, len(working_df), "缺少板块字段，已跳过")
        else:
            mask = pd.Series(True, index=working_df.index)
            if working_df["__sector_rank__"].notna().sum() > 0:
                mask &= pd.to_numeric(working_df["__sector_rank__"], errors="coerce") <= sector_rank_top_n
            if sector_hot_min is not None and working_df["__sector_hot__"].notna().sum() > 0:
                mask &= pd.to_numeric(working_df["__sector_hot__"], errors="coerce") >= sector_hot_min
            if sector_breadth_min is not None and working_df["__sector_breadth__"].notna().sum() > 0:
                mask &= pd.to_numeric(working_df["__sector_breadth__"], errors="coerce") >= sector_breadth_min

            working_df = working_df.loc[mask].copy()
            _append_stage(
                diag_rows,
                "板块增强过滤",
                before,
                len(working_df),
                f"板块排名前N={sector_rank_top_n} 热度下限={sector_hot_min} 广度下限={sector_breadth_min}",
            )

    if bool(params.get("enable_sector_inner_filter")):
        before = len(working_df)
        inner_top_n = _to_int(params.get("sector_inner_rank_top_n"), 5)
        amount_top_n = _to_int(params.get("sector_amount_rank_top_n"), 3)
        allow_mid_sector = bool(params.get("allow_mid_sector"))

        if working_df["__sector_inner_rank__"].notna().sum() == 0 and working_df["__sector_amount_rank__"].notna().sum() == 0:
            notes.append("板块内地位过滤已开启，但缺少相关字段，已跳过。")
            _append_stage(diag_rows, "板块内地位过滤", before, len(working_df), "缺少板块内地位字段，已跳过")
        else:
            mask = pd.Series(False, index=working_df.index)
            if working_df["__sector_inner_rank__"].notna().sum() > 0:
                mask |= pd.to_numeric(working_df["__sector_inner_rank__"], errors="coerce") <= inner_top_n
            if working_df["__sector_amount_rank__"].notna().sum() > 0:
                mask |= pd.to_numeric(working_df["__sector_amount_rank__"], errors="coerce") <= amount_top_n
            if allow_mid_sector:
                mask |= True

            working_df = working_df.loc[mask].copy()
            _append_stage(
                diag_rows,
                "板块内地位过滤",
                before,
                len(working_df),
                f"板块内涨幅排名前N={inner_top_n} 板块内成交额排名前N={amount_top_n} 前排或中军={allow_mid_sector}",
            )

    if bool(params.get("enable_model_threshold")):
        before = len(working_df)
        min_score = _to_float(params.get("min_score"), 0.0)
        max_risk = _to_float(params.get("max_risk"), 10.0)
        score_s = pd.to_numeric(working_df["__score__"], errors="coerce")
        risk_s = pd.to_numeric(working_df["__risk__"], errors="coerce")
        working_df = working_df[(score_s >= min_score) & (risk_s <= max_risk)].copy()
        _append_stage(diag_rows, "模型阈值过滤", before, len(working_df), f"最低综合分={min_score} 最大预测风险={max_risk}")

    diag_df = pd.DataFrame(diag_rows)

    sort_cols = []
    ascending = []

    if "__score__" in working_df.columns:
        sort_cols.append("__score__")
        ascending.append(False)
    if "__risk__" in working_df.columns:
        sort_cols.append("__risk__")
        ascending.append(True)
    if "__amount_yi__" in working_df.columns:
        sort_cols.append("__amount_yi__")
        ascending.append(False)

    if sort_cols:
        working_df = working_df.sort_values(sort_cols, ascending=ascending, na_position="last").copy()

    top_n = _to_int(params.get("top_n"), 5)
    selected_df = working_df.head(top_n).copy()

    def _safe_round(series_name: str, digits: int = 2) -> pd.Series:
        if series_name not in selected_df.columns:
            return pd.Series([None] * len(selected_df), index=selected_df.index)
        s = pd.to_numeric(selected_df[series_name], errors="coerce")
        return s.round(digits)

    output_df = pd.DataFrame(
        {
            "代码": selected_df["__code__"] if "__code__" in selected_df.columns else pd.Series([], dtype="object"),
            "名称": selected_df["__name__"] if "__name__" in selected_df.columns else pd.Series([], dtype="object"),
            "板块": selected_df["__sector__"] if "__sector__" in selected_df.columns else pd.Series([], dtype="object"),
            "综合评分": _safe_round("__score__", 2),
            "预测风险": _safe_round("__risk__", 2),
            "成交额(亿)": _safe_round("__amount_yi__", 2),
            "换手率(%)": _safe_round("__turnover_pct__", 2),
            "量比": _safe_round("__volume_ratio__", 2),
            "近3日涨跌幅": _safe_round("__ret_3d__", 4),
            "近10日涨跌幅": _safe_round("__ret_10d__", 4),
            "收盘位置": _safe_round("__close_position__", 4),
            "上影线": _safe_round("__upper_shadow__", 4),
            "突破区间": _safe_round("__breakout_range__", 4),
        }
    )

    most_kill = None
    if not diag_df.empty:
        diag_non_date = diag_df[diag_df["阶段"] != "按日期取样"].copy()
        if not diag_non_date.empty:
            idx = diag_non_date["减少数量"].idxmax()
            if pd.notna(idx):
                most_kill = diag_non_date.loc[idx].to_dict()

    if output_df.empty:
        summary_text = "当前条件下没有筛出候选股。"
        if most_kill:
            summary_text += f" 最致命条件是：{most_kill['阶段']}（减少 {most_kill['减少数量']} 只）。"
        max_amount_rows = base_range_df.loc[base_range_df["字段"] == "成交额(亿)", "max"]
        if bool(params.get("enable_amount")) and not max_amount_rows.empty:
            max_amount = max_amount_rows.iloc[0]
            if max_amount is not None and pd.notna(max_amount) and max_amount < _to_float(params.get("amount_min_yi"), 0):
                notes.append("当天样本的成交额最大值都低于你的成交额下限，极可能是成交额单位或阈值设置问题。")
    else:
        summary_text = f"共筛出 {len(output_df)} 只候选股。"

    return {
        "created_at": _now_str(),
        "candidate_date": candidate_date,
        "params": deepcopy(params),
        "selected_stocks": output_df.to_dict(orient="records"),
        "result_df": output_df,
        "diagnostic_df": diag_df,
        "field_range_df": base_range_df,
        "summary_text": summary_text,
        "notes": notes,
        "column_mapping": vars(col_map),
        "base_sample_count": len(day_df),
        "after_sector_sample_count": after_sector_count,
    }


def select_candidates(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = build_custom_params_from_ui(params or {})
    return run_factor_selection(normalized)


def save_factor_result_to_session(session_state: Any, result: Dict[str, Any]) -> None:
    session_state["current_factor_result"] = deepcopy(result)

    if "factor_history" not in session_state or not isinstance(session_state.get("factor_history"), list):
        session_state["factor_history"] = []
    session_state["factor_history"].insert(0, deepcopy(result))

    report_record = {
        "id": f"factor_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "module": "factor",
        "page": "因子选股",
        "title": f"{result.get('candidate_date', '-')} 因子选股结果",
        "summary": result.get("summary_text", ""),
        "created_at": result.get("created_at", _now_str()),
        "payload": deepcopy(result),
    }

    if "report_history" not in session_state or not isinstance(session_state.get("report_history"), list):
        session_state["report_history"] = []
    session_state["report_history"].insert(0, report_record)