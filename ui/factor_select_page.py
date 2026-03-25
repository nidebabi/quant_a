from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from services.data_service import get_candidate_pool, get_data_status, load_features
from services.model_service import score_candidates_with_models, score_manual_candidates_with_models


NUMERIC_BASE_COLUMNS = ["close", "high", "low", "volume", "amount", "turnover", "vol_ratio", "ma5", "ma10", "ma20"]


def _parse_number(text: str) -> Optional[float]:
    value = pd.to_numeric(str(text).strip(), errors="coerce")
    return float(value) if pd.notna(value) else None


def _apply_min_max(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float],
    max_value: Optional[float],
    scale: float = 1.0,
) -> pd.DataFrame:
    if column not in df.columns:
        return df
    series = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if min_value is not None:
        mask &= series >= min_value * scale
    if max_value is not None:
        mask &= series <= max_value * scale
    return df[mask].copy()


def _enrich_trade_slice(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features

    df = features.sort_values(["code", "date"]).copy()
    for column in NUMERIC_BASE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    df["ema12"] = df.groupby("code")["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    df["ema26"] = df.groupby("code")["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd_dif"] = df["ema12"] - df["ema26"]
    df["macd_dea"] = df.groupby("code")["macd_dif"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = (df["macd_dif"] - df["macd_dea"]) * 2

    df["delta_close"] = df.groupby("code")["close"].diff()
    df["up_move"] = df["delta_close"].clip(lower=0)
    df["down_move"] = (-df["delta_close"]).clip(lower=0)
    df["avg_up"] = df.groupby("code")["up_move"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    df["avg_down"] = df.groupby("code")["down_move"].transform(lambda s: s.rolling(14, min_periods=14).mean())
    rs = df["avg_up"] / df["avg_down"].replace(0, pd.NA)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    df["low_9"] = df.groupby("code")["low"].transform(lambda s: s.rolling(9, min_periods=9).min())
    df["high_9"] = df.groupby("code")["high"].transform(lambda s: s.rolling(9, min_periods=9).max())
    rsv = (close - df["low_9"]) / (df["high_9"] - df["low_9"]).replace(0, pd.NA) * 100
    df["kdj_k"] = rsv.groupby(df["code"]).transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    df["kdj_d"] = df.groupby("code")["kdj_k"].transform(lambda s: s.ewm(alpha=1 / 3, adjust=False).mean())
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    df["boll_mid"] = df.groupby("code")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["boll_std"] = df.groupby("code")["close"].transform(lambda s: s.rolling(20, min_periods=20).std())
    df["boll_up"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_low"] = df["boll_mid"] - 2 * df["boll_std"]

    df["high_14"] = df.groupby("code")["high"].transform(lambda s: s.rolling(14, min_periods=14).max())
    df["low_14"] = df.groupby("code")["low"].transform(lambda s: s.rolling(14, min_periods=14).min())
    df["wr_14"] = (df["high_14"] - close) / (df["high_14"] - df["low_14"]).replace(0, pd.NA) * 100

    prev_close = df.groupby("code")["close"].shift(1)
    if "pct_chg" not in df.columns:
        df["pct_chg"] = (close / prev_close - 1) * 100
    else:
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")

    if "amplitude" not in df.columns:
        df["amplitude"] = (high - low) / prev_close.replace(0, pd.NA) * 100
    else:
        df["amplitude"] = pd.to_numeric(df["amplitude"], errors="coerce")

    df["ma_bull_flag"] = (
        (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["close"] > df["ma5"])
    ).fillna(False).astype(int)
    df["boll_mid_flag"] = (df["close"] > df["boll_mid"]).fillna(False).astype(int)

    keep_cols = [col for col in df.columns if col not in {"ema12", "ema26", "delta_close", "up_move", "down_move", "avg_up", "avg_down", "low_9", "high_9", "boll_std", "high_14", "low_14"}]
    return df[keep_cols].copy()


@st.cache_data(show_spinner=False, ttl=300)
def _get_trade_slice(trade_date: str) -> pd.DataFrame:
    features = load_features()
    if features.empty:
        return pd.DataFrame()
    features = _enrich_trade_slice(features)
    target_date = pd.to_datetime(trade_date, errors="coerce")
    if pd.isna(target_date):
        target_date = pd.to_datetime(features["date"], errors="coerce").max()
    return features[pd.to_datetime(features["date"], errors="coerce") == target_date].copy()


def _compact_metric_card(title: str, min_key: str, max_key: str) -> tuple[Optional[float], Optional[float]]:
    st.markdown(f'<div class="mini-filter-title">{title}</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="small")
    with left:
        min_value = _parse_number(st.text_input("最小", key=min_key, value="", label_visibility="collapsed", placeholder="最小"))
    with right:
        max_value = _parse_number(st.text_input("最大", key=max_key, value="", label_visibility="collapsed", placeholder="最大"))
    return min_value, max_value


def _build_manual_selection(trade_date: str, params: dict) -> pd.DataFrame:
    df = _get_trade_slice(trade_date)
    if df.empty:
        return pd.DataFrame()

    for column in [
        "close",
        "pct_chg",
        "turnover",
        "amplitude",
        "volume",
        "amount",
        "vol_ratio",
        "macd_hist",
        "rsi_14",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "wr_14",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

    df = _apply_min_max(df, "close", params["price_min"], params["price_max"])
    df = _apply_min_max(df, "pct_chg", params["pct_min"], params["pct_max"])
    df = _apply_min_max(df, "turnover", params["turnover_min"], params["turnover_max"])
    df = _apply_min_max(df, "amplitude", params["amplitude_min"], params["amplitude_max"])
    df = _apply_min_max(df, "volume", params["volume_min"], params["volume_max"], scale=10000)
    df = _apply_min_max(df, "amount", params["amount_min"], params["amount_max"], scale=1e8)
    df = _apply_min_max(df, "vol_ratio", params["vol_ratio_min"], params["vol_ratio_max"])
    df = _apply_min_max(df, "macd_hist", params["macd_min"], params["macd_max"])
    df = _apply_min_max(df, "rsi_14", params["rsi_min"], params["rsi_max"])
    df = _apply_min_max(df, "kdj_k", params["k_min"], params["k_max"])
    df = _apply_min_max(df, "kdj_d", params["d_min"], params["d_max"])
    df = _apply_min_max(df, "kdj_j", params["j_min"], params["j_max"])
    df = _apply_min_max(df, "wr_14", params["wr_min"], params["wr_max"])

    if params["ma_bull_only"]:
        df = df[df["ma_bull_flag"] == 1].copy()
    if params["boll_mid_only"]:
        df = df[df["boll_mid_flag"] == 1].copy()

    manual_codes = [item.strip() for item in str(params["manual_codes"]).split(",") if item.strip()]
    if manual_codes:
        df = df[df["code"].astype(str).isin(manual_codes)].copy()

    if df.empty:
        return pd.DataFrame()

    result = pd.DataFrame(
        {
            "交易日": pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "代码": df["code"].astype(str),
            "名称": df["name"].astype(str),
            "股价": pd.to_numeric(df["close"], errors="coerce").round(2),
            "涨跌幅(%)": pd.to_numeric(df["pct_chg"], errors="coerce").round(2),
            "换手率(%)": pd.to_numeric(df["turnover"], errors="coerce").round(2),
            "振幅(%)": pd.to_numeric(df["amplitude"], errors="coerce").round(2),
            "成交量(万股)": (pd.to_numeric(df["volume"], errors="coerce") / 10000).round(2),
            "成交额(亿)": (pd.to_numeric(df["amount"], errors="coerce") / 1e8).round(2),
            "量比": pd.to_numeric(df["vol_ratio"], errors="coerce").round(2),
            "MACD柱": pd.to_numeric(df["macd_hist"], errors="coerce").round(3),
            "RSI14": pd.to_numeric(df["rsi_14"], errors="coerce").round(2),
            "K值": pd.to_numeric(df["kdj_k"], errors="coerce").round(2),
            "D值": pd.to_numeric(df["kdj_d"], errors="coerce").round(2),
            "J值": pd.to_numeric(df["kdj_j"], errors="coerce").round(2),
            "WR14": pd.to_numeric(df["wr_14"], errors="coerce").round(2),
        }
    )
    return result.sort_values(["成交额(亿)", "换手率(%)"], ascending=[False, False]).reset_index(drop=True)


def render_factor_select_page() -> None:
    st.markdown('<div class="page-title">候选筛选</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">自动候选池继续保留。手动条件选股改成更紧凑的区间输入，只填一个条件也能直接筛，不需要把所有格子都填满。</div>',
        unsafe_allow_html=True,
    )

    status = get_data_status()
    default_date = status.latest_feature_date

    c1, c2, c3 = st.columns([1.1, 0.8, 1.1], gap="medium")
    with c1:
        trade_date = st.text_input("交易日", value=default_date)
    with c2:
        top_n = st.number_input("候选数量", min_value=5, max_value=100, value=20, step=5)
    with c3:
        sector_filter = st.text_input("板块过滤", value="全部")

    if st.button("生成自动候选池", type="primary"):
        st.session_state["latest_candidate_pool"] = get_candidate_pool(
            trade_date=trade_date,
            top_n=int(top_n),
            sector_filter=sector_filter,
        )
        artifacts = st.session_state.get("model_artifacts")
        if artifacts is not None:
            st.session_state["candidate_predictions"] = score_candidates_with_models(
                artifacts,
                trade_date=trade_date,
                top_n=int(top_n),
                sector_filter=sector_filter,
            )

    candidate_df = st.session_state.get("latest_candidate_pool", pd.DataFrame())
    prediction_df = st.session_state.get("candidate_predictions", pd.DataFrame())

    left, right = st.columns(2)
    with left:
        st.subheader("自动因子候选池")
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            st.dataframe(candidate_df, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无自动候选池结果。")
    with right:
        st.subheader("自动 AI 重排结果")
        if isinstance(prediction_df, pd.DataFrame) and not prediction_df.empty:
            st.dataframe(prediction_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无 AI 重排结果。")

    st.divider()
    st.subheader("手动条件选股")
    st.caption("示例：股价最小填 10、最大填 20，只会筛出股价在 10 到 20 之间的股票。")

    st.markdown('<div class="section-label">行情指标</div>', unsafe_allow_html=True)
    row1 = st.columns(4, gap="small")
    with row1[0]:
        price_min, price_max = _compact_metric_card("股价", "price_min", "price_max")
    with row1[1]:
        pct_min, pct_max = _compact_metric_card("涨跌幅(%)", "pct_min", "pct_max")
    with row1[2]:
        turnover_min, turnover_max = _compact_metric_card("换手率(%)", "turnover_min", "turnover_max")
    with row1[3]:
        amplitude_min, amplitude_max = _compact_metric_card("振幅(%)", "amplitude_min", "amplitude_max")

    row2 = st.columns(3, gap="small")
    with row2[0]:
        volume_min, volume_max = _compact_metric_card("成交量(万股)", "volume_min", "volume_max")
    with row2[1]:
        amount_min, amount_max = _compact_metric_card("成交额(亿)", "amount_min", "amount_max")
    with row2[2]:
        vol_ratio_min, vol_ratio_max = _compact_metric_card("量比", "vol_ratio_min", "vol_ratio_max")

    st.markdown('<div class="section-label">技术指标</div>', unsafe_allow_html=True)
    row3 = st.columns(3, gap="small")
    with row3[0]:
        macd_min, macd_max = _compact_metric_card("MACD柱", "macd_min", "macd_max")
    with row3[1]:
        rsi_min, rsi_max = _compact_metric_card("RSI14", "rsi_min", "rsi_max")
    with row3[2]:
        wr_min, wr_max = _compact_metric_card("WR14", "wr_min", "wr_max")

    row4 = st.columns(3, gap="small")
    with row4[0]:
        k_min, k_max = _compact_metric_card("K值", "k_min", "k_max")
    with row4[1]:
        d_min, d_max = _compact_metric_card("D值", "d_min", "d_max")
    with row4[2]:
        j_min, j_max = _compact_metric_card("J值", "j_min", "j_max")

    row5 = st.columns([1, 1, 2], gap="small")
    with row5[0]:
        ma_bull_only = st.checkbox("仅保留均线多头")
    with row5[1]:
        boll_mid_only = st.checkbox("仅保留站上布林中轨")
    with row5[2]:
        manual_codes = st.text_input(
            "手动股票代码",
            value="",
            help="多个代码用英文逗号分隔，例如：000001,600519,300750",
        )

    if st.button("生成手动条件结果", use_container_width=True):
        try:
            params = {
                "price_min": price_min,
                "price_max": price_max,
                "pct_min": pct_min,
                "pct_max": pct_max,
                "turnover_min": turnover_min,
                "turnover_max": turnover_max,
                "amplitude_min": amplitude_min,
                "amplitude_max": amplitude_max,
                "volume_min": volume_min,
                "volume_max": volume_max,
                "amount_min": amount_min,
                "amount_max": amount_max,
                "vol_ratio_min": vol_ratio_min,
                "vol_ratio_max": vol_ratio_max,
                "macd_min": macd_min,
                "macd_max": macd_max,
                "rsi_min": rsi_min,
                "rsi_max": rsi_max,
                "wr_min": wr_min,
                "wr_max": wr_max,
                "k_min": k_min,
                "k_max": k_max,
                "d_min": d_min,
                "d_max": d_max,
                "j_min": j_min,
                "j_max": j_max,
                "ma_bull_only": ma_bull_only,
                "boll_mid_only": boll_mid_only,
                "manual_codes": manual_codes,
            }
            manual_df = _build_manual_selection(trade_date, params)
            st.session_state["manual_candidate_pool"] = manual_df

            artifacts = st.session_state.get("model_artifacts")
            if artifacts is not None and not manual_df.empty:
                st.session_state["manual_candidate_predictions"] = score_manual_candidates_with_models(
                    artifacts,
                    trade_date=trade_date,
                    selected_codes=manual_df["代码"].astype(str).tolist(),
                )
            else:
                st.session_state["manual_candidate_predictions"] = pd.DataFrame()
        except Exception as exc:
            st.session_state["manual_candidate_pool"] = pd.DataFrame()
            st.session_state["manual_candidate_predictions"] = pd.DataFrame()
            st.error(f"手动选股执行失败：{exc}")

    manual_candidate_df = st.session_state.get("manual_candidate_pool", pd.DataFrame())
    manual_prediction_df = st.session_state.get("manual_candidate_predictions", pd.DataFrame())

    left, right = st.columns(2)
    with left:
        st.subheader("手动条件结果")
        if isinstance(manual_candidate_df, pd.DataFrame) and not manual_candidate_df.empty:
            st.dataframe(manual_candidate_df, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无手动条件结果。")
    with right:
        st.subheader("手动条件 AI 评分")
        if isinstance(manual_prediction_df, pd.DataFrame) and not manual_prediction_df.empty:
            st.dataframe(manual_prediction_df.head(30), use_container_width=True, hide_index=True)
        else:
            st.caption("暂无手动条件 AI 评分。")
