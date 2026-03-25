from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from strategy import select_candidates, get_default_params, prepare_sector_features
from settings import BUY_FEE_RATE, SELL_FEE_RATE

ROOT = Path(__file__).resolve().parent
DATA_FEATURES = ROOT / "data" / "features" / "features_all.parquet"
DATA_LABELS = ROOT / "data" / "labels" / "labels_all.parquet"

st.set_page_config(
    page_title="A股量化选股、回测与决策面板",
    page_icon="📈",
    layout="wide"
)

# =========================
# 数据读取
# =========================
@st.cache_data
def load_data():
    features = pd.read_parquet(DATA_FEATURES)
    labels = pd.read_parquet(DATA_LABELS)
    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    features, sector_ready, sector_source = prepare_sector_features(features)
    return features, labels, sector_ready, sector_source


features, labels, sector_ready, sector_source = load_data()
defaults = get_default_params()

# =========================
# 中文表头映射（原选股/回测模块）
# =========================
COLUMN_MAP = {
    "saved_at": "保存时间",
    "exp_name": "实验名称",
    "mode": "模式",
    "selected_date": "候选股预览日期",
    "start_dt": "开始日期",
    "end_dt": "结束日期",
    "stock_code": "股票代码/名称",
    "active_days": "活跃交易日",
    "top_n": "候选股数量",
    "total_return": "总收益",
    "win_days_ratio": "胜率",
    "avg_daily_ret": "平均收益",
    "max_drawdown": "最大回撤",
    "total_trade_count": "总交易数",
    "cannot_buy_count": "不可买入数",
    "unique_stock_count": "股票数量",

    "date": "日期",
    "entry_date": "进场日期",
    "exit_date": "离场日期",
    "code": "代码",
    "name": "名称",
    "sector_name": "板块",
    "sector_rank": "板块强度排名",
    "sector_hot_3d": "板块近3日热度",
    "sector_breadth": "板块广度",
    "stock_sector_ret_rank": "板块内涨幅排名",
    "stock_sector_amount_rank": "板块内成交额排名",
    "is_sector_leader": "板块前排",
    "is_sector_core": "容量中军",

    "close": "收盘价",
    "score": "评分",
    "ret_3": "近3日涨跌幅",
    "ret_10": "近10日涨跌幅",
    "close_loc": "收盘位置",
    "vol_ratio": "量比",
    "upper_shadow_ratio": "上影线比例",
    "breakout20": "突破20日区间",
    "trade_count": "交易数",
    "avg_ret": "平均收益",
    "win_rate_day": "当日胜率",

    "entry_price": "买入价",
    "exit_price": "卖出价",
    "buy_point": "买点",
    "sell_point": "卖点",

    "next_open": "次日开盘",
    "next_high": "次日最高",
    "next_low": "次日最低",
    "next_close": "次日收盘",

    "trade_result": "结果代码",
    "trade_result_cn": "结果说明",
    "trade_ret": "单笔收益",
    "hold_days": "持有天数",
    "highest_high_used": "止损参考高点",
    "stop_price": "止损价",

    "eval_score": "综合评分",
    "decision": "自动结论",
    "style_tag": "风格标签",

    "stock_total_return": "个股总收益",
    "avg_trade_ret": "平均单笔收益",
}


def zh_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_MAP)


# =========================
# 工具函数（原选股/回测模块）
# =========================
def max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min()) if len(dd) else 0.0


def evaluate_nextday_trade(row, stop_loss_pct, buy_fee, sell_fee):
    next_date = row.get("next_date", pd.NaT)

    if pd.isna(row["next_open"]) or row["tradable_flag"] != 1:
        return pd.Series({
            "entry_date": next_date,
            "exit_date": next_date,
            "entry_price": np.nan,
            "exit_price": np.nan,
            "stop_price": np.nan,
            "buy_point": "次日开盘买入",
            "sell_point": "不可买入",
            "trade_result": "cannot_buy",
            "trade_result_cn": "不可买入",
            "trade_ret": np.nan,
            "hold_days": 1
        })

    buy = float(row["next_open"])
    stop_price = buy * (1 - stop_loss_pct)

    if row["next_low"] <= stop_price:
        exit_price = float(row["next_open"]) if float(row["next_open"]) < stop_price else stop_price
        result = "stop_loss"
        result_cn = "触发固定止损卖出"
        sell_point = "次日盘中触发止损卖出"
    else:
        exit_price = float(row["next_close"])
        result = "close_exit"
        result_cn = "次日收盘卖出"
        sell_point = "次日收盘卖出"

    gross_ret = exit_price / buy - 1
    net_ret = gross_ret - buy_fee - sell_fee

    return pd.Series({
        "entry_date": next_date,
        "exit_date": next_date,
        "entry_price": buy,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "buy_point": "次日开盘买入",
        "sell_point": sell_point,
        "trade_result": result,
        "trade_result_cn": result_cn,
        "trade_ret": net_ret,
        "hold_days": 1
    })


def summarize_stock_returns(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return pd.DataFrame()

    required = {"code", "name", "trade_ret"}
    if not required.issubset(set(trade_df.columns)):
        return pd.DataFrame()

    x = trade_df[trade_df["trade_ret"].notna()].copy()
    if x.empty:
        return pd.DataFrame()

    summary = (
        x.groupby(["code", "name"], dropna=False)
        .agg(
            trade_count=("trade_ret", "count"),
            avg_trade_ret=("trade_ret", "mean"),
            stock_total_return=("trade_ret", lambda s: (1 + s).prod() - 1),
        )
        .reset_index()
        .sort_values("stock_total_return", ascending=False)
        .reset_index(drop=True)
    )
    return summary


@st.cache_data(show_spinner=False)
def run_candidate_backtest(features, labels, params_tuple, top_n, start_dt, end_dt, stop_loss_pct, buy_fee, sell_fee):
    params = dict(params_tuple)

    feat = features[(features["date"] >= start_dt) & (features["date"] <= end_dt)].copy()
    lab = labels[(labels["date"] >= start_dt) & (labels["date"] <= end_dt)].copy()

    all_dates = sorted(feat["date"].dropna().unique())

    daily_results = []
    trade_logs = []

    for d in all_dates:
        feature_day = feat[feat["date"] == d].copy()
        ranked = select_candidates(feature_day, params)

        if ranked.empty:
            continue

        picks = ranked.head(top_n).copy()

        label_day = lab[lab["date"] == d][[
            "date", "code", "name", "next_date", "next_open", "next_high",
            "next_low", "next_close", "next_red_close_flag",
            "next_touch_tp_flag", "tradable_flag"
        ]].copy()

        picks = picks.merge(label_day, on=["date", "code", "name"], how="left")
        eval_df = picks.apply(
            evaluate_nextday_trade,
            axis=1,
            stop_loss_pct=stop_loss_pct,
            buy_fee=buy_fee,
            sell_fee=sell_fee
        )
        picks = pd.concat([picks, eval_df], axis=1)

        valid = picks["trade_ret"].notna()
        if valid.sum() == 0:
            continue

        avg_ret = float(picks.loc[valid, "trade_ret"].mean())
        win_rate_day = float((picks.loc[valid, "trade_ret"] > 0).mean())

        daily_results.append({
            "date": pd.to_datetime(d),
            "trade_count": int(valid.sum()),
            "cannot_buy_count": int((picks["trade_result"] == "cannot_buy").sum()),
            "avg_ret": avg_ret,
            "win_rate_day": win_rate_day
        })

        trade_logs.append(picks)

    if not daily_results:
        return {}, pd.DataFrame(), pd.DataFrame()

    daily_df = pd.DataFrame(daily_results).sort_values("date").reset_index(drop=True)
    daily_df["equity"] = (1 + daily_df["avg_ret"]).cumprod()

    trade_df = pd.concat(trade_logs, ignore_index=True)

    metrics = {
        "mode": "候选股策略回测",
        "active_days": int(len(daily_df)),
        "top_n": int(top_n),
        "total_return": float(daily_df["equity"].iloc[-1] - 1),
        "win_days_ratio": float((daily_df["avg_ret"] > 0).mean()),
        "avg_daily_ret": float(daily_df["avg_ret"].mean()),
        "max_drawdown": max_drawdown(daily_df["equity"]),
        "total_trade_count": int(trade_df["trade_ret"].notna().sum()),
        "cannot_buy_count": int((trade_df["trade_result"] == "cannot_buy").sum())
    }

    return metrics, daily_df, trade_df


@st.cache_data(show_spinner=False)
def run_single_stock_backtest(features, stock_code, start_dt, end_dt, stop_loss_pct, buy_fee, sell_fee):
    keyword = str(stock_code).strip()

    code_mask = features["code"].astype(str) == keyword
    name_mask = features["name"].astype(str) == keyword

    df = features[(code_mask | name_mask) &
                  (features["date"] >= start_dt) &
                  (features["date"] <= end_dt)].copy()

    if df.empty or len(df) < 1:
        return {}, pd.DataFrame(), pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)

    entry_row = df.iloc[0]
    entry_date = entry_row["date"]
    entry_price = float(entry_row["open"])

    trailing_high_prev = entry_price
    trades = []
    exited = False

    for i in range(len(df)):
        day = df.iloc[i]
        stop_price = trailing_high_prev * (1 - stop_loss_pct)

        if float(day["low"]) <= stop_price:
            exit_price = float(day["open"]) if float(day["open"]) < stop_price else stop_price
            gross_ret = exit_price / entry_price - 1
            net_ret = gross_ret - buy_fee - sell_fee

            trades.append({
                "entry_date": entry_date,
                "exit_date": day["date"],
                "code": keyword,
                "name": entry_row["name"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "buy_point": "区间首日开盘买入",
                "sell_point": "盘中触发移动止损卖出",
                "highest_high_used": trailing_high_prev,
                "stop_price": stop_price,
                "hold_days": int((day["date"] - entry_date).days),
                "trade_result": "trailing_stop",
                "trade_result_cn": "触发移动止损卖出",
                "trade_ret": net_ret
            })
            exited = True
            break

        trailing_high_prev = max(trailing_high_prev, float(day["high"]))

    if not exited:
        last_day = df.iloc[-1]
        stop_price = trailing_high_prev * (1 - stop_loss_pct)
        exit_price = float(last_day["close"])
        gross_ret = exit_price / entry_price - 1
        net_ret = gross_ret - buy_fee - sell_fee

        trades.append({
            "entry_date": entry_date,
            "exit_date": last_day["date"],
            "code": keyword,
            "name": entry_row["name"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "buy_point": "区间首日开盘买入",
            "sell_point": "结束日收盘卖出",
            "highest_high_used": trailing_high_prev,
            "stop_price": stop_price,
            "hold_days": int((last_day["date"] - entry_date).days),
            "trade_result": "end_date_exit",
            "trade_result_cn": "到结束日收盘卖出",
            "trade_ret": net_ret
        })

    trade_df = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    trade_df["equity"] = (1 + trade_df["trade_ret"]).cumprod()

    daily_df = trade_df[["exit_date", "trade_ret", "equity"]].copy()
    daily_df = daily_df.rename(columns={"exit_date": "date", "trade_ret": "avg_ret"})
    daily_df["trade_count"] = 1
    daily_df["cannot_buy_count"] = 0
    daily_df["win_rate_day"] = (daily_df["avg_ret"] > 0).astype(float)

    metrics = {
        "mode": "固定股票区间回测",
        "stock_code": keyword,
        "active_days": int(len(df)),
        "total_return": float(trade_df["equity"].iloc[-1] - 1),
        "win_days_ratio": float((trade_df["trade_ret"] > 0).mean()),
        "avg_daily_ret": float(trade_df["trade_ret"].mean()),
        "max_drawdown": max_drawdown(trade_df["equity"]),
        "total_trade_count": int(len(trade_df)),
        "cannot_buy_count": 0
    }

    return metrics, daily_df, trade_df


def calc_eval_score(metrics: dict):
    if not metrics:
        return None, "暂无结果", "未知"

    total_return = float(metrics.get("total_return", 0))
    win_rate = float(metrics.get("win_days_ratio", 0))
    max_dd = abs(float(metrics.get("max_drawdown", 0)))
    trade_count = int(metrics.get("total_trade_count", 0))

    score = 50.0
    score += np.clip(total_return * 100, -30, 30)
    score += np.clip((win_rate - 0.5) * 100, -20, 20)
    score += np.clip((0.20 - max_dd) * 100, -20, 20)

    if trade_count <= 5:
        style_tag = "低频"
    elif trade_count <= 30:
        style_tag = "均衡"
    else:
        style_tag = "高频"

    if score >= 75:
        decision = "建议保留"
    elif score >= 55:
        decision = "继续观察"
    else:
        decision = "建议淘汰"

    return round(float(score), 2), decision, style_tag


def compare_with_baseline(current_metrics: dict, baseline_metrics: dict):
    if not current_metrics or not baseline_metrics:
        return "暂无基线对比"

    tr_delta = current_metrics.get("total_return", 0) - baseline_metrics.get("total_return", 0)
    wr_delta = current_metrics.get("win_days_ratio", 0) - baseline_metrics.get("win_days_ratio", 0)
    dd_delta = current_metrics.get("max_drawdown", 0) - baseline_metrics.get("max_drawdown", 0)

    better_count = 0
    if tr_delta > 0:
        better_count += 1
    if wr_delta > 0:
        better_count += 1
    if dd_delta > 0:
        better_count += 1

    if better_count >= 2:
        return f"优于基线：总收益 {tr_delta:.2%}，胜率 {wr_delta:.2%}，最大回撤 {dd_delta:.2%}"
    return f"弱于或接近基线：总收益 {tr_delta:.2%}，胜率 {wr_delta:.2%}，最大回撤 {dd_delta:.2%}"


# =========================
# 工具函数（次日决策模块）
# =========================
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def calc_risk_score(
    index_change_pct: float,
    breadth_pct: float,
    sentiment_score: float,
    overnight_score: float,
    hot_sector_continuity: float,
    news_risk_score: float
):
    score = 50.0
    score += np.clip(index_change_pct * 5, -15, 15)
    score += np.clip((breadth_pct - 50) * 0.4, -15, 15)
    score += np.clip((sentiment_score - 50) * 0.35, -18, 18)
    score += np.clip((overnight_score - 50) * 0.25, -10, 10)
    score += np.clip((hot_sector_continuity - 50) * 0.25, -10, 10)
    score -= np.clip((news_risk_score - 50) * 0.35, -20, 20)

    score = float(np.clip(score, 0, 100))

    if score >= 65:
        level = "低风险日"
        advice = "允许正常找买点，优先做强板块前排。"
    elif score >= 45:
        level = "中性日"
        advice = "可做，但要收缩出手次数，只做高确定性票。"
    else:
        level = "高风险日"
        advice = "优先控仓或空仓观察，少做追高。"

    return {
        "风险评分": round(score, 2),
        "风险等级": level,
        "风险建议": advice,
    }


def classify_stock_type(y_amp: float, y_turnover: float) -> str:
    if y_amp >= 8 or y_turnover >= 20:
        return "高波动"
    return "趋势中军"


def analyze_holding_row(row: pd.Series, risk_level: str):
    code = str(row.get("代码", ""))
    name = str(row.get("名称", ""))
    shares = safe_int(row.get("持股数", 0))
    cost = safe_float(row.get("成本价", 0))
    y_open = safe_float(row.get("昨开", 0))
    y_high = safe_float(row.get("昨高", 0))
    y_low = safe_float(row.get("昨低", 0))
    y_close = safe_float(row.get("昨收", 0))
    y_turnover = safe_float(row.get("昨换手(%)", 0))
    y_amp = safe_float(row.get("昨振幅(%)", 0))
    is_limit_up = safe_int(row.get("是否涨停", 0))

    stock_type = classify_stock_type(y_amp, y_turnover)

    reduce_price = round(max(y_close * 1.02, cost * 1.03), 2)
    clear_price = round(max(cost * 0.95, y_low * 0.98), 2)

    hold_conditions = []
    hold_conditions.append(f"不跌破昨日低点附近（{y_low:.2f}）")
    hold_conditions.append(f"若高开后能站稳昨日收盘（{y_close:.2f}）之上，可继续持有")
    if is_limit_up == 1:
        hold_conditions.append("若出现明显炸板或放量回落，优先兑现")

    if risk_level == "高风险日":
        reduce_price = round(max(y_close * 1.01, cost * 1.01), 2)
        clear_price = round(max(cost * 0.97, y_low * 0.99), 2)

    if y_close >= cost * 1.08:
        action_hint = "已有较厚浮盈，优先考虑冲高减仓。"
    elif y_close < cost:
        action_hint = "仍在成本下方，优先盯防低开和失守关键位。"
    else:
        action_hint = "小幅浮盈，先看开盘承接，强则留，弱则减。"

    return {
        "代码": code,
        "名称": name,
        "持股数": shares,
        "成本价": round(cost, 2),
        "昨日开盘": round(y_open, 2),
        "昨日最高": round(y_high, 2),
        "昨日最低": round(y_low, 2),
        "昨日收盘": round(y_close, 2),
        "票型判定": stock_type,
        "减仓点": reduce_price,
        "清仓点": clear_price,
        "继续持有条件": "；".join(hold_conditions),
        "处理建议": action_hint,
    }


def score_candidate_row(row: pd.Series):
    code = str(row.get("代码", ""))
    name = str(row.get("名称", ""))
    sector = str(row.get("板块", ""))
    sector_rank = safe_int(row.get("板块排名", 99))
    sector_continuity = safe_float(row.get("板块持续性", 0))
    is_front_row = safe_int(row.get("前排龙头", 0))
    is_core = safe_int(row.get("容量中军", 0))
    strong_clean = safe_float(row.get("强得干净", 0))
    buy_comfort = safe_float(row.get("买点舒适度", 0))
    open_below_expect_risk = safe_float(row.get("低于预期风险", 100))
    linkage_score = safe_float(row.get("联动分", 0))
    y_turnover = safe_float(row.get("换手率(%)", 0))
    y_amount_yi = safe_float(row.get("成交额(亿)", 0))
    y_amp = safe_float(row.get("振幅(%)", 0))
    y_close = safe_float(row.get("昨收", 0))
    y_low = safe_float(row.get("昨低", 0))

    reject_reason = []
    if sector_rank > 2:
        reject_reason.append("不是最强或次强板块")
    if is_front_row != 1 and is_core != 1:
        reject_reason.append("不是板块前排或容量中军")
    if strong_clean < 60:
        reject_reason.append("昨天强度不够干净")
    if open_below_expect_risk >= 60:
        reject_reason.append("开盘弱于预期风险偏大")
    if linkage_score < 50:
        reject_reason.append("板块联动偏弱")

    filtered_out = 1 if reject_reason else 0

    base_score = 0.0
    base_score += max(0, 30 - (sector_rank - 1) * 12)
    base_score += sector_continuity * 0.20
    base_score += strong_clean * 0.22
    base_score += buy_comfort * 0.18
    base_score += linkage_score * 0.15
    base_score += min(y_amount_yi, 50) * 0.20
    base_score += min(y_turnover, 30) * 0.30
    base_score -= open_below_expect_risk * 0.25
    if is_front_row == 1:
        base_score += 8
    if is_core == 1:
        base_score += 6
    if y_amp > 12:
        base_score -= 4

    score = round(float(base_score), 2)

    buy_point = f"观察 {y_close:.2f} 附近承接，优先低吸或分时回踩不破昨日收盘再考虑"
    stop_loss = round(max(y_low * 0.98, y_close * 0.95), 2)
    give_up = "；".join([
        "开盘明显弱于预期",
        "板块不联动",
        "分时跌破昨日低点且无承接"
    ])

    return {
        "代码": code,
        "名称": name,
        "板块": sector,
        "板块排名": sector_rank,
        "前排龙头": is_front_row,
        "容量中军": is_core,
        "候选评分": score,
        "是否剔除": filtered_out,
        "剔除原因": "；".join(reject_reason),
        "买点": buy_point,
        "止损点": stop_loss,
        "放弃条件": give_up,
    }


def make_market_conclusion(
    risk_info: dict,
    index_change_pct: float,
    breadth_pct: float,
    sentiment_score: float,
    overnight_score: float,
    hot_sector_continuity: float,
):
    parts = []
    parts.append(f"风险等级：{risk_info['风险等级']}（评分 {risk_info['风险评分']}）")
    parts.append(f"昨日指数表现 {index_change_pct:.2f}%")
    parts.append(f"市场广度 {breadth_pct:.1f}%")
    parts.append(f"短线情绪 {sentiment_score:.0f}/100")
    parts.append(f"隔夜环境 {overnight_score:.0f}/100")
    parts.append(f"热点持续性 {hot_sector_continuity:.0f}/100")
    parts.append(risk_info["风险建议"])
    return "｜".join(parts)


def build_final_plan(
    risk_info: dict,
    holdings_result: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    available_cash: float,
):
    parts = []

    if risk_info["风险等级"] == "高风险日":
        parts.append("明日优先防守，原则上以处理持仓为主，新开仓从严。")
    elif risk_info["风险等级"] == "中性日":
        parts.append("明日可以出手，但只做高确定性票，控制出手数量。")
    else:
        parts.append("明日可以正常寻找买点，但仍优先做最强方向前排。")

    if not holdings_result.empty:
        parts.append("持仓先按预设减仓点/清仓点执行，不符合继续持有条件就先兑现。")

    if available_cash <= 0:
        parts.append("当前可用资金不足，优先做持仓管理，不建议新增仓位。")
    else:
        if selected_candidates.empty:
            parts.append("当前没有通过筛选的候选股，建议以观察为主。")
        else:
            top_names = "、".join(selected_candidates["名称"].head(3).tolist())
            parts.append(f"若允许开仓，优先观察：{top_names}。")
            parts.append("开盘弱于预期、板块不联动、不是当日最强时，直接放弃。")

    return " ".join(parts)


def df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    return df.to_markdown(index=False)


# =========================
# 默认数据（次日决策模块）
# =========================
DEFAULT_HOLDINGS_CN = pd.DataFrame([
    {
        "代码": "603817",
        "名称": "海峡环保",
        "持股数": 1000,
        "成本价": 7.65,
        "昨开": 7.70,
        "昨高": 8.19,
        "昨低": 7.69,
        "昨收": 8.18,
        "昨换手(%)": 11.75,
        "昨振幅(%)": 6.50,
        "是否涨停": 0,
        "备注": ""
    }
])

DEFAULT_CANDIDATES_CN = pd.DataFrame([
    {
        "代码": "000001",
        "名称": "示例A",
        "板块": "示例板块",
        "板块排名": 1,
        "板块持续性": 80,
        "前排龙头": 1,
        "容量中军": 1,
        "强得干净": 85,
        "买点舒适度": 75,
        "低于预期风险": 25,
        "联动分": 80,
        "换手率(%)": 12.5,
        "成交额(亿)": 18.2,
        "振幅(%)": 6.3,
        "昨收": 12.80,
        "昨高": 13.10,
        "昨低": 12.20
    }
])

# =========================
# 会话状态
# =========================
if "bt_result" not in st.session_state:
    st.session_state.bt_result = {
        "metrics": {},
        "daily_df": pd.DataFrame(),
        "trade_df": pd.DataFrame()
    }

if "exp_records" not in st.session_state:
    st.session_state.exp_records = []

if "baseline_metrics" not in st.session_state:
    st.session_state.baseline_metrics = None

if "decision_holdings_df" not in st.session_state:
    st.session_state.decision_holdings_df = DEFAULT_HOLDINGS_CN.copy()

if "decision_candidates_df" not in st.session_state:
    st.session_state.decision_candidates_df = DEFAULT_CANDIDATES_CN.copy()

if "decision_result" not in st.session_state:
    st.session_state.decision_result = None


# =========================
# 页面标题
# =========================
st.title("A股量化选股、回测与决策面板")

if sector_ready:
    st.caption(f"当前板块层来源：{sector_source}")
else:
    st.caption("当前板块层未启用或数据源不可用，不影响你先搭整体框架。")

# =========================
# 左侧参数区（保留给选股/回测）
# =========================
st.sidebar.title("控制面板")

with st.sidebar.expander("候选股参数", expanded=False):
    top_n = st.number_input("候选股数量 TOP_N", min_value=1, max_value=1000, value=20, step=1)

    use_amount = st.checkbox("启用成交额", value=defaults["use_amount"])
    if use_amount:
        c1, c2 = st.columns(2)
        with c1:
            amount_min_yi = st.number_input("成交额下限(亿)", value=float(defaults["amount_min"] / 1e8), step=0.5, format="%.2f")
        with c2:
            amount_max_yi = st.number_input("成交额上限(亿)", value=999999.0, step=0.5, format="%.2f")
    else:
        amount_min_yi, amount_max_yi = 0.0, 1e12

    use_turnover = st.checkbox("启用换手率", value=defaults["use_turnover"])
    if use_turnover:
        c1, c2 = st.columns(2)
        with c1:
            turnover_min = st.number_input("换手率下限", value=float(defaults["turnover_min"]), step=0.1, format="%.4f")
        with c2:
            turnover_max = st.number_input("换手率上限", value=float(defaults["turnover_max"]), step=0.1, format="%.4f")
    else:
        turnover_min, turnover_max = -1e12, 1e12

    use_ret3 = st.checkbox("启用近3日涨跌幅", value=defaults["use_ret3"])
    if use_ret3:
        c1, c2 = st.columns(2)
        with c1:
            ret3_min = st.number_input("近3日下限", value=float(defaults["ret3_min"]), step=0.01, format="%.4f")
        with c2:
            ret3_max = st.number_input("近3日上限", value=float(defaults["ret3_max"]), step=0.01, format="%.4f")
    else:
        ret3_min, ret3_max = -1e12, 1e12

    use_ret10 = st.checkbox("启用近10日涨跌幅", value=defaults["use_ret10"])
    if use_ret10:
        c1, c2 = st.columns(2)
        with c1:
            ret10_min = st.number_input("近10日下限", value=float(defaults["ret10_min"]), step=0.01, format="%.4f")
        with c2:
            ret10_max = st.number_input("近10日上限", value=float(defaults["ret10_max"]), step=0.01, format="%.4f")
    else:
        ret10_min, ret10_max = -1e12, 1e12

    use_close_loc = st.checkbox("启用收盘位置", value=defaults["use_close_loc"])
    if use_close_loc:
        close_loc_min = st.number_input("收盘位置下限", value=float(defaults["close_loc_min"]), step=0.01, format="%.4f")
    else:
        close_loc_min = -1e12

    use_vol_ratio = st.checkbox("启用量比", value=defaults["use_vol_ratio"])
    if use_vol_ratio:
        c1, c2 = st.columns(2)
        with c1:
            vol_ratio_min = st.number_input("量比下限", value=float(defaults["vol_ratio_min"]), step=0.1, format="%.4f")
        with c2:
            vol_ratio_max = st.number_input("量比上限", value=float(defaults["vol_ratio_max"]), step=0.1, format="%.4f")
    else:
        vol_ratio_min, vol_ratio_max = -1e12, 1e12

    use_upper_shadow = st.checkbox("启用上影线", value=defaults["use_upper_shadow"])
    if use_upper_shadow:
        upper_shadow_max = st.number_input("上影线最大值", value=float(defaults["upper_shadow_max"]), step=0.01, format="%.4f")
    else:
        upper_shadow_max = 1e12

    use_breakout = st.checkbox("启用突破区间", value=defaults["use_breakout"])
    if use_breakout:
        c1, c2 = st.columns(2)
        with c1:
            breakout_min = st.number_input("突破下限", value=float(defaults["breakout_min"]), step=0.01, format="%.4f")
        with c2:
            breakout_max = st.number_input("突破上限", value=float(defaults["breakout_max"]), step=0.01, format="%.4f")
    else:
        breakout_min, breakout_max = -1e12, 1e12

    st.markdown("---")
    st.markdown("### 板块/龙头层")

    use_sector_hot = st.checkbox("启用强板块过滤", value=False, disabled=not sector_ready)
    if use_sector_hot:
        c1, c2, c3 = st.columns(3)
        with c1:
            sector_rank_max = st.number_input("板块排名前N", min_value=1, max_value=20, value=2, step=1)
        with c2:
            sector_hot_3d_min = st.number_input("板块近3日热度下限", min_value=0.0, max_value=1.0, value=0.55, step=0.01, format="%.4f")
        with c3:
            sector_breadth_min = st.number_input("板块广度下限", min_value=0.0, max_value=1.0, value=0.35, step=0.01, format="%.4f")
    else:
        sector_rank_max, sector_hot_3d_min, sector_breadth_min = 9999, -1e12, -1e12

    use_sector_position = st.checkbox("启用板块内地位过滤", value=False, disabled=not sector_ready)
    if use_sector_position:
        c1, c2 = st.columns(2)
        with c1:
            stock_sector_ret_rank_max = st.number_input("板块内涨幅排名前N", min_value=1, max_value=20, value=5, step=1)
        with c2:
            stock_sector_amount_rank_max = st.number_input("板块内成交额排名前N", min_value=1, max_value=20, value=3, step=1)
        require_leader_or_core = st.checkbox("前排或中军即可", value=True)
    else:
        stock_sector_ret_rank_max, stock_sector_amount_rank_max, require_leader_or_core = 9999, 9999, False

with st.sidebar.expander("交易参数", expanded=False):
    stop_loss_pct = st.number_input("止损比例", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.4f")
    buy_fee = st.number_input("买入费率", min_value=0.0, value=float(BUY_FEE_RATE), step=0.0001, format="%.4f")
    sell_fee = st.number_input("卖出费率", min_value=0.0, value=float(SELL_FEE_RATE), step=0.0001, format="%.4f")

with st.sidebar.expander("日期与模式", expanded=False):
    available_dates = sorted(pd.to_datetime(features["date"]).dropna().unique())
    min_date = pd.to_datetime(min(available_dates)).date()
    max_date = pd.to_datetime(max(available_dates)).date()

    selected_date = st.selectbox(
        "候选股预览日期",
        available_dates,
        index=len(available_dates) - 1,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m-%d")
    )

    backtest_mode = st.radio("回测模式", ["候选股策略回测", "固定股票区间回测"])
    start_dt = pd.to_datetime(st.date_input("开始日期", min_date))
    end_dt = pd.to_datetime(st.date_input("结束日期", max_date))

    stock_code = ""
    if backtest_mode == "固定股票区间回测":
        stock_code = st.text_input("股票代码/名称", "603817").strip()

with st.sidebar.expander("执行控制", expanded=True):
    enable_backtest = st.toggle("启用回测", value=False)
    exp_name = st.text_input("实验名称", value="")
    run_bt = st.button("开始运行回测")
    st.caption("左侧这里只控制选股/回测模块。次日决策模块在页面内单独输入。")

params = {
    "use_trend": True,
    "use_amount": use_amount,
    "amount_min": amount_min_yi * 1e8,
    "amount_max": amount_max_yi * 1e8,
    "use_turnover": use_turnover,
    "turnover_min": turnover_min,
    "turnover_max": turnover_max,
    "use_ret3": use_ret3,
    "ret3_min": ret3_min,
    "ret3_max": ret3_max,
    "use_ret10": use_ret10,
    "ret10_min": ret10_min,
    "ret10_max": ret10_max,
    "use_close_loc": use_close_loc,
    "close_loc_min": close_loc_min,
    "use_vol_ratio": use_vol_ratio,
    "vol_ratio_min": vol_ratio_min,
    "vol_ratio_max": vol_ratio_max,
    "use_upper_shadow": use_upper_shadow,
    "upper_shadow_max": upper_shadow_max,
    "use_breakout": use_breakout,
    "breakout_min": breakout_min,
    "breakout_max": breakout_max,
    "use_sector_hot": bool(use_sector_hot and sector_ready),
    "sector_rank_max": int(sector_rank_max),
    "sector_hot_3d_min": float(sector_hot_3d_min),
    "sector_breadth_min": float(sector_breadth_min),
    "use_sector_position": bool(use_sector_position and sector_ready),
    "stock_sector_ret_rank_max": int(stock_sector_ret_rank_max),
    "stock_sector_amount_rank_max": int(stock_sector_amount_rank_max),
    "require_leader_or_core": bool(require_leader_or_core),
}
params_tuple = tuple(sorted(params.items()))

# =========================
# 候选股预览
# =========================
day_df = features[features["date"] == pd.to_datetime(selected_date)].copy()
picks_df = select_candidates(day_df, params).head(top_n)

# =========================
# 执行回测
# =========================
if enable_backtest and run_bt:
    if backtest_mode == "候选股策略回测":
        metrics, daily_df, trade_df = run_candidate_backtest(
            features, labels, params_tuple, top_n, start_dt, end_dt,
            stop_loss_pct, buy_fee, sell_fee
        )
    else:
        metrics, daily_df, trade_df = run_single_stock_backtest(
            features, stock_code, start_dt, end_dt,
            stop_loss_pct, buy_fee, sell_fee
        )

    st.session_state.bt_result = {
        "metrics": metrics,
        "daily_df": daily_df,
        "trade_df": trade_df
    }

metrics = st.session_state.bt_result["metrics"]
daily_df = st.session_state.bt_result["daily_df"]
trade_df = st.session_state.bt_result["trade_df"]

eval_score, decision, style_tag = calc_eval_score(metrics)
baseline_text = compare_with_baseline(metrics, st.session_state.baseline_metrics)
stock_summary_df = summarize_stock_returns(trade_df)

# =========================
# 顶部指标
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("活跃交易日", metrics.get("active_days", "-"))
col2.metric("总收益", f"{metrics.get('total_return', 0):.2%}" if metrics else "-")
col3.metric("胜率", f"{metrics.get('win_days_ratio', 0):.2%}" if metrics else "-")
col4.metric("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}" if metrics else "-")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["候选股", "回测", "交易明细", "实验记录", "次日决策"])

# =========================
# TAB1 候选股
# =========================
with tab1:
    st.subheader(f"候选股预览 - {pd.to_datetime(selected_date).strftime('%Y-%m-%d')}")
    st.caption("这里只展示预览日期当天盘后筛出来的候选股，不代表整个回测区间只用这一天的票。")
    if picks_df.empty:
        st.warning("当前参数下没有候选股")
    else:
        show_cols = [c for c in [
            "date", "code", "name", "sector_name",
            "sector_rank", "sector_hot_3d", "sector_breadth",
            "stock_sector_ret_rank", "stock_sector_amount_rank",
            "is_sector_leader", "is_sector_core",
            "close", "score", "ret_3", "ret_10", "close_loc",
            "vol_ratio", "upper_shadow_ratio", "breakout20"
        ] if c in picks_df.columns]
        st.dataframe(zh_df(picks_df[show_cols]), use_container_width=True, hide_index=True)

        top10 = picks_df.head(10).copy()
        fig = px.bar(top10, x="name", y="score", title="候选股 Top10 评分")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB2 回测
# =========================
with tab2:
    st.subheader("回测")
    st.caption("候选股策略回测：会在开始日期到结束日期之间，每天各自筛候选股，再汇总整段结果。")
    if not enable_backtest:
        st.info("左侧先打开“启用回测”，再点“开始运行回测”。")
    elif daily_df.empty:
        st.warning("当前还没有回测结果，请点“开始运行回测”。")
    else:
        if "mode" in metrics:
            st.caption(f"当前模式：{metrics['mode']}")

        a1, a2, a3 = st.columns(3)
        a1.metric("综合评分", eval_score if eval_score is not None else "-")
        a2.metric("自动结论", decision)
        a3.metric("风格标签", style_tag)

        st.info(baseline_text)

        fig = px.line(daily_df, x="date", y="equity", title="收益曲线", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        show_cols = [c for c in ["date", "trade_count", "cannot_buy_count", "avg_ret", "win_rate_day"] if c in daily_df.columns]
        st.dataframe(zh_df(daily_df[show_cols]), use_container_width=True, hide_index=True)

        if len(stock_summary_df) > 1:
            best_row = stock_summary_df.iloc[0]
            worst_row = stock_summary_df.iloc[-1]

            st.markdown("### 个股收益概览")
            b1, b2 = st.columns(2)
            b1.metric(
                f"收益最高：{best_row['name']}（{best_row['code']}）",
                f"{best_row['stock_total_return']:.2%}"
            )
            b2.metric(
                f"收益最低：{worst_row['name']}（{worst_row['code']}）",
                f"{worst_row['stock_total_return']:.2%}"
            )

            summary_cols = [c for c in ["code", "name", "trade_count", "avg_trade_ret", "stock_total_return"] if c in stock_summary_df.columns]
            st.dataframe(
                zh_df(stock_summary_df[summary_cols]),
                use_container_width=True,
                hide_index=True
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("设为当前基线"):
                st.session_state.baseline_metrics = metrics.copy()
                st.success("已设为基线")

        with c2:
            if st.button("保存当前结果到会话记录"):
                if metrics:
                    row = {
                        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "exp_name": exp_name.strip() if exp_name.strip() else "未命名实验",
                        "mode": metrics.get("mode", ""),
                        "selected_date": str(pd.to_datetime(selected_date).date()),
                        "start_dt": str(pd.to_datetime(start_dt).date()),
                        "end_dt": str(pd.to_datetime(end_dt).date()),
                        "stock_code": stock_code,
                        "active_days": metrics.get("active_days", None),
                        "top_n": metrics.get("top_n", top_n),
                        "total_return": metrics.get("total_return", None),
                        "win_days_ratio": metrics.get("win_days_ratio", None),
                        "avg_daily_ret": metrics.get("avg_daily_ret", None),
                        "max_drawdown": metrics.get("max_drawdown", None),
                        "total_trade_count": metrics.get("total_trade_count", None),
                        "cannot_buy_count": metrics.get("cannot_buy_count", None),
                        "eval_score": eval_score,
                        "decision": decision,
                        "style_tag": style_tag,
                        "unique_stock_count": int(stock_summary_df["code"].nunique()) if not stock_summary_df.empty else 0,
                    }
                    st.session_state.exp_records.append(row)
                    st.success("已保存到当前会话记录")
                else:
                    st.warning("当前没有可保存的回测结果")

        with c3:
            if st.button("清空当前会话记录"):
                st.session_state.exp_records = []
                st.success("已清空")

# =========================
# TAB3 交易明细
# =========================
with tab3:
    st.subheader("交易明细")
    if not enable_backtest:
        st.info("左侧先打开“启用回测”，再点“开始运行回测”。")
    elif trade_df.empty:
        st.warning("当前还没有交易明细，请点“开始运行回测”。")
    else:
        left, right = st.columns([1, 2])

        with left:
            if "trade_result_cn" in trade_df.columns:
                options = ["全部"] + sorted(trade_df["trade_result_cn"].dropna().unique().tolist())
                selected = st.selectbox("按结果筛选", options)
                if selected != "全部":
                    trade_df = trade_df[trade_df["trade_result_cn"] == selected].copy()

        with right:
            keyword = st.text_input("按股票代码或名称搜索", "")
            if keyword:
                code_mask = trade_df["code"].astype(str).str.contains(keyword, na=False) if "code" in trade_df.columns else False
                name_mask = trade_df["name"].astype(str).str.contains(keyword, na=False) if "name" in trade_df.columns else False
                trade_df = trade_df[code_mask | name_mask].copy()

        show_cols = [c for c in [
            "entry_date", "exit_date",
            "date", "code", "name", "sector_name", "score",
            "buy_point", "sell_point",
            "entry_price", "exit_price",
            "next_open", "next_high", "next_low", "next_close",
            "highest_high_used", "stop_price",
            "trade_result_cn", "trade_result", "trade_ret", "hold_days"
        ] if c in trade_df.columns]

        st.dataframe(zh_df(trade_df[show_cols]), use_container_width=True, hide_index=True)

# =========================
# TAB4 实验记录
# =========================
with tab4:
    st.subheader("实验记录（仅当前会话）")

    if not st.session_state.exp_records:
        st.info("当前会话还没有保存的实验记录。")
    else:
        exp_df = pd.DataFrame(st.session_state.exp_records)

        show_cols = [c for c in [
            "saved_at", "exp_name", "mode", "selected_date",
            "start_dt", "end_dt", "stock_code", "active_days", "top_n",
            "total_return", "win_days_ratio", "avg_daily_ret",
            "max_drawdown", "total_trade_count", "cannot_buy_count",
            "unique_stock_count", "eval_score", "decision", "style_tag"
        ] if c in exp_df.columns]

        sort_by = st.selectbox("排序字段", ["eval_score", "total_return", "win_days_ratio", "max_drawdown", "saved_at"])
        ascending = st.checkbox("升序", value=False)

        if sort_by in exp_df.columns:
            exp_df = exp_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        st.dataframe(zh_df(exp_df[show_cols]), use_container_width=True, hide_index=True)

# =========================
# TAB5 次日决策（并到主面板）
# =========================
with tab5:
    st.subheader("次日操作决策模块")
    st.caption("这一版先把框架并到主面板里。当前采用手工输入 + 规则判断，后面再接自动数据、AI 和走势推演。")

    dtab1, dtab2 = st.tabs(["输入区", "结果区"])

    with dtab1:
        left, right = st.columns([1, 2])

        with left:
            st.markdown("### 基础信息")
            decision_date = st.date_input("日期", value=date.today(), key="decision_date_tab")
            decision_timepoint = st.selectbox("决策时点", ["盘前", "盘中", "盘后"], index=0, key="decision_timepoint_tab")
            total_capital = st.number_input("账户总资金", min_value=0.0, value=100000.0, step=1000.0, key="decision_total_capital")
            available_cash = st.number_input("账户可用资金", min_value=0.0, value=30000.0, step=1000.0, key="decision_available_cash")

            st.markdown("### 市场环境输入")
            index_change_pct = st.number_input("上一个交易日指数涨跌幅(%)", value=0.50, step=0.10, key="decision_index_change")
            breadth_pct = st.slider("上涨家数占比(%)", min_value=0.0, max_value=100.0, value=58.0, step=1.0, key="decision_breadth")
            sentiment_score = st.slider("短线情绪评分", min_value=0, max_value=100, value=60, step=1, key="decision_sentiment")
            overnight_score = st.slider("隔夜外围/消息评分", min_value=0, max_value=100, value=55, step=1, key="decision_overnight")
            hot_sector_continuity = st.slider("热点持续性评分", min_value=0, max_value=100, value=65, step=1, key="decision_sector_continuity")
            news_risk_score = st.slider("突发消息风险评分", min_value=0, max_value=100, value=40, step=1, key="decision_news_risk")

            generate_decision = st.button("生成决策", key="generate_decision_btn", use_container_width=True)

        with right:
            st.markdown("### 当前持仓输入")
            st.session_state.decision_holdings_df = st.data_editor(
                st.session_state.decision_holdings_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="decision_holdings_editor"
            )

            st.markdown("### 候选股输入")
            st.session_state.decision_candidates_df = st.data_editor(
                st.session_state.decision_candidates_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="decision_candidates_editor"
            )

            st.info("这里全部是中文表头。后面你可以把选股结果、持仓、新闻、板块、模型输出自动接进来。")

        if generate_decision:
            risk_info = calc_risk_score(
                index_change_pct=index_change_pct,
                breadth_pct=breadth_pct,
                sentiment_score=sentiment_score,
                overnight_score=overnight_score,
                hot_sector_continuity=hot_sector_continuity,
                news_risk_score=news_risk_score,
            )

            holdings_df_input = st.session_state.decision_holdings_df.copy()
            candidates_df_input = st.session_state.decision_candidates_df.copy()

            holdings_result = pd.DataFrame()
            if not holdings_df_input.empty:
                holdings_result = pd.DataFrame(
                    [analyze_holding_row(row, risk_info["风险等级"]) for _, row in holdings_df_input.iterrows()]
                )

            candidates_scored = pd.DataFrame()
            selected_candidates = pd.DataFrame()
            if not candidates_df_input.empty:
                candidates_scored = pd.DataFrame(
                    [score_candidate_row(row) for _, row in candidates_df_input.iterrows()]
                )
                selected_candidates = (
                    candidates_scored[candidates_scored["是否剔除"] == 0]
                    .sort_values("候选评分", ascending=False)
                    .reset_index(drop=True)
                )

            market_conclusion = make_market_conclusion(
                risk_info=risk_info,
                index_change_pct=index_change_pct,
                breadth_pct=breadth_pct,
                sentiment_score=sentiment_score,
                overnight_score=overnight_score,
                hot_sector_continuity=hot_sector_continuity,
            )

            final_plan = build_final_plan(
                risk_info=risk_info,
                holdings_result=holdings_result,
                selected_candidates=selected_candidates,
                available_cash=available_cash,
            )

            st.session_state.decision_result = {
                "基础信息": {
                    "日期": str(decision_date),
                    "决策时点": decision_timepoint,
                    "账户总资金": total_capital,
                    "账户可用资金": available_cash,
                },
                "风险信息": risk_info,
                "市场结论": market_conclusion,
                "持仓结果": holdings_result,
                "候选股评分": candidates_scored,
                "优先候选股": selected_candidates,
                "最终方案": final_plan,
            }
            st.success("决策结果已生成，请切到“结果区”查看。")

    with dtab2:
        result = st.session_state.decision_result

        if result is None:
            st.warning("先在“输入区”里点击“生成决策”。")
        else:
            risk_info = result["风险信息"]
            holdings_result = result["持仓结果"]
            candidates_scored = result["候选股评分"]
            selected_candidates = result["优先候选股"]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("风险等级", risk_info["风险等级"])
            k2.metric("风险评分", risk_info["风险评分"])
            k3.metric("候选股通过数", int(len(selected_candidates)))
            k4.metric("是否建议空仓观察", "是" if risk_info["风险等级"] == "高风险日" else "否")

            r1, r2, r3, r4 = st.tabs(["市场结论", "持仓决策", "候选股决策", "最终执行方案"])

            with r1:
                st.subheader("市场环境结论")
                st.info(result["市场结论"])

            with r2:
                st.subheader("持仓处理建议")
                if holdings_result.empty:
                    st.info("当前没有持仓。")
                else:
                    st.dataframe(holdings_result, use_container_width=True, hide_index=True)

            with r3:
                st.subheader("候选股评分结果")
                if candidates_scored.empty:
                    st.info("当前没有候选股输入。")
                else:
                    st.markdown("#### 全部候选股")
                    st.dataframe(candidates_scored, use_container_width=True, hide_index=True)

                    st.markdown("#### 最终优先候选股（1-3只）")
                    if selected_candidates.empty:
                        st.warning("当前没有通过规则筛选的候选股。")
                    else:
                        top_selected = selected_candidates.head(3).copy()
                        st.dataframe(top_selected, use_container_width=True, hide_index=True)

            with r4:
                st.subheader("最终执行方案")
                st.success(result["最终方案"])

                report_md = f"""
# 次日操作决策报告

## 基础信息
- 日期：{result['基础信息']['日期']}
- 决策时点：{result['基础信息']['决策时点']}
- 账户总资金：{result['基础信息']['账户总资金']}
- 账户可用资金：{result['基础信息']['账户可用资金']}

## 市场结论
{result['市场结论']}

## 持仓决策
{df_to_md_table(holdings_result)}

## 候选股评分
{df_to_md_table(candidates_scored)}

## 最终优先候选股
{df_to_md_table(selected_candidates.head(3) if not selected_candidates.empty else pd.DataFrame())}

## 最终执行方案
{result['最终方案']}
"""
                st.download_button(
                    label="下载决策报告(MD)",
                    data=report_md,
                    file_name=f"decision_report_{result['基础信息']['日期']}.md",
                    mime="text/markdown"
                )

st.divider()
st.caption("close_exit = 没触发止损，按收盘卖出。")
st.caption("stop_loss = 触发固定止损卖出。")
st.caption("trailing_stop = 触发移动止损卖出。")
st.caption("end_date_exit = 到回测结束日，按收盘卖出。")
st.caption("这一版已经把“次日操作决策模块”并到主面板里了，后面再继续补“走势推演模块”即可。")