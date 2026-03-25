from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from strategy import select_candidates, get_default_params, prepare_sector_features
from settings import BUY_FEE_RATE, SELL_FEE_RATE


ROOT = Path(__file__).resolve().parent.parent
DATA_FEATURES = ROOT / "data" / "features" / "features_all.parquet"
DATA_LABELS = ROOT / "data" / "labels" / "labels_all.parquet"


COLUMN_MAP = {
    "saved_at": "保存时间",
    "exp_name": "实验名称",
    "mode": "模式",
    "selected_date": "候选股预览日期",
    "start_dt": "开始日期",
    "end_dt": "结束日期",
    "stock_code": "股票代码/名称",
    "sector_filter": "板块筛选",
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
    if df is None or df.empty:
        return df
    return df.rename(columns=COLUMN_MAP)


@st.cache_data(show_spinner=False)
def load_backtest_data():
    if not DATA_FEATURES.exists() or not DATA_LABELS.exists():
        return pd.DataFrame(), pd.DataFrame(), False, "未找到回测数据文件", {}

    features = pd.read_parquet(DATA_FEATURES)
    labels = pd.read_parquet(DATA_LABELS)

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    features, sector_ready, sector_source = prepare_sector_features(features)
    defaults = get_default_params()

    return features, labels, sector_ready, sector_source, defaults


def get_backtest_page_meta():
    features, labels, sector_ready, sector_source, defaults = load_backtest_data()

    if features.empty:
        return {
            "available_dates": [],
            "min_date": None,
            "max_date": None,
            "sector_ready": False,
            "sector_source": sector_source,
            "defaults": defaults if isinstance(defaults, dict) else {},
            "buy_fee": float(BUY_FEE_RATE),
            "sell_fee": float(SELL_FEE_RATE),
        }

    available_dates = sorted(pd.to_datetime(features["date"]).dropna().unique())
    min_date = pd.to_datetime(min(available_dates)).date()
    max_date = pd.to_datetime(max(available_dates)).date()

    return {
        "available_dates": [pd.to_datetime(x) for x in available_dates],
        "min_date": min_date,
        "max_date": max_date,
        "sector_ready": sector_ready,
        "sector_source": sector_source,
        "defaults": defaults if isinstance(defaults, dict) else {},
        "buy_fee": float(BUY_FEE_RATE),
        "sell_fee": float(SELL_FEE_RATE),
    }


def max_drawdown(equity):
    if equity is None or len(equity) == 0:
        return 0.0
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
    if trade_df is None or trade_df.empty:
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
def run_candidate_backtest(params_tuple, top_n, start_dt, end_dt, stop_loss_pct, buy_fee, sell_fee, sector_filter="全部"):
    features, labels, _, _, _ = load_backtest_data()

    if features.empty or labels.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    params = dict(params_tuple)

    feat = features[(features["date"] >= pd.to_datetime(start_dt)) & (features["date"] <= pd.to_datetime(end_dt))].copy()
    lab = labels[(labels["date"] >= pd.to_datetime(start_dt)) & (labels["date"] <= pd.to_datetime(end_dt))].copy()

    all_dates = sorted(feat["date"].dropna().unique())

    daily_results = []
    trade_logs = []

    for d in all_dates:
        feature_day = feat[feat["date"] == d].copy()

        if sector_filter != "全部" and "sector_name" in feature_day.columns:
            feature_day = feature_day[feature_day["sector_name"].astype(str) == str(sector_filter)].copy()

        if feature_day.empty:
            continue

        ranked = select_candidates(feature_day, params)

        if ranked.empty:
            continue

        picks = ranked.head(int(top_n)).copy()

        label_day = lab[lab["date"] == d][[
            "date", "code", "name", "next_date", "next_open", "next_high",
            "next_low", "next_close", "next_red_close_flag",
            "next_touch_tp_flag", "tradable_flag"
        ]].copy()

        picks = picks.merge(label_day, on=["date", "code", "name"], how="left")
        eval_df = picks.apply(
            evaluate_nextday_trade,
            axis=1,
            stop_loss_pct=float(stop_loss_pct),
            buy_fee=float(buy_fee),
            sell_fee=float(sell_fee)
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
def run_single_stock_backtest(stock_code, start_dt, end_dt, stop_loss_pct, buy_fee, sell_fee):
    features, _, _, _, _ = load_backtest_data()

    if features.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    keyword = str(stock_code).strip()

    code_mask = features["code"].astype(str) == keyword
    name_mask = features["name"].astype(str) == keyword

    df = features[
        (code_mask | name_mask) &
        (features["date"] >= pd.to_datetime(start_dt)) &
        (features["date"] <= pd.to_datetime(end_dt))
    ].copy()

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


def ensure_backtest_state():
    if "latest_backtest_result" not in st.session_state:
        st.session_state["latest_backtest_result"] = {
            "metrics": {},
            "daily_df": pd.DataFrame(),
            "trade_df": pd.DataFrame(),
            "stock_summary_df": pd.DataFrame(),
            "eval_score": None,
            "decision": "暂无结果",
            "style_tag": "未知",
            "baseline_text": "暂无基线对比",
        }

    if "backtest_history" not in st.session_state:
        st.session_state["backtest_history"] = []

    if "baseline_metrics" not in st.session_state:
        st.session_state["baseline_metrics"] = None


def build_backtest_result_bundle(metrics, daily_df, trade_df):
    stock_summary_df = summarize_stock_returns(trade_df)
    eval_score, decision, style_tag = calc_eval_score(metrics)
    baseline_text = compare_with_baseline(metrics, st.session_state.get("baseline_metrics"))

    return {
        "metrics": metrics if metrics else {},
        "daily_df": daily_df if daily_df is not None else pd.DataFrame(),
        "trade_df": trade_df if trade_df is not None else pd.DataFrame(),
        "stock_summary_df": stock_summary_df,
        "eval_score": eval_score,
        "decision": decision,
        "style_tag": style_tag,
        "baseline_text": baseline_text,
    }


def save_latest_backtest_result(metrics, daily_df, trade_df):
    st.session_state["latest_backtest_result"] = build_backtest_result_bundle(metrics, daily_df, trade_df)


def append_backtest_history(record: dict):
    ensure_backtest_state()
    st.session_state["backtest_history"].append(record)


def get_backtest_history_df():
    ensure_backtest_state()
    if not st.session_state["backtest_history"]:
        return pd.DataFrame()

    rows = []
    for item in st.session_state["backtest_history"]:
        rows.append({
            "saved_at": item.get("saved_at"),
            "exp_name": item.get("exp_name"),
            "mode": item.get("mode"),
            "selected_date": item.get("selected_date"),
            "start_dt": item.get("start_dt"),
            "end_dt": item.get("end_dt"),
            "stock_code": item.get("stock_code"),
            "sector_filter": item.get("sector_filter"),
            "active_days": item.get("active_days"),
            "top_n": item.get("top_n"),
            "total_return": item.get("total_return"),
            "win_days_ratio": item.get("win_days_ratio"),
            "avg_daily_ret": item.get("avg_daily_ret"),
            "max_drawdown": item.get("max_drawdown"),
            "total_trade_count": item.get("total_trade_count"),
            "cannot_buy_count": item.get("cannot_buy_count"),
            "unique_stock_count": item.get("unique_stock_count"),
            "eval_score": item.get("eval_score"),
            "decision": item.get("decision"),
            "style_tag": item.get("style_tag"),
        })
    return pd.DataFrame(rows)


def get_backtest_history_items():
    ensure_backtest_state()
    return st.session_state["backtest_history"]


def set_history_item_as_latest(index: int):
    ensure_backtest_state()
    items = st.session_state["backtest_history"]
    if index < 0 or index >= len(items):
        return False

    item = items[index]
    st.session_state["latest_backtest_result"] = {
        "metrics": item.get("metrics", {}),
        "daily_df": item.get("daily_df", pd.DataFrame()),
        "trade_df": item.get("trade_df", pd.DataFrame()),
        "stock_summary_df": item.get("stock_summary_df", pd.DataFrame()),
        "eval_score": item.get("eval_score"),
        "decision": item.get("decision", "暂无结果"),
        "style_tag": item.get("style_tag", "未知"),
        "baseline_text": compare_with_baseline(item.get("metrics", {}), st.session_state.get("baseline_metrics")),
    }
    return True


def clear_backtest_history():
    ensure_backtest_state()
    st.session_state["backtest_history"] = []


def get_dashboard_metrics():
    ensure_backtest_state()
    latest = st.session_state.get("latest_backtest_result", {})
    metrics = latest.get("metrics", {}) if latest else {}

    if metrics:
        return {
            "活跃交易日": int(metrics.get("active_days", 0)),
            "总收益": f"{float(metrics.get('total_return', 0)):.2%}",
            "胜率": f"{float(metrics.get('win_days_ratio', 0)):.2%}",
            "最大回撤": f"{float(metrics.get('max_drawdown', 0)):.2%}",
            "今日候选数": int(len(st.session_state.get("factor_candidates", pd.DataFrame()))),
            "当前持仓数": int(len(st.session_state.get("decision_holdings_df", pd.DataFrame()))),
        }

    return {
        "活跃交易日": 0,
        "总收益": "-",
        "胜率": "-",
        "最大回撤": "-",
        "今日候选数": int(len(st.session_state.get("factor_candidates", pd.DataFrame()))),
        "当前持仓数": int(len(st.session_state.get("decision_holdings_df", pd.DataFrame()))),
    }


def get_recent_trades():
    ensure_backtest_state()
    latest = st.session_state.get("latest_backtest_result", {})
    trade_df = latest.get("trade_df", pd.DataFrame()) if latest else pd.DataFrame()

    if trade_df is None or trade_df.empty:
        return pd.DataFrame(
            columns=["日期", "代码", "名称", "操作", "成交价", "数量", "收益率"]
        )

    x = trade_df.copy()

    action_col = []
    for _, row in x.iterrows():
        sell_point = str(row.get("sell_point", ""))
        if "卖" in sell_point:
            action_col.append("卖出")
        else:
            action_col.append("交易")

    result = pd.DataFrame({
        "日期": pd.to_datetime(x.get("exit_date", x.get("date"))).dt.strftime("%Y-%m-%d"),
        "代码": x.get("code", ""),
        "名称": x.get("name", ""),
        "操作": action_col,
        "成交价": pd.to_numeric(x.get("exit_price", 0), errors="coerce").fillna(0).round(2),
        "数量": 1,
        "收益率": pd.to_numeric(x.get("trade_ret", 0), errors="coerce").fillna(0).map(lambda v: f"{v:.2%}"),
    })

    return result.tail(8).reset_index(drop=True)