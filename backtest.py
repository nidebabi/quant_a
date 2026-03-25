import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from settings import (
    DATA_FEATURES, DATA_LABELS, REPORTS, TOP_N,
    TAKE_PROFIT, STOP_LOSS, BUY_FEE_RATE, SELL_FEE_RATE
)
from strategy import select_candidates

def evaluate_trade(row):
    # 买不到，直接不计入交易
    if pd.isna(row["next_open"]) or row["tradable_flag"] != 1:
        return pd.Series({
            "trade_ret": np.nan,
            "trade_result": "cannot_buy"
        })

    buy = row["next_open"]
    tp_price = buy * (1 + TAKE_PROFIT)
    sl_price = buy * (1 + STOP_LOSS)

    hit_tp = row["next_high"] >= tp_price
    hit_sl = row["next_low"] <= sl_price

    # 日线回测，无法判断先后顺序，保守处理
    if hit_tp and hit_sl:
        gross_ret = STOP_LOSS
        result = "both_hit_conservative_stop"
    elif hit_tp:
        gross_ret = TAKE_PROFIT
        result = "take_profit"
    elif hit_sl:
        gross_ret = STOP_LOSS
        result = "stop_loss"
    else:
        gross_ret = row["next_close"] / buy - 1
        result = "close_exit"

    net_ret = gross_ret - BUY_FEE_RATE - SELL_FEE_RATE

    return pd.Series({
        "trade_ret": net_ret,
        "trade_result": result
    })

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())

def main():
    features = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    labels = pd.read_parquet(DATA_LABELS / "labels_all.parquet")

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    all_dates = sorted(features["date"].dropna().unique())

    daily_results = []
    trade_logs = []

    for d in all_dates:
        feature_day = features[features["date"] == d].copy()

        # 这里只看特征，不看标签
        ranked = select_candidates(feature_day)
        if ranked.empty:
            continue

        picks = ranked.head(TOP_N).copy()

        # 选完后，才拼 T+1 标签
        label_day = labels[labels["date"] == d][[
            "date", "code", "name", "next_date", "next_open", "next_high",
            "next_low", "next_close", "next_red_close_flag",
            "next_touch_tp_flag", "tradable_flag"
        ]].copy()

        picks = picks.merge(label_day, on=["date", "code", "name"], how="left")

        eval_df = picks.apply(evaluate_trade, axis=1)
        picks = pd.concat([picks, eval_df], axis=1)

        valid = picks["trade_ret"].notna()
        trade_count = int(valid.sum())
        cannot_buy_count = int((picks["trade_result"] == "cannot_buy").sum())

        if trade_count == 0:
            continue

        avg_ret = float(picks.loc[valid, "trade_ret"].mean())
        win_rate_day = float((picks.loc[valid, "trade_ret"] > 0).mean())

        daily_results.append({
            "date": pd.to_datetime(d),
            "trade_count": trade_count,
            "cannot_buy_count": cannot_buy_count,
            "avg_ret": avg_ret,
            "win_rate_day": win_rate_day
        })

        trade_logs.append(picks)

    if not daily_results:
        print("没有可用回测结果")
        return

    daily_df = pd.DataFrame(daily_results).sort_values("date").reset_index(drop=True)
    daily_df["equity"] = (1 + daily_df["avg_ret"]).cumprod()

    trade_log_df = pd.concat(trade_logs, ignore_index=True)

    metrics = {
        "active_days": int(len(daily_df)),
        "top_n": int(TOP_N),
        "total_return": float(daily_df["equity"].iloc[-1] - 1),
        "win_days_ratio": float((daily_df["avg_ret"] > 0).mean()),
        "avg_daily_ret": float(daily_df["avg_ret"].mean()),
        "max_drawdown": max_drawdown(daily_df["equity"]),
        "total_trade_count": int(trade_log_df["trade_ret"].notna().sum()),
        "cannot_buy_count": int((trade_log_df["trade_result"] == "cannot_buy").sum())
    }

    daily_df.to_csv(REPORTS / "backtest_daily.csv", index=False, encoding="utf-8-sig")
    trade_log_df.to_csv(REPORTS / "backtest_trade_log.csv", index=False, encoding="utf-8-sig")

    with open(REPORTS / "backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(12, 6))
    plt.plot(daily_df["date"], daily_df["equity"])
    plt.title("V1 严格防穿越回测净值曲线")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(REPORTS / "equity_curve.png", dpi=150)

    print(metrics)
    print("回测结果已输出到 reports/")

if __name__ == "__main__":
    main()