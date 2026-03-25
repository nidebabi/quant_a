import numpy as np
import pandas as pd
from settings import (
    DATA_FEATURES, DATA_LABELS, REPORTS, TOP_N,
    TAKE_PROFIT, STOP_LOSS, BUY_FEE_RATE, SELL_FEE_RATE
)
from strategy import select_candidates

# 这里改成你想验证的日期
TARGET_DATE = "2026-03-16"

def evaluate_trade(row):
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

def main():
    target_date = pd.to_datetime(TARGET_DATE)

    features = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    labels = pd.read_parquet(DATA_LABELS / "labels_all.parquet")

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    feature_day = features[features["date"] == target_date].copy()
    if feature_day.empty:
        print("该日期没有特征数据")
        return

    ranked = select_candidates(feature_day)
    if ranked.empty:
        print("该日期没有候选股")
        return

    picks = ranked.head(TOP_N).copy()

    label_day = labels[labels["date"] == target_date][[
        "date", "code", "name", "next_date", "next_open", "next_high",
        "next_low", "next_close", "next_red_close_flag",
        "next_touch_tp_flag", "tradable_flag"
    ]].copy()

    result_df = picks.merge(label_day, on=["date", "code", "name"], how="left")
    eval_df = result_df.apply(evaluate_trade, axis=1)
    result_df = pd.concat([result_df, eval_df], axis=1)

    out_path = REPORTS / f"validate_{TARGET_DATE}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(result_df[[
        "date", "code", "name", "score", "next_open", "next_high",
        "next_low", "next_close", "trade_result", "trade_ret"
    ]])
    print(f"已输出：{out_path}")

if __name__ == "__main__":
    main()