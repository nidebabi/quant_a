import pandas as pd
from settings import DATA_RAW, DATA_LABELS, TAKE_PROFIT

RENAME_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_chg",
    "涨跌额": "chg",
    "换手率": "turnover",
}

def process_one(path):
    df = pd.read_parquet(path).rename(columns=RENAME_MAP)

    keep_cols = ["date", "open", "close", "high", "low", "code", "name"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    label_df = df[["date", "code", "name"]].copy()

    # 这些都是 T+1 的结果，但记录在 T 这一天上，用于回测验证
    label_df["next_date"] = df["date"].shift(-1)
    label_df["next_open"] = df["open"].shift(-1)
    label_df["next_high"] = df["high"].shift(-1)
    label_df["next_low"] = df["low"].shift(-1)
    label_df["next_close"] = df["close"].shift(-1)

    # 是否红盘（从开盘到收盘）
    label_df["next_red_close_flag"] = (label_df["next_close"] > label_df["next_open"]).astype("Int64")

    # 次日从开盘出发，盘中是否摸到 +2%
    label_df["next_touch_tp_flag"] = (
        label_df["next_high"] >= label_df["next_open"] * (1 + TAKE_PROFIT)
    ).astype("Int64")

    # 简单可交易性判断：若全天一个价，保守视为无法成交
    label_df["tradable_flag"] = 1
    flat_mask = (
        label_df["next_open"].notna() &
        label_df["next_high"].notna() &
        label_df["next_low"].notna() &
        (label_df["next_high"] == label_df["next_low"])
    )
    label_df.loc[flat_mask, "tradable_flag"] = 0

    return label_df

def main():
    frames = []

    for path in DATA_RAW.glob("*.parquet"):
        if path.suffix != ".parquet":
            continue
        df = process_one(path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("没有可用数据，请先运行 download_data.py")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "code"]).reset_index(drop=True)
    result.to_parquet(DATA_LABELS / "labels_all.parquet", index=False)

    print("标签文件已生成：labels_all.parquet")

if __name__ == "__main__":
    main()