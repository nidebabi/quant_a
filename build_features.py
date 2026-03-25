import numpy as np
import pandas as pd
from settings import DATA_RAW, DATA_FEATURES, MIN_LISTING_BARS

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

    keep_cols = [
        "date", "open", "close", "high", "low", "volume", "amount",
        "amplitude", "pct_chg", "chg", "turnover", "code", "name"
    ]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"])

    num_cols = [
        "open", "close", "high", "low", "volume", "amount",
        "amplitude", "pct_chg", "chg", "turnover"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < MIN_LISTING_BARS:
        return None

    # 均线：只用历史到当日
    for n in [5, 10, 20]:
        df[f"ma{n}"] = df["close"].rolling(n).mean()

    # 收益特征：只看当日及以前
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    # 量比：只看当日及以前
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma5"]

    # 收盘位置
    spread = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_loc"] = (df["close"] - df["low"]) / spread

    # 上影线比例
    df["upper_shadow_ratio"] = (df["high"] - df["close"]) / spread

    # 趋势标记
    df["trend_flag"] = (
        (df["close"] > df["ma5"]) &
        (df["ma5"] > df["ma10"]) &
        (df["ma10"] > df["ma20"])
    ).astype(int)

    # 20日突破位置：只看到前一日高点
    df["high20_prev"] = df["high"].rolling(20).max().shift(1)
    df["breakout20"] = df["close"] / df["high20_prev"] - 1

    return df

def main():
    frames = []

    for path in DATA_RAW.glob("*.parquet"):
        if path.name == "universe_snapshot.csv":
            continue
        df = process_one(path)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("没有可用数据，请先运行 download_data.py")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "code"]).reset_index(drop=True)
    result.to_parquet(DATA_FEATURES / "features_all.parquet", index=False)

    print("特征文件已生成：features_all.parquet")

if __name__ == "__main__":
    main()