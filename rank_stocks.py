import argparse
import pandas as pd
from settings import DATA_FEATURES, REPORTS, TOP_N
from strategy import select_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="指定选股日期，格式：YYYY-MM-DD，例如 2026-03-18；不传则默认最新交易日"
    )
    args = parser.parse_args()

    df = pd.read_parquet(DATA_FEATURES / "features_all.parquet")
    df["date"] = pd.to_datetime(df["date"])

    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = df["date"].max()

    day_df = df[df["date"] == target_date].copy()

    if day_df.empty:
        print(f"没有找到 {target_date.date()} 的特征数据")
        return

    ranked = select_candidates(day_df)

    if ranked.empty:
        print(f"{target_date.date()} 没有符合条件的候选股")
        return

    top = ranked.head(TOP_N).copy()
    out_path = REPORTS / f"top_picks_{target_date.strftime('%Y%m%d')}.csv"
    top.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"日期: {target_date.date()}")
    print(top[["code", "name", "close", "score"]].head(TOP_N))
    print(f"已输出：{out_path}")

if __name__ == "__main__":
    main()