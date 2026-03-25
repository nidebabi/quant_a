from datetime import datetime, timedelta
import time
import traceback

import baostock as bs
import pandas as pd

from settings import DATA_RAW, START_DATE

# ----------------------------
# 测试模式：先只跑前 20 只，跑通后改成 None
# ----------------------------
MAX_STOCKS_TEST = None

# ----------------------------
# 只做主板：sh.60 / sz.00
# ----------------------------
MAIN_PREFIX = ("sh.60", "sz.00")


def bs_result_to_df(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=rs.fields if hasattr(rs, "fields") else [])
    return pd.DataFrame(rows, columns=rs.fields)


def get_recent_trading_days(back_days: int = 30) -> list[str]:
    """
    获取最近一段时间内的交易日列表，按从近到远排序
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=back_days)).strftime("%Y-%m-%d")

    rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
    df = bs_result_to_df(rs)

    if df.empty:
        raise ValueError("无法获取交易日历，请检查网络或 BaoStock 连接。")

    df = df[df["is_trading_day"] == "1"].copy()
    if df.empty:
        raise ValueError("最近区间内没有交易日。")

    days = df["calendar_date"].tolist()
    days.sort(reverse=True)  # 从近到远
    return days


def get_universe_with_fallback(max_lookback_trade_days: int = 15) -> tuple[pd.DataFrame, str]:
    """
    从最近交易日开始往前找，直到 query_all_stock(day=...) 返回非空
    """
    trading_days = get_recent_trading_days(back_days=60)

    checked = 0
    for day in trading_days:
        if checked >= max_lookback_trade_days:
            break

        rs = bs.query_all_stock(day=day)
        df = bs_result_to_df(rs)

        if not df.empty:
            code_col = "code" if "code" in df.columns else df.columns[0]
            name_col = "code_name" if "code_name" in df.columns else (
                "name" if "name" in df.columns else df.columns[1]
            )

            df = df.rename(columns={code_col: "code", name_col: "name"})
            df = df[df["code"].astype(str).str.startswith(MAIN_PREFIX)].copy()
            df = df[~df["name"].astype(str).str.contains("ST", na=False)].copy()
            df = df.reset_index(drop=True)

            if MAX_STOCKS_TEST is not None:
                df = df.head(MAX_STOCKS_TEST).copy()

            return df, day

        checked += 1

    raise ValueError(f"最近向前回退 {max_lookback_trade_days} 个交易日，仍未获取到股票列表。")


def download_one(code: str, name: str, end_date: str, max_retries: int = 3) -> bool:
    """
    BaoStock:
    adjustflag='3' = 不复权
    frequency='d'  = 日线
    """
    start_date = f"{START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:8]}"

    for attempt in range(1, max_retries + 1):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                fields="date,code,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            df = bs_result_to_df(rs)

            if rs.error_code != "0":
                raise RuntimeError(f"BaoStock error {rs.error_code}: {rs.error_msg}")

            if df.empty:
                print(f"[EMPTY] {code} {name}")
                return False

            df = df.rename(columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
                "amount": "成交额",
                "turn": "换手率",
            })

            for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["code"] = code.replace("sh.", "").replace("sz.", "")
            df["name"] = name

            out_path = DATA_RAW / f"{df['code'].iloc[0]}.parquet"
            df.to_parquet(out_path, index=False)

            print(f"[OK] {code} {name}")
            return True

        except Exception as e:
            print(f"[RETRY {attempt}/{max_retries}] {code} {name} -> {type(e).__name__}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print(f"[FAIL] {code} {name}")
                traceback.print_exc()
                return False


def main():
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_code}, {lg.error_msg}")

    try:
        universe, usable_day = get_universe_with_fallback()

        universe.to_csv(DATA_RAW / "universe_snapshot.csv", index=False, encoding="utf-8-sig")

        success = 0
        total = len(universe)
        print(f"实际使用的股票列表日期: {usable_day}")
        print(f"股票数: {total}")

        # 日线下载结束日期也统一用这个可用日
        end_date = usable_day

        for i, row in universe.iterrows():
            ok = download_one(row["code"], row["name"], end_date)
            if ok:
                success += 1

            time.sleep(0.8)

            if (i + 1) % 20 == 0:
                print(f"进度: {i+1}/{total}，成功: {success}")
                print("暂停 3 秒，避免请求过快...")
                time.sleep(3)

        print(f"下载完成，成功 {success}/{total}")

    finally:
        bs.logout()


if __name__ == "__main__":
    main()