from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
from typing import Dict, List, Optional

import json
import re

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from services.data_service import FEATURE_PATH, LABEL_PATH, RAW_DIR, clear_data_caches

import akshare as ak  # type: ignore


SOURCE_PATH = "Sohu hisHq recent daily bars primary; Tonghuashun v6 and Eastmoney minute as conservative fallbacks"
DEFAULT_SAFE_LIMIT = 5
REFRESH_OVERLAP_DAYS = 5
RAW_COLUMNS = [
    "日期",
    "code",
    "开盘",
    "最高",
    "最低",
    "收盘",
    "成交量",
    "成交额",
    "振幅",
    "涨跌幅",
    "涨跌额",
    "换手率",
    "name",
]


@dataclass
class UpdateResult:
    checked_files: int
    updated_files: int
    failed_files: int
    latest_local_date_before: str
    latest_local_date_after: str
    feature_rows: int
    label_rows: int
    errors: List[str]
    source_path: str = SOURCE_PATH


def _read_local_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_parquet(path)
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[RAW_COLUMNS].copy()


def _latest_raw_date() -> str:
    latest_dates: List[pd.Timestamp] = []
    for path in RAW_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(path, columns=["日期"])
            if not df.empty:
                latest_dates.append(pd.to_datetime(df["日期"]).max())
        except Exception:
            continue
    return str(max(latest_dates).date()) if latest_dates else ""


def _estimate_float_shares(local_df: pd.DataFrame) -> Optional[float]:
    base = local_df.copy()
    base["成交量"] = pd.to_numeric(base["成交量"], errors="coerce")
    base["换手率"] = pd.to_numeric(base["换手率"], errors="coerce")
    valid = base[(base["成交量"] > 0) & (base["换手率"] > 0)].copy()
    if valid.empty:
        return None
    last_row = valid.iloc[-1]
    return float(last_row["成交量"] / (last_row["换手率"] / 100))


def _safe_first_open(group: pd.DataFrame) -> float:
    opens = pd.to_numeric(group["开盘"], errors="coerce")
    non_zero = opens[opens > 0]
    if not non_zero.empty:
        return float(non_zero.iloc[0])
    closes = pd.to_numeric(group["收盘"], errors="coerce").dropna()
    if not closes.empty:
        return float(closes.iloc[0])
    return float("nan")


def _fetch_increment_from_minute(code: str) -> pd.DataFrame:
    minute_df = ak.stock_zh_a_hist_min_em(symbol=code, period="1", adjust="")
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    out = minute_df.copy()
    out["时间"] = pd.to_datetime(out["时间"], errors="coerce")
    out = out.dropna(subset=["时间"]).copy()
    out["交易日期"] = out["时间"].dt.date
    return out


def _fetch_ths_daily_increment(code: str) -> pd.DataFrame:
    market = "sh" if code.startswith("6") else "hs"
    symbol_key = f"sh_{code}" if market == "sh" else f"hs_{code}"
    url = f"https://d.10jqka.com.cn/v6/line/{symbol_key}/01/all.js"
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://stockpage.10jqka.com.cn/{code}/",
            "DNT": "1",
        },
        timeout=20,
    )
    response.raise_for_status()
    match = re.search(r"\((\{.*\})\)\s*$", response.text)
    if not match:
        return pd.DataFrame()
    payload = json.loads(match.group(1))
    total = int(payload.get("total", 0) or 0)
    dates = [item for item in str(payload.get("dates", "")).split(",") if item]
    prices = [item for item in str(payload.get("price", "")).split(",") if item]
    volumes = [item for item in str(payload.get("volumn", "")).split(",") if item]
    price_factor = float(payload.get("priceFactor", 100) or 100)
    start = str(payload.get("start", ""))
    if not start or total <= 0:
        return pd.DataFrame()
    if total == len(dates) + 1 and total == len(volumes) + 1:
        total -= 1
    if total * 4 != len(prices) or total != len(dates) or total != len(volumes):
        return pd.DataFrame()

    start_year = int(start[:4])
    current_year = start_year
    last_mmdd = ""
    rows: List[Dict[str, object]] = []
    for idx in range(total):
        mmdd = dates[idx]
        if last_mmdd and mmdd < last_mmdd:
            current_year += 1
        last_mmdd = mmdd
        low = float(prices[idx * 4 + 0]) / price_factor
        open_price = low + float(prices[idx * 4 + 1]) / price_factor
        high_price = low + float(prices[idx * 4 + 2]) / price_factor
        close_price = low + float(prices[idx * 4 + 3]) / price_factor
        rows.append(
            {
                "日期": f"{current_year}-{mmdd[:2]}-{mmdd[2:]}",
                "开盘": round(open_price, 4),
                "最高": round(high_price, 4),
                "最低": round(low, 4),
                "收盘": round(close_price, 4),
                "成交量": pd.to_numeric(volumes[idx], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _fetch_sohu_daily_increment(code: str) -> pd.DataFrame:
    url = (
        f"https://q.stock.sohu.com/hisHq?code=cn_{code}"
        "&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp"
    )
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://q.stock.sohu.com/cn/{code}/lshq.shtml",
        },
        timeout=20,
    )
    response.raise_for_status()
    match = re.search(r"historySearchHandler\((.*)\)\s*$", response.text)
    if not match:
        return pd.DataFrame()
    payload = json.loads(match.group(1))
    if not payload or payload[0].get("status") != 0:
        return pd.DataFrame()
    hq_rows = payload[0].get("hq", [])
    if not hq_rows:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for item in hq_rows:
        if len(item) < 10:
            continue
        pct_text = str(item[4]).replace("%", "")
        turnover_text = str(item[9]).replace("%", "")
        rows.append(
            {
                "日期": item[0],
                "开盘": pd.to_numeric(item[1], errors="coerce"),
                "收盘": pd.to_numeric(item[2], errors="coerce"),
                "涨跌额": pd.to_numeric(item[3], errors="coerce"),
                "涨跌幅": pd.to_numeric(pct_text, errors="coerce"),
                "最低": pd.to_numeric(item[5], errors="coerce"),
                "最高": pd.to_numeric(item[6], errors="coerce"),
                "成交量": pd.to_numeric(item[7], errors="coerce") * 100,
                "成交额": pd.to_numeric(item[8], errors="coerce") * 10000,
                "换手率": pd.to_numeric(turnover_text, errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_minute_to_daily(
    minute_df: pd.DataFrame,
    local_df: pd.DataFrame,
    code: str,
    name: str,
    end_date: str,
    refresh_from_date: date,
) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    target_end = date.fromisoformat(end_date)
    work_df = minute_df[
        (minute_df["交易日期"] >= refresh_from_date) & (minute_df["交易日期"] <= target_end)
    ].copy()
    if work_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    rows: List[Dict[str, object]] = []
    prev_close = float(pd.to_numeric(local_df["收盘"], errors="coerce").dropna().iloc[-1])
    float_shares = _estimate_float_shares(local_df)

    for trade_date, group in work_df.groupby("交易日期", sort=True):
        close_series = pd.to_numeric(group["收盘"], errors="coerce").dropna()
        high_series = pd.to_numeric(group["最高"], errors="coerce").dropna()
        low_series = pd.to_numeric(group["最低"], errors="coerce").dropna()
        volume_series = pd.to_numeric(group["成交量"], errors="coerce").fillna(0)
        amount_series = pd.to_numeric(group["成交额"], errors="coerce").fillna(0)
        if close_series.empty or high_series.empty or low_series.empty:
            continue

        open_price = _safe_first_open(group)
        close_price = float(close_series.iloc[-1])
        high_price = float(high_series.max())
        low_price = float(low_series.min())
        volume = float(volume_series.sum())
        amount = float(amount_series.sum())
        chg = close_price - prev_close
        pct_chg = (chg / prev_close * 100) if prev_close else None
        amplitude = ((high_price - low_price) / prev_close * 100) if prev_close else None
        turnover = (volume / float_shares * 100) if float_shares else None
        rows.append(
            {
                "日期": trade_date.isoformat(),
                "code": code,
                "开盘": round(open_price, 4) if pd.notna(open_price) else None,
                "最高": round(high_price, 4),
                "最低": round(low_price, 4),
                "收盘": round(close_price, 4),
                "成交量": round(volume),
                "成交额": round(amount, 2),
                "振幅": round(amplitude, 4) if amplitude is not None else None,
                "涨跌幅": round(pct_chg, 4) if pct_chg is not None else None,
                "涨跌额": round(chg, 4),
                "换手率": round(turnover, 4) if turnover is not None else None,
                "name": name,
            }
        )
        prev_close = close_price
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _normalize_increment_frame(
    daily_df: pd.DataFrame,
    local_df: pd.DataFrame,
    code: str,
    name: str,
    end_date: str,
    refresh_from_date: date,
) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    work_df = daily_df.copy()
    work_df["日期"] = pd.to_datetime(work_df["日期"], errors="coerce")
    target_end = date.fromisoformat(end_date)
    work_df = work_df[
        (work_df["日期"].dt.date >= refresh_from_date) & (work_df["日期"].dt.date <= target_end)
    ].copy()
    if work_df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    float_shares = _estimate_float_shares(local_df)
    prev_close = float(pd.to_numeric(local_df["收盘"], errors="coerce").dropna().iloc[-1])
    amount_ratio = pd.to_numeric(local_df["成交额"], errors="coerce") / pd.to_numeric(local_df["成交量"], errors="coerce")
    amount_ratio = amount_ratio.replace([pd.NA, pd.NaT], pd.NA).dropna()
    fallback_amount_ratio = float(amount_ratio.iloc[-20:].median()) if not amount_ratio.empty else None

    rows: List[Dict[str, object]] = []
    for _, row in work_df.sort_values("日期").iterrows():
        open_price = float(pd.to_numeric(row.get("开盘"), errors="coerce"))
        high_price = float(pd.to_numeric(row.get("最高"), errors="coerce"))
        low_price = float(pd.to_numeric(row.get("最低"), errors="coerce"))
        close_price = float(pd.to_numeric(row.get("收盘"), errors="coerce"))
        volume = float(pd.to_numeric(row.get("成交量"), errors="coerce") or 0)

        amount_value = pd.to_numeric(row.get("成交额"), errors="coerce")
        if pd.isna(amount_value):
            price_proxy = (open_price + high_price + low_price + close_price) / 4
            amount_value = volume * (fallback_amount_ratio if fallback_amount_ratio is not None else price_proxy)

        turnover_value = pd.to_numeric(row.get("换手率"), errors="coerce")
        if pd.isna(turnover_value) and float_shares:
            turnover_value = volume / float_shares * 100

        chg_value = pd.to_numeric(row.get("涨跌额"), errors="coerce")
        if pd.isna(chg_value):
            chg_value = close_price - prev_close

        pct_value = pd.to_numeric(row.get("涨跌幅"), errors="coerce")
        if pd.isna(pct_value) and prev_close:
            pct_value = chg_value / prev_close * 100

        amplitude_value = pd.to_numeric(row.get("振幅"), errors="coerce")
        if pd.isna(amplitude_value) and prev_close:
            amplitude_value = (high_price - low_price) / prev_close * 100

        rows.append(
            {
                "日期": row["日期"].strftime("%Y-%m-%d"),
                "code": code,
                "开盘": round(open_price, 4),
                "最高": round(high_price, 4),
                "最低": round(low_price, 4),
                "收盘": round(close_price, 4),
                "成交量": round(volume),
                "成交额": round(float(amount_value), 2) if pd.notna(amount_value) else None,
                "振幅": round(float(amplitude_value), 4) if pd.notna(amplitude_value) else None,
                "涨跌幅": round(float(pct_value), 4) if pd.notna(pct_value) else None,
                "涨跌额": round(float(chg_value), 4) if pd.notna(chg_value) else None,
                "换手率": round(float(turnover_value), 4) if pd.notna(turnover_value) else None,
                "name": name,
            }
        )
        prev_close = close_price
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def _update_one_file(path: Path, end_date: str) -> Dict[str, str]:
    local_df = _read_local_raw(path)
    if local_df.empty:
        return {"status": "skip", "code": path.stem, "message": "empty local raw"}

    code = str(local_df["code"].iloc[-1])
    name = str(local_df["name"].iloc[-1])
    latest_local_date = pd.to_datetime(local_df["日期"], errors="coerce").max().date()
    target_end = date.fromisoformat(end_date)
    refresh_anchor = min(target_end, latest_local_date)
    refresh_from_date = (pd.Timestamp(refresh_anchor) - pd.Timedelta(days=REFRESH_OVERLAP_DAYS)).date()

    append_df = pd.DataFrame(columns=RAW_COLUMNS)

    try:
        sohu_df = _fetch_sohu_daily_increment(code)
        append_df = _normalize_increment_frame(
            sohu_df,
            local_df,
            code=code,
            name=name,
            end_date=end_date,
            refresh_from_date=refresh_from_date,
        )
        if not append_df.empty:
            source = "Sohu hisHq"
        else:
            source = "Sohu hisHq empty"
    except Exception:
        source = "Sohu hisHq failed"

    if append_df.empty:
        try:
            minute_df = _fetch_increment_from_minute(code)
            append_df = _aggregate_minute_to_daily(
                minute_df,
                local_df,
                code=code,
                name=name,
                end_date=end_date,
                refresh_from_date=refresh_from_date,
            )
            source = "Eastmoney minute"
        except Exception:
            source = "Eastmoney minute failed"

    if append_df.empty:
        ths_df = _fetch_ths_daily_increment(code)
        append_df = _normalize_increment_frame(
            ths_df,
            local_df,
            code=code,
            name=name,
            end_date=end_date,
            refresh_from_date=refresh_from_date,
        )
        source = "Tonghuashun v6"

    if append_df.empty:
        return {"status": "skip", "code": code, "message": "no incremental bars from Sohu, Eastmoney, or Tonghuashun"}

    merged = pd.concat([local_df, append_df], ignore_index=True)
    merged["日期"] = pd.to_datetime(merged["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset=["日期"], keep="last").sort_values("日期").reset_index(drop=True)
    merged = merged[RAW_COLUMNS].copy()
    merged.to_parquet(path, index=False)
    return {"status": "updated", "code": code, "message": f"+{len(append_df)} rows via {source}"}


def _replace_codes_in_parquet(parquet_path: Path, new_frames: List[pd.DataFrame], updated_codes: List[str]) -> int:
    if not parquet_path.exists():
        combined = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
        if not combined.empty:
            combined.to_parquet(parquet_path, index=False)
            return len(combined)
        return 0

    temp_path = parquet_path.with_suffix(".tmp.parquet")
    if temp_path.exists():
        temp_path.unlink()

    schema = pq.read_schema(parquet_path)
    writer: Optional[pq.ParquetWriter] = None
    parquet_file: Optional[pq.ParquetFile] = None
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=100000):
            table = pa.Table.from_batches([batch], schema=schema)
            if "code" in table.column_names and updated_codes:
                code_array = pc.cast(table["code"], pa.string())
                keep_mask = pc.invert(pc.is_in(code_array, value_set=pa.array(updated_codes)))
                table = table.filter(keep_mask)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema=table.schema)
            writer.write_table(table)

        for frame in new_frames:
            if frame is None or frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(temp_path, schema=table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        if parquet_file is not None and hasattr(parquet_file, "close"):
            parquet_file.close()
        parquet_file = None

    if temp_path.exists():
        os.replace(temp_path, parquet_path)
    metadata = pq.ParquetFile(parquet_path).metadata
    return metadata.num_rows if metadata is not None else 0


def rebuild_features_and_labels(updated_paths: Optional[List[Path]] = None) -> Dict[str, int]:
    import build_features
    import build_labels

    target_paths = updated_paths or list(RAW_DIR.glob("*.parquet"))
    feature_frames: List[pd.DataFrame] = []
    label_frames: List[pd.DataFrame] = []
    updated_codes: List[str] = []

    for path in target_paths:
        try:
            feature_df = build_features.process_one(path)
            if feature_df is not None and not feature_df.empty:
                feature_frames.append(feature_df)
                updated_codes.append(str(feature_df["code"].iloc[0]))
            label_df = build_labels.process_one(path)
            if label_df is not None and not label_df.empty:
                label_frames.append(label_df)
                if not updated_codes:
                    updated_codes.append(str(label_df["code"].iloc[0]))
        except Exception:
            continue

    if updated_paths:
        feature_rows = _replace_codes_in_parquet(FEATURE_PATH, feature_frames, updated_codes)
        label_rows = _replace_codes_in_parquet(LABEL_PATH, label_frames, updated_codes)
    else:
        if feature_frames:
            features = pd.concat(feature_frames, ignore_index=True)
            features.to_parquet(FEATURE_PATH, index=False)
            feature_rows = len(features)
        else:
            feature_rows = 0

        if label_frames:
            labels = pd.concat(label_frames, ignore_index=True)
            labels.to_parquet(LABEL_PATH, index=False)
            label_rows = len(labels)
        else:
            label_rows = 0

    clear_data_caches()
    return {"feature_rows": feature_rows, "label_rows": label_rows}


def update_market_data(
    end_date: Optional[str] = None,
    max_workers: int = 1,
    limit: Optional[int] = DEFAULT_SAFE_LIMIT,
) -> UpdateResult:
    del max_workers

    target_end = end_date or date.today().isoformat()
    files = sorted(RAW_DIR.glob("*.parquet"))
    if limit is None:
        limit = DEFAULT_SAFE_LIMIT
    files = files[:limit]

    latest_before = _latest_raw_date()
    updated_files = 0
    failed_files = 0
    errors: List[str] = []
    updated_paths: List[Path] = []

    for path in files:
        try:
            result = _update_one_file(path, target_end)
            if result["status"] == "updated":
                updated_files += 1
                updated_paths.append(path)
            elif result["status"] == "error":
                failed_files += 1
                errors.append(f"{result['code']}: {result['message']}")
        except Exception as exc:
            failed_files += 1
            errors.append(f"{path.stem}: {exc}")

    rebuilt = rebuild_features_and_labels(updated_paths=updated_paths) if updated_paths else {"feature_rows": len(pd.read_parquet(FEATURE_PATH)) if FEATURE_PATH.exists() else 0, "label_rows": len(pd.read_parquet(LABEL_PATH)) if LABEL_PATH.exists() else 0}
    latest_after = _latest_raw_date()
    return UpdateResult(
        checked_files=len(files),
        updated_files=updated_files,
        failed_files=failed_files,
        latest_local_date_before=latest_before,
        latest_local_date_after=latest_after,
        feature_rows=rebuilt["feature_rows"],
        label_rows=rebuilt["label_rows"],
        errors=errors[:50],
    )
