from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


CURRENT_FACTOR_KEYS = [
    "current_factor_result",
    "factor_result",
    "current_select_result",
    "current_factor_selection",
    "latest_factor_result",
    "selected_result",
]

CURRENT_BACKTEST_KEYS = [
    "current_backtest_result",
    "backtest_result",
    "latest_backtest_result",
    "single_backtest_result",
]

CURRENT_DECISION_KEYS = [
    "current_decision_result",
    "decision_result",
    "latest_decision_result",
    "stock_decision_result",
]

FACTOR_HISTORY_HINTS = ["factor_history", "select_history", "selection_history", "factor_records"]
BACKTEST_HISTORY_HINTS = ["backtest_history", "backtest_records"]
DECISION_HISTORY_HINTS = ["decision_history", "decision_records"]
REPORT_HISTORY_HINTS = ["report_history", "report_records", "panel_history", "session_reports"]
PRESET_HINTS = ["preset", "scheme", "方案"]


def _session_items(session_state: Any) -> List:
    try:
        return list(session_state.items())
    except Exception:
        try:
            return list(dict(session_state).items())
        except Exception:
            return []


def _first_not_null(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    text = _safe_str(value).strip()
    if not text:
        return None

    candidates = [
        text,
        text.replace("/", "-"),
        text.replace("T", " ").replace("Z", ""),
    ]
    for item in candidates:
        try:
            return pd.to_datetime(item).to_pydatetime()
        except Exception:
            pass
    return None


def _format_dt(value: Any) -> str:
    dt = _parse_datetime(value)
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    text = _safe_str(value).strip().replace(",", "")
    if not text:
        return None

    is_percent = "%" in text
    text = text.replace("%", "")
    try:
        number = float(text)
        if is_percent:
            return number / 100.0
        return number
    except Exception:
        return None


def _coerce_ratio(value: Any) -> Optional[float]:
    number = _coerce_float(value)
    if number is None:
        return None
    if abs(number) > 1.5:
        return number / 100.0
    return number


def _coerce_int(value: Any) -> Optional[int]:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(round(number))


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _safe_str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "是", "风险", "高风险"}:
        return True
    if text in {"0", "false", "no", "n", "否", "非风险"}:
        return False
    return None


def _rows_from_obj(obj: Any) -> List[Dict[str, Any]]:
    if obj is None:
        return []

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return []
        return obj.to_dict(orient="records")

    if isinstance(obj, list):
        rows = []
        for item in obj:
            if isinstance(item, dict):
                rows.append(item)
            else:
                rows.append({"value": item})
        return rows

    if isinstance(obj, dict):
        for key in ["records", "data", "items", "rows", "result", "results"]:
            value = obj.get(key)
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="records")
            if isinstance(value, list):
                return _rows_from_obj(value)
        return [obj]

    return []


def _extract_stock_rows(record: Any) -> List[Dict[str, Any]]:
    if record is None:
        return []

    if isinstance(record, pd.DataFrame):
        return record.to_dict(orient="records")

    if isinstance(record, list):
        return _rows_from_obj(record)

    if not isinstance(record, dict):
        return []

    for key in [
        "selected_stocks",
        "stocks",
        "candidates",
        "candidate_stocks",
        "top_stocks",
        "result_df",
        "selected_df",
        "dataframe",
        "result",
        "results",
    ]:
        if key in record:
            rows = _rows_from_obj(record.get(key))
            if rows:
                return rows

    summary_df = record.get("summary_df")
    if isinstance(summary_df, pd.DataFrame):
        return summary_df.to_dict(orient="records")

    return []


def _looks_like_factor(obj: Any, key: str = "") -> bool:
    if isinstance(obj, pd.DataFrame):
        cols = {str(c).lower() for c in obj.columns}
        return bool(cols & {"code", "stock_code", "ts_code", "name", "score"})
    if isinstance(obj, list):
        if not obj:
            return False
        return _looks_like_factor(obj[0], key)
    if not isinstance(obj, dict):
        return False

    keys = {str(k).lower() for k in obj.keys()}
    if any(hint in key.lower() for hint in ["factor", "select"]):
        return True
    if keys & {"selected_stocks", "candidate_stocks", "result_df", "selected_df", "candidates", "stocks"}:
        return True
    if "params" in keys and ("result" in keys or "results" in keys):
        return True
    return False


def _looks_like_backtest(obj: Any, key: str = "") -> bool:
    if isinstance(obj, list):
        if not obj:
            return False
        return _looks_like_backtest(obj[0], key)
    if not isinstance(obj, dict):
        return False

    keys = {str(k).lower() for k in obj.keys()}
    if "backtest" in key.lower():
        return True
    if keys & {"total_return", "annual_return", "max_drawdown", "sharpe", "equity_curve"}:
        return True
    summary = obj.get("summary")
    if isinstance(summary, dict):
        s_keys = {str(k).lower() for k in summary.keys()}
        if s_keys & {"total_return", "annual_return", "max_drawdown", "sharpe"}:
            return True
    return False


def _looks_like_decision(obj: Any, key: str = "") -> bool:
    if isinstance(obj, list):
        if not obj:
            return False
        return _looks_like_decision(obj[0], key)
    if not isinstance(obj, dict):
        return False

    keys = {str(k).lower() for k in obj.keys()}
    if "decision" in key.lower():
        return True
    if keys & {"risk_day", "is_risk_day", "action", "suggestion", "decision", "position_type"}:
        return True
    return False


def _looks_like_report(obj: Any, key: str = "") -> bool:
    if isinstance(obj, list):
        return "report" in key.lower() or "history" in key.lower()
    if not isinstance(obj, dict):
        return False

    keys = {str(k).lower() for k in obj.keys()}
    if "report" in key.lower():
        return True
    if keys & {"page", "module", "title", "payload"} and (keys & {"created_at", "timestamp", "time", "ts"}):
        return True
    return False


def _normalize_stock_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["代码", "名称", "分数", "涨跌幅", "所属概念"])

    df = pd.DataFrame(rows).copy()

    rename_map = {
        "code": "代码",
        "stock_code": "代码",
        "ts_code": "代码",
        "symbol": "代码",
        "ticker": "代码",
        "name": "名称",
        "stock_name": "名称",
        "score": "分数",
        "total_score": "分数",
        "rank_score": "分数",
        "change_pct": "涨跌幅",
        "pct_chg": "涨跌幅",
        "industry": "所属概念",
        "concept": "所属概念",
        "theme": "所属概念",
    }
    df = df.rename(columns=rename_map)

    for col in ["代码", "名称", "分数", "涨跌幅", "所属概念"]:
        if col not in df.columns:
            df[col] = None

    ordered_cols = ["代码", "名称", "分数", "涨跌幅", "所属概念"]
    other_cols = [col for col in df.columns if col not in ordered_cols]
    df = df[ordered_cols + other_cols]

    if "分数" in df.columns:
        try:
            df["分数"] = pd.to_numeric(df["分数"], errors="coerce").round(4)
        except Exception:
            pass

    return df


def _normalize_factor_record(record: Any, source_key: str = "") -> Optional[Dict[str, Any]]:
    if record is None:
        return None

    rows = _extract_stock_rows(record)
    if not rows and isinstance(record, pd.DataFrame):
        rows = record.to_dict(orient="records")
    if not rows and not _looks_like_factor(record, source_key):
        return None

    detail_df = _normalize_stock_df(rows)

    if isinstance(record, dict):
        params = _first_not_null(record.get("params"), record.get("filters"), record.get("conditions"), {})
        name = _first_not_null(record.get("preset_name"), record.get("scheme_name"), record.get("name"), record.get("title"), "当前选股结果")
        created_raw = _first_not_null(record.get("created_at"), record.get("timestamp"), record.get("time"), record.get("ts"))
    else:
        params = {}
        name = "当前选股结果"
        created_raw = None

    created_text = _format_dt(created_raw) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preview_codes = []
    if "代码" in detail_df.columns:
        preview_codes = [str(v) for v in detail_df["代码"].dropna().astype(str).head(10).tolist()]

    return {
        "record_type": "factor",
        "source_key": source_key,
        "created_at": created_text,
        "created_date": created_text[:10],
        "name": _safe_str(name) or "当前选股结果",
        "stock_count": int(len(detail_df)),
        "params_count": len(params) if isinstance(params, dict) else 0,
        "preview_codes": preview_codes,
        "detail_df": detail_df,
    }


def _merge_dicts(*items: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict):
            merged.update(item)
    return merged


def _normalize_backtest_record(record: Any, source_key: str = "") -> Optional[Dict[str, Any]]:
    if record is None or (not isinstance(record, dict) and not _looks_like_backtest(record, source_key)):
        return None

    if not isinstance(record, dict):
        return None

    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    merged = _merge_dicts(record, summary, metrics)

    created_raw = _first_not_null(merged.get("created_at"), merged.get("timestamp"), merged.get("time"), merged.get("ts"))
    created_text = _format_dt(created_raw) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stock_code = _first_not_null(merged.get("stock_code"), merged.get("code"), merged.get("ts_code"), merged.get("symbol"))
    stock_name = _first_not_null(merged.get("stock_name"), merged.get("name"))
    strategy_name = _first_not_null(merged.get("strategy_name"), merged.get("strategy"), merged.get("name"), "单股回测")
    start_date = _safe_str(_first_not_null(merged.get("start_date"), merged.get("begin_date"), merged.get("from_date")))
    end_date = _safe_str(_first_not_null(merged.get("end_date"), merged.get("finish_date"), merged.get("to_date")))

    total_return = _coerce_ratio(_first_not_null(merged.get("total_return"), merged.get("累计收益"), merged.get("strategy_return")))
    annual_return = _coerce_ratio(_first_not_null(merged.get("annual_return"), merged.get("年化收益"), merged.get("cagr")))
    max_drawdown = _coerce_ratio(_first_not_null(merged.get("max_drawdown"), merged.get("最大回撤"), merged.get("drawdown")))
    sharpe = _coerce_float(_first_not_null(merged.get("sharpe"), merged.get("sharp_ratio"), merged.get("夏普比率")))
    trade_count = _coerce_int(_first_not_null(merged.get("trade_count"), merged.get("trades"), merged.get("交易次数")))
    win_rate = _coerce_ratio(_first_not_null(merged.get("win_rate"), merged.get("胜率")))
    benchmark_return = _coerce_ratio(_first_not_null(merged.get("benchmark_return"), merged.get("bench_return"), merged.get("基准收益")))

    if all(v is None for v in [total_return, annual_return, max_drawdown, sharpe, trade_count, win_rate, benchmark_return]):
        return None

    return {
        "record_type": "backtest",
        "source_key": source_key,
        "created_at": created_text,
        "created_date": created_text[:10],
        "strategy_name": _safe_str(strategy_name) or "单股回测",
        "stock_code": _safe_str(stock_code),
        "stock_name": _safe_str(stock_name),
        "start_date": start_date,
        "end_date": end_date,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "benchmark_return": benchmark_return,
        "summary_text": _safe_str(_first_not_null(merged.get("summary_text"), merged.get("comment"), merged.get("conclusion"))),
    }


def _normalize_decision_record(record: Any, source_key: str = "") -> Optional[Dict[str, Any]]:
    if record is None or (not isinstance(record, dict) and not _looks_like_decision(record, source_key)):
        return None

    if not isinstance(record, dict):
        return None

    created_raw = _first_not_null(record.get("created_at"), record.get("timestamp"), record.get("time"), record.get("ts"))
    created_text = _format_dt(created_raw) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    risk_day = _coerce_bool(_first_not_null(record.get("risk_day"), record.get("is_risk_day"), record.get("risk_flag")))
    action = _first_not_null(record.get("action"), record.get("suggestion"), record.get("decision"), record.get("操作建议"))
    position_type = _first_not_null(record.get("position_type"), record.get("ticket_type"), record.get("持仓票型"))
    stock_code = _first_not_null(record.get("stock_code"), record.get("code"), record.get("ts_code"), record.get("symbol"))
    stock_name = _first_not_null(record.get("stock_name"), record.get("name"))
    score = _coerce_float(_first_not_null(record.get("score"), record.get("decision_score"), record.get("总分")))
    conclusion = _safe_str(_first_not_null(record.get("conclusion"), record.get("summary"), record.get("结论")))
    reduce_point = _safe_str(_first_not_null(record.get("reduce_point"), record.get("减仓点")))
    clear_point = _safe_str(_first_not_null(record.get("clear_point"), record.get("清仓点")))
    hold_condition = _safe_str(_first_not_null(record.get("hold_condition"), record.get("继续持有条件")))

    if all(v in [None, ""] for v in [action, position_type, stock_code, stock_name, conclusion]) and risk_day is None:
        return None

    return {
        "record_type": "decision",
        "source_key": source_key,
        "created_at": created_text,
        "created_date": created_text[:10],
        "stock_code": _safe_str(stock_code),
        "stock_name": _safe_str(stock_name),
        "risk_day": risk_day,
        "action": _safe_str(action),
        "position_type": _safe_str(position_type),
        "score": score,
        "conclusion": conclusion,
        "reduce_point": reduce_point,
        "clear_point": clear_point,
        "hold_condition": hold_condition,
    }


def _normalize_report_record(record: Any, source_key: str = "") -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    if not isinstance(record, dict):
        return None

    created_raw = _first_not_null(record.get("created_at"), record.get("timestamp"), record.get("time"), record.get("ts"))
    created_text = _format_dt(created_raw) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    page = _safe_str(_first_not_null(record.get("page"), record.get("module"), record.get("type"), source_key))
    title = _safe_str(_first_not_null(record.get("title"), record.get("name"), page and f"{page}报告"))
    summary = _safe_str(_first_not_null(record.get("summary"), record.get("desc"), record.get("description")))

    return {
        "record_type": "report",
        "source_key": source_key,
        "created_at": created_text,
        "created_date": created_text[:10],
        "page": page or "未标记",
        "title": title or "未命名报告",
        "summary": summary,
    }


def _normalize_preset_records(obj: Any, source_key: str = "") -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if obj is None:
        return records

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                name = _first_not_null(item.get("name"), item.get("preset_name"), item.get("scheme_name"), item.get("title"))
                params = _first_not_null(item.get("params"), item.get("filters"), item.get("conditions"), {})
                created_text = _format_dt(_first_not_null(item.get("created_at"), item.get("timestamp"), item.get("time"), item.get("ts")))
                if name:
                    records.append({
                        "name": _safe_str(name),
                        "created_at": created_text,
                        "condition_count": len(params) if isinstance(params, dict) else 0,
                        "source_key": source_key,
                    })
        return records

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                name = _first_not_null(value.get("name"), value.get("preset_name"), key)
                params = _first_not_null(value.get("params"), value.get("filters"), value.get("conditions"), value)
                created_text = _format_dt(_first_not_null(value.get("created_at"), value.get("timestamp"), value.get("time"), value.get("ts")))
                records.append({
                    "name": _safe_str(name),
                    "created_at": created_text,
                    "condition_count": len(params) if isinstance(params, dict) else 0,
                    "source_key": source_key,
                })
            elif isinstance(value, list):
                records.extend(_normalize_preset_records(value, source_key=source_key))
        return records

    return records


def _dedupe(records: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    result = []
    seen = set()
    for item in records:
        mark = tuple(_safe_str(item.get(key)) for key in keys)
        if mark in seen:
            continue
        seen.add(mark)
        result.append(item)
    return result


def _sort_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(item: Dict[str, Any]):
        dt = _parse_datetime(item.get("created_at"))
        return dt or datetime.min
    return sorted(records, key=_sort_key, reverse=True)


def _percent_text(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _collect_all_data(session_state: Any) -> Dict[str, Any]:
    session_items = _session_items(session_state)

    factor_records: List[Dict[str, Any]] = []
    backtest_records: List[Dict[str, Any]] = []
    decision_records: List[Dict[str, Any]] = []
    report_records: List[Dict[str, Any]] = []
    preset_records: List[Dict[str, Any]] = []
    detected_keys: List[str] = []

    current_factor = None
    current_backtest = None
    current_decision = None

    for key, value in session_items:
        key_lower = str(key).lower()

        if key in CURRENT_FACTOR_KEYS or key_lower in CURRENT_FACTOR_KEYS:
            normalized = _normalize_factor_record(value, source_key=str(key))
            if normalized:
                current_factor = normalized
                factor_records.append(normalized)
                detected_keys.append(str(key))
            continue

        if key in CURRENT_BACKTEST_KEYS or key_lower in CURRENT_BACKTEST_KEYS:
            normalized = _normalize_backtest_record(value, source_key=str(key))
            if normalized:
                current_backtest = normalized
                backtest_records.append(normalized)
                detected_keys.append(str(key))
            continue

        if key in CURRENT_DECISION_KEYS or key_lower in CURRENT_DECISION_KEYS:
            normalized = _normalize_decision_record(value, source_key=str(key))
            if normalized:
                current_decision = normalized
                decision_records.append(normalized)
                detected_keys.append(str(key))
            continue

        if any(hint in key_lower for hint in PRESET_HINTS):
            presets = _normalize_preset_records(value, source_key=str(key))
            if presets:
                preset_records.extend(presets)
                detected_keys.append(str(key))

        if isinstance(value, list):
            if any(hint in key_lower for hint in FACTOR_HISTORY_HINTS):
                for item in value:
                    normalized = _normalize_factor_record(item, source_key=str(key))
                    if normalized:
                        factor_records.append(normalized)
                detected_keys.append(str(key))
                continue

            if any(hint in key_lower for hint in BACKTEST_HISTORY_HINTS):
                for item in value:
                    normalized = _normalize_backtest_record(item, source_key=str(key))
                    if normalized:
                        backtest_records.append(normalized)
                detected_keys.append(str(key))
                continue

            if any(hint in key_lower for hint in DECISION_HISTORY_HINTS):
                for item in value:
                    normalized = _normalize_decision_record(item, source_key=str(key))
                    if normalized:
                        decision_records.append(normalized)
                detected_keys.append(str(key))
                continue

            if any(hint in key_lower for hint in REPORT_HISTORY_HINTS):
                for item in value:
                    normalized = _normalize_report_record(item, source_key=str(key))
                    if normalized:
                        report_records.append(normalized)

                    if isinstance(item, dict):
                        nested = _first_not_null(item.get("payload"), item.get("data"), item.get("result"), item.get("content"))
                        if _looks_like_factor(nested, key):
                            factor_n = _normalize_factor_record(nested, source_key=f"{key}.payload")
                            if factor_n:
                                factor_records.append(factor_n)
                        if _looks_like_backtest(nested, key):
                            back_n = _normalize_backtest_record(nested, source_key=f"{key}.payload")
                            if back_n:
                                backtest_records.append(back_n)
                        if _looks_like_decision(nested, key):
                            dec_n = _normalize_decision_record(nested, source_key=f"{key}.payload")
                            if dec_n:
                                decision_records.append(dec_n)
                detected_keys.append(str(key))
                continue

            for item in value:
                if _looks_like_factor(item, key_lower):
                    normalized = _normalize_factor_record(item, source_key=str(key))
                    if normalized:
                        factor_records.append(normalized)
                if _looks_like_backtest(item, key_lower):
                    normalized = _normalize_backtest_record(item, source_key=str(key))
                    if normalized:
                        backtest_records.append(normalized)
                if _looks_like_decision(item, key_lower):
                    normalized = _normalize_decision_record(item, source_key=str(key))
                    if normalized:
                        decision_records.append(normalized)
                if _looks_like_report(item, key_lower):
                    normalized = _normalize_report_record(item, source_key=str(key))
                    if normalized:
                        report_records.append(normalized)

        elif isinstance(value, dict):
            if _looks_like_factor(value, key_lower):
                normalized = _normalize_factor_record(value, source_key=str(key))
                if normalized:
                    factor_records.append(normalized)
                    detected_keys.append(str(key))
            if _looks_like_backtest(value, key_lower):
                normalized = _normalize_backtest_record(value, source_key=str(key))
                if normalized:
                    backtest_records.append(normalized)
                    detected_keys.append(str(key))
            if _looks_like_decision(value, key_lower):
                normalized = _normalize_decision_record(value, source_key=str(key))
                if normalized:
                    decision_records.append(normalized)
                    detected_keys.append(str(key))
            if _looks_like_report(value, key_lower):
                normalized = _normalize_report_record(value, source_key=str(key))
                if normalized:
                    report_records.append(normalized)
                    detected_keys.append(str(key))

    factor_records = _sort_records(_dedupe(factor_records, ["created_at", "name", "stock_count", "source_key"]))
    backtest_records = _sort_records(_dedupe(backtest_records, ["created_at", "stock_code", "strategy_name", "start_date", "end_date"]))
    decision_records = _sort_records(_dedupe(decision_records, ["created_at", "stock_code", "action", "risk_day", "source_key"]))
    report_records = _sort_records(_dedupe(report_records, ["created_at", "page", "title", "source_key"]))
    preset_records = _dedupe(preset_records, ["name", "source_key"])

    if current_factor is None and factor_records:
        current_factor = factor_records[0]
    if current_backtest is None and backtest_records:
        current_backtest = backtest_records[0]
    if current_decision is None and decision_records:
        current_decision = decision_records[0]

    total_returns = [item["total_return"] for item in backtest_records if item.get("total_return") is not None]
    max_drawdowns = [item["max_drawdown"] for item in backtest_records if item.get("max_drawdown") is not None]
    risk_flags = [item["risk_day"] for item in decision_records if item.get("risk_day") is not None]

    overview = {
        "factor_count": len(factor_records),
        "backtest_count": len(backtest_records),
        "decision_count": len(decision_records),
        "report_count": len(report_records),
        "preset_count": len(preset_records),
        "current_stock_count": current_factor["stock_count"] if current_factor else 0,
        "avg_backtest_return": sum(total_returns) / len(total_returns) if total_returns else None,
        "avg_max_drawdown": sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else None,
        "risk_day_ratio": (sum(1 for x in risk_flags if x) / len(risk_flags)) if risk_flags else None,
    }

    activity_rows = []
    for row in factor_records:
        activity_rows.append({"日期": row["created_date"], "类型": "选股", "数量": 1})
    for row in backtest_records:
        activity_rows.append({"日期": row["created_date"], "类型": "回测", "数量": 1})
    for row in decision_records:
        activity_rows.append({"日期": row["created_date"], "类型": "决策", "数量": 1})
    for row in report_records:
        activity_rows.append({"日期": row["created_date"], "类型": "报告", "数量": 1})

    activity_df = pd.DataFrame(activity_rows)
    if not activity_df.empty:
        activity_df = activity_df.groupby(["日期", "类型"], as_index=False)["数量"].sum()
        activity_df = activity_df.sort_values(["日期", "类型"])

    backtest_scatter_df = pd.DataFrame([
        {
            "股票代码": item["stock_code"],
            "股票名称": item["stock_name"],
            "策略名称": item["strategy_name"],
            "累计收益": item["total_return"],
            "最大回撤": item["max_drawdown"],
            "夏普比率": item["sharpe"],
            "交易次数": item["trade_count"],
            "时间": item["created_at"],
        }
        for item in backtest_records
        if item.get("total_return") is not None and item.get("max_drawdown") is not None
    ])

    action_counter = Counter()
    for item in decision_records:
        action = item.get("action") or "未标记"
        action_counter[action] += 1
    decision_action_df = pd.DataFrame(
        [{"动作": k, "数量": v} for k, v in action_counter.items()]
    ).sort_values("数量", ascending=False) if action_counter else pd.DataFrame(columns=["动作", "数量"])

    stock_counter = Counter()
    for item in factor_records:
        for code in item.get("preview_codes", []):
            stock_counter[code] += 1
    factor_hot_df = pd.DataFrame(
        [{"股票代码": k, "出现次数": v} for k, v in stock_counter.most_common(10)]
    ) if stock_counter else pd.DataFrame(columns=["股票代码", "出现次数"])

    report_page_counter = Counter()
    for item in report_records:
        report_page_counter[item.get("page") or "未标记"] += 1
    report_page_df = pd.DataFrame(
        [{"页面": k, "数量": v} for k, v in report_page_counter.items()]
    ).sort_values("数量", ascending=False) if report_page_counter else pd.DataFrame(columns=["页面", "数量"])

    factor_recent_df = pd.DataFrame([
        {
            "时间": item["created_at"],
            "名称": item["name"],
            "候选股数量": item["stock_count"],
            "条件数量": item["params_count"],
            "来源": item["source_key"],
        }
        for item in factor_records[:20]
    ])

    backtest_recent_df = pd.DataFrame([
        {
            "时间": item["created_at"],
            "股票代码": item["stock_code"],
            "股票名称": item["stock_name"],
            "策略名称": item["strategy_name"],
            "累计收益": _percent_text(item["total_return"]),
            "年化收益": _percent_text(item["annual_return"]),
            "最大回撤": _percent_text(item["max_drawdown"]),
            "夏普比率": "-" if item["sharpe"] is None else round(item["sharpe"], 4),
            "交易次数": "-" if item["trade_count"] is None else item["trade_count"],
        }
        for item in backtest_records[:20]
    ])

    decision_recent_df = pd.DataFrame([
        {
            "时间": item["created_at"],
            "股票代码": item["stock_code"],
            "股票名称": item["stock_name"],
            "是否风险日": "是" if item["risk_day"] else ("否" if item["risk_day"] is not None else "-"),
            "动作": item["action"] or "-",
            "票型": item["position_type"] or "-",
            "结论": item["conclusion"] or "-",
        }
        for item in decision_records[:20]
    ])

    report_recent_df = pd.DataFrame([
        {
            "时间": item["created_at"],
            "页面": item["page"],
            "标题": item["title"],
            "摘要": item["summary"] or "-",
            "来源": item["source_key"],
        }
        for item in report_records[:20]
    ])

    preset_df = pd.DataFrame([
        {
            "方案名": item["name"],
            "条件数量": item["condition_count"],
            "创建时间": item["created_at"] or "-",
            "来源": item["source_key"],
        }
        for item in preset_records
    ])

    latest_backtest_df = pd.DataFrame()
    if current_backtest:
        latest_backtest_df = pd.DataFrame([{
            "时间": current_backtest["created_at"],
            "股票代码": current_backtest["stock_code"] or "-",
            "股票名称": current_backtest["stock_name"] or "-",
            "策略名称": current_backtest["strategy_name"],
            "开始日期": current_backtest["start_date"] or "-",
            "结束日期": current_backtest["end_date"] or "-",
            "累计收益": _percent_text(current_backtest["total_return"]),
            "年化收益": _percent_text(current_backtest["annual_return"]),
            "最大回撤": _percent_text(current_backtest["max_drawdown"]),
            "夏普比率": "-" if current_backtest["sharpe"] is None else round(current_backtest["sharpe"], 4),
            "交易次数": "-" if current_backtest["trade_count"] is None else current_backtest["trade_count"],
            "胜率": _percent_text(current_backtest["win_rate"]),
        }])

    latest_decision_df = pd.DataFrame()
    if current_decision:
        latest_decision_df = pd.DataFrame([{
            "时间": current_decision["created_at"],
            "股票代码": current_decision["stock_code"] or "-",
            "股票名称": current_decision["stock_name"] or "-",
            "是否风险日": "是" if current_decision["risk_day"] else ("否" if current_decision["risk_day"] is not None else "-"),
            "动作": current_decision["action"] or "-",
            "票型": current_decision["position_type"] or "-",
            "减仓点": current_decision["reduce_point"] or "-",
            "清仓点": current_decision["clear_point"] or "-",
            "继续持有条件": current_decision["hold_condition"] or "-",
            "结论": current_decision["conclusion"] or "-",
        }])

    return {
        "overview": overview,
        "current_factor": current_factor,
        "current_backtest": current_backtest,
        "current_decision": current_decision,
        "activity_df": activity_df,
        "backtest_scatter_df": backtest_scatter_df,
        "decision_action_df": decision_action_df,
        "factor_hot_df": factor_hot_df,
        "report_page_df": report_page_df,
        "factor_recent_df": factor_recent_df,
        "backtest_recent_df": backtest_recent_df,
        "decision_recent_df": decision_recent_df,
        "report_recent_df": report_recent_df,
        "preset_df": preset_df,
        "latest_factor_df": current_factor["detail_df"] if current_factor else pd.DataFrame(),
        "latest_backtest_df": latest_backtest_df,
        "latest_decision_df": latest_decision_df,
        "detected_keys": sorted(set(detected_keys)),
    }


def collect_dashboard_data(session_state: Any) -> Dict[str, Any]:
    return _collect_all_data(session_state)