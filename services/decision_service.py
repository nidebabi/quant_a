from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        text = str(value).strip()
        return text if text else default
    except Exception:
        return default


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return default
        except Exception:
            pass
        return float(value)

    text = _safe_str(value)
    if not text:
        return default

    text = text.replace(",", "").replace("%", "")
    try:
        return float(text)
    except Exception:
        return default


def _to_ratio(value: Any, default: Optional[float] = None) -> Optional[float]:
    number = _to_float(value, default=None)
    if number is None:
        return default
    if abs(number) > 1.5:
        return number / 100.0
    return number


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value, default=None)
    if number is None:
        return default
    return int(round(number))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
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


# =========================
# 兼容旧版：因子选股 -> 决策候选
# =========================
def build_decision_candidates_from_factor(factor_result: Any) -> List[Dict[str, Any]]:
    rows = _extract_stock_rows(factor_result)

    if not rows:
        if isinstance(factor_result, dict):
            rows = [factor_result]
        elif isinstance(factor_result, pd.DataFrame):
            rows = factor_result.to_dict(orient="records")
        elif isinstance(factor_result, list):
            rows = [x for x in factor_result if isinstance(x, dict)]

    candidates: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue

        stock_code = _safe_str(
            _first_not_none(
                row.get("stock_code"),
                row.get("code"),
                row.get("ts_code"),
                row.get("symbol"),
                row.get("ticker"),
                row.get("代码"),
            )
        )
        stock_name = _safe_str(
            _first_not_none(
                row.get("stock_name"),
                row.get("name"),
                row.get("股票名称"),
                row.get("名称"),
            )
        )
        score = _to_float(
            _first_not_none(
                row.get("score"),
                row.get("total_score"),
                row.get("rank_score"),
                row.get("综合评分"),
                row.get("分数"),
            )
        )
        concept = _safe_str(
            _first_not_none(
                row.get("concept"),
                row.get("industry"),
                row.get("theme"),
                row.get("所属概念"),
                row.get("板块"),
            )
        )
        change_pct = _to_ratio(
            _first_not_none(
                row.get("change_pct"),
                row.get("pct_chg"),
                row.get("涨跌幅"),
            )
        )

        if not stock_code and not stock_name:
            continue

        candidate = {
            "id": f"candidate_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{idx}",
            "stock_code": stock_code,
            "stock_name": stock_name,
            "score": score,
            "concept": concept,
            "change_pct": change_pct,
            "source": "factor",
            "created_at": _now_str(),
            "raw": deepcopy(row),
        }
        candidates.append(candidate)

    return candidates


def _find_first_session_value(session_state: Any, keys: List[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if key in session_state and session_state.get(key) is not None:
            return key, session_state.get(key)
    return None, None


def get_latest_factor_candidates(session_state: Any) -> Dict[str, Any]:
    """
    从当前会话里尽量兼容地获取最近一次选股结果，并转换成决策候选。
    """
    candidate_keys = [
        "current_factor_result",
        "factor_result",
        "current_select_result",
        "current_factor_selection",
        "latest_factor_result",
        "selected_result",
        "restored_factor_result",
    ]

    source_key, raw = _find_first_session_value(session_state, candidate_keys)

    if raw is None:
        history = session_state.get("factor_history", [])
        if isinstance(history, list) and history:
            raw = history[0]
            source_key = "factor_history[0]"

    if raw is None:
        report_history = session_state.get("report_history", [])
        if isinstance(report_history, list):
            for item in report_history:
                if not isinstance(item, dict):
                    continue
                payload = item.get("payload")
                page = _safe_str(item.get("page"))
                module = _safe_str(item.get("module"))
                if module == "factor" or page == "因子选股":
                    raw = payload
                    source_key = "report_history.factor"
                    break

    if raw is None:
        return {
            "found": False,
            "source_key": "",
            "count": 0,
            "candidates": [],
        }

    candidates = build_decision_candidates_from_factor(raw)
    return {
        "found": len(candidates) > 0,
        "source_key": source_key or "",
        "count": len(candidates),
        "candidates": candidates,
    }


def get_latest_backtest_snapshot(session_state: Any) -> Dict[str, Any]:
    candidate_keys = [
        "current_backtest_result",
        "backtest_result",
        "latest_backtest_result",
        "single_backtest_result",
        "restored_backtest_result",
    ]

    raw = None
    for key in candidate_keys:
        if key in session_state and session_state.get(key) is not None:
            raw = session_state.get(key)
            break

    if raw is None:
        return {
            "found": False,
            "stock_code": "",
            "stock_name": "",
            "strategy_name": "",
            "total_return": None,
            "annual_return": None,
            "max_drawdown": None,
            "win_rate": None,
            "sharpe": None,
            "trade_count": None,
            "start_date": "",
            "end_date": "",
            "summary": {},
        }

    if not isinstance(raw, dict):
        return {
            "found": False,
            "stock_code": "",
            "stock_name": "",
            "strategy_name": "",
            "total_return": None,
            "annual_return": None,
            "max_drawdown": None,
            "win_rate": None,
            "sharpe": None,
            "trade_count": None,
            "start_date": "",
            "end_date": "",
            "summary": {},
        }

    summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}

    merged = {}
    merged.update(raw)
    merged.update(summary)
    merged.update(metrics)

    stock_code = _safe_str(
        _first_not_none(
            merged.get("stock_code"),
            merged.get("code"),
            merged.get("ts_code"),
            merged.get("symbol"),
        )
    )
    stock_name = _safe_str(
        _first_not_none(
            merged.get("stock_name"),
            merged.get("name"),
        )
    )
    strategy_name = _safe_str(
        _first_not_none(
            merged.get("strategy_name"),
            merged.get("strategy"),
            merged.get("name"),
        )
    )

    return {
        "found": True,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "strategy_name": strategy_name or "单股回测",
        "total_return": _to_ratio(_first_not_none(merged.get("total_return"), merged.get("strategy_return"))),
        "annual_return": _to_ratio(merged.get("annual_return")),
        "max_drawdown": _to_ratio(_first_not_none(merged.get("max_drawdown"), merged.get("drawdown"))),
        "win_rate": _to_ratio(merged.get("win_rate")),
        "sharpe": _to_float(_first_not_none(merged.get("sharpe"), merged.get("sharp_ratio"))),
        "trade_count": _to_int(_first_not_none(merged.get("trade_count"), merged.get("trades")), default=0),
        "start_date": _safe_str(_first_not_none(merged.get("start_date"), merged.get("begin_date"))),
        "end_date": _safe_str(_first_not_none(merged.get("end_date"), merged.get("finish_date"))),
        "summary": raw,
    }


def classify_position_type(
    volatility_pct: Optional[float],
    max_drawdown: Optional[float],
    trade_count: int,
) -> str:
    vol = volatility_pct if volatility_pct is not None else 0.0
    dd = max_drawdown if max_drawdown is not None else 0.0

    if vol >= 8.0 or dd <= -0.18:
        return "高波动题材"
    if vol >= 5.0 or dd <= -0.10 or trade_count >= 12:
        return "情绪波动型"
    if vol <= 3.0 and abs(dd) <= 0.08:
        return "趋势中军"
    return "弱趋势震荡"


def build_score_breakdown(
    total_return: Optional[float],
    max_drawdown: Optional[float],
    win_rate: Optional[float],
    sharpe: Optional[float],
    volatility_pct: Optional[float],
    pnl_pct: Optional[float],
) -> Dict[str, float]:
    score_return = 10.0
    score_drawdown = 10.0
    score_win_rate = 10.0
    score_sharpe = 10.0
    score_risk = 10.0

    if total_return is not None:
        if total_return >= 0.20:
            score_return = 20
        elif total_return >= 0.10:
            score_return = 17
        elif total_return >= 0.03:
            score_return = 14
        elif total_return >= 0:
            score_return = 11
        elif total_return >= -0.03:
            score_return = 8
        elif total_return >= -0.08:
            score_return = 5
        else:
            score_return = 2

    if max_drawdown is not None:
        dd_abs = abs(max_drawdown)
        if dd_abs <= 0.03:
            score_drawdown = 20
        elif dd_abs <= 0.06:
            score_drawdown = 17
        elif dd_abs <= 0.10:
            score_drawdown = 14
        elif dd_abs <= 0.15:
            score_drawdown = 10
        elif dd_abs <= 0.20:
            score_drawdown = 6
        else:
            score_drawdown = 2

    if win_rate is not None:
        if win_rate >= 0.70:
            score_win_rate = 20
        elif win_rate >= 0.60:
            score_win_rate = 17
        elif win_rate >= 0.50:
            score_win_rate = 14
        elif win_rate >= 0.40:
            score_win_rate = 10
        elif win_rate >= 0.30:
            score_win_rate = 6
        else:
            score_win_rate = 2

    if sharpe is not None:
        if sharpe >= 2.0:
            score_sharpe = 20
        elif sharpe >= 1.3:
            score_sharpe = 17
        elif sharpe >= 0.8:
            score_sharpe = 14
        elif sharpe >= 0.3:
            score_sharpe = 10
        elif sharpe >= 0.0:
            score_sharpe = 6
        else:
            score_sharpe = 2

    risk_penalty = 0.0
    if volatility_pct is not None:
        if volatility_pct >= 10:
            risk_penalty += 10
        elif volatility_pct >= 7:
            risk_penalty += 6
        elif volatility_pct >= 5:
            risk_penalty += 3

    if pnl_pct is not None:
        if pnl_pct <= -12:
            risk_penalty += 8
        elif pnl_pct <= -7:
            risk_penalty += 5
        elif pnl_pct <= -3:
            risk_penalty += 2

    score_risk = _clamp(20 - risk_penalty, 2, 20)

    total_score = round(score_return + score_drawdown + score_win_rate + score_sharpe + score_risk, 2)

    return {
        "收益表现": round(score_return, 2),
        "回撤控制": round(score_drawdown, 2),
        "胜率表现": round(score_win_rate, 2),
        "稳定性": round(score_sharpe, 2),
        "风险状态": round(score_risk, 2),
        "综合评分": total_score,
    }


def infer_risk_level(total_score: float) -> str:
    if total_score >= 80:
        return "低风险"
    if total_score >= 60:
        return "中风险"
    if total_score >= 40:
        return "偏高风险"
    return "高风险"


def infer_action(
    mode: str,
    risk_level: str,
    pnl_pct: Optional[float],
    position_type: str,
) -> str:
    if mode == "观察":
        if risk_level in {"低风险", "中风险"}:
            return "可加入观察名单"
        return "仅跟踪，不急于介入"

    if pnl_pct is not None and pnl_pct <= -10:
        return "优先控制风险，逢反弹减仓或清仓"
    if pnl_pct is not None and pnl_pct >= 12 and risk_level in {"偏高风险", "高风险"}:
        return "建议锁定利润，主动减仓"
    if risk_level == "低风险":
        return "可继续持有"
    if risk_level == "中风险":
        return "以持有观察为主"
    if position_type in {"高波动题材", "情绪波动型"}:
        return "建议轻仓博弈，设好止损"
    return "建议减仓观察"


def infer_position_advice(
    mode: str,
    risk_level: str,
    position_type: str,
    pnl_pct: Optional[float],
) -> str:
    if mode == "观察":
        if risk_level == "低风险":
            return "试错仓 10%~20%"
        if risk_level == "中风险":
            return "轻仓 5%~10%"
        return "先空仓观察"

    if risk_level == "低风险":
        if position_type == "趋势中军":
            return "可维持原仓位，强则续抱"
        return "控制仓位在中低水平，防波动"
    if risk_level == "中风险":
        return "建议半仓以内，边走边看"
    if pnl_pct is not None and pnl_pct < 0:
        return "建议主动降仓，先把回撤压住"
    return "建议轻仓或空仓"


def build_price_conditions(
    current_price: Optional[float],
    cost_price: Optional[float],
    position_type: str,
) -> Dict[str, str]:
    if current_price is None:
        current_price = 0.0
    if cost_price is None:
        cost_price = 0.0

    ref_price = current_price if current_price and current_price > 0 else cost_price

    if ref_price <= 0:
        return {
            "reduce_point": "跌破短线关键支撑位时减仓",
            "clear_point": "放量跌破防守位时清仓",
            "hold_condition": "量价稳定、走势不破坏则继续观察",
        }

    if position_type == "趋势中军":
        reduce_price = ref_price * 0.97
        clear_price = ref_price * 0.93
    elif position_type == "高波动题材":
        reduce_price = ref_price * 0.96
        clear_price = ref_price * 0.91
    elif position_type == "情绪波动型":
        reduce_price = ref_price * 0.965
        clear_price = ref_price * 0.92
    else:
        reduce_price = ref_price * 0.97
        clear_price = ref_price * 0.92

    return {
        "reduce_point": f"跌破 {reduce_price:.2f} 附近先减仓观察",
        "clear_point": f"跌破 {clear_price:.2f} 附近优先执行清仓",
        "hold_condition": "价格稳在防守位之上，且无明显放量转弱，可继续持有/观察",
    }


def build_reasons(
    total_return: Optional[float],
    max_drawdown: Optional[float],
    win_rate: Optional[float],
    sharpe: Optional[float],
    pnl_pct: Optional[float],
    volatility_pct: Optional[float],
    position_type: str,
    factor_score: Optional[float],
    factor_concept: str,
    factor_change_pct: Optional[float],
) -> List[str]:
    reasons: List[str] = []

    if total_return is not None:
        reasons.append(f"历史回测累计收益为 {total_return * 100:.2f}%")
    else:
        reasons.append("暂未获取到完整的历史回测收益数据")

    if max_drawdown is not None:
        reasons.append(f"最大回撤为 {max_drawdown * 100:.2f}%")
    if win_rate is not None:
        reasons.append(f"历史胜率为 {win_rate * 100:.2f}%")
    if sharpe is not None:
        reasons.append(f"稳定性指标（夏普）为 {sharpe:.2f}")
    if pnl_pct is not None:
        reasons.append(f"按当前价格估算，持仓浮盈亏约为 {pnl_pct:.2f}%")
    if volatility_pct is not None:
        reasons.append(f"你输入的日内振幅参考为 {volatility_pct:.2f}%")
    if factor_score is not None:
        reasons.append(f"该标的在最近一次选股结果中的参考分数为 {factor_score:.2f}")
    if factor_concept:
        reasons.append(f"该标的关联概念/方向：{factor_concept}")
    if factor_change_pct is not None:
        reasons.append(f"该标的选股结果里的参考涨跌幅为 {factor_change_pct * 100:.2f}%")

    reasons.append(f"当前票型识别为：{position_type}")
    return reasons


def generate_decision_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = _safe_str(payload.get("mode"), "持仓")
    stock_code = _safe_str(payload.get("stock_code"))
    stock_name = _safe_str(payload.get("stock_name"))
    current_price = _to_float(payload.get("current_price"))
    cost_price = _to_float(payload.get("cost_price"))
    position_size = _to_int(payload.get("position_size"), default=0)
    volatility_pct = _to_float(payload.get("volatility_pct"))
    note = _safe_str(payload.get("note"))

    factor_score = _to_float(payload.get("factor_score"))
    factor_concept = _safe_str(payload.get("factor_concept"))
    factor_change_pct = _to_ratio(payload.get("factor_change_pct"))

    backtest_total_return = _to_ratio(payload.get("backtest_total_return"))
    backtest_annual_return = _to_ratio(payload.get("backtest_annual_return"))
    backtest_max_drawdown = _to_ratio(payload.get("backtest_max_drawdown"))
    backtest_win_rate = _to_ratio(payload.get("backtest_win_rate"))
    backtest_sharpe = _to_float(payload.get("backtest_sharpe"))
    backtest_trade_count = _to_int(payload.get("backtest_trade_count"), default=0)
    backtest_strategy_name = _safe_str(payload.get("backtest_strategy_name"), "单股回测")
    backtest_start_date = _safe_str(payload.get("backtest_start_date"))
    backtest_end_date = _safe_str(payload.get("backtest_end_date"))

    pnl_pct = None
    if current_price is not None and cost_price is not None and cost_price > 0:
        pnl_pct = round((current_price / cost_price - 1) * 100, 2)

    position_type = classify_position_type(
        volatility_pct=volatility_pct,
        max_drawdown=backtest_max_drawdown,
        trade_count=backtest_trade_count,
    )

    score_breakdown = build_score_breakdown(
        total_return=backtest_total_return,
        max_drawdown=backtest_max_drawdown,
        win_rate=backtest_win_rate,
        sharpe=backtest_sharpe,
        volatility_pct=volatility_pct,
        pnl_pct=pnl_pct,
    )
    total_score = score_breakdown["综合评分"]
    risk_level = infer_risk_level(total_score)
    action = infer_action(
        mode=mode,
        risk_level=risk_level,
        pnl_pct=pnl_pct,
        position_type=position_type,
    )
    position_advice = infer_position_advice(
        mode=mode,
        risk_level=risk_level,
        position_type=position_type,
        pnl_pct=pnl_pct,
    )

    conditions = build_price_conditions(
        current_price=current_price,
        cost_price=cost_price,
        position_type=position_type,
    )

    reasons = build_reasons(
        total_return=backtest_total_return,
        max_drawdown=backtest_max_drawdown,
        win_rate=backtest_win_rate,
        sharpe=backtest_sharpe,
        pnl_pct=pnl_pct,
        volatility_pct=volatility_pct,
        position_type=position_type,
        factor_score=factor_score,
        factor_concept=factor_concept,
        factor_change_pct=factor_change_pct,
    )

    is_risk_day = risk_level in {"偏高风险", "高风险"}

    summary_text = (
        f"{stock_code or stock_name or '当前标的'} 当前识别为【{position_type}】，"
        f"综合评分 {total_score:.2f}，风险等级【{risk_level}】，"
        f"建议动作：{action}。"
    )

    return {
        "id": f"decision_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "module": "decision",
        "title": f"{stock_code or stock_name or '未命名标的'} 决策结果",
        "created_at": _now_str(),
        "mode": mode,
        "stock_code": stock_code,
        "stock_name": stock_name,
        "current_price": current_price,
        "cost_price": cost_price,
        "position_size": position_size,
        "volatility_pct": volatility_pct,
        "pnl_pct": pnl_pct,
        "position_type": position_type,
        "score": total_score,
        "score_breakdown": score_breakdown,
        "risk_level": risk_level,
        "risk_day": is_risk_day,
        "action": action,
        "position_advice": position_advice,
        "reduce_point": conditions["reduce_point"],
        "clear_point": conditions["clear_point"],
        "hold_condition": conditions["hold_condition"],
        "conclusion": summary_text,
        "summary": summary_text,
        "reasons": reasons,
        "note": note,
        "factor_reference": {
            "score": factor_score,
            "concept": factor_concept,
            "change_pct": factor_change_pct,
        },
        "backtest_summary": {
            "strategy_name": backtest_strategy_name,
            "start_date": backtest_start_date,
            "end_date": backtest_end_date,
            "total_return": backtest_total_return,
            "annual_return": backtest_annual_return,
            "max_drawdown": backtest_max_drawdown,
            "win_rate": backtest_win_rate,
            "sharpe": backtest_sharpe,
            "trade_count": backtest_trade_count,
        },
        "payload": deepcopy(payload),
    }


def _ensure_list(session_state: Any, key: str) -> None:
    if key not in session_state or not isinstance(session_state.get(key), list):
        session_state[key] = []


def save_decision_to_session(session_state: Any, result: Dict[str, Any]) -> None:
    session_state["current_decision_result"] = deepcopy(result)

    _ensure_list(session_state, "decision_history")
    session_state["decision_history"].insert(0, deepcopy(result))

    report_record = {
        "id": result["id"],
        "module": "decision",
        "page": "决策中心",
        "title": result["title"],
        "summary": result["summary"],
        "created_at": result["created_at"],
        "payload": deepcopy(result),
    }

    _ensure_list(session_state, "report_history")
    session_state["report_history"].insert(0, report_record)


def get_decision_history_df(session_state: Any) -> pd.DataFrame:
    history = session_state.get("decision_history", [])
    if not isinstance(history, list) or not history:
        return pd.DataFrame(
            columns=[
                "时间",
                "股票代码",
                "股票名称",
                "模式",
                "票型",
                "综合评分",
                "风险等级",
                "是否风险日",
                "动作",
                "仓位建议",
            ]
        )

    rows = []
    for item in history:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "时间": _safe_str(item.get("created_at")),
                "股票代码": _safe_str(item.get("stock_code")),
                "股票名称": _safe_str(item.get("stock_name")),
                "模式": _safe_str(item.get("mode")),
                "票型": _safe_str(item.get("position_type")),
                "综合评分": item.get("score"),
                "风险等级": _safe_str(item.get("risk_level")),
                "是否风险日": "是" if item.get("risk_day") else "否",
                "动作": _safe_str(item.get("action")),
                "仓位建议": _safe_str(item.get("position_advice")),
            }
        )
    return pd.DataFrame(rows)


def restore_decision_from_history(session_state: Any, index: int) -> Optional[Dict[str, Any]]:
    history = session_state.get("decision_history", [])
    if not isinstance(history, list):
        return None
    if index < 0 or index >= len(history):
        return None

    item = history[index]
    if not isinstance(item, dict):
        return None

    session_state["current_decision_result"] = deepcopy(item)
    return deepcopy(item)