import json
from datetime import date
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="次日操作决策模块",
    page_icon="🧭",
    layout="wide",
)

st.title("次日操作决策模块")
st.caption("先搭完整框架，当前版本以手工输入 + 规则判断为主，后面可再接自动数据、AI、推演模块。")


# =========================
# 默认数据
# =========================
DEFAULT_HOLDINGS = pd.DataFrame([
    {
        "code": "603817",
        "name": "海峡环保",
        "shares": 1000,
        "cost": 7.65,
        "y_open": 7.70,
        "y_high": 8.19,
        "y_low": 7.69,
        "y_close": 8.18,
        "y_turnover": 11.75,
        "y_amp": 6.50,
        "is_limit_up": 0,
        "remark": ""
    }
])

DEFAULT_CANDIDATES = pd.DataFrame([
    {
        "code": "000001",
        "name": "示例A",
        "sector": "示例板块",
        "sector_rank": 1,
        "sector_continuity": 80,
        "is_front_row": 1,
        "is_core": 1,
        "strong_clean": 85,
        "buy_comfort": 75,
        "open_below_expect_risk": 25,
        "linkage_score": 80,
        "y_turnover": 12.5,
        "y_amount_yi": 18.2,
        "y_amp": 6.3,
        "y_close": 12.80,
        "y_high": 13.10,
        "y_low": 12.20
    }
])

if "holdings_df" not in st.session_state:
    st.session_state.holdings_df = DEFAULT_HOLDINGS.copy()

if "candidates_df" not in st.session_state:
    st.session_state.candidates_df = DEFAULT_CANDIDATES.copy()

if "decision_result" not in st.session_state:
    st.session_state.decision_result = None


# =========================
# 工具函数
# =========================
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def calc_risk_score(
    index_change_pct: float,
    breadth_pct: float,
    sentiment_score: float,
    overnight_score: float,
    hot_sector_continuity: float,
    news_risk_score: float
) -> Dict[str, Any]:
    score = 50.0

    score += np.clip(index_change_pct * 5, -15, 15)
    score += np.clip((breadth_pct - 50) * 0.4, -15, 15)
    score += np.clip((sentiment_score - 50) * 0.35, -18, 18)
    score += np.clip((overnight_score - 50) * 0.25, -10, 10)
    score += np.clip((hot_sector_continuity - 50) * 0.25, -10, 10)
    score -= np.clip((news_risk_score - 50) * 0.35, -20, 20)

    score = float(np.clip(score, 0, 100))

    if score >= 65:
        level = "低风险日"
        advice = "允许正常找买点，优先做强板块前排。"
    elif score >= 45:
        level = "中性日"
        advice = "可做，但要收缩出手次数，只做高确定性票。"
    else:
        level = "高风险日"
        advice = "优先控仓或空仓观察，少做追高。"

    return {
        "risk_score": round(score, 2),
        "risk_level": level,
        "risk_advice": advice,
    }


def classify_stock_type(y_amp: float, y_turnover: float) -> str:
    if y_amp >= 8 or y_turnover >= 20:
        return "高波动"
    return "趋势中军"


def analyze_holding_row(row: pd.Series, risk_level: str) -> Dict[str, Any]:
    code = str(row.get("code", ""))
    name = str(row.get("name", ""))
    shares = safe_int(row.get("shares", 0))
    cost = safe_float(row.get("cost", 0))
    y_open = safe_float(row.get("y_open", 0))
    y_high = safe_float(row.get("y_high", 0))
    y_low = safe_float(row.get("y_low", 0))
    y_close = safe_float(row.get("y_close", 0))
    y_turnover = safe_float(row.get("y_turnover", 0))
    y_amp = safe_float(row.get("y_amp", 0))
    is_limit_up = safe_int(row.get("is_limit_up", 0))

    stock_type = classify_stock_type(y_amp, y_turnover)

    # 简单规则版，后续可替换成更复杂逻辑
    reduce_price = round(max(y_close * 1.02, cost * 1.03), 2)
    clear_price = round(max(cost * 0.95, y_low * 0.98), 2)

    hold_conditions = []
    hold_conditions.append(f"不跌破昨日低点附近（{y_low:.2f}）")
    hold_conditions.append(f"若高开后能站稳昨日收盘（{y_close:.2f}）之上，可继续持有")
    if is_limit_up == 1:
        hold_conditions.append("若出现明显炸板或放量回落，优先兑现")

    if risk_level == "高风险日":
        reduce_price = round(max(y_close * 1.01, cost * 1.01), 2)
        clear_price = round(max(cost * 0.97, y_low * 0.99), 2)

    if y_close >= cost * 1.08:
        action_hint = "已有较厚浮盈，优先考虑冲高减仓。"
    elif y_close < cost:
        action_hint = "仍在成本下方，优先盯防低开和失守关键位。"
    else:
        action_hint = "小幅浮盈，先看开盘承接，强则留，弱则减。"

    return {
        "code": code,
        "name": name,
        "shares": shares,
        "cost": round(cost, 2),
        "昨日开盘": round(y_open, 2),
        "昨日最高": round(y_high, 2),
        "昨日最低": round(y_low, 2),
        "昨日收盘": round(y_close, 2),
        "票型判定": stock_type,
        "减仓点": reduce_price,
        "清仓点": clear_price,
        "继续持有条件": "；".join(hold_conditions),
        "处理建议": action_hint,
    }


def score_candidate_row(row: pd.Series) -> Dict[str, Any]:
    code = str(row.get("code", ""))
    name = str(row.get("name", ""))
    sector = str(row.get("sector", ""))
    sector_rank = safe_int(row.get("sector_rank", 99))
    sector_continuity = safe_float(row.get("sector_continuity", 0))
    is_front_row = safe_int(row.get("is_front_row", 0))
    is_core = safe_int(row.get("is_core", 0))
    strong_clean = safe_float(row.get("strong_clean", 0))
    buy_comfort = safe_float(row.get("buy_comfort", 0))
    open_below_expect_risk = safe_float(row.get("open_below_expect_risk", 100))
    linkage_score = safe_float(row.get("linkage_score", 0))
    y_turnover = safe_float(row.get("y_turnover", 0))
    y_amount_yi = safe_float(row.get("y_amount_yi", 0))
    y_amp = safe_float(row.get("y_amp", 0))
    y_close = safe_float(row.get("y_close", 0))
    y_high = safe_float(row.get("y_high", 0))
    y_low = safe_float(row.get("y_low", 0))

    # 先做硬过滤
    reject_reason = []
    if sector_rank > 2:
        reject_reason.append("不是最强或次强板块")
    if is_front_row != 1 and is_core != 1:
        reject_reason.append("不是板块前排或容量中军")
    if strong_clean < 60:
        reject_reason.append("昨天强度不够干净")
    if open_below_expect_risk >= 60:
        reject_reason.append("开盘弱于预期风险偏大")
    if linkage_score < 50:
        reject_reason.append("板块联动偏弱")

    filtered_out = 1 if reject_reason else 0

    base_score = 0.0
    base_score += max(0, 30 - (sector_rank - 1) * 12)
    base_score += sector_continuity * 0.20
    base_score += strong_clean * 0.22
    base_score += buy_comfort * 0.18
    base_score += linkage_score * 0.15
    base_score += min(y_amount_yi, 50) * 0.20
    base_score += min(y_turnover, 30) * 0.30
    base_score -= open_below_expect_risk * 0.25
    if is_front_row == 1:
        base_score += 8
    if is_core == 1:
        base_score += 6
    if y_amp > 12:
        base_score -= 4

    score = round(float(base_score), 2)

    buy_point = f"观察 {y_close:.2f} 附近承接，优先低吸或分时回踩不破昨日收盘再考虑"
    stop_loss = round(max(y_low * 0.98, y_close * 0.95), 2)
    give_up = "；".join([
        "开盘明显弱于预期",
        "板块不联动",
        "分时跌破昨日低点且无承接"
    ])

    return {
        "code": code,
        "name": name,
        "sector": sector,
        "sector_rank": sector_rank,
        "is_front_row": is_front_row,
        "is_core": is_core,
        "score": score,
        "filtered_out": filtered_out,
        "reject_reason": "；".join(reject_reason),
        "buy_point": buy_point,
        "stop_loss": stop_loss,
        "give_up": give_up,
    }


def make_market_conclusion(
    risk_info: Dict[str, Any],
    index_change_pct: float,
    breadth_pct: float,
    sentiment_score: float,
    overnight_score: float,
    hot_sector_continuity: float,
) -> str:
    parts = []
    parts.append(f"风险等级：{risk_info['risk_level']}（评分 {risk_info['risk_score']}）")
    parts.append(f"昨日指数表现 {index_change_pct:.2f}%")
    parts.append(f"市场广度 {breadth_pct:.1f}%")
    parts.append(f"短线情绪 {sentiment_score:.0f}/100")
    parts.append(f"隔夜环境 {overnight_score:.0f}/100")
    parts.append(f"热点持续性 {hot_sector_continuity:.0f}/100")
    parts.append(risk_info["risk_advice"])
    return "｜".join(parts)


def build_final_plan(
    risk_info: Dict[str, Any],
    holdings_result: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    available_cash: float,
) -> str:
    parts = []

    if risk_info["risk_level"] == "高风险日":
        parts.append("明日优先防守，原则上以处理持仓为主，新开仓从严。")
    elif risk_info["risk_level"] == "中性日":
        parts.append("明日可以出手，但只做高确定性票，控制出手数量。")
    else:
        parts.append("明日可以正常寻找买点，但仍优先做最强方向前排。")

    if not holdings_result.empty:
        parts.append("持仓先按预设减仓点/清仓点执行，不符合继续持有条件就先兑现。")

    if available_cash <= 0:
        parts.append("当前可用资金不足，优先做持仓管理，不建议新增仓位。")
    else:
        if selected_candidates.empty:
            parts.append("当前没有通过筛选的候选股，建议以观察为主。")
        else:
            top_names = "、".join(selected_candidates["name"].head(3).tolist())
            parts.append(f"若允许开仓，优先观察：{top_names}。")
            parts.append("开盘弱于预期、板块不联动、不是当日最强时，直接放弃。")

    return " ".join(parts)


def df_to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无数据"
    return df.to_markdown(index=False)


# =========================
# 左侧输入区
# =========================
st.sidebar.header("基础信息")

decision_date = st.sidebar.date_input("日期", value=date.today())
decision_timepoint = st.sidebar.selectbox("决策时点", ["盘前", "盘中", "盘后"], index=0)
total_capital = st.sidebar.number_input("账户总资金", min_value=0.0, value=100000.0, step=1000.0)
available_cash = st.sidebar.number_input("账户可用资金", min_value=0.0, value=30000.0, step=1000.0)

st.sidebar.header("市场环境输入")
index_change_pct = st.sidebar.number_input("上一个交易日指数涨跌幅(%)", value=0.50, step=0.10)
breadth_pct = st.sidebar.slider("上涨家数占比(%)", min_value=0.0, max_value=100.0, value=58.0, step=1.0)
sentiment_score = st.sidebar.slider("短线情绪评分", min_value=0, max_value=100, value=60, step=1)
overnight_score = st.sidebar.slider("隔夜外围/消息评分", min_value=0, max_value=100, value=55, step=1)
hot_sector_continuity = st.sidebar.slider("热点持续性评分", min_value=0, max_value=100, value=65, step=1)
news_risk_score = st.sidebar.slider("突发消息风险评分", min_value=0, max_value=100, value=40, step=1)

st.sidebar.header("执行")
run_btn = st.sidebar.button("生成决策", use_container_width=True)


# =========================
# 主体输入
# =========================
tab_input, tab_result = st.tabs(["输入区", "结果区"])

with tab_input:
    st.subheader("当前持仓输入")
    st.session_state.holdings_df = st.data_editor(
        st.session_state.holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="holdings_editor"
    )

    st.subheader("候选股输入")
    st.session_state.candidates_df = st.data_editor(
        st.session_state.candidates_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="candidates_editor"
    )

    st.info("这一版先用手工输入把框架搭起来。后面可以把选股结果、持仓、新闻、板块、模型输出自动接进来。")


# =========================
# 生成结果
# =========================
if run_btn:
    risk_info = calc_risk_score(
        index_change_pct=index_change_pct,
        breadth_pct=breadth_pct,
        sentiment_score=sentiment_score,
        overnight_score=overnight_score,
        hot_sector_continuity=hot_sector_continuity,
        news_risk_score=news_risk_score,
    )

    holdings_df = st.session_state.holdings_df.copy()
    candidates_df = st.session_state.candidates_df.copy()

    holdings_result = pd.DataFrame()
    if not holdings_df.empty:
        holdings_result = pd.DataFrame([analyze_holding_row(row, risk_info["risk_level"]) for _, row in holdings_df.iterrows()])

    candidates_scored = pd.DataFrame()
    selected_candidates = pd.DataFrame()
    if not candidates_df.empty:
        candidates_scored = pd.DataFrame([score_candidate_row(row) for _, row in candidates_df.iterrows()])
        selected_candidates = (
            candidates_scored[candidates_scored["filtered_out"] == 0]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

    market_conclusion = make_market_conclusion(
        risk_info=risk_info,
        index_change_pct=index_change_pct,
        breadth_pct=breadth_pct,
        sentiment_score=sentiment_score,
        overnight_score=overnight_score,
        hot_sector_continuity=hot_sector_continuity,
    )

    final_plan = build_final_plan(
        risk_info=risk_info,
        holdings_result=holdings_result,
        selected_candidates=selected_candidates,
        available_cash=available_cash,
    )

    result = {
        "base_info": {
            "decision_date": str(decision_date),
            "decision_timepoint": decision_timepoint,
            "total_capital": total_capital,
            "available_cash": available_cash,
        },
        "risk_info": risk_info,
        "market_conclusion": market_conclusion,
        "holdings_result": holdings_result,
        "candidates_scored": candidates_scored,
        "selected_candidates": selected_candidates,
        "final_plan": final_plan,
    }
    st.session_state.decision_result = result


# =========================
# 展示结果
# =========================
with tab_result:
    result = st.session_state.decision_result

    if result is None:
        st.warning("先在左侧点击“生成决策”。")
    else:
        risk_info = result["risk_info"]
        holdings_result = result["holdings_result"]
        candidates_scored = result["candidates_scored"]
        selected_candidates = result["selected_candidates"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("风险等级", risk_info["risk_level"])
        c2.metric("风险评分", risk_info["risk_score"])
        c3.metric("候选股通过数", int(len(selected_candidates)))
        c4.metric("是否建议空仓观察", "是" if risk_info["risk_level"] == "高风险日" else "否")

        sub1, sub2, sub3, sub4 = st.tabs(["市场结论", "持仓决策", "候选股决策", "最终执行方案"])

        with sub1:
            st.subheader("市场环境结论")
            st.info(result["market_conclusion"])

        with sub2:
            st.subheader("持仓处理建议")
            if holdings_result.empty:
                st.info("当前没有持仓。")
            else:
                st.dataframe(holdings_result, use_container_width=True, hide_index=True)

        with sub3:
            st.subheader("候选股评分结果")
            if candidates_scored.empty:
                st.info("当前没有候选股输入。")
            else:
                st.markdown("#### 全部候选股")
                st.dataframe(candidates_scored, use_container_width=True, hide_index=True)

                st.markdown("#### 最终优先候选股（1-3只）")
                if selected_candidates.empty:
                    st.warning("当前没有通过规则筛选的候选股。")
                else:
                    top_selected = selected_candidates.head(3).copy()
                    st.dataframe(top_selected, use_container_width=True, hide_index=True)

        with sub4:
            st.subheader("最终执行方案")
            st.success(result["final_plan"])

            report_md = f"""
# 次日操作决策报告

## 基础信息
- 日期：{result['base_info']['decision_date']}
- 决策时点：{result['base_info']['decision_timepoint']}
- 账户总资金：{result['base_info']['total_capital']}
- 可用资金：{result['base_info']['available_cash']}

## 市场结论
{result['market_conclusion']}

## 持仓决策
{df_to_md_table(holdings_result)}

## 候选股评分
{df_to_md_table(candidates_scored)}

## 最终优先候选股
{df_to_md_table(selected_candidates.head(3) if not selected_candidates.empty else pd.DataFrame())}

## 最终执行方案
{result['final_plan']}
"""
            st.download_button(
                label="下载决策报告(MD)",
                data=report_md,
                file_name=f"decision_report_{result['base_info']['decision_date']}.md",
                mime="text/markdown",
                use_container_width=False,
            )

st.divider()
st.caption("这一版是完整框架版：先把输入、输出、规则链、报告导出全部搭起来。后面你可以继续接自动数据、选股结果、AI 风险判断、次日推演模块。")