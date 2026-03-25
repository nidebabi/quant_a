from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from services.ai_service import summarize_with_ai
from services.data_service import (
    get_candidate_pool,
    get_numeric_feature_columns,
    load_features,
    merge_features_and_labels,
)


TARGETS = {
    "gap_up_flag": "次日高开概率",
    "intraday_up_flag": "次日开盘后走强概率",
    "close_up_flag": "次日收盘上涨概率",
    "next_touch_tp_flag": "次日触达止盈概率",
}


@dataclass
class ModelArtifacts:
    features: List[str]
    models: Dict[str, Pipeline]
    metrics_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    holdout_predictions: pd.DataFrame
    train_summary: str


def _split_by_date(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(df["date"]).dropna().unique())
    if len(dates) < 10:
        return df.copy(), pd.DataFrame()
    split_index = max(int(len(dates) * (1 - test_ratio)), 1)
    split_date = dates[split_index - 1]
    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()
    return train_df, test_df


def _build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=8,
                    min_samples_leaf=12,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _metric_dict(y_true: pd.Series, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        result["auc"] = float("nan")
    return result


def _extract_feature_importance(model: Pipeline, features: List[str], target_name: str) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["target", "feature", "importance"])
    return (
        pd.DataFrame(
            {
                "target": target_name,
                "feature": features,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _build_condition_summary(
    train_df: pd.DataFrame,
    positive_flag: str,
    top_features: List[str],
) -> List[str]:
    summary: List[str] = []
    positive_df = train_df[train_df[positive_flag] == 1]
    negative_df = train_df[train_df[positive_flag] == 0]
    if positive_df.empty or negative_df.empty:
        return summary

    for feature in top_features[:6]:
        if feature not in positive_df.columns:
            continue
        pos_med = pd.to_numeric(positive_df[feature], errors="coerce").median()
        neg_med = pd.to_numeric(negative_df[feature], errors="coerce").median()
        if pd.isna(pos_med) or pd.isna(neg_med):
            continue
        direction = "高于" if pos_med > neg_med else "低于"
        summary.append(
            f"{feature} 在成功样本中通常{direction}失败样本，中位数 {pos_med:.4f} vs {neg_med:.4f}"
        )
    return summary


def train_quant_models() -> ModelArtifacts:
    merged = merge_features_and_labels()
    if merged.empty:
        return ModelArtifacts(
            features=[],
            models={},
            metrics_df=pd.DataFrame(),
            feature_importance_df=pd.DataFrame(),
            holdout_predictions=pd.DataFrame(),
            train_summary="未找到可训练的特征与标签数据。",
        )

    feature_columns = get_numeric_feature_columns(merged)
    if not feature_columns:
        return ModelArtifacts(
            features=[],
            models={},
            metrics_df=pd.DataFrame(),
            feature_importance_df=pd.DataFrame(),
            holdout_predictions=pd.DataFrame(),
            train_summary="样本数据存在，但没有检测到可用的数值特征列。",
        )

    train_df, test_df = _split_by_date(merged)
    if train_df.empty:
        train_df = merged.copy()
    if test_df.empty:
        test_df = train_df.tail(min(1000, len(train_df))).copy()
        train_df = train_df.iloc[:-len(test_df)].copy() if len(train_df) > len(test_df) else train_df.copy()

    models: Dict[str, Pipeline] = {}
    metric_rows: List[Dict[str, Any]] = []
    importance_frames: List[pd.DataFrame] = []
    holdout_output = test_df[
        ["date", "code", "name", "close", "next_open", "next_close", "next_day_return"]
    ].copy()

    for target, label in TARGETS.items():
        if target not in merged.columns:
            continue
        model = _build_pipeline()
        model.fit(train_df[feature_columns], train_df[target].astype(int))
        models[target] = model

        probabilities = model.predict_proba(test_df[feature_columns])[:, 1]
        holdout_output[target.replace("_flag", "_prob")] = probabilities
        metrics = _metric_dict(test_df[target].astype(int), probabilities)
        metric_rows.append({"target": label, **metrics})
        importance_frames.append(_extract_feature_importance(model, feature_columns, label))

    importance_df = pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    top_features = (
        importance_df.sort_values("importance", ascending=False)["feature"].head(8).tolist()
        if not importance_df.empty
        else []
    )
    condition_summary = _build_condition_summary(train_df, "close_up_flag", top_features)

    fallback_summary = (
        "模型已基于本地历史样本完成训练。建议重点观察成交额、换手率、近3日涨幅、"
        "收盘位置和量比等变量。"
    )
    summary_payload = {
        "metrics": metrics_df.to_dict(orient="records"),
        "top_features": top_features,
        "conditions": condition_summary,
    }
    summary_text = summarize_with_ai("A股量化模型训练总结", summary_payload, fallback_summary)

    return ModelArtifacts(
        features=feature_columns,
        models=models,
        metrics_df=metrics_df,
        feature_importance_df=importance_df,
        holdout_predictions=holdout_output,
        train_summary=summary_text,
    )


def _prepare_prediction_frame(candidate_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()

    if "交易日" in candidate_df.columns:
        selected_date = pd.to_datetime(candidate_df["交易日"], errors="coerce").max()
    else:
        selected_date = pd.to_datetime(merged["date"], errors="coerce").max()

    feature_slice = merged[pd.to_datetime(merged["date"], errors="coerce") == selected_date].copy()
    candidate_codes = candidate_df["代码"].astype(str).tolist() if "代码" in candidate_df.columns else []
    if candidate_codes:
        feature_slice = feature_slice[feature_slice["code"].astype(str).isin(candidate_codes)].copy()
    return feature_slice


def _score_feature_frame(model_artifacts: ModelArtifacts, feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df.empty:
        return pd.DataFrame()

    output = feature_df[
        ["date", "code", "name", "close", "amount", "turnover", "ret_3", "ret_10", "close_loc"]
    ].copy()
    for target in TARGETS:
        model = model_artifacts.models.get(target)
        if model is None:
            continue
        output[target.replace("_flag", "_prob")] = model.predict_proba(feature_df[model_artifacts.features])[:, 1]

    output["综合AI评分"] = (
        output.get("gap_up_prob", 0) * 0.25
        + output.get("intraday_up_prob", 0) * 0.25
        + output.get("close_up_prob", 0) * 0.35
        + output.get("next_touch_tp_prob", 0) * 0.15
    )
    return output.sort_values("综合AI评分", ascending=False).reset_index(drop=True)


def score_candidates_with_models(
    model_artifacts: ModelArtifacts,
    trade_date: Optional[str] = None,
    top_n: int = 20,
    sector_filter: str = "全部",
) -> pd.DataFrame:
    if not model_artifacts.models:
        return pd.DataFrame()

    merged = merge_features_and_labels()
    candidate_df = get_candidate_pool(trade_date=trade_date, top_n=top_n, sector_filter=sector_filter)
    prediction_df = _prepare_prediction_frame(candidate_df, merged)
    return _score_feature_frame(model_artifacts, prediction_df)


def score_manual_candidates_with_models(
    model_artifacts: ModelArtifacts,
    trade_date: Optional[str],
    selected_codes: List[str],
) -> pd.DataFrame:
    if not model_artifacts.models or not selected_codes:
        return pd.DataFrame()

    features = load_features()
    if features.empty:
        return pd.DataFrame()

    if trade_date:
        dt = pd.to_datetime(trade_date, errors="coerce")
        if pd.notna(dt):
            features = features[pd.to_datetime(features["date"], errors="coerce") == dt].copy()
    else:
        latest_dt = pd.to_datetime(features["date"], errors="coerce").max()
        features = features[pd.to_datetime(features["date"], errors="coerce") == latest_dt].copy()

    features = features[features["code"].astype(str).isin([str(code) for code in selected_codes])].copy()
    return _score_feature_frame(model_artifacts, features)


def run_prediction_backtest(holdout_predictions: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    if holdout_predictions.empty:
        return {"summary": "暂无留出集预测结果。", "daily_df": pd.DataFrame(), "metrics": {}}

    df = holdout_predictions.copy()
    if "close_up_prob" not in df.columns:
        return {"summary": "留出集里缺少 close_up_prob 结果。", "daily_df": pd.DataFrame(), "metrics": {}}

    daily_rows: List[Dict[str, Any]] = []
    for trade_date, group in df.groupby("date"):
        picks = group.sort_values("close_up_prob", ascending=False).head(top_n).copy()
        daily_return = pd.to_numeric(picks["next_day_return"], errors="coerce").mean()
        win_rate = (pd.to_numeric(picks["next_day_return"], errors="coerce") > 0).mean()
        daily_rows.append(
            {
                "date": pd.to_datetime(trade_date),
                "pick_count": int(len(picks)),
                "avg_return": float(daily_return) if pd.notna(daily_return) else 0.0,
                "win_rate": float(win_rate) if pd.notna(win_rate) else 0.0,
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily_df["equity"] = (1 + daily_df["avg_return"]).cumprod()
    metrics = {
        "active_days": int(len(daily_df)),
        "total_return": float(daily_df["equity"].iloc[-1] - 1) if not daily_df.empty else 0.0,
        "avg_daily_return": float(daily_df["avg_return"].mean()) if not daily_df.empty else 0.0,
        "win_rate": float((daily_df["avg_return"] > 0).mean()) if not daily_df.empty else 0.0,
    }
    summary = (
        f"留出集共 {metrics['active_days']} 个交易日，按模型评分每日选前 {top_n} 只，"
        f"累计收益 {metrics['total_return']:.2%}，日胜率 {metrics['win_rate']:.2%}。"
    )
    return {"summary": summary, "daily_df": daily_df, "metrics": metrics}
