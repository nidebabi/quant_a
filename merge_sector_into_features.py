from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

FEATURES_PATH = ROOT / "data" / "features" / "features_all.parquet"
SECTOR_PATH = ROOT / "data" / "raw" / "sector" / "baostock_industry.parquet"


def normalize_code(code: str) -> str:
    if pd.isna(code):
        return ""
    code = str(code).strip()
    if "." in code:
        code = code.split(".")[1]
    return code.zfill(6)


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"找不到 features 文件: {FEATURES_PATH}")

    if not SECTOR_PATH.exists():
        raise FileNotFoundError(
            f"找不到行业分类文件: {SECTOR_PATH}\n"
            f"请先运行: python download_sector_data.py"
        )

    print("读取 features...")
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])
    features["code"] = features["code"].map(normalize_code)

    print("读取行业分类...")
    sector_df = pd.read_parquet(SECTOR_PATH)
    sector_df["code"] = sector_df["code"].map(normalize_code)

    merge_cols = [c for c in ["code", "industry_name", "sector_name", "sector_type", "industry_classification"] if c in sector_df.columns]
    sector_df = sector_df[merge_cols].drop_duplicates(subset=["code"]).copy()

    # 删除旧字段，避免重复
    old_cols = ["industry_name", "sector_name", "sector_type", "industry_classification"]
    for c in old_cols:
        if c in features.columns:
            features = features.drop(columns=[c])

    features = features.merge(sector_df, on="code", how="left")

    # 如果 sector_name 为空，就用 industry_name 顶上
    if "sector_name" in features.columns and "industry_name" in features.columns:
        features["sector_name"] = features["sector_name"].fillna(features["industry_name"])
    elif "industry_name" in features.columns and "sector_name" not in features.columns:
        features["sector_name"] = features["industry_name"]

    features.to_parquet(FEATURES_PATH, index=False)
    print(f"已把行业分类合并进 features: {FEATURES_PATH}")

    # 输出覆盖情况
    if "sector_name" in features.columns:
        cover = features["sector_name"].notna().mean()
        print(f"sector_name 覆盖率: {cover:.2%}")
    if "industry_name" in features.columns:
        cover2 = features["industry_name"].notna().mean()
        print(f"industry_name 覆盖率: {cover2:.2%}")


if __name__ == "__main__":
    main()