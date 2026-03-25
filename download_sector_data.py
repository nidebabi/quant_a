from __future__ import annotations

from pathlib import Path
import pandas as pd
import baostock as bs

ROOT = Path(__file__).resolve().parent
SECTOR_DIR = ROOT / "data" / "raw" / "sector"
SECTOR_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = SECTOR_DIR / "baostock_industry.parquet"


def normalize_code(code: str) -> str:
    if pd.isna(code):
        return ""
    code = str(code).strip().lower()
    # baostock 格式类似 sh.600000 / sz.000001
    if "." in code:
        code = code.split(".")[1]
    return code.zfill(6)


def main():
    print("开始连接 BaoStock...")
    lg = bs.login()
    print(f"login respond error_code: {lg.error_code}")
    print(f"login respond error_msg : {lg.error_msg}")

    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    try:
        print("开始下载行业分类数据...")
        rs = bs.query_stock_industry()

        if rs.error_code != "0":
            raise RuntimeError(f"query_stock_industry 失败: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            raise RuntimeError("没有拿到任何行业分类数据")

        df = pd.DataFrame(rows, columns=rs.fields)

        # 官方字段通常是：
        # updateDate, code, code_name, industry, industryClassification
        # 这里统一成我们项目里要用的字段
        rename_map = {
            "updateDate": "update_date",
            "code": "raw_code",
            "code_name": "name",
            "industry": "industry_name",
            "industryClassification": "industry_classification",
        }
        df = df.rename(columns=rename_map)

        if "raw_code" not in df.columns or "industry_name" not in df.columns:
            raise RuntimeError(f"返回字段异常，实际字段: {list(df.columns)}")

        df["code"] = df["raw_code"].map(normalize_code)
        df["name"] = df["name"].astype(str).str.strip()
        df["industry_name"] = df["industry_name"].astype(str).str.strip()
        if "industry_classification" in df.columns:
            df["industry_classification"] = df["industry_classification"].astype(str).str.strip()

        # 清理空值
        df = df[(df["code"] != "") & (df["industry_name"] != "")]
        df = df.drop_duplicates(subset=["code"]).reset_index(drop=True)

        # 同时给一份 sector_name，方便你现有逻辑直接识别
        df["sector_name"] = df["industry_name"]
        df["sector_type"] = "baostock_industry"

        keep_cols = [
            c for c in [
                "update_date", "code", "name",
                "industry_name", "industry_classification",
                "sector_name", "sector_type"
            ] if c in df.columns
        ]
        df = df[keep_cols].copy()

        df.to_parquet(OUT_PATH, index=False)
        print(f"行业分类已保存到: {OUT_PATH}")
        print(f"共 {len(df)} 条")
    finally:
        bs.logout()
        print("BaoStock 已登出")


if __name__ == "__main__":
    main()