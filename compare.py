import argparse
from pathlib import Path
import pandas as pd


def _find_one_column(df: pd.DataFrame, candidates: list[str], role_name: str) -> str:
    """在 DataFrame 中按候选名寻找列（大小写不敏感，支持部分匹配）"""
    lower_to_orig = {c.lower(): c for c in df.columns}

    # 1) 精确匹配
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]

    # 2) 子串匹配
    for cand in candidates:
        cand_l = cand.lower()
        for col in df.columns:
            if cand_l in col.lower():
                return col

    raise ValueError(
        f"找不到 {role_name} 列。当前列有：{list(df.columns)}；候选：{candidates}"
    )


def load_aggregator_parquet(exp_dir: str) -> pd.DataFrame:
    """
    读取实验目录下 aggregator_metric 的所有 parquet，并纵向拼接。
    """
    agg_dir = Path(exp_dir) / "aggregator_metric"
    if not agg_dir.exists():
        raise FileNotFoundError(f"目录不存在：{agg_dir}")

    parquet_files = sorted(agg_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"在 {agg_dir} 下未找到 parquet 文件")

    dfs = [pd.read_parquet(p) for p in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    # 去重（如果同一 scenario 重复出现，保留第一次）
    # 先尝试找到 scenario id 列，再去重
    scenario_id_col = _find_one_column(
        df,
        candidates=[
            "scenario_token",
            "scenario_id",
            "scenario",
            "token",
            "log_name_token",
        ],
        role_name="scenario id",
    )
    df = df.drop_duplicates(subset=[scenario_id_col], keep="first").reset_index(drop=True)
    return df


def build_comparison_csv(
    exp1_dir: str,
    exp2_dir: str,
    exp1_name: str,
    exp2_name: str,
    out_csv: str,
) -> None:
    df1 = load_aggregator_parquet(exp1_dir)
    df2 = load_aggregator_parquet(exp2_dir)

    # 自动定位关键列
    id_col_1 = _find_one_column(
        df1, ["scenario_token", "scenario_id", "scenario", "token"], "scenario id (exp1)"
    )
    type_col_1 = _find_one_column(
        df1, ["scenario_type", "type"], "scenario type (exp1)"
    )
    score_col_1 = _find_one_column(
        df1, ["scenario_score", "score", "final_score"], "scenario score (exp1)"
    )

    id_col_2 = _find_one_column(
        df2, ["scenario_token", "scenario_id", "scenario", "token"], "scenario id (exp2)"
    )
    type_col_2 = _find_one_column(
        df2, ["scenario_type", "type"], "scenario type (exp2)"
    )
    score_col_2 = _find_one_column(
        df2, ["scenario_score", "score", "final_score"], "scenario score (exp2)"
    )

    sub1 = df1[[id_col_1, type_col_1, score_col_1]].copy()
    sub2 = df2[[id_col_2, type_col_2, score_col_2]].copy()

    sub1.columns = ["scenario_id", "scenario_type", f"scenario_score_{exp1_name}"]
    sub2.columns = ["scenario_id", "scenario_type_exp2", f"scenario_score_{exp2_name}"]

    merged = sub1.merge(sub2, on="scenario_id", how="inner")

    # 如果两个实验的 scenario_type 不一致，优先保留 exp1，同时给出提醒
    mismatch = (merged["scenario_type"] != merged["scenario_type_exp2"]).sum()
    if mismatch > 0:
        print(f"[Warning] 有 {mismatch} 行 scenario_type 不一致，已保留 exp1 的类型。")

    merged[f"score_diff_{exp2_name}_minus_{exp1_name}"] = (
        merged[f"scenario_score_{exp2_name}"] - merged[f"scenario_score_{exp1_name}"]
    )

    out_cols = [
        "scenario_id",
        "scenario_type",
        f"scenario_score_{exp1_name}",
        f"scenario_score_{exp2_name}",
        f"score_diff_{exp2_name}_minus_{exp1_name}",
    ]
    out_df = merged[out_cols].sort_values("scenario_id").reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"已输出：{out_csv}")
    print(f"总行数：{len(out_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="对比两个实验 aggregator_metric parquet，并导出 scenario 级别对比 CSV"
    )
    parser.add_argument("--exp1_dir", required=True, help="实验1目录（包含 aggregator_metric）")
    parser.add_argument("--exp2_dir", required=True, help="实验2目录（包含 aggregator_metric）")
    parser.add_argument("--exp1_name", required=True, help="实验1命名（用于列名）")
    parser.add_argument("--exp2_name", required=True, help="实验2命名（用于列名）")
    parser.add_argument("--out_csv", required=True, help="输出 CSV 路径")
    args = parser.parse_args()

    build_comparison_csv(
        exp1_dir=args.exp1_dir,
        exp2_dir=args.exp2_dir,
        exp1_name=args.exp1_name,
        exp2_name=args.exp2_name,
        out_csv=args.out_csv,
    )