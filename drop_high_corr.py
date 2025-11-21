# -*- coding: utf-8 -*-
"""
自动扫描 ./data 目录下的 xxx_features_filtered.csv / xxx_between_class_var.csv /
xxx_feature_correlation.csv 三件套，对每一组前缀 xxx 执行：

  1. 读取 xxx_features_filtered.csv（特征表）；
  2. 使用 xxx_between_class_var.csv 中的类间方差信息（若存在）；
  3. 重新计算数值特征之间的相关矩阵；
  4. 查找 |corr| >= CORR_THRESHOLD 的特征对；
  5. 在每一对中删除“较差”的特征（优先保留类间方差更大者）；
  6. 输出到 ./cleaned_data 目录：
       - xxx_features_dedup_corr.csv              清洗后特征表
       - xxx_dropped_high_corr_features.csv       被删除的高度相关特征明细
       - xxx_feature_correlation_cleaned.csv      清洗后特征相关矩阵
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


# ===================== 参数区（主公按需修改） =====================

# 相关性阈值：|corr| >= 该值视为“高度相关”
CORR_THRESHOLD = 0.8

# 是否使用 between_class_var 作为优先保留依据
USE_BETWEEN_CLASS_VAR = True

# 标签列名与元信息列名（不参与相关性与删除）
LABEL_COL = "label"
# 这里兼容 subject-level 和 window-level，两种情况下不存在的列会自动跳过
META_COLS = ["subject_id", "study", "segment_type", "window_idx"]

# 输入 / 输出目录
INPUT_DIR = Path("./data")
OUTPUT_DIR = Path("./cleaned_data")


# ===================== 工具函数 =====================

def load_between_class_var(path: Path) -> Dict[str, float]:
    """
    从 between_class_var.csv 中读取 {feature -> between_class_var} 映射。
    预期列：feature, between_class_var, dropped
    """
    df = pd.read_csv(path)
    if "feature" not in df.columns or "between_class_var" not in df.columns:
        raise ValueError(f"between_class_var CSV 中缺少 'feature' 或 'between_class_var' 列: {path}")
    feat2bcv = dict(zip(df["feature"].astype(str), df["between_class_var"].astype(float)))
    return feat2bcv


def compute_corr_matrix(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    对指定特征列计算皮尔逊相关系数矩阵。
    """
    feat_df = df[feature_cols].copy()
    # 用列均值填补缺失，避免相关性计算出现 NaN
    feat_df = feat_df.fillna(feat_df.mean(numeric_only=True))
    corr = feat_df.corr(method="pearson")
    return corr


def find_high_corr_pairs(
    corr_df: pd.DataFrame,
    threshold: float,
) -> List[Tuple[str, str, float]]:
    """
    在相关矩阵中查找 |corr| >= threshold 的特征对（只返回上三角，避免重复）。

    返回 list[(feat_i, feat_j, corr_ij)]，其中 i < j。
    """
    feats = list(corr_df.columns)
    high_pairs: List[Tuple[str, str, float]] = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            fi, fj = feats[i], feats[j]
            c = corr_df.iloc[i, j]
            if np.isnan(c):
                continue
            if abs(c) >= threshold:
                high_pairs.append((fi, fj, float(c)))
    return high_pairs


def choose_feature_to_drop(
    f1: str,
    f2: str,
    corr_value: float,
    df: pd.DataFrame,
    feat2bcv: Dict[str, float] = None,
) -> Tuple[str, str]:
    """
    在高度相关的特征对 (f1, f2) 中，决定删除哪个特征。

    返回 (keep_feature, drop_feature)。

    规则：
      1. 若提供 feat2bcv（类间方差映射）：
           - 保留 between_class_var 较大的特征；
           - 若相等，则转 2。
      2. 否则 / 平手：
           - 保留非缺失样本数更多的特征；
           - 再平手：按特征名字典序，保留“更小”的名字。
    """
    # 1. 类间方差优先
    if feat2bcv is not None:
        b1 = feat2bcv.get(f1, 0.0)
        b2 = feat2bcv.get(f2, 0.0)
        if b1 > b2:
            return f1, f2
        elif b2 > b1:
            return f2, f1
        # 若相等则继续走下一步

    # 2. 非缺失样本数
    non_null_1 = df[f1].notna().sum()
    non_null_2 = df[f2].notna().sum()
    if non_null_1 > non_null_2:
        return f1, f2
    elif non_null_2 > non_null_1:
        return f2, f1

    # 3. 名字字典序
    if f1 < f2:
        return f1, f2
    else:
        return f2, f1


def process_one_dataset(features_path: Path):
    """
    对单个前缀 xxx 对应的三件套文件执行去高相关特征操作。

    输入：
      features_path: ./data 下的 xxx_features_filtered.csv 路径
    """
    # 解析前缀 xxx
    stem = features_path.stem  # 例如: 'pd_subtype_win60s_windowlevel_features_filtered'
    if not stem.endswith("_features_filtered"):
        print(f"[警告] 文件名不符合 '*_features_filtered.csv' 约定，已跳过: {features_path.name}")
        return

    prefix = stem[:-len("_features_filtered")]  # 'pd_subtype_win60s_windowlevel'

    # 构造同目录下的另外两个文件路径（仍位于 ./data 中）
    between_path = features_path.with_name(f"{prefix}_between_class_var.csv")
    corr_input_path = features_path.with_name(f"{prefix}_feature_correlation.csv")

    if not between_path.exists() or not corr_input_path.exists():
        print(
            f"[警告] 前缀 '{prefix}' 缺少三件套文件："
            f"{between_path.name if between_path.exists() else '[缺失 between_class_var]'}, "
            f"{corr_input_path.name if corr_input_path.exists() else '[缺失 feature_correlation]'}，已跳过。"
        )
        return

    print(f"\n[信息] ===== 处理前缀: {prefix} =====")
    print(f"[信息] 特征表:       {features_path.name}")
    print(f"[信息] 类间方差表:   {between_path.name}")
    print(f"[信息] 原始相关矩阵: {corr_input_path.name}（仅用于存在性检查，本脚本会重新计算相关）")

    # 读取特征表（从 ./data）
    df = pd.read_csv(features_path)

    # label / meta / feature 列集合
    meta_cols = [c for c in META_COLS if c in df.columns]
    if LABEL_COL not in df.columns:
        raise ValueError(f"在 {features_path.name} 中未找到标签列 '{LABEL_COL}'，请检查。")

    cols_to_exclude = set(meta_cols + [LABEL_COL])
    feature_cols = [
        c for c in df.columns
        if c not in cols_to_exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"[信息] 数值特征维度数量: {len(feature_cols)}")

    # 读取类间方差（可选）
    feat2bcv = None
    if USE_BETWEEN_CLASS_VAR:
        try:
            feat2bcv = load_between_class_var(between_path)
            print(f"[信息] 已载入 {len(feat2bcv)} 条类间方差记录。")
        except Exception as e:
            print(f"[警告] 读取类间方差失败，将忽略：{e}")
            feat2bcv = None

    # 计算相关矩阵
    print("[信息] 重新计算特征相关性矩阵...")
    corr_df = compute_corr_matrix(df, feature_cols=feature_cols)

    # 查找高相关特征对
    high_pairs = find_high_corr_pairs(corr_df, threshold=CORR_THRESHOLD)
    print(f"[信息] |corr| >= {CORR_THRESHOLD} 的特征对数量: {len(high_pairs)}")

    dropped = set()
    drop_records = []

    for f1, f2, c in high_pairs:
        if f1 in dropped or f2 in dropped:
            continue

        keep_feat, drop_feat = choose_feature_to_drop(
            f1, f2, corr_value=c, df=df, feat2bcv=feat2bcv
        )
        dropped.add(drop_feat)

        bcv_keep = feat2bcv.get(keep_feat, np.nan) if feat2bcv is not None else np.nan
        bcv_drop = feat2bcv.get(drop_feat, np.nan) if feat2bcv is not None else np.nan

        drop_records.append({
            "feature_dropped": drop_feat,
            "feature_kept": keep_feat,
            "corr_value": c,
            "between_class_var_dropped": bcv_drop,
            "between_class_var_kept": bcv_keep,
        })

    print(f"[信息] 将删除的高度相关特征数量: {len(dropped)}")

    # 构建清洗后的特征表
    cleaned_feature_cols = [f for f in feature_cols if f not in dropped]
    print(f"[信息] 清洗后保留特征数量: {len(cleaned_feature_cols)}")

    final_cols = meta_cols + [LABEL_COL] + cleaned_feature_cols
    df_clean = df[final_cols].copy()

    # 重新计算清洗后特征的相关性矩阵
    corr_clean = compute_corr_matrix(df_clean, feature_cols=cleaned_feature_cols)

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 输出文件名（写到 ./cleaned_data 下）
    out_features = OUTPUT_DIR / f"{prefix}_features_dedup_corr.csv"
    out_dropped = OUTPUT_DIR / f"{prefix}_dropped_high_corr_features.csv"
    out_corr = OUTPUT_DIR / f"{prefix}_feature_correlation_cleaned.csv"

    df_clean.to_csv(out_features, index=False, encoding="utf-8-sig")
    print(f"[完成] 清洗后的特征表已保存至: {out_features}")

    if drop_records:
        df_drop = pd.DataFrame(drop_records)
        df_drop.to_csv(out_dropped, index=False, encoding="utf-8-sig")
        print(f"[完成] 被删除的高度相关特征明细已保存至: {out_dropped}")
    else:
        print("[信息] 没有特征被删除，因此不生成 dropped 特征明细表。")

    corr_clean.to_csv(out_corr, encoding="utf-8-sig")
    print(f"[完成] 清洗后特征的相关矩阵已保存至: {out_corr}")


def main():
    # 输入目录 = ./data
    root = INPUT_DIR.resolve()
    print(f"[信息] 扫描输入目录: {root}")

    if not INPUT_DIR.exists():
        print(f"[警告] 输入目录不存在: {INPUT_DIR}")
        return

    # 查找所有 *_features_filtered.csv
    features_files = sorted(INPUT_DIR.glob("*_features_filtered.csv"))
    if not features_files:
        print("[警告] 未在 ./data 下找到任何 '*_features_filtered.csv' 文件。")
        return

    print(f"[信息] 共发现 {len(features_files)} 个候选特征文件。")

    for f in features_files:
        try:
            process_one_dataset(f)
        except Exception as e:
            print(f"[错误] 处理文件 {f.name} 时发生异常，已跳过: {e}")


if __name__ == "__main__":
    main()
