#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用已保存的 AutoGluon 模型，在按 subject_id 划分的测试集上打印混淆矩阵和分类报告。

假设：
  - 特征文件来自 cleaned_data/pd_nc_subtype_win{win}s_windowlevel_features_dedup_corr.csv
  - 训练脚本 train_and_eval.py 使用相同的划分规则和 RANDOM_STATE=42
  - 训练好的模型保存在 ./models_tabular_multiclass/win{win}s 目录下
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from autogluon.tabular import TabularPredictor


# 与 train_and_eval.py 保持一致的配置
CLEANED_DATA_DIR = Path("./cleaned_data")
MODEL_ROOT = Path("./models_tabular_multiclass")
PREFIX = "pd_nc_subtype"
WINDOW_SIZES = [30, 60, 120, 180, 300]
FILENAME_TEMPLATE = "{prefix}_win{win}s_windowlevel_features_dedup_corr.csv"

LABEL_COL = "label"
SUBJECT_COL = "subject_id"

TEST_SUBJECT_FRACTION = 0.2
VAL_FRACTION_WITHIN_REST = 0.25
RANDOM_STATE = 42


def split_by_subject(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 subject_id 划分 train / val / test，被试级分层，与 train_and_eval.py 保持一致。
    返回: train_df, val_df, test_df
    """
    if SUBJECT_COL not in df.columns:
        raise ValueError(f"数据集中未找到 subject_id 列 '{SUBJECT_COL}'")

    subj_df = df[[SUBJECT_COL, LABEL_COL]].drop_duplicates(subset=[SUBJECT_COL]).reset_index(drop=True)
    n_subjects = len(subj_df)
    if n_subjects < 3:
        # 太少就不做外部划分
        return df.copy(), None, None

    subj_labels = subj_df[LABEL_COL].to_numpy()
    unique_labels, counts = np.unique(subj_labels, return_counts=True)

    if len(unique_labels) == 1:
        raise ValueError("在被试级上只有一个类别，无法进行 3 分类划分")

    # 第一步：划分 test 被试
    min_count = counts.min()
    can_stratify_subjects = min_count >= 2

    rest_subj_df, test_subj_df = train_test_split(
        subj_df,
        test_size=TEST_SUBJECT_FRACTION,
        stratify=subj_labels if can_stratify_subjects else None,
        random_state=RANDOM_STATE,
    )

    # 第二步：在剩余被试中划分 val 被试
    val_subj_df = None
    train_subj_df = rest_subj_df

    if len(rest_subj_df) >= 3:
        rest_labels = rest_subj_df[LABEL_COL].to_numpy()
        uniq_rest, counts_rest = np.unique(rest_labels, return_counts=True)
        can_stratify_rest = (len(uniq_rest) > 1 and counts_rest.min() >= 2)

        train_subj_df, val_subj_df = train_test_split(
            rest_subj_df,
            test_size=VAL_FRACTION_WITHIN_REST,
            stratify=rest_labels if can_stratify_rest else None,
            random_state=RANDOM_STATE,
        )

    train_subjects = set(train_subj_df[SUBJECT_COL].tolist())
    test_subjects = set(test_subj_df[SUBJECT_COL].tolist())
    val_subjects = set(val_subj_df[SUBJECT_COL].tolist()) if val_subj_df is not None else set()

    train_df = df[df[SUBJECT_COL].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df[SUBJECT_COL].isin(val_subjects)].reset_index(drop=True) if val_subjects else None
    test_df = df[df[SUBJECT_COL].isin(test_subjects)].reset_index(drop=True)

    return train_df, val_df, test_df


def eval_for_window(win_size: int) -> None:
    """对某个窗口大小加载数据与模型，在测试集上打印混淆矩阵。"""
    filename = FILENAME_TEMPLATE.format(prefix=PREFIX, win=win_size)
    data_path = CLEANED_DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"[评估] 窗口 {win_size}s")
    print(f"[信息] 读取特征文件: {data_path.resolve()}")

    if not data_path.exists():
        print(f"[警告] 特征文件不存在，跳过: {data_path}")
        return

    df = pd.read_csv(data_path)
    if LABEL_COL not in df.columns:
        print(f"[错误] {data_path.name} 中未找到标签列 '{LABEL_COL}'")
        return
    if SUBJECT_COL not in df.columns:
        print(f"[错误] {data_path.name} 中未找到 subject 列 '{SUBJECT_COL}'")
        return

    # 与训练脚本相同的 label 清洗
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[(df[LABEL_COL].notna()) & (df[LABEL_COL] != "") & (df[LABEL_COL] != "-")].copy()
    if len(df) == 0:
        print("[警告] 有效样本为 0，跳过该窗口")
        return

    _, _, test_df = split_by_subject(df)
    if test_df is None or len(test_df) == 0:
        print("[警告] 当前窗口没有测试集，无法评估")
        return

    exp_model_dir = MODEL_ROOT / f"win{win_size}s"
    if not exp_model_dir.exists():
        print(f"[警告] 模型目录不存在: {exp_model_dir.resolve()}")
        print("       请先运行 train_and_eval.py 训练并保存模型。")
        return

    print(f"[信息] 加载模型: {exp_model_dir.resolve()}")
    predictor = TabularPredictor.load(str(exp_model_dir))

    y_true = test_df[LABEL_COL].to_numpy()
    y_pred = predictor.predict(test_df)
    classes = np.unique(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("[结果] 测试集混淆矩阵（行=真实标签, 列=预测标签）:")
    print("labels:", list(classes))
    print(cm)
    print("[结果] 测试集分类报告：")
    print(classification_report(y_true, y_pred, target_names=[str(c) for c in classes]))


def main() -> None:
    print("[信息] 使用已保存模型评估各时间窗口的测试集混淆矩阵")
    for win in WINDOW_SIZES:
        eval_for_window(win)


if __name__ == "__main__":
    main()

