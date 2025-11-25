#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多窗口（三分类）PD/NC 驾驶行为分类训练脚本（AutoGluon Tabular）

功能：
  - 针对不同时间窗口（例如 30s / 60s / 120s / 180s / 300s），依次读取 ./cleaned_data 下的特征文件；
  - 每个时间窗口作为一次独立实验顺序执行；
  - 任务为 3 分类：两类 PD 亚型 + 正常 nc；
  - 按 subject_id 严格划分 train / val / test（同一 subject 只会出现在其中一种划分）；
  - 使用 AutoGluon TabularPredictor 自动建模；
  - 对每个实验：
      * 打印训练/验证/测试集规模与标签分布；
      * 打印完整 leaderboard（含各模型验证/测试分数与超参数）；
      * 标记最优模型名称、验证/测试分数与其超参数；
      * 将整套 Predictor（包含所有候选模型与最优模型）保存到指定目录；
      * 将每个窗口的最优模型及测试指标追加写入 window_results.csv。
"""

from pathlib import Path
import csv
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor


# ===================== 全局配置 =====================

# 清洗后的特征文件目录（由 extract_features.py + drop_high_corr.py 生成）
# 例如：./cleaned_data/pd_nc_subtype_win30s_windowlevel_features_dedup_corr.csv
CLEANED_DATA_DIR = Path("./cleaned_data")
PREFIX = "pd_nc_subtype"
WINDOW_SIZES = [30, 60, 120, 180, 300]  # 想比较哪些时间窗，就写哪些秒数
FILENAME_TEMPLATE = "{prefix}_win{win}s_windowlevel_features_dedup_corr.csv"

# 汇总不同窗口结果的 CSV
RESULTS_CSV = Path("window_results.csv")

# 列名配置
LABEL_COL = "label"
SUBJECT_COL = "subject_id"

# 被试级划分比例（近似）
TEST_SUBJECT_FRACTION = 0.2          # 测试集被试占全部被试的比例
VAL_FRACTION_WITHIN_REST = 0.25      # 在非测试被试中，验证集被试比例（≈最终各 0.2）

RANDOM_STATE = 42

# AutoGluon Tabular 配置
TAB_PRESETS = "best_quality"         # 可改成 "medium_quality" 等
TIME_LIMIT = None                    # 训练时间限制（秒），None 表示不限制
NUM_GPUS = 1                         # 无 GPU 可改为 0 或 None

# 模型保存根目录，每个时间窗口单独一个子目录
MODEL_ROOT = Path("./models_tabular_multiclass")


def append_window_result(
    window_name: str,
    best_model_name: str,
    metrics: Dict[str, float],
) -> None:
    """
    把单个时间窗的结果追加写入 window_results.csv：
      window_name, best_model_name, 以及 metrics 字典里的所有键值对。
    """
    row = {
        "window": window_name,
        "best_model": best_model_name,
    }
    row.update(metrics)

    file_exists = RESULTS_CSV.exists()

    # 第一次写入需要写表头，后续只追加
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ===================== 工具函数：按被试划分 =====================

def split_by_subject(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    在 subject 级别上划分 train / val / test 被试集合，并映射回样本行（window-level）。

    返回:
        train_df, val_df, test_df
    """
    if SUBJECT_COL not in df.columns:
        raise ValueError(f"数据集中未找到 subject_id 列 '{SUBJECT_COL}'，无法按人划分。")

    # 被试级表：每个 subject_id 一行
    subj_df = df[[SUBJECT_COL, LABEL_COL]].drop_duplicates(subset=[SUBJECT_COL]).reset_index(drop=True)

    n_subjects = len(subj_df)
    if n_subjects < 3:
        print(f"[警告] 被试数量仅为 {n_subjects}，不足以划分有效的 train/val/test。")
        print("       本次将不做外部划分，所有样本都作为训练集，"
              "AutoGluon 仍会在内部做行级划分（无法完全避免同人泄漏）。")
        return df.copy(), None, None

    subj_labels = subj_df[LABEL_COL].to_numpy()
    unique_labels, counts = np.unique(subj_labels, return_counts=True)
    subj_label_counts = dict(zip(unique_labels, counts))
    print(f"[信息] 被试级标签分布: {subj_label_counts}")

    if len(unique_labels) == 1:
        raise ValueError("在被试级上只有一个类别，无法进行 3 分类训练，请检查 label 列。")

    # ---------- 第一步：划分 test 被试 ----------
    min_count = counts.min()
    can_stratify_subjects = min_count >= 2

    print(f"[信息] 目标测试集被试比例: {TEST_SUBJECT_FRACTION} (按 subject 计)")
    rest_subj_df, test_subj_df = train_test_split(
        subj_df,
        test_size=TEST_SUBJECT_FRACTION,
        stratify=subj_labels if can_stratify_subjects else None,
        random_state=RANDOM_STATE,
    )

    print(f"[信息] 被试划分后：训练+验证候选被试数量 = {len(rest_subj_df)}，测试被试数量 = {len(test_subj_df)}")

    # ---------- 第二步：在剩余被试中划分 val 被试 ----------
    val_subj_df = None
    train_subj_df = rest_subj_df

    if len(rest_subj_df) >= 3:
        rest_labels = rest_subj_df[LABEL_COL].to_numpy()
        uniq_rest, counts_rest = np.unique(rest_labels, return_counts=True)
        can_stratify_rest = (len(uniq_rest) > 1 and counts_rest.min() >= 2)

        print(f"[信息] 在非测试被试中划分验证集，被试比例约为 {VAL_FRACTION_WITHIN_REST}")
        train_subj_df, val_subj_df = train_test_split(
            rest_subj_df,
            test_size=VAL_FRACTION_WITHIN_REST,
            stratify=rest_labels if can_stratify_rest else None,
            random_state=RANDOM_STATE,
        )
        print(f"[信息] 被试级划分结果：训练被试 = {len(train_subj_df)}，验证被试 = {len(val_subj_df)}")
    else:
        print("[警告] 训练+验证候选被试数量过少，跳过验证被试划分，本次无独立验证集。")

    # ---------- 映射回样本行 ----------
    train_subjects = set(train_subj_df[SUBJECT_COL].tolist())
    test_subjects = set(test_subj_df[SUBJECT_COL].tolist())
    val_subjects = set(val_subj_df[SUBJECT_COL].tolist()) if val_subj_df is not None else set()

    assert train_subjects.isdisjoint(test_subjects)
    assert train_subjects.isdisjoint(val_subjects)
    assert test_subjects.isdisjoint(val_subjects)

    train_df = df[df[SUBJECT_COL].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df[SUBJECT_COL].isin(val_subjects)].reset_index(drop=True) if val_subjects else None
    test_df = df[df[SUBJECT_COL].isin(test_subjects)].reset_index(drop=True)

    print(
        f"[信息] 样本级划分结果（行数）：训练 = {len(train_df)}, "
        f"验证 = {len(val_df) if val_df is not None else 0}, 测试 = {len(test_df)}"
    )

    return train_df, val_df, test_df


# ===================== 单个窗口实验流程 =====================

def run_experiment_for_window(win_size: int) -> None:
    """
    对某个时间窗口大小（秒）执行一次完整实验：
      - 读取对应的 cleaned_data CSV；
      - 按 subject 划分 train/val/test；
      - 训练多模型，输出 leaderboard；
      - 评估测试集；
      - 保存整个 Predictor；
      - 将最优模型与测试指标写入 CSV。
    """
    filename = FILENAME_TEMPLATE.format(prefix=PREFIX, win=win_size)
    data_path = CLEANED_DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"[实验] 开始窗口 {win_size}s 的实验")
    print(f"[信息] 读取特征文件: {data_path.resolve()}")

    if not data_path.exists():
        print(f"[警告] 特征文件不存在，跳过该实验: {data_path}")
        return

    df = pd.read_csv(data_path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"在 {data_path.name} 中未找到标签列 '{LABEL_COL}'，请检查。")
    if SUBJECT_COL not in df.columns:
        raise ValueError(f"在 {data_path.name} 中未找到 subject 列 '{SUBJECT_COL}'，请检查。")

    # 1. 丢弃 label 缺失 / '-' 的样本（理论上 PD/NC 都不再有 '-'，这里保险起见）
    before = len(df)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[(df[LABEL_COL].notna()) & (df[LABEL_COL] != "") & (df[LABEL_COL] != "-")].copy()
    after = len(df)
    if before != after:
        print(f"[信息] 丢弃 label 为空 / '-' 的样本 {before - after} 条，剩余 {after} 条。")

    if after == 0:
        print("[警告] 有效样本为 0，跳过该实验。")
        return

    # 标签分布（行级）
    labels = df[LABEL_COL].to_numpy()
    uniq, cnts = np.unique(labels, return_counts=True)
    print(f"[信息] 行级标签分布: {dict(zip(uniq, cnts))}")
    print(f"[信息] 总样本行数: {len(df)}, 被试数: {df[SUBJECT_COL].nunique()}")

    # 2. 按 subject 划分 train / val / test
    train_df, val_df, test_df = split_by_subject(df)

    # 各集合标签分布
    def print_label_dist(name: str, sub_df: pd.DataFrame) -> None:
        if sub_df is None or len(sub_df) == 0:
            print(f"  - {name}: 0 行")
            return
        labs = sub_df[LABEL_COL].to_numpy()
        u, c = np.unique(labs, return_counts=True)
        print(f"  - {name}: 行数 = {len(sub_df)}, 标签分布 = {dict(zip(u, c))}")

    print("[信息] 各数据集标签分布：")
    print_label_dist("train", train_df)
    print_label_dist("val", val_df)
    print_label_dist("test", test_df)

    # 3. 创建并训练 TabularPredictor（多分类）
    from shutil import rmtree

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    exp_model_dir = MODEL_ROOT / f"win{win_size}s"
    if exp_model_dir.exists():
        # AutoGluon 在 path 已存在且非空时会报错，这里先清空
        rmtree(exp_model_dir)

    print("[信息] 开始训练 AutoGluon TabularPredictor (multiclass)...")
    predictor = TabularPredictor(
        label=LABEL_COL,
        problem_type="multiclass",
        eval_metric="balanced_accuracy",
        path=str(exp_model_dir),
    )

    fit_kwargs = dict(
        presets=TAB_PRESETS,
        time_limit=TIME_LIMIT,
        num_gpus=NUM_GPUS,
    )

    fit_kwargs = dict(
        presets=TAB_PRESETS,
        time_limit=TIME_LIMIT,
        num_gpus=NUM_GPUS,
    )

    if val_df is not None and len(val_df) > 0:
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,     # subject 级验证集
            use_bag_holdout=True,   # 允许 bagging 用 tuning_data 做 holdout
            **fit_kwargs,
        )
    else:
        predictor.fit(
            train_data=train_df,
            **fit_kwargs,
        )


    print("[信息] 训练完成。")

    # 4. 打印 leaderboard 并确定最优模型
    print("[信息] 生成 leaderboard（含超参数）...")
    ref_data = val_df if (val_df is not None and len(val_df) > 0) else train_df
    lb = predictor.leaderboard(
        extra_info=True,
        silent=True,
    )

    print("[结果] Leaderboard（按 score_val 降序）：")

    sort_col = "score_val" if "score_val" in lb.columns else lb.columns[1]
    lb_sorted = lb.sort_values(by=sort_col, ascending=False)
    cols_to_show = [
        c
        for c in [
            "model",
            "score_val",
            "score_test",
            "fit_time",
            "pred_time_val",
            "hyperparameters",
        ]
        if c in lb_sorted.columns
    ]
    print(lb_sorted[cols_to_show].to_string(index=False))

    best_model = None
    best_score_val = None
    best_score_test = None
    best_hparams = None

    if len(lb_sorted) == 0:
        print("[警告] leaderboard 为空，无法确定最优模型。")
    else:
        best_row = lb_sorted.iloc[0]
        best_model = best_row["model"]
        print(f"[信息] AutoGluon 选出的最优模型名（按 {sort_col}）: {best_model}")

        best_score_val = best_row.get("score_val", None)
        best_score_test = best_row.get("score_test", None)
        best_hparams = best_row.get("hyperparameters", None)

        print(f"[信息] 最优模型验证集分数(score_val): {best_score_val}")
        if best_score_test is not None and not (
            isinstance(best_score_test, float) and np.isnan(best_score_test)
        ):
            print(f"[信息] 最优模型测试集分数(score_test): {best_score_test}")
        print(f"[信息] 最优模型超参数: {best_hparams}")

    # 5. 在测试集上评估
    test_metrics: Dict[str, float] = {}
    if test_df is not None and len(test_df) > 0:
        print("[信息] 在外部测试集上评估模型性能...")
        test_metrics = predictor.evaluate(test_df)
        print("[结果] 测试集评估指标：")
        for k, v in test_metrics.items():
            print(f"    {k}: {v}")
    else:
        print("[信息] 本次未划分外部测试集，可参考 leaderboard 中的验证集分数。")

    # 6. 汇总结果写入 CSV，并确保模型落盘
    summary_metrics: Dict[str, float] = {}
    if best_score_val is not None:
        summary_metrics["score_val"] = float(best_score_val)
    if best_score_test is not None and not (
        isinstance(best_score_test, float) and np.isnan(best_score_test)
    ):
        summary_metrics["score_test"] = float(best_score_test)

    # 测试集上的各类指标，统一前缀为 test_
    for k, v in test_metrics.items():
        try:
            summary_metrics[f"test_{k}"] = float(v)
        except (TypeError, ValueError):
            # 避免把无法转成 float 的复杂结构写进 CSV
            continue

    window_name = f"win{win_size}s"
    append_window_result(
        window_name=window_name,
        best_model_name=best_model or "",
        metrics=summary_metrics,
    )

    # AutoGluon 在训练过程中已经把模型保存到了 path，对应目录中应包含最佳模型及候选模型
    predictor.save()  # 显式调用一次以确保元数据写入完整
    print(f"[完成] 窗口 {win_size}s 的完整模型已保存至: {exp_model_dir.resolve()}")


# ===================== 主入口 =====================

def main() -> None:
    print("[信息] 多窗口三分类实验开始。")
    print(f"[信息] 清洗特征目录: {CLEANED_DATA_DIR.resolve()}")
    print(f"[信息] 模型保存根目录: {MODEL_ROOT.resolve()}")

    # 清空旧的汇总结果文件
    if RESULTS_CSV.exists():
        RESULTS_CSV.unlink()

    for win in WINDOW_SIZES:
        run_experiment_for_window(win)

    print("[完成] 所有时间窗口实验已结束。")


if __name__ == "__main__":
    main()

