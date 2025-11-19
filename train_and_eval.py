# -*- coding: utf-8 -*-
"""
使用 AutoGluon TabularPredictor 在
  ./cleaned_data/pd_subtype_win30s_features_dedup_corr.csv
上的一键训练脚本。

特点：
  - 自动丢弃 label 为空 / '-' 的样本；
  - 按 subject_id 去重，保证一人一行；
  - 检查每个类别的样本数：
      * 若每个类 >= 2 且可以分出测试集，则做 stratify 划分 train/test；
      * 若某个类样本数 < 2，则不划外部 test，全部数据用于训练，
        只依赖 AutoGluon 内部的 validation 评估。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor


DATA_PATH = "./cleaned_data/pd_subtype_win30s_features_dedup_corr.csv"
LABEL_COL = "label"
SUBJECT_COL = "subject_id"

TEST_SIZE = 0.2
RANDOM_STATE = 42

TAB_PRESETS = "best_quality"   # 或 "medium_quality"
TIME_LIMIT = None              # 限时(秒)，不限制可设为 None

MODEL_SAVE_PATH = "./autogluon_tabular_pd_win30s_model"


def main():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"特征文件不存在：{data_path.resolve()}")

    print(f"[信息] 读取数据文件: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"在 CSV 中未找到标签列 '{LABEL_COL}'，请检查列名。")

    # 1. 丢弃 label 缺失或为 '-' 的样本
    before = len(df)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[(df[LABEL_COL].notna()) & (df[LABEL_COL] != "") & (df[LABEL_COL] != "-")].copy()
    after = len(df)
    print(f"[信息] 丢弃 label 为空或为 '-' 的样本 {before - after} 条，剩余 {after} 条。")

    if after == 0:
        raise ValueError("过滤后没有任何带有效 label 的样本，无法训练。")

    # 2. 按 subject_id 去重，一人一行
    if SUBJECT_COL in df.columns:
        before = len(df)
        df = df.sort_values(SUBJECT_COL)
        df = df.drop_duplicates(subset=[SUBJECT_COL], keep="first").reset_index(drop=True)
        after = len(df)
        print(f"[信息] 按 '{SUBJECT_COL}' 去重，一人一行：从 {before} 行降为 {after} 行。")
    else:
        print(f"[警告] 未找到 '{SUBJECT_COL}' 列，将直接对行做随机划分（假定一行=一人）。")

    labels = df[LABEL_COL].to_numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print(f"[信息] 标签分布：{label_counts}")

    if len(unique_labels) == 1:
        raise ValueError("当前数据只包含一个类别，无法进行二分类训练。请检查 label 列。")

    # 3. 判断是否可以安全地做 stratified train/test split
    min_count = counts.min()

    # 估计每个类别在测试集里期望的样本数（向下取整）
    expected_test_per_class = (counts * TEST_SIZE).astype(int)

    can_stratify = (
        min_count >= 2 and             # 每个类至少 2 个
        (expected_test_per_class >= 1).all()  # 测试集中每类至少能分到 1 个
    )

    if can_stratify:
        print("[信息] 每个类别样本数足够，将使用 stratified train/test 划分。")
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df[LABEL_COL],
            random_state=RANDOM_STATE,
        )
        print(f"[信息] 训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
    else:
        # 类别样本太少，放弃外部 test 划分
        print("[警告] 至少有一个类别的样本数过少，无法安全进行分层 train/test 划分。")
        print("       本次将不单独划分外部测试集，全部样本用于训练，")
        print("       泛化性能请以 AutoGluon 内部的 validation 为准，")
        print("       建议后续尽量增加样本量或补齐各类样本。")
        train_df = df
        test_df = None

    # 4. 训练 TabularPredictor
    print("[信息] 开始训练 AutoGluon TabularPredictor...")
    predictor = TabularPredictor(
        label=LABEL_COL,
        problem_type="binary",
        eval_metric="accuracy",
    )

    predictor.fit(
        train_df,
        presets=TAB_PRESETS,
        time_limit=TIME_LIMIT,
        num_gpus=1,
        # 不显式给 tuning_data，让 AutoGluon 自动从 train_df 中切 validation
    )

    print("[信息] 训练完成。")

    # 5. 如有外部测试集，则在其上评估
    if test_df is not None:
        print("[信息] 在外部测试集上评估模型性能...")
        test_metrics = predictor.evaluate(test_df)
        print("[结果] 测试集评估指标：")
        for k, v in test_metrics.items():
            print(f"    {k}: {v}")
    else:
        print("[信息] 本次未划分外部测试集，可使用 predictor.leaderboard(train_df) 查看内部验证效果。")

    # 6. 保存模型
    save_path = Path(MODEL_SAVE_PATH)
    save_path.mkdir(parents=True, exist_ok=True)
    predictor.save(str(save_path))
    print(f"[完成] 模型已保存至目录: {save_path.resolve()}")


if __name__ == "__main__":
    main()
