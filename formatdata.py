# -*- coding: utf-8 -*-
"""
帕金森驾驶数据特征提取脚本（A2 + 时间窗口 + 特征筛选 + 相关性分析）

功能概览：
1. 读取 DATA_ROOT 下所有 *_study1_pedal.txt / *_study2_pedal.txt 文件，
   其中 study1 = 城市路段，study2 = 高速路段。
2. 使用时间窗口方法在每个 study 内提取窗口级特征，再对窗口做聚合，
   得到一人一行的特征向量（A2：城市 + 高速两套特征拼在一起）。
3. 从 label_excel（副本被试标签.xlsx）读取编号-标签映射，
   丢弃 label == '-' 的被试，仅保留参与二分类的样本。
4. 对每个时间窗口长度，生成一个特征表 CSV：
      {OUTPUT_PREFIX}_win{win_size}s_features_filtered.csv
   同时输出：
      - {OUTPUT_PREFIX}_win{win_size}s_between_class_var.csv ：各特征的归一化类间方差和是否被删除
      - {OUTPUT_PREFIX}_win{win_size}s_feature_correlation.csv ：保留特征之间的相关系数矩阵
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===================== 参数区（主公可按需修改） =====================

# 数据根目录，下面会递归查找 *_study1_pedal.txt / *_study2_pedal.txt
DATA_ROOT = r"."

# 标签 Excel 路径（例如 "副本被试标签.xlsx"）
LABEL_EXCEL_PATH = r"./label.xlsx"

# 多种时间窗口长度（单位：秒）
WINDOW_SIZES = [30, 60, 120, 180, 300]  # 主公可自行调整，例如 [30, 60, 90]

# 最小窗口内样本点数（小于该值的窗口将被丢弃；如果最终没有窗口，会退化为整段）
MIN_SAMPLES_PER_WINDOW = 5

# 类间方差阈值：低于该值的特征视为无区分度，将被删除
BETWEEN_CLASS_VAR_THRESHOLD = 0.01

# 输出文件名前缀（会在当前工作目录下生成若干 CSV）
OUTPUT_PREFIX = "pd_subtype"


# ===================== 工具函数：读取与窗口划分 =====================

def parse_pedal_file(file_path: Path, encoding: str = "gbk") -> pd.DataFrame:
    """
    解析单个 *_pedal.txt 文件，提取时间戳、方向盘角度、油门、刹车，返回 DataFrame。

    预期行格式示例（GBK 编码）：
        时间戳 = 1723428988.027000  方向盘角度 = 0  油门 = 0  刹车 = 0

    返回列：
        timestamp: float
        steer: float
        throttle: float
        brake: float
    """
    pattern = re.compile(
        r"时间戳\s*=\s*([0-9.]+)\s+"
        r"方向盘角度\s*=\s*([\-0-9.]+)\s+"
        r"油门\s*=\s*([\-0-9.]+)\s+"
        r"刹车\s*=\s*([\-0-9.]+)"
    )

    rows: List[Tuple[float, float, float, float]] = []

    with file_path.open("r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            ts, steer, throttle, brake = m.groups()
            rows.append(
                (
                    float(ts),
                    float(steer),
                    float(throttle),
                    float(brake),
                )
            )

    if not rows:
        raise ValueError(f"文件中未解析到有效 pedal 数据: {file_path}")

    df = pd.DataFrame(rows, columns=["timestamp", "steer", "throttle", "brake"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_into_time_windows(
    df: pd.DataFrame,
    window_size_sec: float,
    min_samples_per_window: int = 5,
) -> List[pd.DataFrame]:
    """
    按时间戳将一次 session 划分为若干非重叠时间窗口，每个窗口返回一个 DataFrame。

    若按设定参数划分后没有任何窗口（例如太短），则退化为单一窗口：整段 df。
    """
    if df.empty:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)
    t0 = df["timestamp"].iloc[0]
    rel_t = df["timestamp"] - t0

    windows: List[pd.DataFrame] = []
    start = 0.0
    t_max = float(rel_t.iloc[-1])

    while start < t_max:
        end = start + window_size_sec
        mask = (rel_t >= start) & (rel_t < end)
        wdf = df.loc[mask]
        if len(wdf) >= min_samples_per_window:
            windows.append(wdf.copy())
        start = end

    # 如果一个窗口都没有（比如总时长 < window_size 且点数太少），则退化为整段
    if not windows:
        windows = [df.copy()]

    return windows


# ===================== 特征提取：窗口级 + 聚合 =====================

def extract_window_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    对单个时间窗口（城市或高速中的一小段）提取特征，返回一个 dict。

    输入列：
        timestamp, steer, throttle, brake

    输出：
        不含前缀的基础特征（后续会在城市/高速 & 聚合时加前缀）
    """
    if df.empty:
        raise ValueError("传入的窗口 DataFrame 为空，无法提取特征。")

    df = df.sort_values("timestamp").reset_index(drop=True)

    ts = df["timestamp"].to_numpy(dtype=float)
    steering = df["steer"].to_numpy(dtype=float)
    throttle = df["throttle"].to_numpy(dtype=float)
    brake = df["brake"].to_numpy(dtype=float)

    feats: Dict[str, float] = {}
    T = len(df)

    # 样本点数量与近似时长
    feats["n_samples"] = float(T)
    if T > 1:
        duration = float(ts[-1] - ts[0])
    else:
        duration = 0.0
    feats["duration_sec"] = duration
    feats["sampling_rate_hz"] = float((T - 1) / duration) if duration > 0 and T > 1 else 0.0

    # 基础统计
    def add_basic_stats(prefix: str, x: np.ndarray):
        feats[f"{prefix}_mean"] = float(np.mean(x))
        feats[f"{prefix}_std"] = float(np.std(x))
        feats[f"{prefix}_min"] = float(np.min(x))
        feats[f"{prefix}_max"] = float(np.max(x))
        feats[f"{prefix}_median"] = float(np.median(x))
        feats[f"{prefix}_q25"] = float(np.percentile(x, 25))
        feats[f"{prefix}_q75"] = float(np.percentile(x, 75))

    add_basic_stats("steer", steering)
    add_basic_stats("throttle", throttle)
    add_basic_stats("brake", brake)

    # 方向盘活动性
    abs_steer = np.abs(steering)
    feats["steer_abs_mean"] = float(np.mean(abs_steer))
    feats["steer_abs_std"] = float(np.std(abs_steer))
    feats["steer_pos_ratio"] = float(np.mean(steering > 0))
    feats["steer_neg_ratio"] = float(np.mean(steering < 0))
    feats["steer_zero_ratio"] = float(np.mean(steering == 0))

    if T > 1:
        sign_changes = np.sum(np.sign(steering[1:]) != np.sign(steering[:-1]))
        feats["steer_signchange_rate"] = float(sign_changes / (T - 1))
    else:
        feats["steer_signchange_rate"] = 0.0

    # 油门 / 刹车使用模式
    for name, x in [("throttle", throttle), ("brake", brake)]:
        feats[f"{name}_active_ratio"] = float(np.mean(x > 0))
        feats[f"{name}_zero_ratio"] = float(np.mean(x == 0))

        active_x = x[x > 0]
        if active_x.size > 0:
            feats[f"{name}_active_mean"] = float(np.mean(active_x))
            feats[f"{name}_active_std"] = float(np.std(active_x))
        else:
            feats[f"{name}_active_mean"] = 0.0
            feats[f"{name}_active_std"] = 0.0

    # 一阶差分平滑度
    if T > 1:
        d_steer = np.diff(steering)
        d_throttle = np.diff(throttle)
        d_brake = np.diff(brake)

        def add_diff_stats(prefix: str, x: np.ndarray):
            feats[f"{prefix}_diff_mean"] = float(np.mean(x))
            feats[f"{prefix}_diff_std"] = float(np.std(x))
            feats[f"{prefix}_diff_abs_mean"] = float(np.mean(np.abs(x)))

        add_diff_stats("steer", d_steer)
        add_diff_stats("throttle", d_throttle)
        add_diff_stats("brake", d_brake)
    else:
        for prefix in ["steer", "throttle", "brake"]:
            feats[f"{prefix}_diff_mean"] = 0.0
            feats[f"{prefix}_diff_std"] = 0.0
            feats[f"{prefix}_diff_abs_mean"] = 0.0

    return feats


def aggregate_window_features(
    window_feature_list: List[Dict[str, float]],
    prefix: str,
) -> Dict[str, float]:
    """
    将同一 study 下多个窗口的特征做聚合，产生一组带前缀的特征。

    输入：
      window_feature_list: 每个元素是 extract_window_features 的输出
      prefix: 例如 "city_" 或 "highway_"

    输出：
      对每个基础特征 f，计算在窗口维度上的 mean 和 std：
         {f}_mean_over_windows, {f}_std_over_windows
      然后加上给定前缀。
    """
    if not window_feature_list:
        return {}

    df = pd.DataFrame(window_feature_list)

    agg_feats: Dict[str, float] = {}
    for col in df.columns:
        col_values = df[col].to_numpy(dtype=float)
        agg_feats[f"{prefix}{col}_mean_over_windows"] = float(np.mean(col_values))
        agg_feats[f"{prefix}{col}_std_over_windows"] = float(np.std(col_values))

    return agg_feats


# ===================== 读取标签 & 构造 subject-study 映射 =====================

def load_label_mapping(label_excel_path: str) -> Dict[str, str]:
    """
    从 Excel 读取编号-标签映射，丢弃 label == '-' 的样本。

    预期列：
        编号: e.g. 'P1', 'P2', ...
        label: 'jz', 'zc', 或 '-'
    """
    df_label = pd.read_excel(label_excel_path)
    if "编号" not in df_label.columns or "label" not in df_label.columns:
        raise ValueError("标签表中缺少 “编号” 或 “label” 列，请检查 Excel。")

    mapping: Dict[str, str] = {}
    for _, row in df_label.iterrows():
        subj = str(row["编号"]).strip()
        lab = str(row["label"]).strip()
        if not subj or lab == "-" or lab == "" or lab.lower() == "nan":
            continue
        mapping[subj] = lab

    return mapping


def discover_sessions(data_root: str) -> Dict[str, Dict[int, Path]]:
    """
    在数据根目录下发现所有 *_study1_pedal.txt / *_study2_pedal.txt 文件，
    返回映射： {subject_id: {1: path_to_study1, 2: path_to_study2}}。

    subject_id 取自文件名中的 'Pxx' 部分：
        例如 P1_study2_pedal.txt -> subject_id = 'P1', study = 2
    """
    root = Path(data_root)
    sessions: Dict[str, Dict[int, Path]] = {}

    for f in root.rglob("*_study*_pedal.txt"):
        name = f.stem
        m = re.search(r"(P\d+)_study(\d+)_pedal", name, re.IGNORECASE)
        if not m:
            continue
        subj = m.group(1)
        study_num = int(m.group(2))
        if study_num not in (1, 2):
            continue
        sessions.setdefault(subj, {})[study_num] = f

    return sessions


# ===================== 类间方差 & 相关性分析 =====================

def compute_between_class_variance(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    对每一维特征：
      1. 先做 z-score 归一化；
      2. 计算归一化后的类间方差 (Fisher-style between-class variance)。

    返回：
        DataFrame: [feature, between_class_var]
    """
    labels = df[label_col].to_numpy()

    rows = []
    for feat in feature_cols:
        x = df[feat].to_numpy(dtype=float)

        # 去除 NaN
        mask = ~np.isnan(x)
        x = x[mask]
        y = labels[mask]
        if x.size <= 1:
            bc_var = 0.0
        else:
            mean = x.mean()
            std = x.std(ddof=0)
            if std == 0:
                bc_var = 0.0
            else:
                z = (x - mean) / std
                z_mean = z.mean()
                bc_var = 0.0
                N = float(len(z))
                # 各类均值
                for cls in np.unique(y):
                    cls_mask = (y == cls)
                    n_k = cls_mask.sum()
                    if n_k == 0:
                        continue
                    m_k = z[cls_mask].mean()
                    p_k = n_k / N
                    bc_var += p_k * (m_k - z_mean) ** 2

        rows.append({"feature": feat, "between_class_var": float(bc_var)})

    return pd.DataFrame(rows)


def compute_feature_correlation(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    计算特征间的皮尔逊相关系数矩阵。
    """
    feat_df = df[feature_cols].copy()
    # 用列均值填补缺失，避免 corr 中出现大量 NaN
    feat_df = feat_df.fillna(feat_df.mean(numeric_only=True))
    corr = feat_df.corr(method="pearson")
    return corr


# ===================== 主流程：按窗口长度生成数据集 =====================

def build_subject_level_features_for_window(
    sessions: Dict[str, Dict[int, Path]],
    label_map: Dict[str, str],
    window_size_sec: float,
    min_samples_per_window: int,
) -> pd.DataFrame:
    """
    针对某个固定的时间窗口长度，生成一行一人的特征表（城市 + 高速 A2 拼接）。

    仅保留出现在 label_map 且 label != '-' 的被试。
    """
    rows = []
    for subj_id, studies in sorted(sessions.items()):
        if subj_id not in label_map:
            # 该编号在二分类标签中不存在，跳过
            continue

        label = label_map[subj_id]

        # 需要同时考虑 study1 (city) 和 study2 (highway)
        # 若缺失某一段，可选择跳过该被试，或降级为单段；此处为了干净分析，直接跳过
        if 1 not in studies or 2 not in studies:
            print(f"[警告] {subj_id} 缺少 study1 或 study2，已跳过。")
            continue

        subj_feats: Dict[str, float] = {"subject_id": subj_id, "label": label}

        # study1 -> city
        try:
            df1 = parse_pedal_file(studies[1])
            windows1 = split_into_time_windows(
                df1,
                window_size_sec=window_size_sec,
                min_samples_per_window=min_samples_per_window,
            )
            win_feats1 = [extract_window_features(wdf) for wdf in windows1]
            agg_city = aggregate_window_features(win_feats1, prefix="city_")
            subj_feats.update(agg_city)
        except Exception as e:
            print(f"[错误] {subj_id} study1 处理失败: {e}")
            continue

        # study2 -> highway
        try:
            df2 = parse_pedal_file(studies[2])
            windows2 = split_into_time_windows(
                df2,
                window_size_sec=window_size_sec,
                min_samples_per_window=min_samples_per_window,
            )
            win_feats2 = [extract_window_features(wdf) for wdf in windows2]
            agg_highway = aggregate_window_features(win_feats2, prefix="highway_")
            subj_feats.update(agg_highway)
        except Exception as e:
            print(f"[错误] {subj_id} study2 处理失败: {e}")
            continue

        rows.append(subj_feats)

    if not rows:
        raise ValueError("在当前窗口长度下，没有成功生成任何被试的特征行。")

    df = pd.DataFrame(rows)
    return df


def main():
    data_root = DATA_ROOT
    label_excel = LABEL_EXCEL_PATH

    print(f"[信息] 数据根目录: {data_root}")
    print(f"[信息] 标签文件:   {label_excel}")

    # 读取标签
    label_map = load_label_mapping(label_excel)
    print(f"[信息] 有效二分类标签数量: {len(label_map)}")

    # 构造 subject -> {study1, study2} 映射
    sessions = discover_sessions(data_root)
    print(f"[信息] 共发现 {len(sessions)} 个被试的 pedal 文件。")

    for win_size in WINDOW_SIZES:
        print(f"\n[信息] === 处理时间窗口: {win_size} 秒 ===")

        # 1. 生成 A2（城市+高速）的一人一行特征表
        df = build_subject_level_features_for_window(
            sessions=sessions,
            label_map=label_map,
            window_size_sec=win_size,
            min_samples_per_window=MIN_SAMPLES_PER_WINDOW,
        )

        # 将非数值/元信息列分离
        label_col = "label"
        meta_cols = ["subject_id", label_col]
        numeric_cols = [
            c for c in df.columns
            if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        # 2. 计算归一化后的类间方差
        bc_df = compute_between_class_variance(
            df=df,
            label_col=label_col,
            feature_cols=numeric_cols,
        )
        # 标记哪些特征被删除
        bc_df["dropped"] = bc_df["between_class_var"] < BETWEEN_CLASS_VAR_THRESHOLD

        # 3. 删除类间方差很小的特征
        keep_features = bc_df.loc[~bc_df["dropped"], "feature"].tolist()
        print(
            f"[信息] 窗口 {win_size}s：原始特征数 = {len(numeric_cols)}, "
            f"保留特征数 = {len(keep_features)}"
        )

        # 构造用于 AutoMM 的最终特征表（保留元信息 + 经过筛选的特征）
        final_cols = meta_cols + keep_features
        df_final = df[final_cols].copy()

        # 4. 计算保留特征之间的相关性矩阵
        corr_df = compute_feature_correlation(df_final, feature_cols=keep_features)

        # 5. 保存到 CSV
        base_name = f"{OUTPUT_PREFIX}_win{int(win_size)}s"

        features_csv = f"{base_name}_features_filtered.csv"
        bc_csv = f"{base_name}_between_class_var.csv"
        corr_csv = f"{base_name}_feature_correlation.csv"

        df_final.to_csv(features_csv, index=False, encoding="utf-8-sig")
        bc_df.to_csv(bc_csv, index=False, encoding="utf-8-sig")
        corr_df.to_csv(corr_csv, encoding="utf-8-sig")

        print(f"[完成] 特征表已保存至:        {features_csv}")
        print(f"[完成] 类间方差统计已保存至:  {bc_csv}")
        print(f"[完成] 特征相关性矩阵已保存至:{corr_csv}")


if __name__ == "__main__":
    main()
