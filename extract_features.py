# -*- coding: utf-8 -*-
"""
帕金森 + 正常对照驾驶数据特征提取脚本（window-level + 3分类）

目录结构假定为：
    DATA_ROOT/
        PD/
            P1/
                P1_study1_pedal.txt
                P1_study2_pedal.txt
            P2/...
        NC/
            P1/
                P1_study1_pedal.txt
                P1_study2_pedal.txt
            P2/...

标签：
    - PD 组：从 Excel 中读取 “编号”–“label”，原来的 2 个 PD 亚型标签继续使用；
    - NC 组：统一标签为 'nc'，不依赖 Excel；
    - Excel 中 label 为 '-' 的样本不参与 3 分类。

输出：
    每个时间窗口一行（window-level），每行包含：
        subject_id   : 'PD_P1' / 'NC_P1'  （带组信息）
        raw_id       : 'P1'（原始编号，方便对照 Excel）
        group        : 'PD' or 'NC'
        study        : 1 / 2
        segment_type : 'city' / 'highway'
        window_idx   : 窗口序号
        label        : 3 类之一（2 个 PD 亚型 + 'nc'）
        + 统计特征列

    写入 ./data 目录：
        {OUTPUT_PREFIX}_win{win}s_windowlevel_features_filtered.csv
        {OUTPUT_PREFIX}_win{win}s_windowlevel_between_class_var.csv
        {OUTPUT_PREFIX}_win{win}s_windowlevel_feature_correlation.csv
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ===================== 参数区（主公按需修改） =====================

# 数据根目录：上层目录，下面有 PD / NC 两个子目录
DATA_ROOT = r"./"   # 例如 "/data1/gushengda/hci/raw"

# 标签 Excel 路径（原来 2 分类的那张表）
LABEL_EXCEL_PATH = r"./label.xlsx"

# 多种时间窗口长度（单位：秒）
WINDOW_SIZES = [30, 60, 120, 180, 300]

# 最小窗口内样本点数（小于该值的窗口将被丢弃；如果最终没有窗口，会退化为整段）
MIN_SAMPLES_PER_WINDOW = 5

# 类间方差阈值：低于该值的特征视为无区分度，将被删除
BETWEEN_CLASS_VAR_THRESHOLD = 0.01

# 输出文件名前缀
OUTPUT_PREFIX = "pd_nc_subtype"

# 输出目录
OUTPUT_DIR = Path("./data")


# ===================== 工具函数：读取与时间窗口划分 =====================

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


# ===================== 单窗口特征提取 =====================

def extract_window_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    对单个时间窗口提取特征，返回一个 dict（不含前缀）。
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


# ===================== 标签 & session 发现 =====================

def load_label_mapping(label_excel_path: str) -> Dict[str, str]:
    """
    从 Excel 读取编号-标签映射，用于 PD 组。

    预期列：
        编号: e.g. 'P1', 'P2', ...
        label: 两种 PD 亚型标签 或 '-'

    仅保留 label != '-' 的记录。
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
    递归查找所有 *_study*_pedal.txt 文件，构造会话映射：

        { (group, raw_id): {1: path_to_study1, 2: path_to_study2} }

    其中：
        group  : 'PD' or 'NC'
        raw_id : 'P1', 'P2', ... （不带组前缀）

    注意：真正用于训练的 subject_id = f"{group}_{raw_id}"，
          用来区分 PD/P1 和 NC/P1 两个不同个体。
    """
    root = Path(data_root)
    sessions: Dict[Tuple[str, str], Dict[int, Path]] = {}

    for f in root.rglob("*_study*_pedal.txt"):
        # 识别 group
        group = None
        for part in f.parts:
            up = part.upper()
            if up in ("PD", "NC"):
                group = up
                break
        if group is None:
            # 既不在 PD 也不在 NC 下，跳过
            continue

        # 解析 raw_id 和 study_num
        # 1) 优先从文件名中解析：P1_study1_pedal.txt
        name = f.stem
        m = re.search(r"(P\d+)_study(\d+)_pedal", name, re.IGNORECASE)
        if m:
            raw_id = m.group(1)          # 'P1'
            study_num = int(m.group(2))  # 1 / 2
        else:
            # 2) 回退：目录名用作 raw_id，文件名中找 study
            raw_id = f.parent.name       # 假定为 'P1'
            m2 = re.search(r"study(\d+)", name, re.IGNORECASE)
            if not m2:
                continue
            study_num = int(m2.group(1))

        if study_num not in (1, 2):
            continue

        key = (group, raw_id)
        sessions.setdefault(key, {})[study_num] = f

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


# ===================== 构造 window-level 特征表 =====================

def build_window_level_features_for_window(
    sessions: Dict[Tuple[str, str], Dict[int, Path]],
    label_map_pd: Dict[str, str],
    window_size_sec: float,
    min_samples_per_window: int,
) -> pd.DataFrame:
    """
    针对某个固定的时间窗口长度，生成“每个窗口一行”的特征表。

    sessions 的 key 为 (group, raw_id)，其中：
        group: 'PD' or 'NC'
        raw_id: 'P1' 等

    行结构：
      subject_id, raw_id, group, label, study, segment_type, window_idx, 以及所有统计特征列
    """
    rows = []

    for (group, raw_id), studies in sorted(sessions.items()):
        if group == "PD":
            # PD 组：从 Excel 映射里取 2 分类标签
            if raw_id not in label_map_pd:
                # 不在 2 分类范围内（label='-' 等），跳过
                continue
            label = label_map_pd[raw_id]
        else:
            # NC 组：直接赋 'nc'
            label = "nc"

        subject_id = f"{group}_{raw_id}"  # 如 "PD_P1" / "NC_P1"

        for study_num, path in sorted(studies.items()):
            segment_type = "city" if study_num == 1 else "highway"

            try:
                df_session = parse_pedal_file(path)
                windows = split_into_time_windows(
                    df_session,
                    window_size_sec=window_size_sec,
                    min_samples_per_window=min_samples_per_window,
                )
                for idx, wdf in enumerate(windows):
                    feats = extract_window_features(wdf)
                    row = {
                        "subject_id": subject_id,
                        "raw_id": raw_id,
                        "group": group,
                        "label": label,
                        "study": study_num,
                        "segment_type": segment_type,
                        "window_idx": idx,
                    }
                    row.update(feats)
                    rows.append(row)
            except Exception as e:
                print(f"[错误] 处理 {group}/{raw_id} study{study_num} 文件 {path} 失败: {e}")
                continue

    if not rows:
        raise ValueError("在当前窗口长度下，没有生成任何窗口级样本，请检查数据和参数。")

    df = pd.DataFrame(rows)
    return df


# ===================== 主流程 =====================

def main():
    data_root = DATA_ROOT
    label_excel = LABEL_EXCEL_PATH

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[信息] 数据根目录: {Path(data_root).resolve()}")
    print(f"[信息] 标签文件:   {Path(label_excel).resolve()}")
    print(f"[信息] 输出目录:   {OUTPUT_DIR.resolve()}")

    # 读取 PD 组标签
    label_map_pd = load_label_mapping(label_excel)
    print(f"[信息] PD 组有效二分类标签数量: {len(label_map_pd)}")

    # 构造 (group, raw_id) -> {study1, study2} 映射
    sessions = discover_sessions(data_root)
    print(f"[信息] 共发现 {len(sessions)} 个 (group, raw_id) 会话目录。")

    # 统计一下 PD / NC 被试数
    pd_subjects = {key for key in sessions if key[0] == "PD"}
    nc_subjects = {key for key in sessions if key[0] == "NC"}
    print(f"[信息] PD 被试个数(原始 raw_id): {len(pd_subjects)}")
    print(f"[信息] NC 被试个数(原始 raw_id): {len(nc_subjects)}")

    for win_size in WINDOW_SIZES:
        print(f"\n[信息] === 处理时间窗口: {win_size} 秒 (window-level, 3分类) ===")

        # 1. 生成 window-level 特征表
        df = build_window_level_features_for_window(
            sessions=sessions,
            label_map_pd=label_map_pd,
            window_size_sec=win_size,
            min_samples_per_window=MIN_SAMPLES_PER_WINDOW,
        )

        label_col = "label"
        meta_cols = ["subject_id", "raw_id", "group", "study", "segment_type", "window_idx", label_col]
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
            f"保留特征数 = {len(keep_features)}, 窗口样本数 = {len(df)}"
        )

        # 构造用于 Tabular/AutoMM 的最终特征表（保留元信息 + 经过筛选的特征）
        final_cols = meta_cols + keep_features
        df_final = df[final_cols].copy()

        # 4. 计算保留特征之间的相关性矩阵
        corr_df = compute_feature_correlation(df_final, feature_cols=keep_features)

        # 5. 保存到 ./data
        base_name = f"{OUTPUT_PREFIX}_win{int(win_size)}s_windowlevel"

        features_csv = OUTPUT_DIR / f"{base_name}_features_filtered.csv"
        bc_csv = OUTPUT_DIR / f"{base_name}_between_class_var.csv"
        corr_csv = OUTPUT_DIR / f"{base_name}_feature_correlation.csv"

        df_final.to_csv(features_csv, index=False, encoding="utf-8-sig")
        bc_df.to_csv(bc_csv, index=False, encoding="utf-8-sig")
        corr_df.to_csv(corr_csv, encoding="utf-8-sig")

        print(f"[完成] 特征表已保存至:        {features_csv}")
        print(f"[完成] 类间方差统计已保存至:  {bc_csv}")
        print(f"[完成] 特征相关性矩阵已保存至:{corr_csv}")


if __name__ == "__main__":
    main()
