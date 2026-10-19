"""映射与聚合：自由标签嵌入 → genome 标签 top-k 余弦映射 + (电影, genome 标签) 分数聚合。

纯函数模块：嵌入矩阵由调用方用 Task 10 的 embedder 生成后按行序对齐传入
（free_matrix 第 i 行对应 free_tags 第 i 行，genome_matrix 同理），本模块
不做嵌入、不做 I/O。矩阵须已 L2 归一化（三个 embedder 的输出契约），
余弦即内积：sims = free_matrix @ genome_matrix.T。

排序确定性：候选按 (cosine 降序, tag_id 升序) 全序排列（np.lexsort，末键
为主键）——并列分数由 tag_id 裁决而非矩阵行序，同分不同行序的输入产出恒
同的映射结果，无任何浮点不确定性来源。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from movietaggen.config import VALID_AGG

MAPPING_COLUMNS = ["movie_id", "tag", "tag_id", "cosine", "rank", "low_conf"]
SCORE_COLUMNS = ["movie_id", "tag_id", "score"]


def map_free_tags(free_tags: pd.DataFrame, genome_tags: pd.DataFrame,
                  free_matrix: np.ndarray, genome_matrix: np.ndarray,
                  top_k: int, threshold: float) -> pd.DataFrame:
    """逐自由标签取 genome 余弦 top-k 候选（列 movie_id, tag, tag_id, cosine, rank, low_conf）。

    - 行数守卫：矩阵行数与对应 DataFrame 行数不一致 → ValueError（错位矩阵
      会静默产出垃圾映射，宁可在接缝处大声失败）。
    - 每对 (自由标签, 候选) 一行；rank 为该候选在其自由标签 top-k 中的名次
      （1 起），非全局名次。top_k 超过 genome 标签数时按可用数截断。
    - low_conf 仅为旗标（cosine < threshold），不丢行——是否剔除由调用方决定。
    - 自由标签或 genome 标签任一为空 → 带 MAPPING_COLUMNS 全列 schema 的空
      DataFrame（与 data.align 零匹配产物同型）。
    """
    if free_matrix.shape[0] != len(free_tags):
        raise ValueError(
            f"free_matrix 行数 {free_matrix.shape[0]} 与 free_tags 行数 "
            f"{len(free_tags)} 不一致（矩阵行序须与 DataFrame 行序逐行对齐）")
    if genome_matrix.shape[0] != len(genome_tags):
        raise ValueError(
            f"genome_matrix 行数 {genome_matrix.shape[0]} 与 genome_tags 行数 "
            f"{len(genome_tags)} 不一致（矩阵行序须与 DataFrame 行序逐行对齐）")

    sims = np.asarray(free_matrix) @ np.asarray(genome_matrix).T  # (n_free, n_genome)
    tag_ids = genome_tags["tag_id"].to_numpy()
    movie_ids = free_tags["movie_id"].to_numpy()
    tags = free_tags["tag"].to_numpy()

    rows = []
    for i in range(sims.shape[0]):
        # (cosine 降序, tag_id 升序) 全序取 top-k：lexsort 末键为主键
        order = np.lexsort((tag_ids, -sims[i]))[:top_k]
        for rank, j in enumerate(order, start=1):
            cosine = float(sims[i, j])
            rows.append({"movie_id": movie_ids[i], "tag": tags[i],
                         "tag_id": tag_ids[j], "cosine": cosine,
                         "rank": rank, "low_conf": cosine < threshold})
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def aggregate_scores(mappings: pd.DataFrame, agg: str) -> pd.DataFrame:
    """同 (movie_id, tag_id) 多条映射聚合为单一分数（列 movie_id, tag_id, score）。

    - 聚合作用于传入的全部行：low_conf 行的剔除是调用方职责（如
      ``mappings[~mappings.low_conf]``），本函数不过滤——审计侧可对同一份
      mappings 复算不同口径。
    - agg="max" 取组内最大余弦，"mean" 取均值；其他取值 ValueError。
    - 输出按 (movie_id, tag_id) 排序（groupby 键序），确定性。
    - 空输入 → 带 SCORE_COLUMNS 全列 schema 的空 DataFrame。
    """
    if agg not in VALID_AGG:
        raise ValueError(f"agg 必须是 {VALID_AGG}，得到 {agg!r}")
    if len(mappings) == 0:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    grouped = mappings.groupby(["movie_id", "tag_id"], sort=True)["cosine"]
    score = grouped.max() if agg == "max" else grouped.mean()
    return score.reset_index().rename(columns={"cosine": "score"})
