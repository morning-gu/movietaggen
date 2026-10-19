"""评估指标：纯函数、numpy/pandas 输入、手算值锁定（tests/test_metrics.py）。

排序类指标约定：truth ≥ threshold 为相关集；按 pred 降序取 top-k。
（rmse / mae / precision_at_k / ndcg_at_k / paired_bootstrap 逐字复制自
genometag-mllm 项目的 genometag/eval/metrics.py——A/B 共用同一套打分口径；
不复制 consistency_std / icc_2_1：方向 B 无逐轮一致性论证，YAGNI。）

向量类约定：输入一律 2-D 行矩阵 (n, d)；余弦分母截断——零向量与任何向量的
余弦记 0（不产生 NaN）；空侧 / 少于两行的退化输入一律返回 0.0，由调用方
决定如何上报。

dual_rmse 双口径：
- rmse_pred_only：pred × truth 内连接，只评"敢打分"的 (movie, tag) 对——
  惩罚"错报"；
- rmse_full：pred 电影 × all_tag_ids 全叉积，缺失预测补 0（未打分 = 预测
  不相关），稀疏 truth 缺失行按 relevance 0 处理——补 0 项同时惩罚"漏报"。
  二者合读才能区分保守（漏报多）与冒进（错报多）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(truth))))


def precision_at_k(pred: np.ndarray, truth: np.ndarray, k: int, threshold: float) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    top = np.argsort(-pred)[:k]
    relevant = truth >= threshold
    hits = relevant[top].sum()
    return float(hits / k)


def ndcg_at_k(pred: np.ndarray, truth: np.ndarray, k: int, threshold: float) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    relevant = truth >= threshold
    top = np.argsort(-pred)[:k]
    gains = relevant[top].astype(float)
    dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
    n_rel = int(relevant.sum())
    ideal = np.ones(min(k, n_rel))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_resamples: int = 10000, seed: int = 42) -> tuple[float, float]:
    """逐单元配对 bootstrap：返回 (mean(a-b), 双侧 p 值)。"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    samples = rng.choice(diffs, size=(n_resamples, n), replace=True).mean(axis=1)
    observed = float(diffs.mean())
    p = 2 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    return observed, min(1.0, p)


def spearman_corr(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman 秩相关（scipy 包装）。

    空输入、或任一侧常数列（秩全并列、相关分母为 0，spearmanr 返回 NaN）
    → 保护返回 0.0。
    """
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    if pred.size == 0 or truth.size == 0:
        return 0.0
    if np.all(pred == pred[0]) or np.all(truth == truth[0]):
        return 0.0
    rho, _ = spearmanr(pred, truth)
    return 0.0 if np.isnan(rho) else float(rho)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa：κ = (po - pe) / (1 - pe)，po=观测一致率、pe=随机期望一致率。

    长度不一致 → 中文 ValueError；任一侧为空、或双侧全同类（po=pe=1，
    κ=0/0 无定义）→ 保护返回 0.0，调用方据此决定是否跳过该指标。
    """
    if len(a) != len(b):
        raise ValueError(f"cohen_kappa：两侧样本数长度不一致（len(a)={len(a)}, len(b)={len(b)}）")
    if not a:
        return 0.0
    if len(set(a)) == 1 and len(set(b)) == 1:
        return 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n)
        for c in set(a) | set(b)
    )
    return (po - pe) / (1.0 - pe)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行向量两两余弦矩阵 (n_a, n_b)；分母截断——零向量与任何向量的余弦记 0。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    return (a / np.maximum(na, 1e-12)) @ (b / np.maximum(nb, 1e-12)).T


def gen_relevance(gen_vecs: np.ndarray, user_vecs: np.ndarray) -> float:
    """生成相关性：双向 top-1 余弦均值。

    0.5 · (mean_i max_j cos(g_i, u_j) + mean_j max_i cos(g_i, u_j))——
    生成侧每个向量找最贴合的用户向量、用户侧每个向量找最贴合的生成向量，
    两个方向的可达性对称平均。任一侧为空 → 0.0。
    """
    gen_vecs, user_vecs = np.asarray(gen_vecs, dtype=float), np.asarray(user_vecs, dtype=float)
    if gen_vecs.size == 0 or user_vecs.size == 0:
        return 0.0
    cos = _cosine_matrix(gen_vecs, user_vecs)
    return float(0.5 * (cos.max(axis=1).mean() + cos.max(axis=0).mean()))


def gen_coverage(user_vecs: np.ndarray, gen_vecs: np.ndarray, tau: float) -> float:
    """生成覆盖度：user 侧每行对生成侧的最大余弦 ≥ tau 的比例（≥ 语义，边界计入）。

    user 侧为空 → 0.0；gen 侧为空（无生成内容可覆盖）→ 0.0。
    """
    user_vecs, gen_vecs = np.asarray(user_vecs, dtype=float), np.asarray(gen_vecs, dtype=float)
    if user_vecs.size == 0 or gen_vecs.size == 0:
        return 0.0
    cos = _cosine_matrix(user_vecs, gen_vecs)
    return float((cos.max(axis=1) >= tau).mean())


def gen_redundancy(gen_vecs: np.ndarray) -> float:
    """生成冗余度：两两余弦绝对值均值（i<j）——方向相反也算冗余。

    少于 2 行 → 0.0。
    """
    gen_vecs = np.asarray(gen_vecs, dtype=float)
    if gen_vecs.shape[0] < 2:
        return 0.0
    cos = _cosine_matrix(gen_vecs, gen_vecs)
    iu = np.triu_indices(cos.shape[0], k=1)
    return float(np.abs(cos[iu]).mean())


def _pair_map(df: pd.DataFrame, value_col: str, name: str) -> dict[tuple[int, int], float]:
    """(movie_id, tag_id) → value 查找表；重复键 → 中文 ValueError（上游聚合层
    应保证 (movie_id, tag_id) 唯一，静默去重/覆盖会悄悄改错口径）。"""
    keys = list(zip(df["movie_id"].astype(int), df["tag_id"].astype(int)))
    if len(set(keys)) != len(keys):
        raise ValueError(f"dual_rmse：{name} 存在重复 (movie_id, tag_id) 对——"
                         "上游聚合层应保证 (movie_id, tag_id) 唯一")
    return dict(zip(keys, df[value_col].astype(float)))


def dual_rmse(pred_df: pd.DataFrame, truth_df: pd.DataFrame, all_tag_ids: set[int]) -> dict:
    """双口径 RMSE：返回 {"rmse_full", "rmse_pred_only", "n_pairs_full"}。

    - pred_df 列 (movie_id, tag_id, score)；truth_df 列 (movie_id, tag_id, relevance)。
    - rmse_pred_only：内连接逐对比较。
    - rmse_full：pred 电影 × all_tag_ids 全叉积；缺失预测补 0、稀疏 truth
      缺失行按 relevance 0 处理（见模块 docstring 的口径说明）。
    - 空 pred → 全 0 字典（无以评估的对象，如何上报由调用方决定）；
      all_tag_ids 为空而 pred 非空 → 中文 ValueError（标签宇宙为空是上游
      契约被破坏，静默给 0 会掩盖问题）。
    """
    if pred_df.empty:
        return {"rmse_full": 0.0, "rmse_pred_only": 0.0, "n_pairs_full": 0}
    if not all_tag_ids:
        raise ValueError("dual_rmse：all_tag_ids 为空，无法对齐 full 口径")
    pred = pred_df[["movie_id", "tag_id", "score"]]
    truth = truth_df[["movie_id", "tag_id", "relevance"]]

    # pred_only：内连接——只评"敢打分"的对。
    merged = pred.merge(truth, on=["movie_id", "tag_id"], how="inner")
    rmse_pred_only = 0.0 if merged.empty else rmse(
        merged["score"].to_numpy(dtype=float), merged["relevance"].to_numpy(dtype=float))

    # full 口径：全叉积对齐，缺失预测 / 缺失 truth 都补 0。
    pred_movies = sorted({int(m) for m in pred["movie_id"]})
    pairs = [(m, t) for m in pred_movies for t in sorted(all_tag_ids)]
    pred_map = _pair_map(pred, "score", "pred")
    truth_map = _pair_map(truth, "relevance", "truth")
    pred_full = np.array([pred_map.get(p, 0.0) for p in pairs])
    truth_full = np.array([truth_map.get(p, 0.0) for p in pairs])
    return {
        "rmse_full": rmse(pred_full, truth_full),
        "rmse_pred_only": rmse_pred_only,
        "n_pairs_full": len(pairs),
    }
