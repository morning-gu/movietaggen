"""评估指标：手算值锁定。

- rmse / mae / precision_at_k / ndcg_at_k / paired_bootstrap 抄 A 项目
  （genometag-mllm）的测试值，验证逐字复制无漂移。
- spearman / kappa / 生成质量三件套 / 双口径 RMSE 为 B 新增，全部手算锁值。
"""
import numpy as np
import pandas as pd
import pytest

from movietaggen.eval.metrics import (
    cohen_kappa, dual_rmse, gen_coverage, gen_redundancy, gen_relevance, mae,
    ndcg_at_k, paired_bootstrap, precision_at_k, rmse, spearman_corr,
)

# ---- A 项目手算值（与 genometag-mllm/tests/test_metrics.py 逐字对齐）----


def test_rmse_mae():
    pred = np.array([1.0, 0.0, 1.0])
    truth = np.array([0.0, 1.0, 1.0])
    assert rmse(pred, truth) == pytest.approx(np.sqrt(2 / 3))
    assert mae(pred, truth) == pytest.approx(2 / 3)


def test_precision_at_k():
    pred = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([0.9, 0.1, 0.8, 0.2])
    # 相关集(threshold=0.5) = {0,2}; top-2 pred = {0,1} → 命中 {0}
    assert precision_at_k(pred, truth, k=2, threshold=0.5) == pytest.approx(0.5)
    assert precision_at_k(pred, truth, k=4, threshold=0.5) == pytest.approx(0.5)


def test_ndcg_at_k():
    pred = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([0.9, 0.1, 0.8, 0.2])
    # DCG = 1/log2(2) = 1; IDCG = 1/log2(2) + 1/log2(3) = 1.63093
    assert ndcg_at_k(pred, truth, k=2, threshold=0.5) == pytest.approx(1 / 1.63093, rel=1e-4)


def test_perfect_prediction():
    pred = np.array([0.9, 0.1, 0.8])
    assert precision_at_k(pred, pred, k=2, threshold=0.5) == pytest.approx(1.0)
    assert ndcg_at_k(pred, pred, k=2, threshold=0.5) == pytest.approx(1.0)


def test_paired_bootstrap():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    diff, p = paired_bootstrap(a, b, n_resamples=10000, seed=42)
    assert diff == pytest.approx(-1.0)
    assert p < 0.05                      # b 稳定优于 a
    diff0, p0 = paired_bootstrap(a, a, n_resamples=1000, seed=42)
    assert diff0 == pytest.approx(0.0) and p0 > 0.9


# ---- spearman_corr：秩相关（scipy 包装）----


def test_spearman_swap():
    # 手算：pred 秩 [1,2,3]；truth [3,1,2] 的秩也是 [3,1,2]，Σd²=4+1+1=6
    #   → ρ = 1 - 6·6/(3·(3²-1)) = 1 - 36/24 = -0.5
    assert spearman_corr([1, 2, 3], [3, 1, 2]) == pytest.approx(-0.5)
    # Σd²=1+1+0=2 → ρ = 1 - 12/24 = +0.5
    assert spearman_corr([1, 2, 3], [2, 1, 3]) == pytest.approx(0.5)
    assert spearman_corr([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman_corr([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_degenerate():
    # 任一侧常数列：秩全并列、相关分母为 0（spearmanr 给 NaN）→ 保护返回 0.0
    assert spearman_corr([2, 2, 2], [1, 2, 3]) == 0.0
    assert spearman_corr([1, 2, 3], [5, 5, 5]) == 0.0
    # 空输入 → 0.0
    assert spearman_corr([], []) == 0.0


# ---- cohen_kappa：手写 po/pe ----


def test_cohen_kappa_perfect():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == pytest.approx(1.0)
    # 完全不一致（二类互换）：po=0, pe=0.5 → κ = -1（负值，断言 ≤ 0）
    assert cohen_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) <= 0.0


def test_cohen_kappa_chance_level():
    # 恰在随机水平：po=0.5、pe=0.25+0.25=0.5 → κ=0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_cohen_kappa_single_class_protected():
    # 双侧全同类：po=pe=1，κ=0/0 无定义 → 保护返回 0.0
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 0.0
    assert cohen_kappa(["a"], ["a"]) == 0.0


def test_cohen_kappa_length_mismatch():
    with pytest.raises(ValueError, match="长度不一致"):
        cohen_kappa(["a", "b"], ["a"])


# ---- 生成质量三件套：3 维单位正交基 e1/e2/e3 上手算 ----

E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0])
E3 = np.array([0.0, 0.0, 1.0])


def test_gen_relevance_bounds():
    # 双向 top-1 余弦均值：gen=[e1] 对 user=[e1,e2]
    #   正向：mean_i max_j cos = 1；反向：user 行 max cos = [1, 0] → 均值 0.5
    #   → 0.5·(1 + 0.5) = 0.75
    assert gen_relevance(np.array([E1]), np.array([E1, E2])) == pytest.approx(0.75)
    # 完美对应 → 1.0；正交零重叠 → 0.0
    assert gen_relevance(np.array([E1, E2]), np.array([E1, E2])) == pytest.approx(1.0)
    assert gen_relevance(np.array([E1]), np.array([E2])) == pytest.approx(0.0)


def test_gen_relevance_normalization():
    # 非单位向量按余弦归一化：2·e1 与 [e1, e2] 的余弦仍是 [1, 0]
    assert gen_relevance(np.array([2 * E1]), np.array([E1, E2])) == pytest.approx(0.75)


def test_gen_relevance_empty():
    assert gen_relevance(np.zeros((0, 3)), np.array([E1])) == 0.0
    assert gen_relevance(np.array([E1]), np.zeros((0, 3))) == 0.0


def test_gen_coverage():
    # user 侧每行 max cos ≥ tau 的比例：[e1,e2] 对 gen=[e1] → 只覆盖 e1 行
    assert gen_coverage(np.array([E1, E2]), np.array([E1]), tau=0.5) == pytest.approx(0.5)
    assert gen_coverage(np.array([E1, E2]), np.array([E1, E2]), tau=0.5) == pytest.approx(1.0)
    # 边界：cos 恰等于 tau 计入（≥ 语义）
    assert gen_coverage(np.array([E1]), np.array([E1]), tau=1.0) == pytest.approx(1.0)


def test_gen_coverage_empty():
    assert gen_coverage(np.zeros((0, 3)), np.array([E1]), tau=0.5) == 0.0
    # gen 侧为空：无生成内容可覆盖 → 0.0
    assert gen_coverage(np.array([E1]), np.zeros((0, 3)), tau=0.5) == 0.0


def test_gen_redundancy():
    # 两两余弦绝对值均值（i<j）：正交 → 0；相同/相反 → |cos| = 1
    assert gen_redundancy(np.array([E1, E2])) == pytest.approx(0.0)
    assert gen_redundancy(np.array([E1, E1])) == pytest.approx(1.0)
    assert gen_redundancy(np.array([E1, -E1])) == pytest.approx(1.0)
    # [e1, e1, e2]：三对 (e1,e1)=1、(e1,e2)=0、(e1,e2)=0 → 均值 1/3
    assert gen_redundancy(np.array([E1, E1, E2])) == pytest.approx(1 / 3)


def test_gen_redundancy_degenerate():
    # 少于 2 行 → 0.0
    assert gen_redundancy(np.array([E1])) == 0.0
    assert gen_redundancy(np.zeros((0, 3))) == 0.0


# ---- dual_rmse：双口径 RMSE（full 补 0 对齐叉积，pred_only 只看内连接）----


def _pred_df():
    return pd.DataFrame([
        {"movie_id": 1, "tag_id": 10, "score": 1.0},
        {"movie_id": 2, "tag_id": 20, "score": 0.5},
    ])


def _truth_df():
    # 稀疏 truth：叉积 6 对中只标注 4 对；(1,20)/(2,30) 缺失 → relevance 按 0 处理
    return pd.DataFrame([
        {"movie_id": 1, "tag_id": 10, "relevance": 1.0},
        {"movie_id": 1, "tag_id": 30, "relevance": 1.0},
        {"movie_id": 2, "tag_id": 10, "relevance": 0.0},
        {"movie_id": 2, "tag_id": 20, "relevance": 1.0},
    ])


def test_dual_rmse_fill_zero():
    out = dual_rmse(_pred_df(), _truth_df(), all_tag_ids={10, 20, 30})
    # full 口径 = 2 电影 × 3 tag = 6 对；pred 缺失的 4 对补 0：
    #   按 (1,10),(1,20),(1,30),(2,10),(2,20),(2,30) 排：
    #   pred  = [1,   0, 0,   0,   0.5, 0]
    #   truth = [1,   0, 1,   0,   1,   0]
    #   误差平方和 = 0 + 0 + 1 + 0 + 0.25 + 0 = 1.25
    assert out["rmse_full"] == pytest.approx(np.sqrt(1.25 / 6))
    assert out["n_pairs_full"] == 6
    # pred_only 只有内连接 2 对：(1,10) 误差 0；(2,20) 误差 0.5² → 均值 0.125
    assert out["rmse_pred_only"] == pytest.approx(np.sqrt(0.125))


def test_dual_rmse_empty_pred():
    empty = pd.DataFrame({"movie_id": pd.Series(dtype=int),
                          "tag_id": pd.Series(dtype=int),
                          "score": pd.Series(dtype=float)})
    out = dual_rmse(empty, _truth_df(), all_tag_ids={10, 20, 30})
    assert out == {"rmse_full": 0.0, "rmse_pred_only": 0.0, "n_pairs_full": 0}


def test_dual_rmse_rejects_duplicate_pairs():
    pred = pd.DataFrame([
        {"movie_id": 1, "tag_id": 10, "score": 0.5},
        {"movie_id": 1, "tag_id": 10, "score": 0.7},
    ])
    with pytest.raises(ValueError, match="重复"):
        dual_rmse(pred, _truth_df(), all_tag_ids={10, 20, 30})
