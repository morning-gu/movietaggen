"""映射与聚合：top-k 余弦映射（并列按 tag_id 裁决）、low_conf 旗标、max/mean 聚合、行数守卫。

手造 4 维正交基 e1..e4 上的单位向量，余弦全部手工可算：
- 自由标签 tag a = e1；tag b = (7/9)e1 + (√32/9)e4（|b| = 49/81 + 32/81 = 1）
- genome t1（tag_id 11）= 0.9e1 + √0.19 e2 → cos(a,t1)=0.9、cos(b,t1)=0.9·7/9=0.7
- genome t2（tag_id 22）= 0.6e1 + 0.8e3 → cos(a,t2)=0.6、cos(b,t2)=0.6·7/9≈0.4667
（genome 向量的其余正交分量互不干扰：t1 无 e3/e4 分量，t2 无 e2/e4 分量。）
"""
import numpy as np
import pandas as pd
import pytest

from movietaggen.align.mapping import aggregate_scores, map_free_tags

MAPPING_COLUMNS = ["movie_id", "tag", "tag_id", "cosine", "rank", "low_conf"]
SCORE_COLUMNS = ["movie_id", "tag_id", "score"]


def _free_tags():
    """两个自由标签同属电影 1（聚合测试需要同 (movie, tag_id) 组内两条映射）。"""
    return pd.DataFrame([
        {"movie_id": 1, "tag": "tag a", "n_runs": 2},
        {"movie_id": 1, "tag": "tag b", "n_runs": 1},
    ])


def _genome_tags():
    return pd.DataFrame([{"tag_id": 11, "tag": "t1"},
                         {"tag_id": 22, "tag": "t2"}])


def _free_matrix():
    return np.array([
        [1.0, 0.0, 0.0, 0.0],                    # tag a = e1
        [7 / 9, 0.0, 0.0, np.sqrt(32) / 9],      # tag b
    ])


def _genome_matrix():
    return np.array([
        [0.9, np.sqrt(0.19), 0.0, 0.0],          # t1：与 e1 余弦 0.9
        [0.6, 0.0, 0.8, 0.0],                    # t2：与 e1 余弦 0.6
    ])


def _row_for(mappings, tag, tag_id):
    """取指定 (自由标签, genome 标签) 的唯一映射行。"""
    rows = mappings[(mappings["tag"] == tag) & (mappings["tag_id"] == tag_id)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------- map_free_tags：top-k 与阈值 ----------


def test_map_topk3_threshold05_keeps_all_low_conf_false():
    """threshold=0.5、top_k=3（> genome 数 2，按可用数截断）→ 两候选均保留、rank 1/2、旗标全 False。"""
    m = map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                      _genome_matrix(), top_k=3, threshold=0.5)
    assert list(m.columns) == MAPPING_COLUMNS
    assert len(m) == 4                       # 2 自由标签 × 2 候选（top_k 截断到可用数）
    # 确定性输出序：自由标签按输入行序，候选按名次
    assert list(m["tag"]) == ["tag a", "tag a", "tag b", "tag b"]
    assert list(m["rank"]) == [1, 2, 1, 2]
    a1, a2 = _row_for(m, "tag a", 11), _row_for(m, "tag a", 22)
    assert a1["cosine"] == pytest.approx(0.9)
    assert a2["cosine"] == pytest.approx(0.6)
    # tag a 两行 0.9 / 0.6 均 ≥ 0.5 → False；tag b 对 t2 仅 0.4667 < 0.5 → True
    assert list(m["low_conf"]) == [False, False, False, True]


def test_map_threshold07_marks_low_conf_but_keeps_row():
    """threshold=0.7 → t2 行（0.6 < 0.7）low_conf=True；旗标只标记不丢行，rank 不变。"""
    m = map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                      _genome_matrix(), top_k=3, threshold=0.7)
    a = m[m["tag"] == "tag a"].sort_values("rank")
    assert list(a["tag_id"]) == [11, 22]
    assert list(a["rank"]) == [1, 2]
    assert list(a["low_conf"]) == [False, True]
    b = m[m["tag"] == "tag b"].sort_values("rank")
    assert list(b["low_conf"]) == [False, True]   # 0.7 恰在阈值上（≥ 保留）；0.4667 < 0.7


def test_map_tag_b_cosines_hand_computed():
    """tag b 的两条余弦按构造精确可算：对 t1 为 0.9·7/9=0.7，对 t2 为 0.6·7/9。"""
    m = map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                      _genome_matrix(), top_k=3, threshold=0.5)
    b1, b2 = _row_for(m, "tag b", 11), _row_for(m, "tag b", 22)
    assert b1["cosine"] == pytest.approx(0.7)
    assert b2["cosine"] == pytest.approx(7 / 9 * 0.6)


def test_map_topk1_only_rank1_rows():
    """top_k=1 → 每自由标签只留 rank=1 行；两标签的首选都是 t1。"""
    m = map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                      _genome_matrix(), top_k=1, threshold=0.5)
    assert len(m) == 2
    assert list(m["rank"]) == [1, 1]
    assert list(m["tag"]) == ["tag a", "tag b"]
    assert list(m["tag_id"]) == [11, 11]
    assert m.iloc[1]["cosine"] == pytest.approx(0.7)


def test_map_tie_broken_by_lower_tag_id_not_row_order():
    """同分并列 → tag_id 升序裁决，与 genome 矩阵行序无关。

    两 genome 向量均与 e1 正交（余弦精确 0.0）；tag_id 5 在矩阵行 0、tag_id 3
    在行 1 —— 若按行序裁决会输出 [5, 3]，按本契约必须输出 [3, 5]。
    """
    genome_tags = pd.DataFrame([{"tag_id": 5, "tag": "g5"},
                                {"tag_id": 3, "tag": "g3"}])
    genome_matrix = np.array([[0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0]])
    m = map_free_tags(_free_tags().iloc[:1], genome_tags,
                      _free_matrix()[:1], genome_matrix, top_k=2, threshold=0.5)
    assert list(m["tag_id"]) == [3, 5]
    assert list(m["rank"]) == [1, 2]
    assert list(m["cosine"]) == [pytest.approx(0.0), pytest.approx(0.0)]


# ---------- map_free_tags：守卫与空输入 ----------


def test_map_free_matrix_row_mismatch_raises():
    """free_matrix 行数 ≠ len(free_tags) → ValueError，中文信息同时点出两个行数。"""
    with pytest.raises(ValueError) as ei:
        map_free_tags(_free_tags(), _genome_tags(), np.zeros((3, 4)),
                      _genome_matrix(), top_k=3, threshold=0.5)
    msg = str(ei.value)
    assert "free_matrix" in msg
    assert "3" in msg and "2" in msg


def test_map_genome_matrix_row_mismatch_raises():
    """genome_matrix 行数 ≠ len(genome_tags) → ValueError，中文信息同时点出两个行数。"""
    with pytest.raises(ValueError) as ei:
        map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                      np.zeros((5, 4)), top_k=3, threshold=0.5)
    msg = str(ei.value)
    assert "genome_matrix" in msg
    assert "5" in msg and "2" in msg


def test_map_empty_free_tags_returns_full_schema():
    """空自由标签 → 空 DataFrame，但列 schema 完整（下游写空 parquet 的同型场景）。"""
    m = map_free_tags(_free_tags().iloc[0:0], _genome_tags(), np.zeros((0, 4)),
                      _genome_matrix(), top_k=3, threshold=0.5)
    assert len(m) == 0
    assert list(m.columns) == MAPPING_COLUMNS


def test_map_empty_genome_tags_returns_full_schema():
    """空 genome 标签 → 空 DataFrame 且 schema 完整（每个自由标签零候选）。"""
    m = map_free_tags(_free_tags(), _genome_tags().iloc[0:0], _free_matrix(),
                      np.zeros((0, 4)), top_k=3, threshold=0.5)
    assert len(m) == 0
    assert list(m.columns) == MAPPING_COLUMNS


# ---------- aggregate_scores ----------


def test_aggregate_max_and_mean_hand_computed():
    """同 (movie, tag_id) 组内两条映射（0.9 / 0.7）→ max=0.9、mean=0.8（brief 关键样例）。"""
    mappings = map_free_tags(_free_tags(), _genome_tags(), _free_matrix(),
                             _genome_matrix(), top_k=3, threshold=0.5)
    mx = aggregate_scores(mappings, "max")
    assert list(mx.columns) == SCORE_COLUMNS
    assert len(mx) == 2                       # 每 (movie, tag_id) 组一行
    assert list(mx["tag_id"]) == [11, 22]     # groupby 排序键序，确定性输出
    row = mx[mx["tag_id"] == 11].iloc[0]
    assert row["movie_id"] == 1
    assert row["score"] == pytest.approx(0.9)
    assert mx[mx["tag_id"] == 22].iloc[0]["score"] == pytest.approx(0.6)

    mn = aggregate_scores(mappings, "mean")
    assert list(mn.columns) == SCORE_COLUMNS
    assert mn[mn["tag_id"] == 11].iloc[0]["score"] == pytest.approx(0.8)
    assert mn[mn["tag_id"] == 22].iloc[0]["score"] == pytest.approx((0.6 + 7 / 9 * 0.6) / 2)


def test_aggregate_accepts_minimal_schema_and_includes_low_conf_rows():
    """聚合作用于传入的全部行——low_conf 行的剔除是调用方职责，本函数不过滤。"""
    mappings = pd.DataFrame([
        {"movie_id": 1, "tag_id": 11, "cosine": 0.9, "low_conf": False},
        {"movie_id": 1, "tag_id": 11, "cosine": 0.1, "low_conf": True},
    ])
    out = aggregate_scores(mappings, "mean")
    assert out.iloc[0]["score"] == pytest.approx(0.5)


def test_aggregate_invalid_agg_raises():
    """agg 非 max/mean → ValueError（中文信息列出合法取值）。"""
    mappings = pd.DataFrame({"movie_id": [1], "tag_id": [11], "cosine": [0.9]})
    with pytest.raises(ValueError) as ei:
        aggregate_scores(mappings, "sum")
    msg = str(ei.value)
    assert "max" in msg and "mean" in msg


def test_aggregate_empty_input_returns_full_schema():
    """空映射表 → 空 DataFrame 且三列 schema 完整。"""
    out = aggregate_scores(pd.DataFrame(columns=MAPPING_COLUMNS), "max")
    assert len(out) == 0
    assert list(out.columns) == SCORE_COLUMNS
