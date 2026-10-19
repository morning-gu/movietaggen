"""审计校准抽样导出：确定性抽样、参考表一致性大声失败与 CSV 回填配对。

覆盖面（brief 三测试 + 控制器裁定钉）：
- 导出列固定 8 列（movie_id, title, tag, tag_id, genome_tag, cosine,
  judge_verdict, human_verdict），human_verdict 全空字符串；标题/genome
  标签分别从 dataset / genome_tags 连接而来，注 sheet 行序即排序键行序；
- 抽样确定性：先按 (movie_id, tag, tag_id) 稳定排序再
  random.Random(seed).sample——同 seed 同样本、异 seed 异样本；
- n > 行数 → 全量；空 judge_df → 8 列空表；
- 一致性：judge 行的 movie_id / tag_id 不在 dataset / genome_tags 中 →
  中文 ValueError 点名缺失 id（校验覆盖全部行，抽样前——审计产物与
  数据集必须一致，注 sheet 静默 NaN 比崩溃更糟）；
- CSV 回填往返：导出 → 文本层手填 human_verdict（含逗号标题/标签考验
  quoting、首尾空格、纯空格格）→ refill_kappa 返回逐行对齐的
  (judge, human) 配对；parse_failed 行跳过；全空 human → None；
  仅 parse_failed 行被填 → None；任一侧全同类 → None（kappa 无定义）。
"""
import pandas as pd
import pytest

from movietaggen.audit.calibration import (
    CALIBRATION_COLUMNS,
    export_calibration_sample,
    refill_kappa,
)


def _judge_df():
    """7 行判定结果（乱序给出，实现内排序）：accept / reject / parse_failed 三态。"""
    return pd.DataFrame({
        "movie_id": [3, 1, 2, 1, 3, 2, 1],
        "tag": ["slow, tense", "mind-bending", "neon visuals", "tense",
                "tense", "mind-bending", "neon visuals"],
        "tag_id": [12, 7, 5, 7, 9, 5, 5],
        "cosine": [0.41, 0.83, 0.66, 0.72, 0.35, 0.58, 0.61],
        "verdict": ["accept", "reject", "accept", "accept",
                    "parse_failed", "reject", "accept"],
        "reason": ["r"] * 7,
        "judge_model": ["qwen3.7-plus"] * 7,
    })


def _dataset():
    """最小 dataset（实现只读 movie_id/title）；标题含逗号考验 quoting 往返。"""
    return pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Paris, Texas", "Neon Alley", "Slow West"],
    })


def _genome_tags():
    return pd.DataFrame({
        "tag_id": [5, 7, 9, 12],
        "tag": ["visual aesthetics", "thought-provoking", "tension",
                "unhurried pace"],
    })


# 排序键 (movie_id, tag, tag_id) 下的稳定行序（手算，供断言引用）：
# 0:(1,mind-bending,7,reject)   1:(1,neon visuals,5,accept)
# 2:(1,tense,7,accept)          3:(2,mind-bending,5,reject)
# 4:(2,neon visuals,5,accept)   5:(3,"slow, tense",12,accept)
# 6:(3,tense,9,parse_failed)


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------

def test_export_columns_sorted_and_joins():
    """8 列固定 schema；human_verdict 全空；行序 = 排序键行序；连接逐格正确。"""
    out = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                    n=7, seed=42)
    assert list(out.columns) == CALIBRATION_COLUMNS
    assert len(out) == 7
    assert (out["human_verdict"] == "").all()
    keys = list(zip(out["movie_id"], out["tag"], out["tag_id"]))
    assert keys == sorted(keys)
    first = out.iloc[0]                     # 排序首行 (1, mind-bending, 7)
    assert first["movie_id"] == 1 and first["tag"] == "mind-bending"
    assert first["title"] == "Paris, Texas"
    assert first["tag_id"] == 7 and first["genome_tag"] == "thought-provoking"
    assert first["cosine"] == pytest.approx(0.83)
    assert first["judge_verdict"] == "reject"


def test_export_sampling_deterministic_same_seed():
    """同 seed 两次调用逐帧一致；样本键是全量键的子集且行数正确。"""
    s1 = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                   n=5, seed=42)
    s2 = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                   n=5, seed=42)
    pd.testing.assert_frame_equal(s1, s2)
    all_keys = set(zip(_judge_df()["movie_id"], _judge_df()["tag"],
                       _judge_df()["tag_id"]))
    assert len(s1) == 5
    assert set(zip(s1["movie_id"], s1["tag"], s1["tag_id"])) <= all_keys


def test_export_sampling_different_seed_differs():
    """异 seed 异样本（seed 42/43 在 7 选 5 上已验证抽样集不相等）。"""
    s42 = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                    n=5, seed=42)
    s43 = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                    n=5, seed=43)
    keys42 = list(zip(s42["movie_id"], s42["tag"], s42["tag_id"]))
    keys43 = list(zip(s43["movie_id"], s43["tag"], s43["tag_id"]))
    assert keys42 != keys43


def test_export_n_exceeds_rows_returns_all():
    """n > 行数 → 全量 7 行，且与 n=7 全量逐帧一致。"""
    out = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                    n=100, seed=42)
    full = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                     n=7, seed=42)
    assert len(out) == 7
    pd.testing.assert_frame_equal(out, full)


def test_export_empty_judge_df_returns_empty_schema(tmp_path):
    """空 judge_df → 8 列空表；对应 CSV 回填 → None（无可配对行）。"""
    empty = pd.DataFrame(columns=["movie_id", "tag", "tag_id", "cosine",
                                  "verdict", "reason", "judge_model"])
    out = export_calibration_sample(empty, _dataset(), _genome_tags(),
                                    n=5, seed=42)
    assert list(out.columns) == CALIBRATION_COLUMNS and len(out) == 0
    csv_path = tmp_path / "empty.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")
    assert refill_kappa(csv_path) is None


def _with_extra_row(judge_df, movie_id, tag_id):
    """追加一行指定 id 的判定（构造参考表缺失场景）。"""
    extra = pd.DataFrame({"movie_id": [movie_id], "tag": ["ghost"],
                          "tag_id": [tag_id], "cosine": [0.5],
                          "verdict": ["accept"], "reason": [""],
                          "judge_model": ["qwen3.7-plus"]})
    return pd.concat([judge_df, extra], ignore_index=True)


def test_export_missing_movie_id_loud_error():
    """judge 行 movie_id 不在 dataset → 中文 ValueError 点名缺失 id。"""
    judge = _with_extra_row(_judge_df(), movie_id=999, tag_id=5)
    with pytest.raises(ValueError) as ei:
        export_calibration_sample(judge, _dataset(), _genome_tags(),
                                  n=7, seed=42)
    msg = str(ei.value)
    assert "movie_id" in msg and "999" in msg and "dataset" in msg


def test_export_missing_tag_id_loud_error():
    """judge 行 tag_id 不在 genome_tags → 中文 ValueError 点名缺失 id。"""
    judge = _with_extra_row(_judge_df(), movie_id=1, tag_id=999)
    with pytest.raises(ValueError) as ei:
        export_calibration_sample(judge, _dataset(), _genome_tags(),
                                  n=7, seed=42)
    msg = str(ei.value)
    assert "tag_id" in msg and "999" in msg and "genome_tags" in msg


def test_export_missing_id_fails_even_if_unsampled():
    """一致性校验覆盖全部 judge 行（抽样前）：坏行即使不被抽中也大声报错。"""
    judge = _with_extra_row(_judge_df(), movie_id=999, tag_id=5)
    with pytest.raises(ValueError):
        export_calibration_sample(judge, _dataset(), _genome_tags(),
                                  n=1, seed=42)


# ---------------------------------------------------------------------------
# CSV 回填配对
# ---------------------------------------------------------------------------

def _export_csv(tmp_path, name, **kwargs):
    out = export_calibration_sample(_judge_df(), _dataset(), _genome_tags(),
                                    **kwargs)
    csv_path = tmp_path / name
    out.to_csv(csv_path, index=False, encoding="utf-8")
    return out, csv_path


def _hand_fill(csv_path, fills):
    """模拟人工在编辑器里改 CSV：按排序行号在行尾空 human_verdict 处补值。"""
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("movie_id")          # 表头在首行
    for idx, value in fills.items():
        line = lines[1 + idx]                       # +1 跳过表头
        assert line.endswith(",")                   # 行尾是空 human_verdict 格
        lines[1 + idx] = line + value
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_csv_roundtrip_hand_fill(tmp_path):
    """写 → 文本层手填几格 → refill_kappa 返回逐行对齐配对；引号保真 + 空白剥离。"""
    _, csv_path = _export_csv(tmp_path, "sample.csv", n=7, seed=42)
    _hand_fill(csv_path, {
        1: "reject",        # judge=accept → 配对 (accept, reject)
        2: "   ",           # 纯空格 → 视为未填
        3: " accept ",      # judge=reject → 配对 (reject, accept)，首尾空白剥离
        5: "accept",        # judge=accept → 配对 (accept, accept)
        6: "accept",        # judge=parse_failed → 跳过
    })

    # quoting 往返：含逗号的标题/标签读回不串列、逐字符保真
    # （3 行属电影 1 → "Paris, Texas" 应完整出现 3 次；若 quoting 破坏则串列为 0）
    back = pd.read_csv(csv_path, encoding="utf-8", dtype=str,
                       keep_default_na=False)
    assert (back["title"] == "Paris, Texas").sum() == 3
    assert set(back["title"]) == {"Paris, Texas", "Neon Alley", "Slow West"}
    assert (back["tag"] == "slow, tense").sum() == 1

    result = refill_kappa(csv_path)
    assert result is not None
    judge_list, human_list = result
    # 排序行 1/3/5 成对；行 2 纯空格未填、行 6 parse_failed 均不入对
    assert judge_list == ["accept", "reject", "accept"]
    assert human_list == ["reject", "accept", "accept"]


def test_refill_all_empty_human_returns_none(tmp_path):
    """没有任何 human_verdict → None（kappa 无从谈起）。"""
    _, csv_path = _export_csv(tmp_path, "sample.csv", n=7, seed=42)
    assert refill_kappa(csv_path) is None


def test_refill_only_parse_failed_filled_returns_none(tmp_path):
    """human 只填在 parse_failed 行上 → 有效配对为空 → None。"""
    _, csv_path = _export_csv(tmp_path, "sample.csv", n=7, seed=42)
    _hand_fill(csv_path, {6: "accept"})   # 唯一 parse_failed 行（排序末行）
    assert refill_kappa(csv_path) is None


def test_refill_single_class_side_returns_none(tmp_path):
    """任一侧全部同类 → None（kappa 无定义，调用方据此跳过）。"""
    # human 侧单类：judge 混合、human 全 accept
    _, csv_path = _export_csv(tmp_path, "human_single.csv", n=7, seed=42)
    _hand_fill(csv_path, {i: "accept" for i in range(7)})
    assert refill_kappa(csv_path) is None

    # judge 侧单类：judge 全 accept、human 混合
    judge = _judge_df().copy()
    judge["verdict"] = "accept"
    out = export_calibration_sample(judge, _dataset(), _genome_tags(),
                                    n=7, seed=42)
    csv_path2 = tmp_path / "judge_single.csv"
    out.to_csv(csv_path2, index=False, encoding="utf-8")
    humans = ["accept", "reject", "accept", "reject", "accept", "accept",
              "reject"]
    _hand_fill(csv_path2, dict(enumerate(humans)))
    assert refill_kappa(csv_path2) is None
