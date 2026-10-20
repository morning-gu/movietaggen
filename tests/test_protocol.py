"""评估协议：时序分割、冷启动、tag 类型分解、防泄露断言。

用例复制自 genometag-mllm 项目的 tests/test_protocol.py（A/B 共用同一协议
语义，锁定值一致；仅 import 目标换为 movietaggen.eval.protocol）。
"""
import pandas as pd
import pytest

from movietaggen.eval.protocol import (
    assert_no_leakage, cold_start_subset, fixed_split,
    sample_limited_movie_ids, tag_type_breakdown, temporal_split,
)


def _dataset():
    return pd.DataFrame({"movie_id": [1, 2, 3, 4], "year": [2018, 2021, 2022, 2023]})


def test_temporal_split():
    train, test = temporal_split(_dataset(), 2022)
    assert set(train) == {1, 2}       # 严格小于 2022
    assert set(test) == {3, 4}        # >= 2022


def test_cold_start_subset():
    user_tags = pd.DataFrame({"movie_id": [1, 1, 2], "tag": ["a", "b", "c"]})
    cold = cold_start_subset(_dataset(), user_tags, max_tags=0)
    assert set(cold) == {3, 4}        # 1 有 2 条标签；2 只有 1 条(>0)；3、4 无标签


def test_tag_type_breakdown():
    tg = pd.DataFrame({"tag_id": [1, 2, 3], "tag_type": ["content", "content", "preference"]})
    breakdown = tag_type_breakdown(tg)
    assert breakdown == {"content": [1, 2], "preference": [3]}


def test_assert_no_leakage():
    """Review Focus #4：拟合集与测试集重叠必须抛错。"""
    assert_no_leakage({1, 2}, {3, 4})            # 无重叠 → 通过
    with pytest.raises(ValueError, match="泄露|重叠"):
        assert_no_leakage({1, 2}, {2, 3})


# —— 固定分割与受限抽样（440 协议新增）——


def _write_eval_test(path, ids):
    path.write_text("".join(f"{mid}\n" for mid in ids), encoding="utf-8")


def test_fixed_split_partition(tmp_path):
    """正常切分：train ∪ test == 全集、无交、顺序按 dataset 出现序；空行跳过。"""
    path = tmp_path / "eval_test.txt"
    path.write_text("2\n\n4\n\n", encoding="utf-8")   # 夹空行
    train, test = fixed_split(_dataset(), path)
    assert list(train) == [1, 3]                  # dataset 出现序
    assert list(test) == [2, 4]
    assert set(train) | set(test) == {1, 2, 3, 4}  # train+test == 全集
    assert not (set(train) & set(test))            # 无交


def test_fixed_split_missing_file(tmp_path):
    """文件缺失 → ValueError 点名路径。"""
    with pytest.raises(ValueError, match="不存在"):
        fixed_split(_dataset(), tmp_path / "nope" / "eval_test.txt")


def test_fixed_split_empty_file(tmp_path):
    """空清单（只有空行）→ ValueError 点名路径与成因。"""
    path = tmp_path / "eval_test.txt"
    path.write_text("\n \n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        fixed_split(_dataset(), path)


def test_fixed_split_bad_line(tmp_path):
    """非整数行 → ValueError 点名行号与内容。"""
    path = tmp_path / "eval_test.txt"
    path.write_text("2\nabc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="第 2 行.*abc"):
        fixed_split(_dataset(), path)


def test_fixed_split_out_of_dataset(tmp_path):
    """清单 id 不在 dataset → ValueError 点名缺失电影，test ⊆ dataset 的守门。"""
    path = tmp_path / "eval_test.txt"
    _write_eval_test(path, [2, 99])
    with pytest.raises(ValueError, match="99"):
        fixed_split(_dataset(), path)


def test_sample_limited_none_full(tmp_path):
    """movie_limit=None → 升序全量，且不读清单（路径不存在也无碍）。"""
    ids = sample_limited_movie_ids(_dataset(), None, seed=7,
                                   eval_test_path=tmp_path / "unused.txt")
    assert ids == [1, 2, 3, 4]


def test_sample_limited_within_train_pool(tmp_path):
    """抽样结果 ⊆ 训练池（fixed_split 的 train 侧），长度 = min(limit, 池)。"""
    path = tmp_path / "eval_test.txt"
    _write_eval_test(path, [4])
    ids = sample_limited_movie_ids(_dataset(), 2, seed=7, eval_test_path=path)
    assert len(ids) == 2
    assert set(ids) <= {1, 2, 3}                  # 4 是测试集，不得入池


def test_sample_limited_limit_ge_pool(tmp_path):
    """Review Focus #3：limit ≥ 池 → 全池（不崩溃、不空集）。"""
    path = tmp_path / "eval_test.txt"
    _write_eval_test(path, [4])
    ids = sample_limited_movie_ids(_dataset(), 10, seed=7, eval_test_path=path)
    assert sorted(ids) == [1, 2, 3]


def test_sample_limited_same_seed_deterministic(tmp_path):
    """同 seed 确定性。"""
    path = tmp_path / "eval_test.txt"
    _write_eval_test(path, [4])
    a = sample_limited_movie_ids(_dataset(), 2, seed=7, eval_test_path=path)
    b = sample_limited_movie_ids(_dataset(), 2, seed=7, eval_test_path=path)
    assert a == b
