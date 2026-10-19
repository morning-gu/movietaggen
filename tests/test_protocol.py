"""评估协议：时序分割、冷启动、tag 类型分解、防泄露断言。

用例复制自 genometag-mllm 项目的 tests/test_protocol.py（A/B 共用同一协议
语义，锁定值一致；仅 import 目标换为 movietaggen.eval.protocol）。
"""
import pandas as pd
import pytest

from movietaggen.eval.protocol import (
    assert_no_leakage, cold_start_subset, tag_type_breakdown, temporal_split,
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
