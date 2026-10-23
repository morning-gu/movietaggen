"""LLM-as-mapper（B②）：prompt 组装、choice 解析、去重选择、并行等价。"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from movietaggen.align.llm_mapper import (
    MAPPER_RUN_ID, MAPPER_SYSTEM_PROMPT, build_mapper_messages,
    llm_select_mappings, parse_choice)
from movietaggen.align.mapping import MAPPING_COLUMNS


GT = {900: "heist", 500: "robbery", 700: "crime",
     400: "slow pacing", 600: "atmosphere"}


class FakeMapperClient:
    """按 tag 字符串返回固定 choice JSON；统计调用数（去重断言用）。"""

    def __init__(self, choices: dict[str, int]):
        self.choices = choices
        self.calls: list[str] = []

    def complete(self, messages, run_id=MAPPER_RUN_ID):
        assert run_id == MAPPER_RUN_ID
        text = messages[1]["content"]
        tag = text.split('"')[1]          # 首个带引号字段 = 自由标签
        self.calls.append(tag)
        choice = self.choices.get(tag, 1)
        return SimpleNamespace(ok=True,
                               content=json.dumps({"choice": choice}),
                               raw={"choices": [{"message": {
                                   "content": json.dumps({"choice": choice})}}]},
                               usage={})


def _cands() -> pd.DataFrame:
    """2 电影共享 tag 'heist film'（各 3 候选）+ 1 电影 'slow burn'（2 候选）。

    候选 tag_id 故意不按数值序（rank 才是余弦序）——选择按 rank 取。"""
    rows = []
    for movie in (1, 2):
        rows += [
            {"movie_id": movie, "tag": "heist film", "tag_id": 900,
             "cosine": 0.90, "rank": 1, "low_conf": False},
            {"movie_id": movie, "tag": "heist film", "tag_id": 500,
             "cosine": 0.85, "rank": 2, "low_conf": False},
            {"movie_id": movie, "tag": "heist film", "tag_id": 700,
             "cosine": 0.80, "rank": 3, "low_conf": False},
        ]
    rows += [
        {"movie_id": 3, "tag": "slow burn", "tag_id": 400,
         "cosine": 0.88, "rank": 1, "low_conf": False},
        {"movie_id": 3, "tag": "slow burn", "tag_id": 600,
         "cosine": 0.82, "rank": 2, "low_conf": False},
    ]
    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def test_mapper_prompt():
    cands = [{"tag_id": 5, "tag": "heist", "definition": "抢劫题材"},
             {"tag_id": 6, "tag": "crime"}]        # 无定义候选
    msgs = build_mapper_messages("heist film", cands)
    assert msgs[0] == {"role": "system", "content": MAPPER_SYSTEM_PROMPT}
    text = msgs[1]["content"]
    assert '自由标签: "heist film"' in text
    assert '1. "heist"（定义: 抢劫题材）' in text
    assert '2. "crime"' in text and "定义" not in text.split('2. "crime"')[1].split("\n")[0]
    assert "输出 0" in text and '{"choice": 编号}' in text


def test_parse_choice_variants():
    assert parse_choice('{"choice": 3}') == 3
    assert parse_choice('```json\n{"choice": 2}\n```') == 2
    assert parse_choice('好的，{"choice": 0}') == 0
    assert parse_choice("垃圾") is None
    assert parse_choice('{"choice": "3"}') is None      # 字符串不收
    assert parse_choice('{"choice": true}') is None    # bool 显式拒绝
    assert parse_choice('{"choice": -1}') is None      # 负数不收
    assert parse_choice('{"other": 1}') is None


def test_llm_select_mappings_dedupe_and_choice():
    """同串跨电影只调用一次（去重）；choice 2 → rank-2 候选被选中（rank/cosine
    原样保留）；choice 0 → 无匹配标签整组剔除。"""
    client = FakeMapperClient({"heist film": 2, "slow burn": 0})
    out = llm_select_mappings(_cands(), GT, {500: "抢劫"}, client, cache=None,
                              model_name="m", extra_body=None)
    assert sorted(client.calls) == ["heist film", "slow burn"]  # 去重：各恰好一次
    assert len(client.calls) == 2
    assert list(out.columns) == MAPPING_COLUMNS
    assert list(out["movie_id"]) == [1, 2]             # 两电影各保留选中行
    assert list(out["tag_id"]) == [500, 500]           # choice 2 = rank-2 候选
    assert list(out["rank"]) == [2, 2] and list(out["cosine"]) == [0.85, 0.85]


def test_llm_select_mappings_parse_fail_and_out_of_range():
    """解析失败（垃圾）与越界（choice > K）都按无有效选择整组剔除。"""
    client = FakeMapperClient({"heist film": 99, "slow burn": 99})  # 越界
    out = llm_select_mappings(_cands(), GT, {}, client, cache=None,
                              model_name="m", extra_body=None)
    assert out.empty

    class GarbageClient(FakeMapperClient):
        def complete(self, messages, run_id=MAPPER_RUN_ID):
            self.calls.append(messages[1]["content"].split('"')[1])
            return SimpleNamespace(ok=True, content="not json",
                                   raw={}, usage={})

    out2 = llm_select_mappings(_cands(), GT, {}, GarbageClient({}), cache=None,
                               model_name="m", extra_body=None)
    assert out2.empty


def test_llm_select_mappings_parallel_equivalence():
    """workers=4 与串行输出逐位一致（pool.map 保序 + 选择任务独立）。"""
    choices = {"heist film": 1, "slow burn": 2}
    c1 = FakeMapperClient(choices)
    c4 = FakeMapperClient(choices)
    serial = llm_select_mappings(_cands(), GT, {}, c1, cache=None,
                                 model_name="m", extra_body=None, workers=1)
    par = llm_select_mappings(_cands(), GT, {}, c4, cache=None,
                              model_name="m", extra_body=None, workers=4)
    pd.testing.assert_frame_equal(serial, par)


def test_llm_select_mappings_inconsistent_candidates_fail():
    """同串跨电影候选序列不一致 → 大声失败（缓存污染守卫）。"""
    cands = _cands()
    cands.loc[(cands["tag"] == "heist film") & (cands["movie_id"] == 2)
              & (cands["rank"] == 2), "tag_id"] = 999   # 破坏一致性
    with pytest.raises(ValueError, match="不一致"):
        llm_select_mappings(cands, GT, {}, FakeMapperClient({}), cache=None,
                            model_name="m", extra_body=None)


def test_llm_select_mappings_empty():
    out = llm_select_mappings(pd.DataFrame(columns=MAPPING_COLUMNS), GT, {},
                              FakeMapperClient({}), cache=None,
                              model_name="m", extra_body=None)
    assert out.empty and list(out.columns) == MAPPING_COLUMNS
