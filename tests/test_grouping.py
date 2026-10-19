"""Tag 分组：parse_grouping_response 与 load_tag_groups 用例（摘自 A）。

group_tags_via_llm 的 LLM 分组循环用例（mock 客户端驱动）未随本任务摘取，
留待生成/标注侧任务覆盖（grouping.py 模块本身已逐字复制）。
"""
import json

from movietaggen.data.grouping import load_tag_groups, parse_grouping_response

GOOD = '{"dark": {"group": "视觉风格", "tag_type": "content", "definition": "色调偏暗"}, '\
       '"overrated": {"group": "受众偏好", "tag_type": "preference", "definition": "名过其实"}}'


def test_parse_grouping_response():
    parsed = parse_grouping_response(GOOD, ["dark", "overrated"])
    assert parsed["dark"]["tag_type"] == "content"
    assert parsed["overrated"]["group"] == "受众偏好"


def test_parse_missing_tag_returns_none():
    assert parse_grouping_response(GOOD, ["dark", "missing"]) is None


def test_parse_bad_type_returns_none():
    bad = json.dumps({"dark": {"group": "g", "tag_type": "weird", "definition": "d"}})
    assert parse_grouping_response(bad, ["dark"]) is None


def test_load_tag_groups(tmp_path):
    data = [{"tag_id": 1, "tag": "dark", "group": "视觉风格",
             "tag_type": "content", "definition": "色调偏暗"}]
    p = tmp_path / "tag_groups.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    df = load_tag_groups(p)
    assert list(df.columns) == ["tag_id", "tag", "group", "tag_type", "definition"]
