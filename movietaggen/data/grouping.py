"""Tag 语义分组：1,128 tag → ~20 组 + content/preference 类型 + 一句话定义。

分组是一次性离线产物（tag_groups.json，入 git）。跨批一致性：
后续批次的 prompt 携带已出现的组名，引导模型复用而非发明新组。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

VALID_TAG_TYPES = ("content", "preference")

_GROUPING_SYSTEM = "你是电影标签体系专家，负责为 MovieLens genome tag 建立语义分组。"


def build_grouping_messages(tag_batch: list[str], known_groups: list[str]) -> list[dict]:
    hint = ""
    if known_groups:
        hint = "已有组名（优先复用，不要新建语义重复的组）：" + "、".join(known_groups) + "\n"
    user = (
        hint
        + "将以下电影标签按语义分组（目标约 20 组，如：视觉风格、情绪氛围、叙事主题、"
        "角色特征、制作要素、受众偏好……），并为每个标签标注类型"
        "（content = 描述电影本身内容特征；preference = 表达观众偏好/评价）"
        "与一句话定义。\n\n标签：\n"
        + "\n".join(f"- {t}" for t in tag_batch)
        + '\n\n只输出 JSON：{"标签名": {"group": "组名", "tag_type": "content|preference", '
          '"definition": "一句话定义"}, ...}\n'
        "重要：JSON 的键必须与上面给出的标签原文逐字符一致——不翻译、不改写、"
        "不去空格（人名如 \"neil gaiman\" 也保持英文原文）；group 与 definition 用中文。"
    )
    return [{"role": "system", "content": _GROUPING_SYSTEM},
            {"role": "user", "content": user}]


def parse_grouping_response(text: str, expected: list[str]) -> dict[str, dict] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 与 parse_scores 一致的降级：模型可能用 ```json 围栏或前后缀文本包裹
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out: dict[str, dict] = {}
    for tag in expected:
        entry = data.get(tag)
        if not isinstance(entry, dict):
            return None
        group, ttype, definition = entry.get("group"), entry.get("tag_type"), entry.get("definition")
        if not isinstance(group, str) or not group or ttype not in VALID_TAG_TYPES \
                or not isinstance(definition, str) or not definition:
            return None
        out[tag] = {"group": group, "tag_type": ttype, "definition": definition}
    return out


def validate_coverage(parsed: dict[str, dict], expected: list[str]) -> list[str]:
    return [t for t in expected if t not in parsed]


def _parse_entries(text: str) -> dict[str, dict]:
    """宽松解析：能解析多少条算多少（结构非法的条目丢弃），不要求覆盖。

    与 parse_grouping_response 的全有或全无契约互补——补漏机制由调用方负责。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for tag, entry in data.items():
        if not isinstance(entry, dict):
            continue
        group, ttype, definition = (entry.get("group"), entry.get("tag_type"),
                                    entry.get("definition"))
        if not isinstance(group, str) or not group or ttype not in VALID_TAG_TYPES \
                or not isinstance(definition, str) or not definition:
            continue
        out[tag] = {"group": group, "tag_type": ttype, "definition": definition}
    return out


def group_tags_via_llm(tags: list[str], client, cache, model: str,
                       batch_size: int = 150, params: dict | None = None) -> pd.DataFrame:
    """分批调用 LLM 分组：宽松接受 + 缺失补漏（不静默丢 tag）。

    params: 采样参数指纹（sampling_params 产物），进缓存键。

    实测教训（qwen3.5-4b，2026-10-16）：40/批的输出可能合法但漏 1-2 个 tag
    （39/40 keys、finish=stop）——全有或全无校验会报废整批。改为：接受已解析
    条目，缺失标签作为小批重新入队补漏（每标签最多 3 次尝试），仍缺才抛错。
    """
    from movietaggen.generate.cache import cache_key

    def call(batch: list[str], attempt: int) -> str:
        messages = build_grouping_messages(batch, known_groups)
        run_id = "grouping" if attempt == 0 else f"grouping#r{attempt}"
        key = cache_key(messages, model, params or {}, run_id)
        raw = cache.get(key) if cache else None
        if raw is None:
            resp = client.complete(messages, run_id=run_id)
            if cache:
                cache.put(key, resp.raw)
            return resp.content
        return raw.get("choices", [{}])[0].get("message", {}).get("content", "")

    rows: list[dict] = []
    known_groups: list[str] = []
    pending: list[tuple[list[str], int]] = [
        (tags[i:i + batch_size], 0) for i in range(0, len(tags), batch_size)]
    while pending:
        batch, attempt = pending.pop(0)
        entries = _parse_entries(call(batch, attempt))
        batch_set = set(batch)
        accepted = {t: e for t, e in entries.items() if t in batch_set}
        missing = [t for t in batch if t not in accepted]
        for tag, entry in accepted.items():
            rows.append({"tag": tag, **entry})
            if entry["group"] not in known_groups:
                known_groups.append(entry["group"])
        if not missing:
            continue
        if attempt < 2:                      # 补漏：缺失标签作为小批重新入队
            pending.append((missing, attempt + 1))
        else:
            raise RuntimeError(f"分组补漏后仍缺失标签: {missing[:5]}")
    return pd.DataFrame(rows)


def load_tag_groups(path: Path) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(data)[["tag_id", "tag", "group", "tag_type", "definition"]]
