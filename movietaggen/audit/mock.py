"""确定性 Mock 评审客户端：测试与冒烟用，零网络、零缓存。

与 MockGenClient（generate/mock.py）同族：不读不写缓存（缓存接线由
run_audit 按 config.mock 门控），输出 OpenAI 信封形状（缓存命中路径从
choices[0].message.content 还原文本的契约）。

判定规则（控制器裁定，确定性哈希）：
- mapping：verdict = "accept" 当且仅当 _hash_unit(f"{free_tag}|{genome_tag}")
  > 0.2（约 80% 接受率，保证过滤测试有 reject 可剔）；reason 为 verdict
  对应的固定中文句子。两个标签从 prompt 的引号字段解析——build_mapping_messages
  的 `自由标签: "X"` / `候选 genome 标签: "Y"` 行即本 mock 的解析契约。
- gen_quality：relevance/coverage 各 = 1 + int(hash × 5)（1-5），哈希键含
  标题与两组标签数组。
prompt 无法识别（既非映射判定也非生成质量）或引号字段缺失 → 大声
ValueError（mock 是测试基座，静默编造结果会掩盖上游契约破坏）。
"""
from __future__ import annotations

import hashlib
import json
import re

from movietaggen.generate.client import LLMResponse

# prompt 文本标记 → 任务类型（两类 prompt 的字段互不重叠，按标记分发）
_MAPPING_MARKER = "嵌入相似度:"
_GEN_MARKER = "真实用户标签:"

_FREE_TAG_RE = re.compile(r'自由标签: "(.+?)"')
_GENOME_TAG_RE = re.compile(r'候选 genome 标签: "(.+?)"')
_GEN_TAGS_RE = re.compile(r"生成标签: (\[.*?\])")
_USER_TAGS_RE = re.compile(r"真实用户标签: (\[.*?\])")
_TITLE_RE = re.compile(r"电影：《(.+?)》")

REASON_ACCEPT = "语义对齐且适用于该电影"
REASON_REJECT = "语义与电影内容不匹配"


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值（与 data/sample.py、generate/mock.py 同一实现）。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def _extract_text(messages: list[dict]) -> str:
    """拼接消息中的全部文本（多模态 content parts 防御性兼容；图像 part 无 text 键自然跳过）。"""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _envelope(content: str) -> LLMResponse:
    """固定 usage 常量 + OpenAI 信封形状的 LLMResponse（与 MockGenClient 同构）。"""
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return LLMResponse(
        content=content, usage=usage,
        raw={"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
             "usage": usage, "mock": True},
        ok=True)


def _required(m: re.Match | None, label: str) -> str:
    if m is None:
        raise ValueError(f"MockJudgeClient 无法从 prompt 解析{label}（"
                         "build_mapping_messages/build_gen_quality_messages "
                         "的格式即本 mock 的解析契约）")
    return m.group(1)


class MockJudgeClient:
    """确定性评审客户端：verdict/打分全部由哈希决定，不联网、不碰缓存。"""

    def __init__(self, model: str = "mock-judge-1"):
        self.model = model

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        text = _extract_text(messages)
        if _MAPPING_MARKER in text:
            return self._mapping(text)
        if _GEN_MARKER in text:
            return self._gen_quality(text)
        raise ValueError(
            "MockJudgeClient 无法识别 prompt 类型（既非映射判定也非生成质量打分）")

    def _mapping(self, text: str) -> LLMResponse:
        free_tag = _required(_FREE_TAG_RE.search(text), "自由标签引号字段")
        genome_tag = _required(_GENOME_TAG_RE.search(text), "genome 标签引号字段")
        accept = _hash_unit(f"{free_tag}|{genome_tag}") > 0.2
        verdict = "accept" if accept else "reject"
        reason = REASON_ACCEPT if accept else REASON_REJECT
        return _envelope(json.dumps(
            {"verdict": verdict, "reason": reason}, ensure_ascii=False))

    def _gen_quality(self, text: str) -> LLMResponse:
        gen_tags = json.loads(_required(_GEN_TAGS_RE.search(text), "生成标签数组"))
        user_tags = json.loads(_required(_USER_TAGS_RE.search(text), "用户标签数组"))
        title_m = _TITLE_RE.search(text)
        title = title_m.group(1) if title_m else "unknown"
        key = f"{title}|{json.dumps(gen_tags, ensure_ascii=False)}" \
              f"|{json.dumps(user_tags, ensure_ascii=False)}"
        scores = {name: 1 + int(_hash_unit(f"{name}|{key}") * 5)
                  for name in ("relevance", "coverage")}
        return _envelope(json.dumps(scores, ensure_ascii=False))
