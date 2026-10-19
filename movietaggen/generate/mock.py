"""确定性 Mock 生成客户端：测试与冒烟用，零网络、零缓存。

与 A 的 MockClient（打分）对应：按 prompt 契约（prompts.py 的标题行 /
维度块 / 内容节标题）解析电影与请求的维度，从与合成数据共享的规范词表
（data/sample.py 的 SYNTHETIC_TAGS × DIMENSION_GROUPS）哈希采样标签——
生成伪标签与用户自由标签天然重叠，相关性 / 覆盖度指标在 mock 路径下非退化。

消融可区分性（spec §10 mock 约束 b）由两条 mock-only 规则保证（真实 MLLM
不受此约束）：
- 维度门控：只输出 prompt 维度块中出现的维度；
- 模态门控：维度 d 仅当 DIMENSION_MODALITIES[d] 与 prompt 实际出现的
  内容节相交时产标签（如 video-only prompt 下 narrative/character 沉默）。
title-only prompt（无任何内容节）只输出一个标题派生标签。

轮次噪声：采样键含 run_id，n_runs 并集因此非平凡；另以哈希决定的小概率
对选中标签做词序交换，模拟 LLM 逐轮的措辞抖动。
"""
from __future__ import annotations

import hashlib
import json
import re

from movietaggen.data.sample import DIMENSION_GROUPS, SYNTHETIC_TAGS
from movietaggen.generate.client import LLMResponse
from movietaggen.generate.prompts import normalize_tag

# 维度 → 该维度生成标签所依赖的模态（mock-only 门控规则）。
# 维度 d 仅当其模态集与 prompt 实际出现的内容节相交时产标签。
DIMENSION_MODALITIES = {
    "style": {"video"},
    "mood": {"video", "dialogue"},
    "theme": {"script", "dialogue"},
    "narrative": {"script"},
    "character": {"script", "dialogue"},
}

# 内容节标题前缀 → 模态（关键帧节标题带张数后缀，故按前缀匹配）。
_SECTION_MODALITIES = {
    "## 关键帧": "video",
    "## 剧本节选": "script",
    "## 台词/字幕": "dialogue",
}

_TITLE_RE = re.compile(r"电影：《(.+?)》（")
_DIMENSION_LINE_RE = re.compile(r"^- ([a-z_]+)：")
_SWAP_PROB = 0.25   # 选中标签被词序交换的小概率（轮次措辞噪声）


def _build_vocab() -> dict[str, list[str]]:
    """维度 → 规范英文标签词表（与合成数据共享，不在本模块重复词表内容）。"""
    vocab: dict[str, list[str]] = {}
    for t in SYNTHETIC_TAGS:
        d = DIMENSION_GROUPS.get(t["group"])
        if d:
            vocab.setdefault(d, []).append(t["tag"])
    return vocab


VOCAB = _build_vocab()


def _extract_text(messages: list[dict]) -> str:
    """拼接消息中的全部文本（多模态 content parts；图像 part 无 text 键自然跳过）。"""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值（与 data/sample.py 同一实现）。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def _parse_title(text: str) -> str:
    m = _TITLE_RE.search(text)
    return m.group(1) if m else "unknown"


def _parse_dimensions(text: str) -> list[str]:
    """解析维度块 `- {dim}：` 行（仅 DIMENSION_MODALITIES 内的已知维度，按出现序）。"""
    dims: list[str] = []
    if "## 生成维度" not in text:
        return dims
    for line in text.split("## 生成维度", 1)[1].splitlines():
        if line.startswith("## "):
            break   # 进入下一个节，维度块结束
        m = _DIMENSION_LINE_RE.match(line)
        if m and m.group(1) in DIMENSION_MODALITIES and m.group(1) not in dims:
            dims.append(m.group(1))
    return dims


def _available_sections(text: str) -> set[str]:
    """prompt 实际出现的内容节 → 模态集合（前缀匹配：关键帧节标题带张数后缀）。"""
    return {mod for prefix, mod in _SECTION_MODALITIES.items() if prefix in text}


def _maybe_swap(tag: str, key: str) -> str:
    """哈希决定的小概率词序交换（两词以上标签交换前两词），模拟轮次措辞噪声。"""
    words = tag.split()
    if len(words) < 2 or _hash_unit(f"swap|{key}") >= _SWAP_PROB:
        return tag
    return " ".join([words[1], words[0]] + words[2:])


class MockGenClient:
    def __init__(self, n_tags_per_dimension: int = 4, model: str = "mock-1"):
        self.n_tags_per_dimension = n_tags_per_dimension
        self.model = model

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        text = _extract_text(messages)
        title = _parse_title(text)
        sections = _available_sections(text)
        if not sections:
            # title-only 探针（modalities 为空）：仅一个标题派生标签
            tags: list[str] = [f"{normalize_tag(title)} movie"]
        else:
            tags = []
            for d in _parse_dimensions(text):
                if not (DIMENSION_MODALITIES[d] & sections):
                    continue   # 模态门控：该维度的依赖模态未出现
                pool = VOCAB.get(d, [])
                picked = sorted(pool, key=lambda tg: (
                    _hash_unit(f"pick|{title}|{d}|{run_id}|{tg}"), tg))
                tags.extend(_maybe_swap(tg, f"{title}|{d}|{run_id}|{tg}")
                            for tg in picked[: self.n_tags_per_dimension])
        content = json.dumps(tags, ensure_ascii=False)
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        # raw 必须是 OpenAI 信封形状：pipeline 缓存命中路径从 choices[0].message.content
        # 还原文本（缓存测试走 cfg.mock=false + 注入 mock 的组合，信封契约在此生效）
        return LLMResponse(content=content, usage=usage,
                           raw={"choices": [{"message": {"content": content},
                                             "finish_reason": "stop"}],
                                "usage": usage, "mock": True},
                           ok=True)
