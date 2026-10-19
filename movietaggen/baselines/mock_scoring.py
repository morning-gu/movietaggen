"""确定性 Mock 打分客户端：测试与冒烟用，零网络、零缓存。

逐字复制自 A 的 genometag/annotate/mock.py（MockClient → MockScoringClient）。

打分规则：score = clip(base_score(title, tag) + hash 噪声(run 维度), 0, 1)。
base_score 与合成样本 ground truth 共用（见 data/sample.py），
因此 mock 预测与 ground truth 天然相关、轮次间小幅波动——
这使指标函数在集成测试中产生非退化输出。
"""
from __future__ import annotations

import hashlib
import json
import re

from movietaggen.data.sample import base_score
from movietaggen.generate.client import LLMResponse

_TITLE_RE = re.compile(r"电影：《(.+?)》（")
_TAG_LINE_RE = re.compile(r"^- (.+?)：.*$", re.MULTILINE)
_TAGS_HEADER = "## 需要打分的标签"


def _extract_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):  # 多模态 content parts
            parts.extend(p.get("text", "") for p in c if isinstance(p, dict))
    return "\n".join(parts)


def _hash_unit(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


class MockScoringClient:
    def __init__(self, correlation_noise: float = 0.05, model: str = "mock-1"):
        self.correlation_noise = correlation_noise
        self.model = model

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        text = _extract_text(messages)
        m = _TITLE_RE.search(text)
        title = m.group(1) if m else "unknown"
        tags: list[str] = []
        if _TAGS_HEADER in text:
            block = text.split(_TAGS_HEADER, 1)[1]
            for line in block.splitlines():
                lm = _TAG_LINE_RE.match(line)
                if lm:
                    tags.append(lm.group(1))
        scores = {}
        for tag in tags:
            noise = (_hash_unit(f"noise|{title}|{tag}|{run_id}") - 0.5) * 2 * self.correlation_noise
            scores[tag] = min(1.0, max(0.0, base_score(title, tag) + noise))
        content = json.dumps(scores, ensure_ascii=False)
        usage = {"prompt_tokens": 100, "completion_tokens": 50}
        # raw 必须是 OpenAI 信封形状：pipeline 缓存命中路径从 choices[0].message.content
        # 还原文本（缓存测试走 cfg.mock=false + 注入 mock 的组合，信封契约在此生效）
        return LLMResponse(content=content, usage=usage,
                           raw={"choices": [{"message": {"content": content},
                                             "finish_reason": "stop"}],
                                "usage": usage, "mock": True},
                           ok=True)
