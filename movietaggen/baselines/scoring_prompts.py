"""Prompt 组装（模态裁剪、预算截断、few-shot）与输出解析。

逐字复制自 A 的 genometag/annotate/prompts.py（消融轴 9 对照：直接评分基线
与 B 的生成式方法必须在同一 prompt 契约下比较）。唯一差异：四个内容助手
（approx_tokens / truncate_to_budget / sample_keyframes / encode_image）改从
movietaggen.generate.content 导入——Task 7 已把它们原样抽出供 A/B 共用
（controller DRY 裁定，不重复实现）。

文本块格式是 MockClient（annotate/mock.py）的解析契约：
- 电影标题行：MOVIE_TITLE_LINE
- 标签块：TAGS_BLOCK_START 后每行 "- {tag}：{definition}"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from movietaggen.config import ModelConfig
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.generate.content import (
    approx_tokens,
    encode_image,
    sample_keyframes,
    truncate_to_budget,
)

SYSTEM_PROMPT = "你是电影内容分析专家。根据提供的电影内容为标签打关联度分，输出严格的 JSON 对象。"
MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"
TAGS_BLOCK_START = "## 需要打分的标签"
_INSTRUCTIONS = (
    "对下面每一个标签，给出该电影与之的关联度评分（0.0-1.0，保留两位小数）。\n"
    "只输出 JSON 对象，键为标签名、值为分数，不要输出其他内容。"
)


@dataclass
class TagDef:
    tag_id: int
    tag: str
    definition: str


@dataclass
class FewShot:
    title: str
    year: int
    scores: dict[str, float]


def build_messages(movie: MovieMultimodal, tag_defs: list[TagDef],
                   modalities: list[str], few_shots: list[FewShot],
                   model_cfg: ModelConfig) -> list[dict]:
    parts: list[str] = [
        MOVIE_TITLE_LINE.format(title=movie.title, year=movie.year if movie.year else "未知年份"),
        _INSTRUCTIONS,
    ]
    if few_shots:
        lines = ["## 示例（已知关联度的电影）"]
        for fs in few_shots:
            lines.append(f"《{fs.title}》（{fs.year}）：" + ", ".join(
                f"{t}: {s}" for t, s in fs.scores.items()))
        parts.append("\n".join(lines))

    sampled: list[Path] = []
    if "video" in modalities and movie.keyframes:
        sampled = sample_keyframes(movie.keyframes, model_cfg.max_frames)
        parts.append(f"## 关键帧（{len(sampled)} 张，见后附图像）")
    if "script" in modalities and movie.script:
        parts.append("## 剧本节选\n" + truncate_to_budget(movie.script, model_cfg.script_token_budget))
    if "dialogue" in modalities and movie.subtitles:
        parts.append("## 台词/字幕\n" + truncate_to_budget(movie.subtitles, model_cfg.dialogue_token_budget))

    tag_lines = [f"- {t.tag}：{t.definition}" for t in tag_defs]
    parts.append(TAGS_BLOCK_START + "\n" + "\n".join(tag_lines))

    content: list[dict] = [{"type": "text", "text": "\n\n".join(parts)}]
    for frame in sampled:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_scores(text: str, expected_tags: list[str]) -> dict[str, float] | None:
    """解析模型输出。任一 tag 缺失/非数值/越界/不可解析 → None（调用方决定重试）。"""
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return None
    scores: dict[str, float] = {}
    for tag in expected_tags:
        if tag not in data:
            return None
        v = _coerce_score(data[tag])
        if v is None:
            return None
        scores[tag] = v
    return scores


def parse_scores_lenient(text: str, expected_tags: list[str]) -> dict[str, float]:
    """宽松解析：返回全部合法条目，缺失/非法条目跳过（数值字符串如 "0.18" 可转）。

    服务器实测（qwen3.5-4b）：完整 JSON 偶尔漏 1-2 个 tag 或输出字符串数值——
    全有或全无会把 58/60 的好结果整任务报废。缺失 tag 由调用方记 null 并统计。
    """
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for tag in expected_tags:
        if tag in data:
            v = _coerce_score(data[tag])
            if v is not None:
                out[tag] = v
    return out


def _extract_json_object(text: str):
    """直接 json.loads；失败则正则截取首个 {...}（模型可能加围栏/前后缀）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None


def _coerce_score(v) -> float | None:
    """数值校验/转换：bool 拒绝；数值字符串可转；范围 [0,1]；否则 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return None
    if not isinstance(v, (int, float)):
        return None
    return float(v) if 0.0 <= float(v) <= 1.0 else None
