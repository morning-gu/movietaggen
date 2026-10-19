"""五维度生成 prompt 组装与输出解析。

文本块格式是 MockGenClient（generate/mock.py，Task 8）的解析契约：
- 电影标题行：MOVIE_TITLE_LINE
- 维度块：DIMENSIONS_BLOCK 下每行 "- {dim}：{desc}"（只渲染配置的维度）
- 内容节标题沿用 A：## 关键帧 / ## 剧本节选 / ## 台词/字幕（有内容才渲染）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from movietaggen.config import VALID_DIMENSIONS, ModelConfig
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.generate.content import encode_image, sample_keyframes, truncate_to_budget

SYSTEM_PROMPT = "你是电影内容分析专家。根据提供的电影内容生成自然语言标签，输出严格的 JSON 数组。"
MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"
DIMENSIONS_BLOCK = "## 生成维度"
_INSTRUCTIONS = (
    "基于提供的电影内容，围绕上述维度生成 15-20 个英文标签（单词或短语），"
    "每个标签描述该电影的一个具体特征。"
    '只输出 JSON 数组，如 ["tag one"]，不要输出其他内容。'
)

# 只渲染配置的维度（按 VALID_DIMENSIONS 固定顺序；非法维度忽略，配置校验在上游兜底）
_DIMENSION_LINES = {
    "style": "- style：视觉风格（画面、色彩、镜头语言）",
    "mood": "- mood：情绪基调",
    "theme": "- theme：叙事主题",
    "narrative": "- narrative：叙事手法",
    "character": "- character：角色特征",
}


@dataclass
class FewShotTags:
    """few-shot 示例：已知 ground-truth 标签的电影。"""

    title: str
    year: int
    tags: list[str]


def build_gen_messages(movie: MovieMultimodal, dimensions: list[str],
                       modalities: list[str], few_shots: list[FewShotTags],
                       model_cfg: ModelConfig) -> list[dict]:
    """组装生成 prompt（结构照 A 的 build_messages）：

    标题行 → 指令 → few-shot 块 → 维度块 → 内容节；图像以 base64 附于文本后。
    modalities 为空（title-only 探针）时纯文本、无任何内容节。
    """
    parts: list[str] = [
        MOVIE_TITLE_LINE.format(title=movie.title, year=movie.year if movie.year else "未知年份"),
        _INSTRUCTIONS,
    ]
    if few_shots:
        lines = ["## 示例（已知标签的电影）"]
        for fs in few_shots:
            lines.append(f"《{fs.title}》（{fs.year}）：" + ", ".join(fs.tags))
        parts.append("\n".join(lines))

    dim_lines = [_DIMENSION_LINES[d] for d in VALID_DIMENSIONS if d in dimensions]
    parts.append(DIMENSIONS_BLOCK + "\n" + "\n".join(dim_lines))

    sampled: list[Path] = []
    if "video" in modalities and movie.keyframes:
        sampled = sample_keyframes(movie.keyframes, model_cfg.max_frames)
        parts.append(f"## 关键帧（{len(sampled)} 张，见后附图像）")
    if "script" in modalities and movie.script:
        parts.append("## 剧本节选\n" + truncate_to_budget(movie.script, model_cfg.script_token_budget))
    if "dialogue" in modalities and movie.subtitles:
        parts.append("## 台词/字幕\n" + truncate_to_budget(movie.subtitles, model_cfg.dialogue_token_budget))

    content: list[dict] = [{"type": "text", "text": "\n\n".join(parts)}]
    for frame in sampled:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"}})
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}]


def normalize_tag(tag: str) -> str:
    """归一化标签：casefold + 压缩内部空白 + 去首尾空白。

    仅作去重键使用，不改变输出标签本身（保留首个出现的原形）。
    """
    return " ".join(tag.casefold().split())


def parse_tags(text: str) -> list[str]:
    """解析模型输出为标签列表。

    三级降级：直接 json.loads → 正则截取首个 [...] 再 json.loads →
    逐行解析 `- `/`•`/`数字.` 前缀。全部经 normalize_tag 去重（保留首个原形）。
    任何情况下不抛异常；解析失败返回空列表（调用方统计失败率）。
    """
    tags = _tags_from_json(text)
    if tags is None:
        m = _JSON_ARRAY_RE.search(text)
        if m:
            tags = _tags_from_json(m.group())
    if tags is not None:
        return _dedup(tags)
    return _dedup(_tags_from_lines(text))


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-•]\s*|\d+[.、)）]\s*)")


def _tags_from_json(candidate: str) -> list[str] | None:
    """JSON 数组 → 字符串列表；不可解析 / 非字符串数组 → None（调用方降级）。"""
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    if isinstance(data, list) and all(isinstance(t, str) for t in data):
        return data
    return None


def _tags_from_lines(text: str) -> list[str]:
    out: list[str] = []
    for ln in text.splitlines():
        if _LINE_PREFIX_RE.match(ln):
            out.append(_LINE_PREFIX_RE.sub("", ln, count=1).strip())
    return out


def _dedup(tags: list[str]) -> list[str]:
    """按 normalize_tag 去重，保留首个出现的原形；空白标签丢弃。"""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = normalize_tag(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out
