"""生成侧共享内容助手（自 A 的 genometag/annotate/prompts.py 原样抽出）。

token 近似、预算截断、关键帧抽样、图像编码——A（打分）与 B（生成）共用，
保证两侧对同一模态内容构建出的 prompt 物料一致，Mock 客户端只需一套解析逻辑。
"""
from __future__ import annotations

import base64
from pathlib import Path


def approx_tokens(text: str) -> int:
    """近似 token 数（英文约 4 字符/token；预算截断用，不追求精确）。"""
    return max(1, len(text) // 4)


def truncate_to_budget(text: str, token_budget: int) -> str:
    if approx_tokens(text) <= token_budget:
        return text
    return text[: token_budget * 4]


def sample_keyframes(frames: list[Path], max_frames: int) -> list[Path]:
    if len(frames) <= max_frames:
        return list(frames)
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()
