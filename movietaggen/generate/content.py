"""生成侧共享内容助手（自 A 的 genometag/annotate/prompts.py 原样抽出）。

token 近似、预算截断、关键帧抽样、图像编码——A（打分）与 B（生成）共用，
保证两侧对同一模态内容构建出的 prompt 物料一致，Mock 客户端只需一套解析逻辑。

to_multimodal / available_modalities（M9a 消重）：B 的 generate/pipeline.py 与
baselines/direct_score.py 原各持一份逐字节相同的副本，现统一收敛于此共享之家
（本模块即共享助手的既定落点）。函数体自 A 的 genometag/annotate/pipeline.py
逐字移动而来（含 NaN 守卫注释），非重写——verbatim 契约由
.superpowers/sdd/方向B_MovieTagGen_实施计划/task-19-verbatim-check.py 继续钉住。
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from movietaggen.data.movienet import MovieMultimodal


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


def available_modalities(mm: MovieMultimodal, wanted: list[str]) -> list[str]:
    avail = []
    if "video" in wanted and mm.keyframes:
        avail.append("video")
    if "script" in wanted and mm.script:
        avail.append("script")
    if "dialogue" in wanted and mm.subtitles:
        avail.append("dialogue")
    return avail


def to_multimodal(row, movienet_root: Path) -> MovieMultimodal:
    # 注意：混合 str/缺失列经 pandas3 string dtype + parquet 往返后缺失为 NaN（float）
    # —— NaN 为真值，必须用 pd.notna 守卫（真实数据 479/1035 有剧本的混合列实测触发）
    frames = [movienet_root / k for k in row["keyframes"]]
    script = None
    if pd.notna(row["script_path"]) and row["script_path"]:
        p = movienet_root / row["script_path"]
        script = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    subtitles = None
    if pd.notna(row["subtitles_path"]) and row["subtitles_path"]:
        p = movienet_root / row["subtitles_path"]
        subtitles = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    return MovieMultimodal(str(row["movienet_id"]), str(row["title"]),
                           int(row["year"]) if pd.notna(row["year"]) else None,
                           None, frames, script, subtitles)
