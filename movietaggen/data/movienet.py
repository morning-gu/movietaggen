"""MovieNet 适配器：唯一接触 MovieNet 原始格式的模块。

假定格式（合成样本即此格式的镜像；真实数据下载后如列名/布局有出入，
只改本文件的常量与 load_movienet，下游不动）:

    <root>/movies.csv              # movie_id,title,year,imdb_id
    <root>/keyframes/<id>/*.jpg
    <root>/scripts/<id>.txt        # 可选缺失
    <root>/subtitles/<id>.srt      # 可选缺失
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MOVIES_COLUMNS = {"movie_id": "movie_id", "title": "title",
                  "year": "year", "imdb_id": "imdb_id"}

_SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}")


def load_srt(path: Path) -> str:
    """SRT → 纯文本（去序号、时间轴行、空行）。"""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    out = []
    for ln in lines:
        if not ln or ln.isdigit() or _SRT_TIME.match(ln) or "-->" in ln:
            continue
        out.append(ln)
    return "\n".join(out)


@dataclass
class MovieMultimodal:
    movienet_id: str
    title: str
    year: int | None
    imdb_id: str | None
    keyframes: list[Path]
    script: str | None
    subtitles: str | None


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def load_movienet(root: Path) -> list[MovieMultimodal]:
    root = Path(root)
    df = pd.read_csv(root / "movies.csv").rename(columns=MOVIES_COLUMNS)
    movies: list[MovieMultimodal] = []
    for _, row in df.iterrows():
        mid = str(row["movie_id"])
        frames = sorted((root / "keyframes" / mid).glob("*.jpg"))
        script_path = root / "scripts" / f"{mid}.txt"
        sub_path = root / "subtitles" / f"{mid}.srt"
        year = row.get("year")
        movies.append(MovieMultimodal(
            movienet_id=mid,
            title=str(row["title"]),
            year=int(year) if pd.notna(year) else None,
            imdb_id=str(row["imdb_id"]) if pd.notna(row.get("imdb_id")) else None,
            keyframes=frames,
            script=_read_optional(script_path),
            subtitles=load_srt(sub_path) if sub_path.exists() else None,
        ))
    return movies
