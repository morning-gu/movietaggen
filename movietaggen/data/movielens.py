"""MovieLens 适配器：genome-scores/tags、movies、links、tags（用户标签）。

统一输出规范列名（snake_case），movie_id/tag_id 为 int。原始文件格式稳定
（ml-latest 与 ml-25m 的 genome 文件结构相同）。
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_TITLE_YEAR_RE = re.compile(r"^(.*) \((\d{4})\)$")


def parse_title_year(title: str) -> tuple[str, int | None]:
    """'Toy Story (1995)' → ('Toy Story', 1995)；无年份 → (原标题, None)。"""
    m = _TITLE_YEAR_RE.match(title.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return title.strip(), None


def load_genome_scores(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"movieId": "movie_id", "tagId": "tag_id"})[
        ["movie_id", "tag_id", "relevance"]]


def load_genome_tags(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"tagId": "tag_id"})[["tag_id", "tag"]]


def load_movies(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "movie_id": df["movieId"],
        "title": df["title"],
    })
    parsed = out["title"].map(parse_title_year)
    out["title"] = parsed.map(lambda t: t[0])
    out["year"] = parsed.map(lambda t: t[1]).astype("Int64")
    out["genres"] = df["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|"))
    return out


def load_links(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(columns={"movieId": "movie_id", "imdbId": "imdb_id"})[
        ["movie_id", "imdb_id"]]


def load_user_tags(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"userId": "user_id", "movieId": "movie_id"})[
        ["user_id", "movie_id", "tag", "timestamp"]]
    # 真实 tags.csv 的 tag 列含 NaN（用户提交过空标签）——pandas3 string dtype
    # 下 astype(str) 不转化 NaN，下游 sorted/set 的 float/str 混型会炸
    # （06 gen_quality 真实数据实测触发）。空标签不是标签，加载层统一剔除，
    # 对 03 分析 / 06 审计 / 08 评估三个消费者一致。
    return df[df["tag"].notna()].reset_index(drop=True)
