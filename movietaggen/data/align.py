"""MovieNet ↔ MovieLens 对齐：IMDb 主键优先，标题+年份 fallback。

产出 aligned 表（仅匹配成功行）与人类可读的对齐报告（匹配率、未匹配清单、歧义清单），
并从 raw 数据构建实验数据集（dataset / genome_scores / genome_tags parquet）。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from movietaggen.config import Paths
from movietaggen.data.movielens import (load_genome_scores, load_genome_tags,
                                        load_links, load_movies)
from movietaggen.data.movienet import MovieMultimodal, load_movienet


def normalize_imdb(value) -> int | None:
    """'tt0123456' / 123456 / 123456.0 → 123456；非法 → None。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower().removeprefix("tt")
    if s.endswith(".0"):  # pandas float 化
        s = s[:-2]
    s = s.lstrip("0") or "0"
    return int(s) if s.isdigit() else None


def _norm_title(title: str) -> str:
    """标题规范化：NFKC、小写、去标点、压缩空白。"""
    t = unicodedata.normalize("NFKC", str(title)).lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class AlignResult:
    aligned: pd.DataFrame
    report: str


ALIGNED_COLUMNS = ["movie_id", "title", "year", "genres", "imdb_id",
                   "movienet_id", "match_method"]
DATASET_COLUMNS = ["movie_id", "title", "year", "genres", "imdb_id",
                   "movienet_id", "keyframes", "script_path", "subtitles_path"]


def align(movies_ml: pd.DataFrame, links: pd.DataFrame,
          movienet_movies: list[MovieMultimodal]) -> AlignResult:
    imdb_to_mn: dict[int, MovieMultimodal] = {}
    for m in movienet_movies:
        norm = normalize_imdb(m.imdb_id)
        if norm is not None:
            imdb_to_mn[norm] = m

    link_map: dict[int, int] = {}  # movie_id(ML) -> imdb(int)
    for _, r in links.iterrows():
        norm = normalize_imdb(r["imdb_id"])
        if norm is not None:
            link_map[int(r["movie_id"])] = norm

    # 标题索引：norm_title|year -> [MovieMultimodal]
    title_idx: dict[tuple[str, int], list[MovieMultimodal]] = {}
    for m in movienet_movies:
        if m.year is not None:
            title_idx.setdefault((_norm_title(m.title), m.year), []).append(m)

    rows = []
    ambiguous, unmatched_ml = [], []
    imdb_hits = title_hits = 0
    for _, ml in movies_ml.iterrows():
        matched, method = None, None
        imdb = link_map.get(int(ml["movie_id"]))
        if imdb is not None and imdb in imdb_to_mn:
            matched, method = imdb_to_mn[imdb], "imdb"
        elif ml["year"] is not None and pd.notna(ml["year"]):
            cands = title_idx.get((_norm_title(ml["title"]), int(ml["year"])), [])
            if len(cands) == 1:
                matched, method = cands[0], "title"
            elif len(cands) > 1:
                ambiguous.append(f'{ml["title"]} ({int(ml["year"])})')
        if matched is None:
            unmatched_ml.append(f'{ml["movie_id"]}: {ml["title"]} ({ml["year"]})')
            continue
        if method == "imdb":
            imdb_hits += 1
        else:
            title_hits += 1
        rows.append({"movie_id": int(ml["movie_id"]), "title": ml["title"],
                     "year": int(ml["year"]) if pd.notna(ml["year"]) else None,
                     "genres": list(ml["genres"]), "imdb_id": imdb,
                     "movienet_id": matched.movienet_id, "match_method": method})

    n_total = len(movies_ml)
    n_matched = len(rows)
    matched_mn = {r["movienet_id"] for r in rows}
    unmatched_mn = [m.movienet_id for m in movienet_movies if m.movienet_id not in matched_mn]
    report = (
        f"# 对齐报告\n\n"
        f"- MovieLens 电影总数: {n_total}\n"
        f"- 匹配成功: {n_matched}（IMDb {imdb_hits} + 标题 {title_hits}）\n"
        f"- 匹配率: {n_matched / n_total:.1%}\n\n"
        f"## 歧义（同名同年多部，未匹配）\n"
        + ("\n".join(f"- {a}" for a in ambiguous) or "- 无") + "\n\n"
        f"## 未匹配的 MovieLens 电影（前 50）\n"
        + ("\n".join(f"- {u}" for u in unmatched_ml[:50]) or "- 无") + "\n\n"
        f"## 未匹配的 MovieNet 电影（前 50）\n"
        + ("\n".join(f"- {u}" for u in unmatched_mn[:50]) or "- 无") + "\n"
    )
    # 显式列名：零匹配时也要有完整 schema（空表可被下游按列访问）
    return AlignResult(aligned=pd.DataFrame(rows, columns=ALIGNED_COLUMNS), report=report)


def build_dataset(paths: Paths) -> dict[str, int]:
    """读 raw → 对齐 → 写 interim/processed，返回统计。02 脚本与测试 fixture 复用。"""
    movies_ml = load_movies(paths.movielens / "movies.csv")
    links = load_links(paths.movielens / "links.csv")
    mn_movies = load_movienet(paths.movienet)

    result = align(movies_ml, links, mn_movies)
    paths.interim.mkdir(parents=True, exist_ok=True)
    paths.processed.mkdir(parents=True, exist_ok=True)
    result.aligned.to_parquet(paths.aligned, index=False)
    (paths.interim / "align_report.md").write_text(result.report, encoding="utf-8")

    # 多模态文件引用（相对 movienet 根，便于仓库迁移）
    mn_by_id = {m.movienet_id: m for m in mn_movies}
    dataset_rows = []
    for _, r in result.aligned.iterrows():
        m = mn_by_id[r["movienet_id"]]
        dataset_rows.append({
            "movie_id": r["movie_id"], "title": r["title"], "year": r["year"],
            "genres": r["genres"], "imdb_id": r["imdb_id"],
            "movienet_id": r["movienet_id"],
            "keyframes": [str(p.relative_to(paths.movienet)) for p in m.keyframes],
            "script_path": f'scripts/{m.movienet_id}.txt' if m.script else None,
            "subtitles_path": f'subtitles/{m.movienet_id}.srt' if m.subtitles else None,
        })
    # 显式列名：零匹配时也要有完整 schema（空表可被下游按列访问，见 DATASET_COLUMNS）
    dataset = pd.DataFrame(dataset_rows, columns=DATASET_COLUMNS)

    scores = load_genome_scores(paths.movielens / "genome-scores.csv")
    # 实验集的诚实定义：对齐且可评估——无 genome 覆盖的电影不进 dataset
    # （真实数据 1,084 对齐中 49 部无覆盖，剔除省 ~4.5% 标注成本）
    covered = set(scores["movie_id"])
    dataset = dataset[dataset["movie_id"].isin(covered)]
    dataset.to_parquet(paths.dataset, index=False)

    scores = scores[scores["movie_id"].isin(set(dataset["movie_id"]))]
    scores.to_parquet(paths.genome_scores, index=False)
    load_genome_tags(paths.movielens / "genome-tags.csv").to_parquet(
        paths.genome_tags, index=False)
    return {"movies": len(dataset), "genome_rows": len(scores)}
