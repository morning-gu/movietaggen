"""genre-only 基线：genre → tag 均值映射（训练参考集统计），未见 genre 组合回退全局均值。"""
from __future__ import annotations

import pandas as pd

from movietaggen.eval.protocol import assert_no_leakage


def genre_map(train_scores: pd.DataFrame, dataset: pd.DataFrame,
              movie_ids: list[int], test_ids: set[int] | None = None) -> pd.DataFrame:
    if test_ids is not None:
        assert_no_leakage(set(train_scores["movie_id"]), test_ids)
    train_meta = dataset[dataset["movie_id"].isin(set(train_scores["movie_id"]))]
    merged = train_scores.merge(train_meta[["movie_id", "genres"]], on="movie_id")
    # 每 genre 的 tag 均值
    genre_rows = []
    for _, r in merged.iterrows():
        for g in r["genres"]:
            genre_rows.append({"genre": g, "tag_id": r["tag_id"], "relevance": r["relevance"]})
    per_genre = pd.DataFrame(genre_rows).groupby(["genre", "tag_id"])["relevance"].mean()
    global_mean = train_scores.groupby("tag_id")["relevance"].mean()
    all_tags = list(global_mean.index)

    meta = dataset.set_index("movie_id")
    rows = []
    for m in movie_ids:
        genres = meta.loc[m, "genres"] if m in meta.index else []
        for t in all_tags:
            vals = [per_genre.get((g, t)) for g in genres]
            vals = [v for v in vals if v is not None]
            score = float(sum(vals) / len(vals)) if vals else float(global_mean[t])
            rows.append({"movie_id": m, "tag_id": t, "score": score, "method": "genre_map"})
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
