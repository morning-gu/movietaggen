"""MovieLens 适配器：格式解析与规范化。"""
import pandas as pd

from movietaggen.data.movielens import (
    load_genome_scores, load_genome_tags, load_links, load_movies,
    load_user_tags, parse_title_year,
)


def test_parse_title_year():
    assert parse_title_year("Toy Story (1995)") == ("Toy Story", 1995)
    assert parse_title_year("No Year Here") == ("No Year Here", None)
    assert parse_title_year("City of God (2002)") == ("City of God", 2002)


def _write(tmp_path, name, rows, columns):
    p = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(p, index=False)
    return p


def test_load_genome_scores(tmp_path):
    p = _write(tmp_path, "genome-scores.csv",
               [(1, 5, 0.5), (1, 6, 0.9)], ["movieId", "tagId", "relevance"])
    df = load_genome_scores(p)
    assert list(df.columns) == ["movie_id", "tag_id", "relevance"]
    assert df["movie_id"].dtype.kind == "i"


def test_load_genome_tags(tmp_path):
    p = _write(tmp_path, "genome-tags.csv", [(5, "dark")], ["tagId", "tag"])
    assert list(load_genome_tags(p).columns) == ["tag_id", "tag"]


def test_load_movies_genres_list(tmp_path):
    p = _write(tmp_path, "movies.csv",
               [(1, "Toy Story (1995)", "Animation|Children"),
                (2, "Unknown (2001)", "(no genres listed)")],
               ["movieId", "title", "genres"])
    df = load_movies(p)
    assert df.loc[0, "genres"] == ["Animation", "Children"]
    assert df.loc[1, "genres"] == []          # (no genres listed) → 空列表
    assert df.loc[0, "year"] == 1995
    assert df.loc[0, "title"] == "Toy Story"


def test_load_links_and_user_tags(tmp_path):
    links = _write(tmp_path, "links.csv", [(1, 114709, 862)], ["movieId", "imdbId", "tmdbId"])
    assert list(load_links(links).columns) == ["movie_id", "imdb_id"]
    tags = _write(tmp_path, "tags.csv",
                  [(9, 1, "classic", 1700000000)], ["userId", "movieId", "tag", "timestamp"])
    df = load_user_tags(tags)
    assert list(df.columns) == ["user_id", "movie_id", "tag", "timestamp"]


def test_raw_dir_override(tmp_path, monkeypatch):
    """raw_dir 覆盖端到端：config YAML → paths_for 落在 fake raw 根下 → 适配器能读出数据。

    test_config.py 的同名用例只断言路径拼接；本用例（Task 5 控制器决议）补真
    文件与加载器消费，验证 paths_for real 分支经过公开 config 面后与适配器咬合。
    """
    import yaml

    from movietaggen.config import load_config, paths_for
    from movietaggen.data.movienet import load_movienet

    raw = tmp_path / "fake-raw"
    ml = raw / "movielens" / "ml-latest"   # paths_for real 分支约定 ml-latest 子目录
    mn = raw / "movienet"
    ml.mkdir(parents=True)
    mn.mkdir(parents=True)
    pd.DataFrame([(1, "Fake Movie (2020)", "Drama")],
                 columns=["movieId", "title", "genres"]).to_csv(ml / "movies.csv", index=False)
    pd.DataFrame([(1, 111)], columns=["movieId", "imdbId"]).to_csv(
        ml / "links.csv", index=False)
    pd.DataFrame([(1, 5, 0.5)], columns=["movieId", "tagId", "relevance"]).to_csv(
        ml / "genome-scores.csv", index=False)
    pd.DataFrame([(5, "dark")], columns=["tagId", "tag"]).to_csv(
        ml / "genome-tags.csv", index=False)
    pd.DataFrame([(9, 1, "classic", 1700000000)],
                 columns=["userId", "movieId", "tag", "timestamp"]).to_csv(
        ml / "tags.csv", index=False)
    pd.DataFrame([("mnX1", "Fake Movie", 2020, "tt0000111")],
                 columns=["movie_id", "title", "year", "imdb_id"]).to_csv(
        mn / "movies.csv", index=False)

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(yaml.safe_dump(
        {"exp_name": "t", "data": {"source": "real", "raw_dir": str(raw)}},
        allow_unicode=True), encoding="utf-8")
    monkeypatch.chdir(tmp_path)   # 绝对 raw_dir 的解析不依赖 cwd
    p = paths_for(load_config(cfg_yaml))
    assert p.movienet == mn
    assert p.movielens == ml
    # 加载器经由解析出的路径读到 fake 数据（config → paths → 适配器 全链咬合）
    movies = load_movies(p.movielens / "movies.csv")
    assert list(movies["title"]) == ["Fake Movie"]
    assert movies.loc[0, "year"] == 2020
    mn_movies = load_movienet(p.movienet)
    assert [m.movienet_id for m in mn_movies] == ["mnX1"]
