"""对齐：IMDb 规范化、主键匹配、翻拍歧义、报告内容。"""
import pandas as pd
import pytest

from movietaggen.data.align import AlignResult, align, normalize_imdb


def test_normalize_imdb():
    assert normalize_imdb("tt0123456") == 123456
    assert normalize_imdb("123456") == 123456
    assert normalize_imdb(123456) == 123456
    assert normalize_imdb(123456.0) == 123456     # pandas 读 int 列可能 float 化
    assert normalize_imdb("tt0000001") == 1
    assert normalize_imdb("junk") is None
    assert normalize_imdb(None) is None


def _movies_ml():
    return pd.DataFrame([
        {"movie_id": 1, "title": "Alpha", "year": 2001, "genres": ["Drama"]},
        {"movie_id": 2, "title": "Beta", "year": 2002, "genres": ["Action"]},
        {"movie_id": 3, "title": "Gamma", "year": 2003, "genres": ["Crime"]},
    ])


def _links():
    return pd.DataFrame([{"movie_id": 1, "imdb_id": 111},
                         {"movie_id": 2, "imdb_id": 222}])  # 3 无 IMDb


def test_align_imdb_primary():
    from movietaggen.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnA", "Alpha", 2001, "tt0000111", [], None, None),
          MovieMultimodal("mnB", "Beta", 2002, "tt0000222", [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    df = result.aligned
    assert set(df["match_method"]) == {"imdb"}
    assert list(df["movie_id"]) == [1, 2]
    assert "IMDb 2" in result.report            # 报告含匹配统计


def test_align_title_fallback_for_missing_imdb():
    from movietaggen.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnC", "gamma ", 2003, None, [], None, None)]  # 大小写/空格差异
    result = align(_movies_ml(), _links(), mn)
    row = result.aligned[result.aligned["movie_id"] == 3]
    assert len(row) == 1
    assert row.iloc[0]["match_method"] == "title"


def test_ambiguous_remake_unmatched():
    """同名同年多部（翻拍撞年份）→ 不匹配并写进报告歧义清单。"""
    from movietaggen.data.movienet import MovieMultimodal
    mn = [MovieMultimodal("mnX", "Gamma", 2003, None, [], None, None),
          MovieMultimodal("mnY", "Gamma", 2003, None, [], None, None)]
    result = align(_movies_ml(), _links(), mn)
    assert result.aligned[result.aligned["movie_id"] == 3].empty
    assert "Gamma" in result.report


def _synthetic_paths(tmp_path):
    """为 tmp_path 构建指向合成数据的 Paths（Task 12 fixture 复用同一模式）。

    B 的 Paths 字段比 A 多（cache_llm/cache_embed/stage_*/baselines_dir），
    fixture 按同名布局补齐占位。
    """
    from movietaggen.config import Paths
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache_llm=tmp_path / ".cache" / "llm",
        cache_embed=tmp_path / ".cache" / "embed",
        results=tmp_path / "results" / "x",
        stage_gen=tmp_path / "results" / "x" / "stage-gen",
        stage_align=tmp_path / "results" / "x" / "stage-align",
        stage_audit=tmp_path / "results" / "x" / "stage-audit",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet",
        baselines_dir=tmp_path / "results" / "baselines")


def test_build_dataset_from_synthetic(tmp_path):
    from movietaggen.data.align import build_dataset
    from movietaggen.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    paths = _synthetic_paths(tmp_path)
    stats = build_dataset(paths)
    assert stats == {"movies": 5, "genome_rows": 180}   # B 样本 36 tags × 5 部（A 为 24 × 5）
    ds = pd.read_parquet(paths.dataset)
    assert len(ds) == 5
    assert ds.loc[0, "keyframes"][0].startswith("keyframes")   # 相对路径
    assert (paths.interim / "align_report.md").exists()


def test_build_dataset_zero_matches(tmp_path):
    """零匹配：build_dataset 写空产物 + 报告 0%，而非崩溃（对齐报告正是为诊断此场景而生）。"""
    from movietaggen.data.align import build_dataset
    from movietaggen.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    # 覆写 MovieNet 元数据：标题/年份/IMDb 均与合成 MovieLens 对不上
    pd.DataFrame([{"movie_id": "mnZ9", "title": "Unrelated Film",
                   "year": 1999, "imdb_id": "tt9999999"}]).to_csv(
        tmp_path / "raw" / "movienet" / "movies.csv", index=False)
    paths = _synthetic_paths(tmp_path)
    stats = build_dataset(paths)
    assert stats == {"movies": 0, "genome_rows": 0}
    assert len(pd.read_parquet(paths.aligned)) == 0
    assert len(pd.read_parquet(paths.dataset)) == 0
    assert len(pd.read_parquet(paths.genome_scores)) == 0
    report = (paths.interim / "align_report.md").read_text(encoding="utf-8")
    assert "0.0%" in report


def test_build_dataset_drops_movies_without_genome(tmp_path):
    """实验集的诚实定义：对齐且可评估——无 genome 覆盖的电影不进 dataset（省标注成本）。"""
    import pandas as pd

    from movietaggen.data.align import build_dataset
    from movietaggen.data.sample import make_synthetic_dataset

    make_synthetic_dataset(tmp_path)
    # 电影 5（Paper Skies）没有 genome 行——模拟 49 部无覆盖的真实场景
    scores_csv = tmp_path / "raw" / "movielens" / "genome-scores.csv"
    df = pd.read_csv(scores_csv)
    df = df[df["movieId"] != 5]
    df.to_csv(scores_csv, index=False)

    paths = _synthetic_paths(tmp_path)
    stats = build_dataset(paths)
    ds = pd.read_parquet(paths.dataset)
    assert stats["movies"] == 4                       # 无覆盖电影被剔除
    assert 5 not in set(ds["movie_id"])
    assert stats["genome_rows"] == 4 * 36             # B 样本 36 tags/部（A 为 4 × 24）
