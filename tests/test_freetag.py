"""自由标签分布分析（B 特有）：过滤口径、casefold 归一、排序、年份分桶、字面重合。"""
import json

import pandas as pd
import pytest

from movietaggen.data.freetag import (
    analyze_freetags,
    build_report,
    freetag_stats,
    literal_overlap,
)


def _user_tags():
    """10 行手造用户标签：movie 99 在实验集外（测过滤），含大小写/错拼/重复。"""
    return pd.DataFrame([
        {"user_id": 1, "movie_id": 1, "tag": "neon visuals", "timestamp": 1},
        {"user_id": 1, "movie_id": 1, "tag": "neon visuals", "timestamp": 2},
        {"user_id": 2, "movie_id": 1, "tag": "Neon Visuals", "timestamp": 3},
        {"user_id": 2, "movie_id": 2, "tag": "dark palette", "timestamp": 4},
        {"user_id": 2, "movie_id": 2, "tag": "dark palettes", "timestamp": 5},
        {"user_id": 3, "movie_id": 2, "tag": "neon visualz", "timestamp": 6},
        {"user_id": 3, "movie_id": 3, "tag": "slow burn", "timestamp": 7},
        {"user_id": 3, "movie_id": 99, "tag": "orphan tag", "timestamp": 8},
        {"user_id": 1, "movie_id": 3, "tag": "NEON VISUALS", "timestamp": 9},
        {"user_id": 1, "movie_id": 2, "tag": "dark palette", "timestamp": 10},
    ])


def _dataset():
    """3 部电影：1975 / 2005 / 空年份 → 覆盖 <1980、2000-2009、unknown 三桶。"""
    return pd.DataFrame([
        {"movie_id": 1, "title": "A", "year": 1975},
        {"movie_id": 2, "title": "B", "year": 2005},
        {"movie_id": 3, "title": "C", "year": None},
    ])


def _genome_tags():
    return pd.DataFrame([
        {"tag_id": 1, "tag": "Neon Visuals"},   # 大小写不同，casefold 后一致
        {"tag_id": 2, "tag": "unrelated tag"},
    ])


# ---------- freetag_stats ----------


def test_freetag_stats_filters_to_dataset_and_counts():
    """先过滤到实验集电影集合，再统计：n_unique / n_rows / 平均标签数。"""
    stats = freetag_stats(_user_tags(), _dataset())
    assert stats["n_unique"] == 5   # neon visuals / dark palette / dark palettes / neon visualz / slow burn
    assert stats["n_rows"] == 9     # 实验集外 movie 99 的 1 行被过滤
    assert stats["tags_per_movie_mean"] == pytest.approx(3.0)   # (3+4+2)/3


def test_freetag_stats_top_tags_order_and_truncation():
    """行数降序、同频按标签升序；大小写变体并入同一计数；top_n 截断。"""
    top = freetag_stats(_user_tags(), _dataset())["top_tags"]
    assert top[0] == ["neon visuals", 4]    # "Neon Visuals" / "NEON VISUALS" 并入
    assert top[1] == ["dark palette", 2]
    assert [t for t, _ in top[2:]] == ["dark palettes", "neon visualz", "slow burn"]
    assert freetag_stats(_user_tags(), _dataset(), top_n=2)["top_tags"] == [
        ["neon visuals", 4], ["dark palette", 2]]


def test_freetag_stats_year_buckets():
    """按 dataset.year 分桶计标签行数；空年份 → unknown；全桶键稳定输出。"""
    buckets = freetag_stats(_user_tags(), _dataset())["year_buckets"]
    assert buckets == {"<1980": 3, "1980-1999": 0, "2000-2009": 4,
                       "2010-2019": 0, ">=2020": 0, "unknown": 2}


def test_year_bucket_boundaries():
    """桶边界逐一检验：1979→<1980，1980/1999→1980-1999，…，2020→>=2020，空→unknown。"""
    years = [1979, 1980, 1999, 2000, 2009, 2010, 2019, 2020, None]
    dataset = pd.DataFrame({"movie_id": range(1, 10), "year": years})
    tags = pd.DataFrame({"user_id": 1, "movie_id": range(1, 10),
                         "tag": [f"t{i}" for i in range(1, 10)], "timestamp": 0})
    stats = freetag_stats(tags, dataset)
    assert stats["year_buckets"] == {"<1980": 1, "1980-1999": 2, "2000-2009": 2,
                                     "2010-2019": 2, ">=2020": 1, "unknown": 1}


def test_freetag_stats_mean_excludes_untagged_movies():
    """tags_per_movie_mean 分母 = 被标注过的电影（未被标注的电影不拉低均值）。"""
    dataset = pd.DataFrame([{"movie_id": 1, "year": 2010},
                            {"movie_id": 2, "year": 2011}])   # movie 2 无任何用户标签
    tags = pd.DataFrame([{"user_id": 1, "movie_id": 1, "tag": "a", "timestamp": 1},
                         {"user_id": 1, "movie_id": 1, "tag": "b", "timestamp": 2}])
    stats = freetag_stats(tags, dataset)
    assert stats["n_rows"] == 2
    assert stats["tags_per_movie_mean"] == 2.0


def test_freetag_stats_empty_user_tags():
    """空输入不崩溃：全零统计（真实管道零匹配数据集的同型场景）。"""
    stats = freetag_stats(_user_tags().iloc[0:0], _dataset())
    assert stats["n_unique"] == 0
    assert stats["n_rows"] == 0
    assert stats["tags_per_movie_mean"] == 0.0
    assert stats["top_tags"] == []
    assert set(stats["year_buckets"].values()) == {0}


# ---------- literal_overlap ----------


def test_literal_overlap_exact_match_and_typo():
    """'neon visuals' 计入字面重合；错拼 'neon visualz' 不计入（brief 关键样例）。

    注意 literal_overlap 的口径：直接对传入的 user_tags 求高频（控制器
    决议无 dataset 参数，不按实验集过滤），10 行含实验集外的 orphan tag，
    故唯一标签数为 6 而非实验集内的 5。
    """
    overlap = literal_overlap(_user_tags(), _genome_tags())
    assert overlap["top_n_total"] == 6
    assert overlap["exact_match"] == 1
    assert overlap["match_list"] == [["neon visuals", "Neon Visuals"]]
    assert all(t != "neon visualz" for t, _ in overlap["match_list"])


def test_literal_overlap_top_n_truncation_and_empty_and_no_match():
    """top_n 截断只比前 top_n 个高频标签；空输入与零重合均返回空清单。"""
    overlap = literal_overlap(_user_tags(), _genome_tags(), top_n=2)
    assert overlap["top_n_total"] == 2      # neon visuals ✓ / dark palette ✗
    assert overlap["exact_match"] == 1
    assert overlap["match_list"] == [["neon visuals", "Neon Visuals"]]
    empty = literal_overlap(_user_tags().iloc[0:0], _genome_tags())
    assert empty == {"top_n_total": 0, "exact_match": 0, "match_list": []}
    nomatch = literal_overlap(_user_tags(), pd.DataFrame([{"tag_id": 1, "tag": "zzz"}]))
    assert nomatch == {"top_n_total": 6, "exact_match": 0, "match_list": []}


# ---------- build_report ----------


def test_build_report_renders_summary():
    """中文 markdown 摘要：概况、Top-10 表、分桶表、重合摘要（含口径注记）。"""
    stats = freetag_stats(_user_tags(), _dataset())
    report = build_report(stats, literal_overlap(_user_tags(), _genome_tags()),
                          movie_count=3)
    assert report.startswith("# 自由标签分布分析报告")
    assert "| 1 | neon visuals | 4 |" in report
    assert "| unknown | 2 |" in report
    assert "1（16.7%）" in report
    assert "未按实验集过滤" in report    # 重合口径与概况节的"实验集内"显式区分


# ---------- analyze_freetags（脚本内核） ----------


def _paths(tmp_path):
    """指向 tmp 的 Paths（与 test_align._synthetic_paths 同一布局模式）。"""
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


def test_analyze_freetags_writes_outputs(tmp_path):
    """端到端（合成数据 → 02 建库 → 03 分析）：JSON 与 markdown 报告落盘。"""
    from movietaggen.data.align import build_dataset
    from movietaggen.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    paths = _paths(tmp_path)
    build_dataset(paths)
    result = analyze_freetags(paths)
    assert result["freetag_stats"]["n_rows"] >= 20      # 5 部 × 4-8 标签
    payload = json.loads((paths.interim / "freetag_stats.json").read_text(encoding="utf-8"))
    assert set(payload) == {"freetag_stats", "literal_overlap"}
    assert payload["freetag_stats"] == result["freetag_stats"]
    report = (paths.interim / "freetag_report.md").read_text(encoding="utf-8")
    assert "# 自由标签分布分析报告" in report
    assert "## 高频标签 Top 10" in report


def test_analyze_freetags_skips_without_tags_csv(tmp_path):
    """tags.csv 缺失 → 优雅跳过（返回 None，不写产物）。"""
    from movietaggen.data.align import build_dataset
    from movietaggen.data.sample import make_synthetic_dataset
    make_synthetic_dataset(tmp_path)
    (tmp_path / "raw" / "movielens" / "tags.csv").unlink()
    paths = _paths(tmp_path)
    build_dataset(paths)
    assert analyze_freetags(paths) is None
    assert not (paths.interim / "freetag_stats.json").exists()


def test_analyze_freetags_requires_dataset(tmp_path):
    """dataset parquet 缺失（未跑 02）→ FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        analyze_freetags(_paths(tmp_path))
