"""直接评分基线：prompt 结构镜像 A、few-shot 防泄露、mock 端到端（稠密 + 相关性）、
缓存往返、tag_groups.json 真实产物迁移。

Review Focus 5（计划头）的测试钉在本文件：few-shot 只能取自测试集之外的电影，
人为把测试电影塞进参考池 → assert_no_leakage 必须触发（test_few_shot_leakage_guard）。
"""
from pathlib import Path

import pandas as pd
import pytest

from movietaggen.baselines.direct_score import (
    few_shot_movie_ids,
    run_direct_score,
    select_few_shots,
)
from movietaggen.baselines.mock_scoring import MockScoringClient
from movietaggen.baselines.scoring_prompts import (
    MOVIE_TITLE_LINE,
    SYSTEM_PROMPT,
    TAGS_BLOCK_START,
    TagDef,
    build_messages,
)
from movietaggen.config import Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.grouping import load_tag_groups
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.eval.protocol import assert_no_leakage, temporal_split
from movietaggen.generate.content import to_multimodal

REPO_ROOT = Path(__file__).resolve().parent.parent


def _paths_for(tmp_path) -> Paths:
    """指向 tmp_path 的 Paths（同 tests/test_pipeline.py 的 _paths_for 模式）。"""
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache_llm=tmp_path / ".cache" / "llm",
        cache_embed=tmp_path / ".cache" / "embed",
        results=tmp_path / "results" / "smoke",
        stage_gen=tmp_path / "results" / "smoke" / "stage-gen",
        stage_align=tmp_path / "results" / "smoke" / "stage-align",
        stage_audit=tmp_path / "results" / "smoke" / "stage-audit",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet",
        baselines_dir=tmp_path / "results" / "baselines",
        eval_test=tmp_path / "processed" / "eval_test.txt",
        direct_score_a=tmp_path / "processed" / "direct_score_a.parquet")


@pytest.fixture()
def fake_setup(tmp_path):
    """临时目录内搭建完整合成数据集（封闭测试，不碰仓库全局 data/synthetic）。"""
    make_synthetic_dataset(tmp_path)
    paths = _paths_for(tmp_path)
    build_dataset(paths)                      # 产出 dataset/genome_scores/tag_groups
    cfg = load_config("configs/smoke.yaml")   # mock=true, temporal_split_year=2019
    yield cfg, paths


def _group_tag_defs(tag_groups: pd.DataFrame, group: str) -> list[TagDef]:
    return [TagDef(int(r.tag_id), r.tag, r.definition)
            for r in tag_groups[tag_groups["group"] == group].itertuples()]


def test_prompt_mirrors_a_structure(fake_setup):
    """build_messages 产物含标题行、指令、few-shot 块、TAGS_BLOCK_START 与
    组内每条 `- {tag}：{definition}` 定义行（与 A 的 prompt 契约逐项对齐）。"""
    cfg, paths = fake_setup
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)

    movie_rows = {int(r["movie_id"]): r for _, r in dataset.iterrows()}
    mm = to_multimodal(movie_rows[1], paths.movienet)
    group = sorted(tag_groups["group"].unique())[0]
    tag_defs = _group_tag_defs(tag_groups, group)

    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    train_ids, _ = temporal_split(dataset, cfg.protocol.temporal_split_year)
    few_shots = select_few_shots(dataset, genome_scores, cfg.baselines.few_shot_examples,
                                 sorted(train_ids), cfg.protocol.top_k, tag_names)
    assert few_shots                           # 合成训练侧非空，few-shot 块应出现

    messages = build_messages(mm, tag_defs, ["video", "script", "dialogue"],
                              few_shots, cfg.model)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    text = messages[1]["content"][0]["text"]
    assert MOVIE_TITLE_LINE.format(title=mm.title, year=mm.year) in text
    assert "## 示例（已知关联度的电影）" in text
    for fs in few_shots:
        assert f"《{fs.title}》（{fs.year}）" in text
    assert TAGS_BLOCK_START in text
    for td in tag_defs:
        assert f"- {td.tag}：{td.definition}" in text


def test_few_shot_leakage_guard(fake_setup):
    """Review Focus 5：few-shot 只能取自测试集之外——干净池选取后
    assert_no_leakage 通过；人为把测试电影塞进参考池 → 断言触发。"""
    cfg, paths = fake_setup
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    train_ids, test_ids = temporal_split(dataset, cfg.protocol.temporal_split_year)

    shots = select_few_shots(dataset, genome_scores, cfg.baselines.few_shot_examples,
                             sorted(train_ids), cfg.protocol.top_k, tag_names)
    ids = few_shot_movie_ids(dataset, shots)
    assert ids == {1}                          # 合成数据 2018 年仅 movie 1（训练侧）
    assert_no_leakage(ids, set(test_ids))      # 干净池 → 防泄露断言通过

    # 人为把测试电影塞进参考池：其 relevance 打满 1.0 → top-k 分数和必居首 →
    # 必被选为 few-shot → few-shot 电影与测试集重叠 → 断言必须触发
    test_movie_id = int(test_ids[0])
    boosted = genome_scores.copy()
    boosted.loc[boosted["movie_id"] == test_movie_id, "relevance"] = 1.0
    dirty_shots = select_few_shots(dataset, boosted, cfg.baselines.few_shot_examples,
                                   sorted(train_ids) + [test_movie_id],
                                   cfg.protocol.top_k, tag_names)
    dirty_ids = few_shot_movie_ids(dataset, dirty_shots)
    assert test_movie_id in dirty_ids          # 注入的测试电影确实被选中
    with pytest.raises(ValueError, match="泄露|重叠"):
        assert_no_leakage(dirty_ids, set(test_ids))


def test_direct_score_end_to_end_mock(fake_setup):
    """合成数据 + MockScoringClient：稠密 (movie, tag) 全覆盖、mock 分数与
    ground truth 相关性 > 0.3（base_score 机制的非退化检查）。"""
    cfg, paths = fake_setup
    df = run_direct_score(cfg, paths, MockScoringClient(), cache=None)

    assert list(df.columns) == ["movie_id", "tag_id", "score", "method"]
    assert (df["method"] == "direct_score").all()

    dataset = pd.read_parquet(paths.dataset)
    tag_groups = load_tag_groups(paths.tag_groups)
    expected_pairs = {(int(m), int(t)) for m in dataset["movie_id"]
                      for t in tag_groups["tag_id"]}
    got_pairs = set(zip(df["movie_id"].astype(int), df["tag_id"].astype(int)))
    assert got_pairs == expected_pairs         # 稠密：电影 × tag 全覆盖，无缺行
    assert len(df) == len(expected_pairs)
    assert df["score"].notna().all()           # mock 全部解析成功，无 NaN 行

    truth = pd.read_parquet(paths.genome_scores)
    merged = df.merge(truth, on=["movie_id", "tag_id"])
    assert len(merged) == len(df)              # 真值同样稠密，逐对可比
    assert merged["score"].corr(merged["relevance"]) > 0.3


def test_cache_roundtrip_all_hits(fake_setup):
    """缓存路径（cfg.mock=false 才读写缓存；client 仍注入 mock）：任务矩阵
    movie × group × run 全部落缓存，二跑全命中零调用且结果一致。"""
    from movietaggen.generate.cache import ResponseCache

    cfg, paths = fake_setup
    cfg.mock = False
    cache = ResponseCache(paths.cache_llm)
    df1 = run_direct_score(cfg, paths, MockScoringClient(), cache=cache)

    n_groups = load_tag_groups(paths.tag_groups)["group"].nunique()
    n_movies = len(pd.read_parquet(paths.dataset))
    assert len(list(paths.cache_llm.glob("*.json"))) == \
        n_movies * n_groups * cfg.baselines.n_runs

    class NoCallMock(MockScoringClient):
        def complete(self, messages, run_id):
            raise AssertionError("缓存全命中不应有新调用")

    df2 = run_direct_score(cfg, paths, NoCallMock(), cache=cache)
    pd.testing.assert_frame_equal(df2, df1)


def test_real_tag_groups_migrated():
    """A 的真实 1,128-tag 分组产物迁入本仓库 data/processed（source:real 的
    paths.tag_groups 指向它）。"""
    groups = load_tag_groups(REPO_ROOT / "data" / "processed" / "tag_groups.json")
    assert len(groups) == 1128
