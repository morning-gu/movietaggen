"""直接评分基线：prompt 结构镜像 A、few-shot 防泄露、mock 端到端（稠密 + 相关性）、
缓存往返、tag_groups.json 真实产物迁移。

Review Focus 5（计划头）的测试钉在本文件：few-shot 只能取自测试集之外的电影，
人为把测试电影塞进参考池 → assert_no_leakage 必须触发（test_few_shot_leakage_guard）。

440 协议接线（440 计划 Task 6）：
- a_borrow 加载器（test_load_direct_score_a_*）：逐对均值口径（缺轮不计入、
  全空 → NaN）、迁移物缺失/电影越界/modality_set 混值大声失败；
- own_run 改接（test_own_run_*）：few-shot 池走 fixed_split 训练侧、movie_limit
  走 sample_limited_movie_ids——测试清单电影绝不入池/入样。
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from movietaggen.baselines.direct_score import (
    few_shot_movie_ids,
    load_direct_score_a,
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


# ---------------------------------------------------------------------------
# 440 计划 Task 6：a_borrow 加载器（load_direct_score_a）与 own_run 改接
# ---------------------------------------------------------------------------

_A_RUN_IDS = ["run-0", "run-1", "run-2"]


def _a_borrow_frame(movie_ids=(101, 102),
                    modality="video|script|dialogue") -> pd.DataFrame:
    """构造小而快的 a_borrow fixture 帧（2 电影 × 3 tag × 3 run）。

    刻意不读仓库真实迁移物（单测要小而快）。口径探针：(101, 22) 仅
    run-0/run-1 有分（0.2/0.6，run-2 缺）→ 均值应为两轮均值 0.4；
    (101, 33) 三轮全 NaN → 均值应保持 NaN；其余 (movie, tag) 各轮同分
    （查表定值，缺省 0.4）。
    """
    fixed = {(101, 11): 0.5, (102, 11): 0.1, (102, 22): 0.7, (102, 33): 0.9}
    rows = []
    for m in movie_ids:
        for t in (11, 22, 33):
            for r, run_id in enumerate(_A_RUN_IDS):
                if (m, t) == (101, 22):
                    score = {0: 0.2, 1: 0.6}.get(r)   # run-2 缺 → NaN
                elif (m, t) == (101, 33):
                    score = None                      # 全空 → NaN
                else:
                    score = fixed.get((m, t), 0.4)
                rows.append({"movie_id": m, "tag_id": t, "group": "视觉风格",
                             "modality_set": modality, "run_id": run_id,
                             "score": score})
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "group",
                                       "modality_set", "run_id", "score"])


def test_load_direct_score_a_pair_mean_semantics(tmp_path):
    """均值口径逐对断言：仅 2 轮有分 → 两轮均值；全 NaN 轮 → NaN 保留；
    输出与 run_direct_score 同构（列序 + method="direct_score"）。"""
    paths = _paths_for(tmp_path)
    paths.direct_score_a.parent.mkdir(parents=True, exist_ok=True)
    _a_borrow_frame().to_parquet(paths.direct_score_a)
    dataset = pd.DataFrame({"movie_id": [101, 102, 103]})  # ⊇ parquet 电影集

    out = load_direct_score_a(paths, dataset)

    assert list(out.columns) == ["movie_id", "tag_id", "score", "method"]
    assert (out["method"] == "direct_score").all()
    assert len(out) == 6                         # 2 电影 × 3 tag 稠密无缺行
    got = {(int(m), int(t)): s for m, t, s
           in zip(out["movie_id"], out["tag_id"], out["score"])}
    assert got[(101, 22)] == pytest.approx(0.4)  # 两轮均值（缺轮不计入）
    assert pd.isna(got[(101, 33)])               # 全空 → NaN
    assert got[(101, 11)] == pytest.approx(0.5)  # 三轮同分 → 该分
    assert got[(102, 33)] == pytest.approx(0.9)


def test_load_direct_score_a_missing_file(tmp_path):
    """迁移物文件缺失 → ValueError 点名路径与成因（绕过 config 校验直接
    调用加载器的调用方也要大声失败）。"""
    paths = _paths_for(tmp_path)                 # direct_score_a 未落盘
    with pytest.raises(ValueError) as ei:
        load_direct_score_a(paths, pd.DataFrame({"movie_id": [101]}))
    assert str(paths.direct_score_a) in str(ei.value)
    assert "a_borrow" in str(ei.value)


def test_load_direct_score_a_movie_out_of_universe(tmp_path):
    """迁移物电影 ∉ dataset 电影 → ValueError 点名越界电影与清单/数据集
    版本错配提示（Review Focus 1）。"""
    paths = _paths_for(tmp_path)
    paths.direct_score_a.parent.mkdir(parents=True, exist_ok=True)
    _a_borrow_frame(movie_ids=(101, 999)).to_parquet(paths.direct_score_a)

    with pytest.raises(ValueError) as ei:
        load_direct_score_a(paths, pd.DataFrame({"movie_id": [101, 102]}))
    assert "999" in str(ei.value)
    assert "错配" in str(ei.value)


def test_load_direct_score_a_bad_modality_set(tmp_path):
    """modality_set 混入其它值 → ValueError 点名实际值集。"""
    paths = _paths_for(tmp_path)
    paths.direct_score_a.parent.mkdir(parents=True, exist_ok=True)
    df = _a_borrow_frame()
    df.loc[df["run_id"] == "run-0", "modality_set"] = "video|script"
    df.to_parquet(paths.direct_score_a)

    with pytest.raises(ValueError) as ei:
        load_direct_score_a(paths, pd.DataFrame({"movie_id": [101, 102]}))
    # 带引号断言以区别于合法值 video|script|dialogue 的子串
    assert "'video|script'" in str(ei.value)


def test_own_run_few_shot_excludes_fixed_test_movies(fake_setup):
    """own_run few-shot 池接 fixed_split：测试清单电影不得进入 few-shot 池。

    合成 eval_test 清单（Random(42) 采样）钉住 movie 1（Crimson Horizon）——
    时序分割下它恰在训练侧（旧接线会把它选进 few-shot）；固定分割后
    few-shot 池 = dataset − 清单，捕获到的 prompt few-shot 块必须不含它。
    """
    cfg, paths = fake_setup
    dataset = pd.read_parquet(paths.dataset)
    test_ids = [int(line) for line in
                paths.eval_test.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    title_of = dict(zip(dataset["movie_id"], dataset["title"]))
    test_titles = {title_of[m] for m in test_ids}
    assert test_titles == {"Crimson Horizon"}    # 清单钉住 movie 1（判别锚点）

    captured: list[list[dict]] = []

    class CapturingClient(MockScoringClient):
        def complete(self, messages, run_id):
            captured.append(messages)
            return super().complete(messages, run_id)

    run_direct_score(cfg, paths, CapturingClient(), cache=None)

    shot_titles: set[str] = set()
    for messages in captured:
        text = messages[1]["content"][0]["text"]
        for block in text.split("\n\n"):
            if block.startswith("## 示例（已知关联度的电影）"):
                shot_titles |= set(re.findall(r"《(.+?)》", block))
    assert shot_titles                           # few-shot 块确实出现（防空洞通过）
    assert not (shot_titles & test_titles)       # 测试电影绝不在 few-shot 池


def test_own_run_movie_limit_excludes_fixed_test_movies(fake_setup):
    """own_run movie_limit 抽样接 sample_limited_movie_ids：受限跑从固定分割
    训练侧抽样，测试清单电影绝不入样（旧接线从全集抽样，seed=42 会抽中
    movie 1，本测试据此判别新旧接线）。"""
    cfg, paths = fake_setup
    cfg.data.movie_limit = 3

    df = run_direct_score(cfg, paths, MockScoringClient(), cache=None)

    test_ids = {int(line) for line in
                paths.eval_test.read_text(encoding="utf-8").splitlines()
                if line.strip()}
    scored = set(df["movie_id"].astype(int))
    assert len(scored) == cfg.data.movie_limit   # 受限跑确实限量
    assert not (scored & test_ids)               # 测试电影不入样
