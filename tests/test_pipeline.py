"""生成管道：续跑跳过、截断 jsonl 容忍、跨轮并集去重、NaN 守卫、空模态、并行等价、缓存、dry-run。

Review Focus 1/2/4（计划头）的测试钉在本文件：
1. 中断残留的截断 jsonl 行 → 跳过并计数，不崩溃（test_truncated_jsonl_line_tolerated）；
2. modalities=[] 是 title-only 探针的显式意图，不判 no_content（test_empty_modalities_not_no_content）；
4. 混合 str/NaN 列经 parquet 往返后 NaN 为真值 → pd.notna 守卫（test_nan_mixed_column_guard）。
"""
import json
import random

import numpy as np
import pandas as pd
import pytest

from movietaggen.config import Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.eval.protocol import fixed_split
from movietaggen.generate.content import to_multimodal
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import (
    build_task_matrix,
    run_generation,
)
from movietaggen.generate.prompts import FewShotTags, normalize_tag


def _paths_for(tmp_path) -> Paths:
    """指向 tmp_path 的 Paths（同 tests/test_align.py 的 _synthetic_paths 模式）。"""
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
    cfg = load_config("configs/smoke.yaml")   # n_runs=2, workers=1, mock=true
    yield cfg, paths


def _counting_client():
    """包装 MockGenClient 计数 complete 调用数（transport 层调用计数）。"""
    class CountingMock(MockGenClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, messages, run_id):
            self.calls += 1
            return super().complete(messages, run_id)

    return CountingMock()


def _manifest(paths):
    return json.loads(
        (paths.stage_gen / "run_manifest.json").read_text(encoding="utf-8"))


def _extend_to_ten_movies(paths, test_ids):
    """数据集复制扩展到 10 部（id 6-10 复用前 5 部的模态文件）+ 重写 eval_test 清单。

    make_synthetic 只产 5 部，不够"10 部 + 清单 3 部 + limit 5"的训练池
    判别结构（440 协议 movie_limit 抽样限训练池）。build_dataset 不读清单，
    故 dataset 产出后再覆盖 eval_test.txt 即可。
    """
    dataset = pd.read_parquet(paths.dataset)
    extra = dataset.head(5).copy()
    extra["movie_id"] = [6, 7, 8, 9, 10]
    pd.concat([dataset, extra], ignore_index=True).to_parquet(
        paths.dataset, index=False)
    paths.eval_test.write_text(
        "".join(f"{m}\n" for m in sorted(test_ids)), encoding="utf-8")


def test_build_task_matrix():
    """任务矩阵无 group 维度：电影 × run-{r}。"""
    tasks = build_task_matrix([10, 11], n_runs=2)
    assert len(tasks) == 4
    assert {t.run_id for t in tasks} == {"run-0", "run-1"}
    assert all(t.movie_id in (10, 11) for t in tasks)


def test_run_generation_end_to_end(fake_setup):
    cfg, paths = fake_setup
    out = run_generation(cfg, paths, MockGenClient(), cache=None)
    assert out == paths.stage_gen / "tags.parquet"
    df = pd.read_parquet(out)
    assert set(df.columns) == {"movie_id", "tag", "n_runs"}
    assert set(df["movie_id"]) == {1, 2, 3, 4, 5}   # 合成数据 5 部全量
    assert len(df) > 0
    # responses.jsonl：每任务一行，字段齐全
    lines = (paths.stage_gen / "responses.jsonl").read_text(
        encoding="utf-8").splitlines()
    total = 5 * cfg.generate.n_runs
    assert len(lines) == total
    rec = json.loads(lines[0])
    assert set(rec) == {"task_key", "status", "tags", "usage"}
    assert rec["status"] == "done" and rec["tags"]
    manifest = _manifest(paths)
    assert manifest["tasks"] == {"done": total, "parse_failed": 0, "no_content": 0}
    assert manifest["total_calls"] == total
    assert manifest["n_tags_total"] == len(df)


def test_resume_skips_done(fake_setup):
    """预写一行 done 的 responses.jsonl → 该任务不再调用 client，历史标签进并集。"""
    cfg, paths = fake_setup
    paths.stage_gen.mkdir(parents=True, exist_ok=True)
    pre = {"task_key": "1|run-0", "status": "done",
           "tags": ["preexisting tag"], "usage": {}}
    (paths.stage_gen / "responses.jsonl").write_text(
        json.dumps(pre, ensure_ascii=False) + "\n", encoding="utf-8")

    client = _counting_client()
    out = run_generation(cfg, paths, client, cache=None)
    total = 5 * cfg.generate.n_runs
    assert client.calls == total - 1            # 预写任务被跳过，只补缺失
    df = pd.read_parquet(out)
    row = df[(df["movie_id"] == 1) & (df["tag"] == "preexisting tag")]
    assert len(row) == 1 and row.iloc[0]["n_runs"] == 1   # 历史行进并集、不丢不重


def test_truncated_jsonl_line_tolerated(fake_setup, capsys):
    """Review Focus 1：崩溃残留的截断 jsonl 行被跳过并计数，续跑不崩、结果完整。"""
    cfg, paths = fake_setup
    paths.stage_gen.mkdir(parents=True, exist_ok=True)
    jsonl = paths.stage_gen / "responses.jsonl"
    good = {"task_key": "1|run-0", "status": "done", "tags": ["good tag"], "usage": {}}
    jsonl.write_text(
        '{"task_key": "999|run-0", "status": "done", "tags": ["half' + "\n"
        + json.dumps(good, ensure_ascii=False) + "\n", encoding="utf-8")

    client = _counting_client()
    out = run_generation(cfg, paths, client, cache=None)
    total = 5 * cfg.generate.n_runs
    assert client.calls == total - 1            # 好行任务跳过；坏行任务不在矩阵内
    df = pd.read_parquet(out)
    assert "good tag" in set(df[df["movie_id"] == 1]["tag"])
    manifest = _manifest(paths)
    assert manifest["tasks"]["done"] == total - 1
    assert "跳过损坏的 jsonl 行: 1" in capsys.readouterr().out


def test_union_dedupe_across_runs(fake_setup):
    """mock 两轮：tags.parquet 的 (movie, 归一化 tag) 唯一，n_runs ∈ {1,2} 且两值都出现。"""
    cfg, paths = fake_setup
    out = run_generation(cfg, paths, MockGenClient(), cache=None)
    df = pd.read_parquet(out)
    keys = df.assign(norm=df["tag"].map(normalize_tag))
    assert keys.duplicated(["movie_id", "norm"]).sum() == 0
    assert set(df["n_runs"]) == {1, 2}          # 轮次噪声 → 既有双轮标签也有单轮变体
    assert df["n_runs"].between(1, cfg.generate.n_runs).all()


def test_nan_mixed_column_guard(fake_setup):
    """Review Focus 4：混合 str/NaN script_path 经 parquet 往返后（pandas3 缺失为
    NaN 且为真值）不得把 NaN 当路径——无剧本电影凭其余模态正常产出。"""
    cfg, paths = fake_setup
    dataset = pd.read_parquet(paths.dataset)
    # 造混合列：仅第 0、2 行保留剧本路径，其余置 NaN（模拟真实 479/1035 有剧本）
    dataset["script_path"] = [dataset.loc[i, "script_path"] if i in (0, 2) else np.nan
                              for i in range(len(dataset))]
    dataset.to_parquet(paths.dataset, index=False)

    # 单元面：to_multimodal（content.py 共享版，M9a 消重）对 NaN 行 script=None（A 的 pd.notna 守卫）
    mm = to_multimodal(dataset.iloc[1], paths.movienet)
    assert mm.script is None

    client = _counting_client()
    out = run_generation(cfg, paths, client, cache=None)
    total = 5 * cfg.generate.n_runs
    manifest = _manifest(paths)
    assert manifest["tasks"]["done"] == total   # 无一 no_content：其余模态可用
    assert manifest["tasks"]["no_content"] == 0
    assert len(pd.read_parquet(out)) > 0


def test_empty_modalities_not_no_content(fake_setup):
    """Review Focus 2：modalities=[] 是 title-only 探针的显式意图，不判 no_content。"""
    cfg, paths = fake_setup
    cfg.generate.modalities = []
    out = run_generation(cfg, paths, MockGenClient(), cache=None)
    total = 5 * cfg.generate.n_runs
    manifest = _manifest(paths)
    assert manifest["tasks"]["no_content"] == 0
    assert manifest["tasks"]["done"] == total
    df = pd.read_parquet(out)
    assert "crimson horizon movie" in set(df[df["movie_id"] == 1]["tag"])  # 标题派生标签


def test_no_content_when_requested_modality_unavailable(fake_setup):
    """A 修订后语义：请求了内容模态但数据全缺 → no_content 且不调用 client。"""
    cfg, paths = fake_setup
    cfg.generate.modalities = ["script"]
    dataset = pd.read_parquet(paths.dataset)
    dataset["script_path"] = None               # 全部电影无剧本
    dataset.to_parquet(paths.dataset, index=False)

    client = _counting_client()
    out = run_generation(cfg, paths, client, cache=None)
    assert client.calls == 0                    # no_content 任务不调用 MLLM
    total = 5 * cfg.generate.n_runs
    manifest = _manifest(paths)
    assert manifest["tasks"]["no_content"] == total
    assert len(pd.read_parquet(out)) == 0       # 空并集 → 空 parquet（schema 完整）


def test_parallel_equivalent_to_serial(fake_setup):
    """workers=2 与串行等价：mock 确定性 → 并集（按归一化键）与 manifest 一致。"""
    cfg, paths = fake_setup
    out_seq = run_generation(cfg, paths, MockGenClient(), cache=None)
    df_seq = pd.read_parquet(out_seq)
    manifest_seq = _manifest(paths)

    cfg.generate.workers = 2
    paths.results = paths.results.with_name("smoke-par")
    paths.stage_gen = paths.results / "stage-gen"
    out_par = run_generation(cfg, paths, MockGenClient(), cache=None)
    df_par = pd.read_parquet(out_par)

    s = df_seq.assign(norm=df_seq["tag"].map(normalize_tag))
    p = df_par.assign(norm=df_par["tag"].map(normalize_tag))
    merged = s.merge(p, on=["movie_id", "norm"], suffixes=("_s", "_p"))
    assert len(merged) == len(s) == len(p)      # 并集逐键一致
    assert (merged["n_runs_s"] == merged["n_runs_p"]).all()
    assert _manifest(paths)["tasks"] == manifest_seq["tasks"]


def test_cache_roundtrip_all_hits(fake_setup):
    """真实缓存路径（mock:false 才读写缓存；client 仍注入 mock）：二跑全部命中零调用。"""
    from movietaggen.generate.cache import ResponseCache

    cfg, paths = fake_setup
    cfg.mock = False                        # 缓存门控只看 config.mock；client 可注入 mock
    cache = ResponseCache(paths.cache_llm)
    out = run_generation(cfg, paths, MockGenClient(), cache=cache)
    n_first = len(pd.read_parquet(out))
    assert len(list(paths.cache_llm.glob("*.json"))) == 5 * cfg.generate.n_runs

    paths.results = paths.results.with_name("smoke-cache2")
    paths.stage_gen = paths.results / "stage-gen"

    class NoCallMock(MockGenClient):
        def complete(self, messages, run_id):
            raise AssertionError("缓存全命中不应有新调用")

    out2 = run_generation(cfg, paths, NoCallMock(), cache=cache)
    df2 = pd.read_parquet(out2)
    assert len(df2) == n_first                # 信封还原 content → 并集一致


def test_movie_limit_sampled_deterministically(fake_setup):
    """movie_limit 抽样：同 seed 同样本；池 = 固定分割训练侧（440 协议，
    测试清单电影不入样）。"""
    cfg, paths = fake_setup
    cfg.data.movie_limit = 3
    run_generation(cfg, paths, MockGenClient(), cache=None)
    df = pd.read_parquet(paths.stage_gen / "tags.parquet")
    assert len(set(df["movie_id"])) == 3
    train_ids, test_ids = fixed_split(pd.read_parquet(paths.dataset), paths.eval_test)
    expected = set(random.Random(cfg.data.seed).sample(
        sorted(train_ids.tolist()), 3))
    assert set(df["movie_id"]) == expected
    assert not (set(df["movie_id"]) & set(test_ids.tolist()))


def test_movie_limit_samples_from_train_pool_only(fake_setup):
    """440 协议：movie_limit 抽样限训练池——测试清单电影不入任务矩阵。

    10 部数据集 + 清单 3 部 + limit 5：旧接线从全集抽样（seed=42 会抽中
    1/2），新接线生成电影集 ⊆ train 7 部，本测试据此判别新旧接线。
    """
    cfg, paths = fake_setup
    test_ids = {1, 2, 3}
    _extend_to_ten_movies(paths, test_ids)
    cfg.data.movie_limit = 5

    run_generation(cfg, paths, MockGenClient(), cache=None)
    df = pd.read_parquet(paths.stage_gen / "tags.parquet")

    generated = set(df["movie_id"])
    assert len(generated) == 5                    # 受限跑确实限量
    assert generated <= {4, 5, 6, 7, 8, 9, 10}    # ⊆ train 7 部
    assert not (generated & test_ids)             # 测试电影不入池


def test_movie_limit_train_pool_same_seed_deterministic(fake_setup):
    """同 seed 两次跑：抽样限训练池后确定性保持（两次生成的电影集相同）。"""
    cfg, paths = fake_setup
    _extend_to_ten_movies(paths, {1, 2, 3})
    cfg.data.movie_limit = 5

    run_generation(cfg, paths, MockGenClient(), cache=None)
    ids_first = set(pd.read_parquet(paths.stage_gen / "tags.parquet")["movie_id"])

    paths.results = paths.results.with_name("smoke-limit2")
    paths.stage_gen = paths.results / "stage-gen"
    run_generation(cfg, paths, MockGenClient(), cache=None)
    ids_second = set(pd.read_parquet(paths.stage_gen / "tags.parquet")["movie_id"])

    assert ids_first == ids_second
    assert len(ids_first) == 5


def test_movie_limit_none_full_dataset(fake_setup):
    """movie_limit=None → 全量 10 部（不读清单，语义与协议单测一致）。"""
    cfg, paths = fake_setup
    _extend_to_ten_movies(paths, {1, 2, 3})
    cfg.data.movie_limit = None

    out = run_generation(cfg, paths, MockGenClient(), cache=None)
    assert set(pd.read_parquet(out)["movie_id"]) == set(range(1, 11))


def test_dry_run_prints_prompts_without_calls(fake_setup, capsys):
    """--dry-run：打印前 3 个任务的 prompt、不调用客户端、不写文件。"""
    cfg, paths = fake_setup

    class NoCallMock(MockGenClient):
        def complete(self, messages, run_id):
            raise AssertionError("dry-run 不应调用 MLLM")

    out = run_generation(cfg, paths, NoCallMock(), cache=None, dry_run=True)
    assert out is None
    assert not paths.stage_gen.exists()       # 不写文件
    out_text = capsys.readouterr().out
    assert out_text.count("[dry-run]") == 3
    assert "电影：《" in out_text


def test_few_shots_plumbed_to_prompt(fake_setup, capsys):
    """few_shots 参数贯通到 prompt（当前默认 []；后续任务接线时管道不用改）。"""
    cfg, paths = fake_setup
    fs = [FewShotTags(title="Old Film", year=2001, tags=["noir", "rain-soaked"])]
    run_generation(cfg, paths, MockGenClient(), cache=None, dry_run=True, few_shots=fs)
    assert "《Old Film》（2001）：noir, rain-soaked" in capsys.readouterr().out
