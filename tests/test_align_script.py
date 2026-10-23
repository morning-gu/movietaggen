"""阶段二对齐脚本（05）：嵌入 → 映射 → 聚合 → 训练分割校准 的端到端接线测试。

覆盖面（brief 测试 + 控制器裁定钉）：
- 端到端 mock 集成：00→02→04(mock)→05 全链在 tmp 树内封闭跑（复用
  test_pipeline.py / test_stages.py 的 fixture 模式），断言五产物、
  cosine/score 值域、scores_raw 与 mappings 聚合一致、防泄露的训练对数；
- 借用不触源：input_exp 借用 stage-gen，源实验目录逐文件不变，
  本实验 stage-gen/ 带 PROVENANCE.json；
- 阶段隔离：05 跑前后本实验 stage-gen 逐字节不变（spec 反目标 2/3）；
- 空实验降级：无自由标签 / 空数据集 → 空 schema 产物 + 零输出 manifest，
  不崩（合法空实验可表示）；
- 固定分割训练对：train_pairs 恰等于 scores_raw ⋈ genome_scores 后
  movie_id ∈ fixed_split 训练侧（dataset − eval_test 清单）的行数（00
  合成产出 processed/eval_test.txt，各测试按需覆写其内容）；
- 训练对为空：数据非空但固定分割后无训练对（eval_test 清单覆盖全部电影）
  → fit_calibration 的 ValueError 传播（配置问题，控制器裁定 6）；
- smoke 配置钉：temporal_split_year: 2019（440 固定分割后已不参与 05 校准，
  钉保护 report 描述性分桶视图的既有配置不被误改）。
"""
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movietaggen.align.calibrate import Calibrator
from movietaggen.align.mapping import MAPPING_COLUMNS, SCORE_COLUMNS, aggregate_scores
from movietaggen.config import Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import run_generation

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载
run_alignment = importlib.import_module("scripts.05_run_alignment").run_alignment

STAGE_ALIGN_ARTIFACTS = (
    "mappings.parquet", "scores_raw.parquet", "scores_calibrated.parquet",
    "calibration.json", "run_manifest.json")


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
def chain(tmp_path):
    """00→02→04（mock）全链就绪：合成数据 → dataset → stage-gen（05 留给各测试跑）。"""
    make_synthetic_dataset(tmp_path)          # 00
    paths = _paths_for(tmp_path)
    build_dataset(paths)                      # 02
    cfg = load_config("configs/smoke.yaml")   # mock=true → make_embedder 得 MockEmbedder
    run_generation(cfg, paths, MockGenClient(), cache=None)   # 04
    return cfg, paths


def _snapshot(root: Path) -> dict:
    """目录逐文件 sha256 快照（内容级比对，不依赖 mtime）。"""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_align_end_to_end(chain):
    """五产物齐全；值域契约；scores_raw = mappings 全量聚合；训练对只含固定分割训练侧电影。"""
    cfg, paths = chain
    out = run_alignment(cfg, paths, "configs/smoke.yaml")
    assert out == paths.stage_align
    for name in STAGE_ALIGN_ARTIFACTS:
        assert (paths.stage_align / name).exists(), f"缺产物 {name}"
    assert (paths.results / "config.yaml").exists()   # 配置快照在 results 根

    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    assert set(mappings.columns) == set(MAPPING_COLUMNS)
    assert len(mappings) > 0
    assert mappings["cosine"].between(-1.0, 1.0).all()

    raw = pd.read_parquet(paths.stage_align / "scores_raw.parquet")
    assert set(raw.columns) == set(SCORE_COLUMNS)
    # 全量聚合一致（low_conf 不过滤——裁定 1；重算口径 = mappings 按 cfg.align.agg 聚合）
    pd.testing.assert_frame_equal(raw, aggregate_scores(mappings, cfg.align.agg))

    # 校准分 = calibration.json 重建的 Calibrator 对 scores_raw 的 transform，且 ∈ [0,1]
    d = json.loads((paths.stage_align / "calibration.json").read_text(encoding="utf-8"))
    assert d["method"] == cfg.align.calib_method
    calib = pd.read_parquet(paths.stage_align / "scores_calibrated.parquet")
    assert set(calib.columns) == set(SCORE_COLUMNS)
    assert calib["score"].between(0.0, 1.0).all()
    calibrator = Calibrator.from_dict(d)
    assert np.allclose(calib["score"].to_numpy(dtype=float),
                       calibrator.transform(raw["score"].to_numpy(dtype=float)))
    # transform 作用于全量行（含测试分割电影），行键与 scores_raw 逐行一致
    assert list(calib["movie_id"]) == list(raw["movie_id"])
    assert list(calib["tag_id"]) == list(raw["tag_id"])

    # manifest：键齐全、计数与产物一致、嵌入模型为 mock 路径
    manifest = json.loads(
        (paths.stage_align / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"n_movies", "n_free_tags", "n_mappings", "n_scores",
                             "calib_method", "train_pairs", "embed_model"}
    assert manifest["n_movies"] == 5
    assert manifest["n_free_tags"] == len(pd.read_parquet(paths.stage_gen / "tags.parquet"))
    assert manifest["n_mappings"] == len(mappings)
    assert manifest["n_scores"] == len(raw)
    assert manifest["calib_method"] == cfg.align.calib_method
    assert manifest["embed_model"] == "mock"

    # 防泄露：train_pairs 恰等于固定分割训练侧（dataset − eval_test 清单）
    # 电影的 (score, truth) 对数（独立重算口径：解析清单文本，不经 fixed_split）
    dataset = pd.read_parquet(paths.dataset)
    scores = pd.read_parquet(paths.genome_scores)
    test_ids = {int(line) for line in
                paths.eval_test.read_text(encoding="utf-8").splitlines() if line.strip()}
    train_movies = set(dataset["movie_id"]) - test_ids
    expected_pairs = raw[raw["movie_id"].isin(train_movies)].merge(
        scores, on=["movie_id", "tag_id"])
    assert manifest["train_pairs"] == len(expected_pairs)
    assert len(expected_pairs) > 0
    # 过滤确实发生：测试清单电影的分数行存在但不进训练对
    assert set(raw["movie_id"]) - train_movies


def test_train_pairs_from_fixed_split(chain):
    """固定分割契约（brief Step 1a）：清单 2 部测试电影 → 校准拟合成功，且
    manifest["train_pairs"] 恰等于 scores_raw ⋈ genome_scores 后 movie_id ∈
    训练侧（dataset − 清单）的行数——测试清单电影的 truth 行数不入计。

    计划文本的 "dataset 8 部" 为撰写时假设（SYNTHETIC_MOVIES 实为 5 部，
    test_constants_shape 守卫）；语义等价物：dataset 5 部 + eval_test 2 部。
    """
    cfg, paths = chain
    # 覆写 00 合成清单（默认 1 部）：钉 2 部测试电影 → 训练侧 3 部
    paths.eval_test.write_text("1\n2\n", encoding="utf-8")

    run_alignment(cfg, paths, "configs/smoke.yaml")

    # 拟合成功：非降级哨兵（fitted:false 占位无拟合参数），且能重建校准器
    # （哨兵缺参数键，from_dict 会 ValueError——能重建即已拟合）
    d = json.loads((paths.stage_align / "calibration.json").read_text(encoding="utf-8"))
    assert "fitted" not in d and d["method"] == cfg.align.calib_method
    Calibrator.from_dict(d)
    manifest = json.loads(
        (paths.stage_align / "run_manifest.json").read_text(encoding="utf-8"))
    dataset = pd.read_parquet(paths.dataset)
    raw = pd.read_parquet(paths.stage_align / "scores_raw.parquet")
    scores = pd.read_parquet(paths.genome_scores)
    test_ids = {1, 2}
    train_movies = set(dataset["movie_id"]) - test_ids
    expected = raw[raw["movie_id"].isin(train_movies)].merge(
        scores, on=["movie_id", "tag_id"])
    assert manifest["train_pairs"] == len(expected)
    assert len(expected) > 0
    # 排除只发生在测试侧：raw 中有测试清单电影的行（不入计），且被排除的
    # 电影恰为测试清单成员（无训练侧电影被误排）
    excluded = set(raw["movie_id"]) - train_movies
    assert excluded
    assert excluded <= test_ids


def test_borrow_does_not_touch_source(tmp_path):
    """input_exp 借用：源实验 stage-gen 逐文件不变；本实验 stage-gen 带溯源的物理副本。"""
    make_synthetic_dataset(tmp_path)
    donor_paths = _paths_for(tmp_path)
    build_dataset(donor_paths)
    cfg = load_config("configs/smoke.yaml")
    run_generation(cfg, donor_paths, MockGenClient(), cache=None)   # 源实验 04
    before = _snapshot(donor_paths.stage_gen)

    # 消费实验（results/consumer）：借用源实验 "smoke" 的 stage-gen
    consumer_results = donor_paths.results.with_name("consumer")
    consumer = replace(donor_paths,
                       results=consumer_results,
                       stage_gen=consumer_results / "stage-gen",
                       stage_align=consumer_results / "stage-align",
                       stage_audit=consumer_results / "stage-audit")
    cfg.align.input_exp = "smoke"
    out = run_alignment(cfg, consumer, "configs/smoke.yaml")

    assert out == consumer.stage_align
    assert _snapshot(donor_paths.stage_gen) == before          # 源目录内容级不变
    prov_path = consumer.stage_gen / "PROVENANCE.json"
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    assert prov["source_exp"] == "smoke" and prov["stage"] == "stage-gen"
    # 物理副本与源逐字节一致（借用即固化，源此后被删/改不影响）
    assert hashlib.sha256((consumer.stage_gen / "tags.parquet").read_bytes()).hexdigest() \
        == before["tags.parquet"]
    # 借用产物正常进入对齐，五产物齐备
    for name in STAGE_ALIGN_ARTIFACTS:
        assert (consumer.stage_align / name).exists()
    assert len(pd.read_parquet(consumer.stage_align / "mappings.parquet")) > 0


def test_stage_gen_untouched_by_align(chain):
    """阶段隔离：05 跑前后 stage-gen 逐文件不变；自有产物不写 PROVENANCE（无借用）。"""
    cfg, paths = chain
    before = _snapshot(paths.stage_gen)
    run_alignment(cfg, paths, "configs/smoke.yaml")
    assert _snapshot(paths.stage_gen) == before
    assert not (paths.stage_gen / "PROVENANCE.json").exists()


def test_empty_stage_gen_writes_empty_artifacts(chain):
    """无自由标签 → 空 schema 产物 + 零输出 manifest，不崩（合法空实验可表示）。"""
    cfg, paths = chain
    pd.DataFrame(columns=["movie_id", "tag", "n_runs"]).to_parquet(
        paths.stage_gen / "tags.parquet", index=False)

    out = run_alignment(cfg, paths, "configs/smoke.yaml")
    assert out == paths.stage_align
    for name in STAGE_ALIGN_ARTIFACTS:
        assert (paths.stage_align / name).exists()
    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    raw = pd.read_parquet(paths.stage_align / "scores_raw.parquet")
    calib = pd.read_parquet(paths.stage_align / "scores_calibrated.parquet")
    assert len(mappings) == len(raw) == len(calib) == 0
    assert set(mappings.columns) == set(MAPPING_COLUMNS)
    assert set(raw.columns) == set(SCORE_COLUMNS)
    manifest = json.loads(
        (paths.stage_align / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_movies"] == 5
    assert manifest["n_free_tags"] == 0
    assert manifest["n_mappings"] == 0
    assert manifest["n_scores"] == 0
    assert manifest["train_pairs"] == 0
    # 校准无从拟合：calibration.json 以 fitted:false 哨兵占位
    d = json.loads((paths.stage_align / "calibration.json").read_text(encoding="utf-8"))
    assert d["fitted"] is False and d["method"] == cfg.align.calib_method


def test_empty_dataset_writes_empty_artifacts(chain):
    """空数据集（02 未产出/零匹配）→ 同样降级为零输出，不崩；输入侧计数如实记录。"""
    cfg, paths = chain
    dataset = pd.read_parquet(paths.dataset)
    dataset.iloc[0:0].to_parquet(paths.dataset, index=False)

    out = run_alignment(cfg, paths, "configs/smoke.yaml")
    assert out == paths.stage_align
    assert len(pd.read_parquet(paths.stage_align / "mappings.parquet")) == 0
    assert len(pd.read_parquet(paths.stage_align / "scores_calibrated.parquet")) == 0
    manifest = json.loads(
        (paths.stage_align / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_movies"] == 0
    assert manifest["n_free_tags"] > 0          # 输入侧如实记录，输出侧全零
    assert manifest["n_mappings"] == 0 and manifest["n_scores"] == 0
    assert manifest["train_pairs"] == 0


def test_empty_train_pairs_propagates_value_error(chain):
    """数据非空但固定分割后训练对为空 → fit_calibration 的 ValueError 传播
    （配置问题，不静默降级，控制器裁定 6）。

    固定分割下的触发形态：eval_test 清单覆盖全部电影 → 训练侧为空集
    （另一等价形态是生成产物只含测试清单电影，joined 过滤后同样为空）。
    """
    cfg, paths = chain
    dataset = pd.read_parquet(paths.dataset)
    paths.eval_test.write_text(
        "".join(f"{mid}\n" for mid in sorted(dataset["movie_id"])), encoding="utf-8")
    with pytest.raises(ValueError, match="校准对为空"):
        run_alignment(cfg, paths, "configs/smoke.yaml")


def test_smoke_config_temporal_split_year_pinned():
    """temporal_split_year 配置钉：smoke.yaml 钉 2019、default.yaml 维持 schema
    默认 2015。440 固定分割后该字段已不参与 05 校准（拟合面统一走 fixed_split），
    仅 report 层描述性分桶视图（年代子集）仍消费——此钉防止视图年份被误改。"""
    assert load_config("configs/smoke.yaml").protocol.temporal_split_year == 2019
    assert load_config("configs/default.yaml").protocol.temporal_split_year == 2015


def test_llm_mapper_end_to_end(chain):
    """B② 集成：mapper=llm（fake 客户端恒选 choice 1）→ 单电影单标签单候选行、
    校准拟合、五产物齐；embed 模式（默认）不触 client 的契约由其它用例钉。"""
    from types import SimpleNamespace
    import json as _json

    cfg, paths = chain
    cfg.align.mapper = "llm"
    cfg.align.mapper_candidates = 3

    class AlwaysFirst:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, run_id="llm-mapper"):
            self.calls += 1
            content = '{"choice": 1}'
            return SimpleNamespace(ok=True, content=content,
                                   raw={"choices": [{"message": {"content": content}}]},
                                   usage={})

    fake = AlwaysFirst()
    out = run_alignment(cfg, paths, "configs/smoke.yaml", client=fake, cache=None)
    mappings = pd.read_parquet(out / "mappings.parquet")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    # 每个唯一 (movie, tag) 行恰好一个候选（choice 1）且发生过真实调用
    assert len(mappings) > 0
    assert mappings.groupby(["movie_id", "tag"]).size().max() == 1
    assert fake.calls > 0
    assert manifest["n_mappings"] == len(mappings)
    # 校准真实拟合（非哨兵）
    calib = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert "fitted" not in calib and calib["method"] == "isotonic"
