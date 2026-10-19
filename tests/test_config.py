"""配置加载、校验、路径推导。"""
from pathlib import Path

import pytest
import yaml

from movietaggen.config import (
    VALID_DIMENSIONS,
    VALID_MODALITIES,
    load_config,
    paths_for,
    validate_config,
)


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


MINIMAL = {"exp_name": "t", "data": {"source": "synthetic"}}


def test_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.generate.n_runs == 3 and cfg.generate.few_shot_examples == 0
    assert cfg.generate.dimensions == list(VALID_DIMENSIONS)
    assert cfg.align.top_k == 3 and cfg.align.threshold == 0.5 and cfg.align.agg == "max"
    assert cfg.audit.judge_tasks == ["mapping", "gen_quality"]
    assert cfg.audit.judge_context == "pair+title" and cfg.audit.apply_filter is True
    assert cfg.eval.coverage_top_k == 10 and cfg.eval.coverage_tau == 0.7


def test_empty_modalities_legal(tmp_path):
    """空 modalities 合法：仅 title 的探针实验。"""
    cfg = load_config(_write(tmp_path, {**MINIMAL, "generate": {"modalities": []}}))
    assert cfg.generate.modalities == []


def test_empty_dimensions_rejected(tmp_path):
    """dimensions 空列表不合法：生成阶段零维度没有意义。"""
    bad = {**MINIMAL, "generate": {"dimensions": []}}
    with pytest.raises(ValueError, match="dimensions"):
        load_config(_write(tmp_path, bad))


def test_invalid_dimension_rejected(tmp_path):
    bad = {**MINIMAL, "generate": {"dimensions": ["style", "plot"]}}
    with pytest.raises(ValueError, match="维度"):
        load_config(_write(tmp_path, bad))


def test_invalid_modality_rejected(tmp_path):
    bad = {**MINIMAL, "generate": {"modalities": ["video", "audio"]}}
    with pytest.raises(ValueError, match="模态"):
        load_config(_write(tmp_path, bad))


def test_invalid_agg_rejected(tmp_path):
    bad = {**MINIMAL, "align": {"agg": "sum"}}
    with pytest.raises(ValueError, match="agg"):
        load_config(_write(tmp_path, bad))


def test_invalid_calib_method_rejected(tmp_path):
    bad = {**MINIMAL, "align": {"calib_method": "platt"}}
    with pytest.raises(ValueError, match="calib_method"):
        load_config(_write(tmp_path, bad))


def test_invalid_judge_context_rejected(tmp_path):
    bad = {**MINIMAL, "audit": {"judge_context": "full"}}
    with pytest.raises(ValueError, match="judge_context"):
        load_config(_write(tmp_path, bad))


def test_invalid_judge_task_rejected(tmp_path):
    bad = {**MINIMAL, "audit": {"judge_tasks": ["mapping", "safety"]}}
    with pytest.raises(ValueError, match="judge_tasks"):
        load_config(_write(tmp_path, bad))


def test_unknown_top_level_key_rejected(tmp_path):
    """未知顶层键必须报错，不得静默忽略（防止拼错/过时配置）。"""
    bad = {**MINIMAL, "annotation": {"n_runs": 5}}
    with pytest.raises(ValueError, match="annotation"):
        load_config(_write(tmp_path, bad))


def test_threshold_range(tmp_path):
    bad = {**MINIMAL, "align": {"threshold": 1.5}}
    with pytest.raises(ValueError, match="threshold"):
        load_config(_write(tmp_path, bad))


def test_sample_rate_range(tmp_path):
    bad = {**MINIMAL, "audit": {"sample_rate": 0.0}}
    with pytest.raises(ValueError, match="sample_rate"):
        load_config(_write(tmp_path, bad))


def test_n_runs_validated(tmp_path):
    bad = {**MINIMAL, "generate": {"n_runs": 0}}
    with pytest.raises(ValueError, match="n_runs"):
        load_config(_write(tmp_path, bad))


def test_extra_body_loads_from_yaml(tmp_path):
    """extra_body 字典透传（如 DashScope enable_thinking）。"""
    cfg_data = {**MINIMAL, "model": {"extra_body": {"enable_thinking": False}}}
    cfg = load_config(_write(tmp_path, cfg_data))
    assert cfg.model.extra_body == {"enable_thinking": False}


def test_paths_synthetic_isolated(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    p = paths_for(cfg)
    assert p.movienet == p.root / "data" / "synthetic" / "raw" / "movienet"
    assert p.movielens == p.root / "data" / "synthetic" / "raw" / "movielens"
    assert p.interim == p.root / "data" / "synthetic" / "interim"
    assert p.processed == p.root / "data" / "synthetic" / "processed"
    assert p.tag_groups == p.processed / "tag_groups.json"
    assert p.results == p.root / "results" / "t"
    assert p.stage_gen == p.results / "stage-gen"
    assert p.stage_align == p.results / "stage-align"
    assert p.stage_audit == p.results / "stage-audit"
    assert p.cache_llm == p.root / ".cache" / "llm"
    assert p.cache_embed == p.root / ".cache" / "embed"
    assert p.baselines_dir == p.root / "results" / "baselines"


def test_paths_real_default_raw_dir(tmp_path):
    """source: real 且未指定 raw_dir：默认指向 A 仓库（genometag-mllm）的原始数据。"""
    cfg = load_config(_write(tmp_path, {**MINIMAL, "data": {"source": "real"}}))
    p = paths_for(cfg)
    raw = p.root / ".." / "genometag-mllm" / "data" / "raw"
    assert p.movienet == raw / "movienet"
    assert p.movielens == raw / "movielens" / "ml-latest"
    assert p.interim == p.root / "data" / "interim"
    assert p.processed == p.root / "data" / "processed"


def test_raw_dir_override(tmp_path):
    """raw_dir 指向后：movielens = raw_dir/"movielens"/"ml-latest"；interim/processed 不随 raw_dir。"""
    raw = tmp_path / "other-raw"
    cfg = load_config(
        _write(tmp_path, {**MINIMAL, "data": {"source": "real", "raw_dir": str(raw)}}))
    p = paths_for(cfg)
    assert p.movienet == raw / "movienet"
    assert p.movielens == raw / "movielens" / "ml-latest"
    assert p.interim == p.root / "data" / "interim"
    assert p.processed == p.root / "data" / "processed"


def test_baselines_dir_fixed_across_exp_names(tmp_path):
    """baselines_dir 固定为 results/baselines，不随 exp_name 变化。"""
    cfg_a = load_config(_write(tmp_path, MINIMAL))
    cfg_b = load_config(_write(tmp_path, {**MINIMAL, "exp_name": "other"}))
    assert paths_for(cfg_a).results != paths_for(cfg_b).results
    assert paths_for(cfg_a).baselines_dir == paths_for(cfg_b).baselines_dir


def test_validate_rejects_bad_threshold_directly():
    """validate_config 也可独立调用（不经 load_config）。"""
    from movietaggen.config import AlignConfig, AppConfig
    bad = AppConfig(align=AlignConfig(threshold=0.0))
    with pytest.raises(ValueError, match="threshold"):
        validate_config(bad)


def test_default_yaml_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.exp_name == "default"
    assert cfg.data.source == "real"
    assert cfg.data.movie_limit is None


def test_pilot_yaml_loads():
    cfg = load_config("configs/pilot.yaml")
    assert cfg.exp_name == "pilot"
    assert cfg.data.movie_limit == 50


def test_smoke_yaml_loads():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.exp_name == "smoke"
    assert cfg.mock is True
    assert cfg.data.source == "synthetic"
    assert cfg.generate.n_runs == 2
