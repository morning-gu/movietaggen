"""全部配置可加载 + 消融配置按预期差异。"""
from pathlib import Path

import pytest

from movietaggen.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_all_configs_load():
    for p in sorted((REPO_ROOT / "configs").rglob("*.yaml")):
        load_config(p)  # 全部合法，未知键/非法值会在此抛错


def test_ablation_differs_as_intended():
    load = lambda rel: load_config(REPO_ROOT / rel)
    assert load("configs/ablation/video-only.yaml").generate.modalities == ["video"]
    t = load("configs/ablation/title-only.yaml")
    assert t.generate.modalities == [] and t.exp_name == "ablation/title-only"
    assert load("configs/ablation/dim-character.yaml").generate.dimensions == ["character"]
    assert load("configs/ablation/nruns-1.yaml").generate.n_runs == 1
    assert load("configs/ablation/topk-1.yaml").align.top_k == 1
    assert load("configs/ablation/threshold-0.80.yaml").align.threshold == 0.80
    assert load("configs/ablation/threshold-0.85.yaml").align.threshold == 0.85
    assert load("configs/ablation/threshold-0.90.yaml").align.threshold == 0.90
    assert load("configs/ablation/agg-mean.yaml").align.agg == "mean"
    a = load("configs/ablation/audit-off.yaml")
    assert a.audit.apply_filter is False
    assert load("configs/ablation/embed-sbert.yaml").align.embed_model == "sentence-transformers/all-MiniLM-L6-v2"
