"""06 审计脚本（判审 + 过滤 + 校准抽样导出）：全链接线与 judge 客户端分支测试。

覆盖面（brief 集成测试 + 控制器裁定钉）：
- 端到端 mock 集成：00→02→04→05→06 全链在 tmp 树内封闭跑（复用
  test_align_script.py / test_judge.py 的 fixture 模式；project_root 打桩使
  main() 的 paths_for 落在 tmp 树），断言 stage-audit 五产物（judge_mapping /
  judge_gen_quality / scores_filtered / run_manifest / calibration_sample）、
  calibration_sample.csv 行数 ≤ 500 且恰 8 列、manifest 8 键、配置快照；
- min() 封顶：calibration_sample 小于判定行数时行数恰为配置值；
- judge 客户端分支（单元，仅构造不联网——OpenAIClient 的 transport 惰性
  构建，构造期无网络）：mock 配置 → MockJudgeClient；非 mock →
  OpenAIClient 且 judge 模型/温度取 audit 配置、端点接线沿用 model 配置。
"""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from movietaggen.audit.calibration import CALIBRATION_COLUMNS
from movietaggen.audit.judge import JUDGE_GEN_COLUMNS, JUDGE_MAPPING_COLUMNS
from movietaggen.audit.mock import MockJudgeClient
from movietaggen.align.mapping import SCORE_COLUMNS
from movietaggen.config import AppConfig, Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.generate.client import OpenAIClient
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import run_generation

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载（同 test_align_script）
run_audit_script = importlib.import_module("scripts.06_run_audit")
run_alignment = importlib.import_module("scripts.05_run_alignment").run_alignment

# stage-audit 五产物（brief 集成断言清单）
STAGE_AUDIT_ARTIFACTS = (
    "judge_mapping.parquet", "judge_gen_quality.parquet",
    "scores_filtered.parquet", "run_manifest.json", "calibration_sample.csv")

# run_manifest.json 固定 8 键（T14 judge.py 的 manifest schema）
AUDIT_MANIFEST_KEYS = {"n_mappings", "n_judged", "n_accept", "n_reject",
                       "n_parse_failed", "n_gen_quality_movies", "judge_model",
                       "sample_rate"}


def _paths_for(tmp_path) -> Paths:
    """指向 tmp 树的 Paths——布局与 config.paths_for 的 synthetic 分支逐路径
    一致（synthetic 基目录 = root/data/synthetic，同 00 脚本的真实布局），使
    main() 经真实 paths_for（project_root 打桩为 tmp_path）落到同一棵 tmp 树。"""
    base = tmp_path / "data" / "synthetic"
    return Paths(
        root=tmp_path,
        movienet=base / "raw" / "movienet",
        movielens=base / "raw" / "movielens",
        interim=base / "interim", processed=base / "processed",
        cache_llm=tmp_path / ".cache" / "llm",
        cache_embed=tmp_path / ".cache" / "embed",
        results=tmp_path / "results" / "smoke",
        stage_gen=tmp_path / "results" / "smoke" / "stage-gen",
        stage_align=tmp_path / "results" / "smoke" / "stage-align",
        stage_audit=tmp_path / "results" / "smoke" / "stage-audit",
        tag_groups=base / "processed" / "tag_groups.json",
        dataset=base / "processed" / "dataset.parquet",
        genome_scores=base / "processed" / "genome_scores.parquet",
        genome_tags=base / "processed" / "genome_tags.parquet",
        aligned=base / "interim" / "aligned.parquet",
        baselines_dir=tmp_path / "results" / "baselines")


@pytest.fixture()
def chain(tmp_path):
    """00→02→04→05（mock）全链就绪：stage-gen + stage-align，06 的 main 留给各测试跑。"""
    make_synthetic_dataset(tmp_path / "data" / "synthetic")   # 00（同 00 脚本布局）
    paths = _paths_for(tmp_path)
    build_dataset(paths)                      # 02
    cfg = load_config("configs/smoke.yaml")   # mock=true
    run_generation(cfg, paths, MockGenClient(), cache=None)   # 04
    run_alignment(cfg, paths, "configs/smoke.yaml")           # 05
    return cfg, paths


def test_audit_script_end_to_end(chain, tmp_path, monkeypatch):
    """00→02→04→05→06 全链：五产物齐全、manifest 8 键、校准表 8 列且行数 ≤ 500。"""
    cfg, paths = chain
    # 打桩 project_root → main() 的 load_config/paths_for 落在 tmp 树（exp_name=smoke）
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    rc = run_audit_script.main("configs/smoke.yaml")
    assert rc == 0

    # 五产物 + 配置快照（results 根，run_audit 经 main 接线传入 config_path）
    for name in STAGE_AUDIT_ARTIFACTS:
        assert (paths.stage_audit / name).exists(), f"缺产物 {name}"
    assert (paths.results / "config.yaml").exists()

    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == AUDIT_MANIFEST_KEYS

    jm = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    jg = pd.read_parquet(paths.stage_audit / "judge_gen_quality.parquet")
    assert set(jm.columns) == set(JUDGE_MAPPING_COLUMNS)
    assert len(jm) == manifest["n_judged"] > 0     # sample_rate=1.0 → 全量判定
    assert set(jg.columns) == set(JUDGE_GEN_COLUMNS)
    assert manifest["n_gen_quality_movies"] == len(jg) == 5

    sf = pd.read_parquet(paths.stage_audit / "scores_filtered.parquet")
    assert set(sf.columns) == set(SCORE_COLUMNS)
    assert len(sf) > 0                             # mock 有 reject 被剔但不致全空

    # 校准抽样表：行数 = min(500, 判定行数) = 判定行数；恰 8 列；human_verdict 全空
    calib = pd.read_csv(paths.stage_audit / "calibration_sample.csv",
                        dtype=str, keep_default_na=False)
    assert list(calib.columns) == list(CALIBRATION_COLUMNS)
    assert len(calib) == len(jm)
    assert len(calib) <= cfg.audit.calibration_sample            # ≤ 500
    assert (calib["human_verdict"] == "").all()                  # 留给人填
    assert set(calib["judge_verdict"]) <= {"accept", "reject", "parse_failed"}


def test_calibration_sample_capped_by_config(chain, tmp_path, monkeypatch):
    """audit.calibration_sample < 判定行数 → 行数恰为配置值（min() 封顶语义）。"""
    _, paths = chain
    cap_cfg = tmp_path / "cap.yaml"
    cap_cfg.write_text(
        "exp_name: smoke\n"
        "mock: true\n"
        "data:\n"
        "  source: synthetic\n"
        "generate:\n"
        "  n_runs: 2\n"
        "audit:\n"
        "  calibration_sample: 1\n"
        "protocol:\n"
        "  temporal_split_year: 2019\n", encoding="utf-8")
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    assert run_audit_script.main(str(cap_cfg)) == 0
    calib = pd.read_csv(paths.stage_audit / "calibration_sample.csv",
                        dtype=str, keep_default_na=False)
    assert len(calib) == 1
    assert list(calib.columns) == list(CALIBRATION_COLUMNS)


def test_make_judge_client_mock_branch():
    """mock 配置 → MockJudgeClient（不建 OpenAI 客户端、不读环境变量）。"""
    cfg = AppConfig(mock=True)
    client = run_audit_script.make_judge_client(cfg)
    assert isinstance(client, MockJudgeClient)


def test_make_judge_client_openai_branch():
    """非 mock → OpenAIClient：judge 模型/温度取 audit 配置，端点接线（环境变量名/
    限速/extra_body）沿用 model 配置；仅构造，transport 惰性、无网络。"""
    cfg = AppConfig(mock=False)
    cfg.audit.judge_model = "judge-x"
    cfg.audit.judge_temperature = 0.3
    client = run_audit_script.make_judge_client(cfg)
    assert isinstance(client, OpenAIClient)
    assert client.model == "judge-x"
    assert client.temperature == 0.3
    assert client.base_url_env == cfg.model.base_url_env
    assert client.api_key_env == cfg.model.api_key_env
    assert client.request_interval == cfg.model.request_interval
    assert client.extra_body == cfg.model.extra_body
