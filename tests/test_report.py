"""评估报告：fake_exp_tree 全链产物上的指标表、生成质量双口径、跨实验合并。

fixture 复用 test_audit_script.py 的模式：00→02→04→05→06（mock）在 tmp 树
内封闭跑（project_root 打桩使 06/08 脚本 main 的 paths_for 落在 tmp 树）。
"""
import importlib
import json

import pandas as pd
import pytest

from movietaggen.config import AppConfig, Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.grouping import load_tag_groups
from movietaggen.data.movielens import load_user_tags
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.eval.report import metrics_table, write_report
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import run_generation

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载（同 test_audit_script）
run_audit_script = importlib.import_module("scripts.06_run_audit")
run_alignment = importlib.import_module("scripts.05_run_alignment").run_alignment
run_evaluate = importlib.import_module("scripts.08_evaluate")

REQUIRED_COLUMNS = {"method", "subset", "rmse_full", "rmse_pred_only",
                    "spearman", "precision@k", "ndcg@k"}


def _paths_for(tmp_path, exp: str = "smoke") -> Paths:
    """指向 tmp 树的 Paths（exp_name 可变）——布局与 config.paths_for 的
    synthetic 分支逐路径一致（同 test_audit_script 的打桩前提）。"""
    base = tmp_path / "data" / "synthetic"
    results = tmp_path / "results" / exp
    return Paths(
        root=tmp_path,
        movienet=base / "raw" / "movienet",
        movielens=base / "raw" / "movielens",
        interim=base / "interim", processed=base / "processed",
        cache_llm=tmp_path / ".cache" / "llm",
        cache_embed=tmp_path / ".cache" / "embed",
        results=results,
        stage_gen=results / "stage-gen",
        stage_align=results / "stage-align",
        stage_audit=results / "stage-audit",
        tag_groups=base / "processed" / "tag_groups.json",
        dataset=base / "processed" / "dataset.parquet",
        genome_scores=base / "processed" / "genome_scores.parquet",
        genome_tags=base / "processed" / "genome_tags.parquet",
        aligned=base / "interim" / "aligned.parquet",
        baselines_dir=tmp_path / "results" / "baselines")


@pytest.fixture()
def fake_exp_tree(tmp_path, monkeypatch):
    """00→02→04→05→06（mock）全链就绪：stage-gen/align/audit 产物齐备，
    08 的 main 留给各测试跑。"""
    make_synthetic_dataset(tmp_path / "data" / "synthetic")    # 00（同 00 脚本布局）
    paths = _paths_for(tmp_path)
    build_dataset(paths)                       # 02
    cfg = load_config("configs/smoke.yaml")    # mock=true
    run_generation(cfg, paths, MockGenClient(), cache=None)    # 04
    run_alignment(cfg, paths, "configs/smoke.yaml")            # 05
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)
    assert run_audit_script.main("configs/smoke.yaml") == 0    # 06
    return cfg, paths, tmp_path


def _second_exp(tmp_path):
    """同树第二实验（exp_name=smoke2）：跑 04+05 产出 stage-align 两评分矩阵
    （无 stage-audit——merged 测试只考 raw/calibrated 的前缀消歧）。"""
    yaml2 = tmp_path / "smoke2.yaml"
    yaml2.write_text(
        "exp_name: smoke2\n"
        "mock: true\n"
        "data:\n"
        "  source: synthetic\n"
        "generate:\n"
        "  n_runs: 2\n"
        "protocol:\n"
        "  temporal_split_year: 2019\n", encoding="utf-8")
    cfg2 = load_config(yaml2)
    paths2 = _paths_for(tmp_path, "smoke2")
    run_generation(cfg2, paths2, MockGenClient(), cache=None)
    run_alignment(cfg2, paths2, str(yaml2))
    return paths2


def test_metrics_table_matches_direct_computation():
    """数值手算锁值（照 A 的 test_report 模式）：perfect 预测 → 双口径 RMSE=0、
    spearman=1、NDCG@K=1、P@K=0.5（k=2、每电影 1 个相关）；反向预测 → RMSE>0.5。"""
    truth = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                          "relevance": [0.9, 0.1, 0.8, 0.2]})
    perfect = truth.rename(columns={"relevance": "score"})[["movie_id", "tag_id", "score"]]
    wrong = truth.assign(score=1.0 - truth["relevance"])[["movie_id", "tag_id", "score"]]
    cfg = AppConfig()
    cfg.protocol.top_k = 2
    cfg.protocol.relevance_threshold = 0.5
    table = metrics_table({"perfect": perfect, "wrong": wrong}, truth, cfg.protocol)
    row = table[table["method"] == "perfect"].iloc[0]
    assert row["rmse_full"] == pytest.approx(0.0)
    assert row["rmse_pred_only"] == pytest.approx(0.0)
    assert row["spearman"] == pytest.approx(1.0)
    assert row["precision@k"] == pytest.approx(0.5)
    assert row["ndcg@k"] == pytest.approx(1.0)
    row_w = table[table["method"] == "wrong"].iloc[0]
    assert row_w["rmse_pred_only"] > 0.5


def test_report_on_synthetic(fake_exp_tree, tmp_path):
    cfg, paths, _ = fake_exp_tree
    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    # 基线文件（Task 19 未落地前手工造——write_report 对 method 名透传）
    baselines_dir = tmp_path / "results" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    base_rows = [{"movie_id": int(m), "tag_id": int(t), "score": 0.5, "method": "genre_map"}
                 for m in truth["movie_id"].unique() for t in truth["tag_id"].unique()]
    pd.DataFrame(base_rows).to_parquet(baselines_dir / "baselines.parquet", index=False)

    out = write_report([paths.results], paths.baselines_dir, tmp_path / "evaluation",
                       cfg, dataset, truth, tag_groups, None)
    df = pd.read_csv(out / "metrics.csv")
    assert {"two-stage[raw]", "two-stage[calibrated]", "two-stage[filtered]",
            "genre_map"} <= set(df["method"])
    assert REQUIRED_COLUMNS <= set(df.columns)
    assert df["rmse_full"].notna().all()
    # 五子集齐全（spec §7.3 描述性分桶；smoke 配置 split=2019）
    assert {"全部", "时序训练(<2019)", "时序测试(≥2019)",
            "tag:content", "tag:preference"} <= set(df["subset"])
    # 每个子集下方法集合一致（过滤不丢方法行；空 pred 以 0 值行表示）
    all_methods = set(df[df["subset"] == "全部"]["method"])
    for sub in ("tag:content", "tag:preference", "时序训练(<2019)", "时序测试(≥2019)"):
        assert set(df[df["subset"] == sub]["method"]) == all_methods
    assert (out / "report.md").exists()
    assert "评估报告" in (out / "report.md").read_text(encoding="utf-8")


def test_gen_quality_json(fake_exp_tree, tmp_path):
    cfg, paths, _ = fake_exp_tree
    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    user_tags = load_user_tags(paths.movielens / "tags.csv")
    out = write_report([paths.results], paths.baselines_dir, tmp_path / "evaluation",
                       cfg, dataset, truth, tag_groups, user_tags)
    gq = json.loads((out / "gen_quality.json").read_text(encoding="utf-8"))
    # 嵌入口径三件套：mock embedder 下非退化（生成词表与用户标签共享 → 有真实重叠）
    assert 0.1 < gq["relevance"] <= 1.0
    assert 0.0 < gq["coverage"] <= 1.0
    assert 0.0 < gq["redundancy"] < 1.0
    # judge 口径（06 已跑 gen_quality 任务）：mock 哈希分 1-5 的均值
    assert 1.0 <= gq["judge_relevance"] <= 5.0
    assert 1.0 <= gq["judge_coverage"] <= 5.0


def test_gen_quality_without_user_tags(fake_exp_tree, tmp_path):
    """user_tags 缺失 → note 占位（嵌入口径跳过），judge 口径仍在。"""
    cfg, paths, _ = fake_exp_tree
    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    out = write_report([paths.results], paths.baselines_dir, tmp_path / "evaluation",
                       cfg, dataset, truth, None, None)
    gq = json.loads((out / "gen_quality.json").read_text(encoding="utf-8"))
    assert gq["note"] == "无用户标签，嵌入口径跳过"
    assert "relevance" not in gq                     # 嵌入口径键不存在
    assert 1.0 <= gq["judge_relevance"] <= 5.0


def test_extra_exps_merged(fake_exp_tree):
    """两个实验目录 → 08 脚本 main（--extra 接线）产出的 metrics.csv 含两者的
    method 行：第一实验保持裸名，第二实验加 |smoke2 后缀。"""
    _, paths, _ = fake_exp_tree
    _second_exp(paths.root)
    assert run_evaluate.main("configs/smoke.yaml", ["smoke2"]) == 0
    df = pd.read_csv(paths.results / "evaluation" / "metrics.csv")
    methods = set(df["method"])
    assert {"two-stage[raw]", "two-stage[calibrated]", "two-stage[filtered]",
            "two-stage[raw|smoke2]", "two-stage[calibrated|smoke2]"} <= methods
    # 第一实验保持裸名（无 |smoke 后缀）
    assert not any(m.endswith("|smoke]") for m in methods)
    gq = json.loads((paths.results / "evaluation" / "gen_quality.json")
                    .read_text(encoding="utf-8"))
    assert "relevance" in gq                          # 生成质量取第一实验


def test_evaluate_script_single_exp(fake_exp_tree):
    """08 脚本接线（不带 --extra）：dataset/truth/tag_groups/user_tags 读取与
    产物落位 results/<exp>/evaluation/，cache_embed 接线（embed_texts 路径）。"""
    _, paths, _ = fake_exp_tree
    assert run_evaluate.main("configs/smoke.yaml") == 0
    for name in ("metrics.csv", "report.md", "gen_quality.json"):
        assert (paths.results / "evaluation" / name).exists()
    df = pd.read_csv(paths.results / "evaluation" / "metrics.csv")
    assert {"two-stage[raw]", "two-stage[calibrated]",
            "two-stage[filtered]"} <= set(df["method"])
    gq = json.loads((paths.results / "evaluation" / "gen_quality.json")
                    .read_text(encoding="utf-8"))
    assert "relevance" in gq and "judge_relevance" in gq


def test_write_report_requires_exp_dir(tmp_path):
    """空实验目录列表 → 中文 ValueError（不静默产出空报告）。"""
    cfg = load_config("configs/smoke.yaml")
    with pytest.raises(ValueError, match="exp_dirs"):
        write_report([], None, tmp_path / "evaluation", cfg,
                     pd.DataFrame(), pd.DataFrame(), None, None)
