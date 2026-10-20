"""评估报告：fake_exp_tree 全链产物上的指标表、生成质量双口径、跨实验合并。

fixture 复用 test_audit_script.py 的模式：00→02→04→05→06（mock）在 tmp 树
内封闭跑（project_root 打桩使 06/08 脚本 main 的 paths_for 落在 tmp 树）。
"""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from movietaggen.config import AppConfig, Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.grouping import load_tag_groups
from movietaggen.data.movielens import load_user_tags
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.eval.protocol import fixed_split
from movietaggen.eval.report import (
    gen_quality_summary,
    metrics_table,
    threshold_filtered_predictions,
    write_report,
)
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import run_generation

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载（同 test_audit_script）
run_audit_script = importlib.import_module("scripts.06_run_audit")
run_alignment = importlib.import_module("scripts.05_run_alignment").run_alignment
run_evaluate = importlib.import_module("scripts.08_evaluate")

REQUIRED_COLUMNS = {"method", "subset", "rmse_full", "rmse_pred_only",
                    "spearman", "precision@k", "ndcg@k", "n_pred"}


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
        baselines_dir=tmp_path / "results" / "baselines",
        eval_test=base / "processed" / "eval_test.txt",
        direct_score_a=base / "processed" / "direct_score_a.parquet")


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
        "  temporal_split_year: 2019\n"
        "baselines:\n"
        "  direct_score_source: own_run\n", encoding="utf-8")
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
            "two-stage[threshold-filtered]", "genre_map"} <= set(df["method"])
    assert REQUIRED_COLUMNS <= set(df.columns)
    # 终审修复波口径：threshold-filtered 变体行——非空子集有真实数值 +
    # n_pred>0，空子集（mock 嵌入下仅内容型标签有高置信映射 → tag:preference
    # 子集为空）按 I1 是 NaN 指标行 + n_pred=0（终审前该行是 0.0 伪装满分）；
    # 其余方法行（预测非空）必须全部有真实数值（C1 剔 NaN 后不被污染）。
    tf = df[df["method"] == "two-stage[threshold-filtered]"]
    assert len(tf) > 0
    assert tf.loc[tf["n_pred"] > 0, "rmse_full"].notna().all()
    assert tf.loc[tf["n_pred"] == 0, "rmse_full"].isna().all()
    assert (tf["n_pred"] >= 0).all()
    assert df.loc[df["method"] != "two-stage[threshold-filtered]",
                  "rmse_full"].notna().all()
    # 五子集齐全（spec §7.3 描述性分桶；smoke 配置 split=2019）
    assert {"全部", "时序训练(<2019)", "时序测试(≥2019)",
            "tag:content", "tag:preference"} <= set(df["subset"])
    # 每个子集下方法集合一致（过滤不丢方法行；空 pred 以 NaN 指标行表示，I1）
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
    # 逐电影口径元数据：5 部合成电影全部有用户标签、全部进入平均
    assert gq["n_movies"] == 5
    assert gq["n_movies_no_user_tags"] == 0


def test_gen_quality_per_movie_no_cross_movie_credit(tmp_path):
    """跨电影污染必须不可得（spec §7.1 逐电影口径）：m1 用户 [alpha, gamma]、
    生成 [alpha]；m2 用户 [beta]、生成 [gamma]。池级口径下 m2 的生成 gamma
    会替 m1 的用户 gamma 记账（coverage = 2/3）；逐电影口径 m2 的 coverage=0
    → 均值恰 0.25，且各电影生成侧仅 1 条 → redundancy 恒 0（池级 > 0）。"""
    exp = tmp_path / "exp"
    (exp / "stage-gen").mkdir(parents=True)
    pd.DataFrame({
        "movie_id": [1, 2],
        "tag": ["alpha one", "gamma three"],
        "n_runs": [1, 1],
    }).to_parquet(exp / "stage-gen" / "tags.parquet", index=False)
    user_tags = pd.DataFrame({
        "user_id": [1, 1, 2],
        "movie_id": [1, 1, 2],
        "tag": ["alpha one", "gamma three", "beta two"],
        "timestamp": [0, 0, 0],
    })
    cfg = AppConfig(mock=True)
    gq = gen_quality_summary(exp, user_tags, cfg, cache_root=None)
    assert gq["coverage"] == pytest.approx(0.25)     # m1: 1/2、m2: 0 → 均值
    assert gq["redundancy"] == pytest.approx(0.0)    # 逐电影无跨电影对可数
    assert gq["n_movies"] == 2
    assert gq["n_movies_no_user_tags"] == 0


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
            "two-stage[threshold-filtered]", "two-stage[raw|smoke2]",
            "two-stage[calibrated|smoke2]"} <= methods
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
    assert {"two-stage[raw]", "two-stage[calibrated]", "two-stage[filtered]",
            "two-stage[threshold-filtered]"} <= set(df["method"])
    gq = json.loads((paths.results / "evaluation" / "gen_quality.json")
                    .read_text(encoding="utf-8"))
    assert "relevance" in gq and "judge_relevance" in gq


def test_write_report_requires_exp_dir(tmp_path):
    """空实验目录列表 → 中文 ValueError（不静默产出空报告）。"""
    cfg = load_config("configs/smoke.yaml")
    with pytest.raises(ValueError, match="exp_dirs"):
        write_report([], None, tmp_path / "evaluation", cfg,
                     pd.DataFrame(), pd.DataFrame(), None, None)


# ---------------------------------------------------------------------------
# 终审修复波回归：C1（NaN 分数不污染）/ I1（空集 NaN 行 + n_pred）/
# I2（NDCG 的 IDCG 取完整 truth 宇宙）/ I6（阈值筛选变体）/ M6 / I7
# ---------------------------------------------------------------------------

_TRUTH = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                       "relevance": [0.9, 0.1, 0.8, 0.2]})


def _protocol_cfg():
    cfg = AppConfig()
    cfg.protocol.top_k = 2
    cfg.protocol.relevance_threshold = 0.5
    return cfg.protocol


def test_metrics_table_drops_nan_scores():
    """C1：预测帧中的 NaN 分数行（direct_score 的合法产物）必须剔除——
    NaN 入 dual_rmse 会把整个方法的 rmse 变 NaN，spearman 守卫还会静默
    返回 0.0 伪装真实分。其余电影对的指标照常计算；全 NaN 方法 → NaN 行
    （n_pred=0），不污染其它方法。"""
    good = pd.DataFrame({"movie_id": [1, 1, 2, 2], "tag_id": [1, 2, 1, 2],
                         "score": [0.9, 0.1, 0.8, float("nan")]})
    all_nan = good.assign(score=float("nan"))
    table = metrics_table({"good": good, "all_nan": all_nan}, _TRUTH,
                          _protocol_cfg())
    row = table[table["method"] == "good"].iloc[0]
    # 非 NaN 三对 perfect：pred_only=0；full 口径电影 2 的 tag 2 缺失补 0
    # 对 truth 0.2 → 误差 0.2²/4 → rmse_full = 0.1；spearman=1；n_pred=3
    assert row["rmse_pred_only"] == pytest.approx(0.0)
    assert row["rmse_full"] == pytest.approx(0.1)
    assert row["spearman"] == pytest.approx(1.0)
    assert row["n_pred"] == 3
    row_nan = table[table["method"] == "all_nan"].iloc[0]
    for col in ("rmse_full", "rmse_pred_only", "spearman",
                "precision@k", "ndcg@k"):
        assert pd.isna(row_nan[col]), col
    assert row_nan["n_pred"] == 0


def test_metrics_table_empty_subset_reports_nan_row():
    """I1：方法×子集的预测集为空 → NaN 指标行 + n_pred=0（不是 0.0 伪装满分）；
    有预测的方法行数值照常。"""
    a = pd.DataFrame({"movie_id": [1, 1], "tag_id": [1, 2],
                      "score": [0.9, 0.1]})            # 只有电影 1
    b = _TRUTH.rename(columns={"relevance": "score"})[["movie_id", "tag_id", "score"]]
    table = metrics_table({"a": a, "b": b}, _TRUTH, _protocol_cfg(),
                          movie_ids=[2])               # 子集只含电影 2
    row_a = table[table["method"] == "a"].iloc[0]
    for col in ("rmse_full", "rmse_pred_only", "spearman",
                "precision@k", "ndcg@k"):
        assert pd.isna(row_a[col]), col
    assert row_a["n_pred"] == 0
    row_b = table[table["method"] == "b"].iloc[0]
    assert row_b["rmse_full"] == pytest.approx(0.0)
    assert row_b["n_pred"] == 2                        # 电影 2 的两对


def test_ndcg_idcg_uses_full_truth_universe():
    """I2：NDCG 的 IDCG 分母取完整 truth 行的相关总数——稀疏方法（2 个预测对）
    未检索到的 relevant 必须扣分，不再被受限宇宙放大成 1.0；P@K 两宇宙数学
    等价，语义不变（分母恒 = k）。"""
    truth = pd.DataFrame({"movie_id": [1] * 5, "tag_id": [1, 2, 3, 4, 5],
                          "relevance": [0.9, 0.8, 0.7, 0.6, 0.5]})
    sparse = pd.DataFrame({"movie_id": [1, 1], "tag_id": [1, 2],
                           "score": [0.9, 0.8]})       # 只预测前 2 个（均相关）
    dense = truth.rename(columns={"relevance": "score"})[["movie_id", "tag_id", "score"]]
    cfg = AppConfig()
    cfg.protocol.top_k = 3
    cfg.protocol.relevance_threshold = 0.5
    table = metrics_table({"sparse": sparse, "dense": dense}, truth, cfg.protocol)
    row = table[table["method"] == "sparse"].iloc[0]
    # 手算：DCG = 1/log2(2) + 1/log2(3)；IDCG（完整宇宙 5 个 relevant，k=3）
    #   = DCG + 1/log2(4) → NDCG = DCG / IDCG ≈ 0.7653（受限宇宙错误口径给 1.0）
    dcg = 1 / np.log2(2) + 1 / np.log2(3)
    assert row["ndcg@k"] == pytest.approx(dcg / (dcg + 1 / np.log2(4)))
    dense_row = table[table["method"] == "dense"].iloc[0]
    assert dense_row["ndcg@k"] == pytest.approx(1.0)
    assert row["ndcg@k"] < dense_row["ndcg@k"]
    # P@K 不受影响：top-3 = 2 个预测对，全相关 → 2/3
    assert row["precision@k"] == pytest.approx(2 / 3)


def _mappings_exp(tmp_path, low_conf_flags, calib_dict):
    """手工 stage-align 目录：mappings（cosine 0.9/0.8/0.7 随旗标）+ calibration.json。"""
    exp = tmp_path / "exp"
    (exp / "stage-align").mkdir(parents=True)
    rows = [{"movie_id": 1, "tag": f"t{i}", "tag_id": 10 + i,
             "cosine": 0.9 - 0.1 * i, "rank": 1, "low_conf": lc}
            for i, lc in enumerate(low_conf_flags)]
    pd.DataFrame(rows).to_parquet(exp / "stage-align" / "mappings.parquet", index=False)
    (exp / "stage-align" / "calibration.json").write_text(
        json.dumps(calib_dict), encoding="utf-8")
    return exp


def test_threshold_filtered_variant_drops_low_conf_and_calibrates(tmp_path):
    """I6：low_conf 行剔除后按 agg 重聚合 + 校准换算——变体与 raw 聚合不同
    （少一行、值已换算），阈值消融臂由此产出真实不同的分数。"""
    exp = _mappings_exp(tmp_path, [False, True, False],
                        {"method": "linear", "slope": 0.5, "intercept": 0.0})
    variant = threshold_filtered_predictions(exp, "max")
    assert variant is not None
    assert sorted(variant["tag_id"].tolist()) == [10, 12]   # tag 11（low_conf）剔除
    got = dict(zip(variant["tag_id"].astype(int), variant["score"].astype(float)))
    assert got[10] == pytest.approx(0.45)                   # 0.9·0.5 校准换算
    assert got[12] == pytest.approx(0.35)
    assert sorted(got) != [10, 11, 12]                      # 与 raw 聚合（含 11）不同


def test_threshold_filtered_variant_all_low_conf_empty(tmp_path):
    """I6：全部映射 low_conf → 空聚合返回（评估层按 I1 以 NaN 行上报空集）。"""
    exp = _mappings_exp(tmp_path, [True, True],
                        {"method": "linear", "slope": 0.5, "intercept": 0.0})
    variant = threshold_filtered_predictions(exp, "max")
    assert variant is not None and len(variant) == 0
    assert list(variant.columns) == ["movie_id", "tag_id", "score"]


def test_threshold_filtered_variant_sentinel_and_missing(tmp_path):
    """I6：空实验哨兵（fitted=false）→ None + 打印说明；mappings 缺失 → None。"""
    exp = _mappings_exp(tmp_path, [False], {"method": "linear", "fitted": False})
    assert threshold_filtered_predictions(exp, "max") is None
    (exp / "stage-align" / "mappings.parquet").unlink()
    assert threshold_filtered_predictions(exp, "max") is None


def test_evaluate_warns_when_main_experiment_missing(fake_exp_tree, capsys):
    """M6：主实验缺 stage-align/scores_raw.parquet → 大声中文警告点名目录；
    评估仍以 --extra 实验继续（过滤后 smoke2 成为首个实验，取裸方法名），
    主实验被剔而非被静默顶替。"""
    _, paths, _ = fake_exp_tree
    (paths.results / "stage-align" / "scores_raw.parquet").unlink()
    _second_exp(paths.root)
    assert run_evaluate.main("configs/smoke.yaml", ["smoke2"]) == 0
    captured = capsys.readouterr()
    assert "警告" in captured.out and str(paths.results) in captured.out
    df = pd.read_csv(paths.results / "evaluation" / "metrics.csv")
    assert "two-stage[raw]" in set(df["method"])        # smoke2（现首个实验）


def test_evaluate_kappa_flag_writes_kappa_json(fake_exp_tree, tmp_path, capsys):
    """I7：--kappa 接线（M4-M5 人工校准回路的出数口）——refill_kappa 配对 →
    cohen_kappa → 中文摘要 + kappa.json（{"cohen_kappa", "n"}）。手算：judge 与
    human 完全不一致（二类互换）→ po=0、pe=0.5 → κ=-1。"""
    _, paths, _ = fake_exp_tree
    csv_path = tmp_path / "filled.csv"
    csv_path.write_text(
        "movie_id,title,tag,tag_id,genome_tag,cosine,judge_verdict,human_verdict\n"
        '1,"Paris, Texas",a,5,visual,0.8,accept,reject\n'
        '1,"Paris, Texas",b,7,tension,0.7,accept,reject\n'
        "2,Neon Alley,c,9,pace,0.6,reject,accept\n"
        "2,Neon Alley,d,12,mood,0.5,reject,accept\n", encoding="utf-8")
    assert run_evaluate.main("configs/smoke.yaml", kappa_csv=str(csv_path)) == 0
    out = capsys.readouterr().out
    assert "kappa" in out and "-1.0000" in out
    kappa = json.loads((paths.results / "evaluation" / "kappa.json")
                       .read_text(encoding="utf-8"))
    assert kappa["cohen_kappa"] == pytest.approx(-1.0)
    assert kappa["n"] == 4


def test_evaluate_kappa_flag_no_valid_pairs_skips(fake_exp_tree, tmp_path, capsys):
    """I7：无有效配对（human 未填）→ 打印原因、不落 kappa.json、rc 仍 0。"""
    _, paths, _ = fake_exp_tree
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "movie_id,title,tag,tag_id,genome_tag,cosine,judge_verdict,human_verdict\n"
        "1,T,a,5,visual,0.8,accept,\n", encoding="utf-8")
    assert run_evaluate.main("configs/smoke.yaml", kappa_csv=str(csv_path)) == 0
    out = capsys.readouterr().out
    assert "无有效配对" in out
    assert not (paths.results / "evaluation" / "kappa.json").exists()


# ---------------------------------------------------------------------------
# 440 协议：report 固定子集行（spec §2.4 论文主表口径——同集严格可比）
# ---------------------------------------------------------------------------


def test_report_fixed_split_rows_same_set_comparability(fake_exp_tree, tmp_path):
    """固定子集行的同集可比钉死（Task 9）：全宇宙型与仅测试集型两方法在
    "固定测试"行 n_pred 相等且 = 测试电影数 × tag 数（同电影集同构——对照
    "全部"行两方法 n_pred 不等，固定行才使方法同集可比）；"固定训练"行
    movie 集为补集（全宇宙型在训练侧的预测数，仅测试集型按 I1 为空集行）；
    固定两行位于"全部"之后、时序行之前；eval_test_ids=None 时无"固定"行
    （向后兼容）。"""
    cfg, paths, _ = fake_exp_tree
    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)
    _, test_ids = fixed_split(dataset, paths.eval_test)
    n_tags = truth["tag_id"].nunique()
    # 两方法经 baselines.parquet 注入（write_report 对 method 名透传）：
    # 全宇宙型（dataset 全部电影 × 全部 tag）与仅测试集型（只 eval_test 电影）
    full = truth.rename(columns={"relevance": "score"})[["movie_id", "tag_id", "score"]]
    test_only = full[full["movie_id"].isin(set(test_ids.tolist()))]
    baselines_dir = tmp_path / "results" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([full.assign(method="full_universe"),
               test_only.assign(method="test_only")]
              ).to_parquet(baselines_dir / "baselines.parquet", index=False)

    out = write_report([paths.results], paths.baselines_dir, tmp_path / "evaluation",
                       cfg, dataset, truth, tag_groups, None,
                       eval_test_ids=list(test_ids))
    df = pd.read_csv(out / "metrics.csv")
    label_test = f"固定测试({len(test_ids)})"
    label_train = f"固定训练({len(dataset) - len(test_ids)})"
    subsets = df["subset"].drop_duplicates().tolist()
    # 行序：固定两行在"全部"之后、时序行之前（论文主表口径紧邻全量视图）
    assert subsets.index("全部") < subsets.index(label_test) \
        < subsets.index(label_train) < subsets.index("时序训练(<2019)")
    # 同集可比钉死：固定测试行两方法 n_pred 相等 = 测试电影数 × tag 数
    fixed = df[df["subset"] == label_test].set_index("method")
    assert fixed.loc["full_universe", "n_pred"] == fixed.loc["test_only", "n_pred"]
    assert fixed.loc["full_universe", "n_pred"] == len(test_ids) * n_tags
    # 对照："全部"行（交集语义）两方法 n_pred 不等——全宇宙 / 仅测试集的
    # 预测宇宙差异只在固定行被抹平
    allrow = df[df["subset"] == "全部"].set_index("method")
    assert allrow.loc["full_universe", "n_pred"] != allrow.loc["test_only", "n_pred"]
    # 固定训练行 movie 集 = 补集：全宇宙型 n_pred = 训练电影数 × tag 数；
    # 仅测试集型在训练侧无预测（I1 空集行 n_pred=0，不以 0.0 伪装满分）
    trainrow = df[df["subset"] == label_train].set_index("method")
    assert trainrow.loc["full_universe", "n_pred"] == \
        (len(dataset) - len(test_ids)) * n_tags
    assert trainrow.loc["test_only", "n_pred"] == 0

    # eval_test_ids=None → 无"固定"行（向后兼容，既有用例口径不变）
    out2 = write_report([paths.results], paths.baselines_dir,
                        tmp_path / "evaluation_none", cfg, dataset, truth,
                        tag_groups, None)
    df2 = pd.read_csv(out2 / "metrics.csv")
    assert not df2["subset"].str.startswith("固定").any()
