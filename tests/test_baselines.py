"""genre_map / random / bert 基线与 07 编排脚本：手算值、确定性、防泄露、端到端。

覆盖面（brief 集成测试 + 控制器裁定钉）：
- genre_map：训练集统计手算值（adapt A 的 test_baselines 手算用例）、未见
  genre 回退全局均值、test_ids 泄露触发（Review Focus 5）；
- random：确定性（同 seed 同帧）、稠密覆盖、score ∈ [0,1)、哈希公式钉；
- bert（离线单元）：_movie_texts 读盘守卫、泄露断言先于 torch 触发；网络
  用例照 A 保留 RUN_BERT_TEST=1 显式豁免（全局无网络约束的唯一豁免）；
- 07 集成：合成 + mock（direct_score 走 MockScoringClient）→ baselines.parquet
  含三 method、manifest {"methods","rows"}、配置快照；smoke.yaml 的
  baselines.run 离线三元组钉（控制器裁定 4——bert 需模型下载，冒烟免下载）。
"""
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from movietaggen.baselines.bert_baseline import _movie_texts, bert_baseline
from movietaggen.baselines.genre_map import genre_map
from movietaggen.baselines.random_baseline import random_baseline
from movietaggen.baselines.mock_scoring import MockScoringClient
from movietaggen.config import AppConfig, Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.sample import _hash_unit, make_synthetic_dataset
from movietaggen.generate.client import OpenAIClient

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载（同 test_audit_script）
run_baselines_script = importlib.import_module("scripts.07_run_baselines")


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
        baselines_dir=tmp_path / "results" / "baselines",
        eval_test=base / "processed" / "eval_test.txt",
        direct_score_a=base / "processed" / "direct_score_a.parquet")


# ---------------------------------------------------------------------------
# genre_map：训练集统计 + 全局均值回退 + 防泄露（Review Focus 5）
# ---------------------------------------------------------------------------

def test_genre_map_hand_computed():
    """adapt A 的 test_genre_baseline_hand_computed：per-genre 均值与空 genre
    回退全局均值的手算值（v_A = mean(0.8, 0.6) = 0.7；v_B = mean(0.6, 0.2) =
    0.4；全局 = 1.6/3）。"""
    dataset = pd.DataFrame({"movie_id": [1, 2, 3, 4, 5, 6],
                            "genres": [["A"], ["A", "B"], ["B"], ["A"], ["A", "B"], []]})
    train = pd.DataFrame({  # 电影 1-3 为训练参考集
        "movie_id": [1, 2, 3], "tag_id": [10, 10, 10], "relevance": [0.8, 0.6, 0.2]})
    df = genre_map(train, dataset, movie_ids=[4, 5, 6])
    got = dict(zip(df["movie_id"], df["score"]))
    assert got[4] == pytest.approx(0.7)                 # [A]
    assert got[5] == pytest.approx(0.55)                # [A, B] → (0.7+0.4)/2
    assert got[6] == pytest.approx(1.6 / 3)             # 无 genre → 全局均值
    assert (df["method"] == "genre_map").all()
    assert list(df["tag_id"]) == [10, 10, 10]           # tag 全集 = 训练参考集 tag 宇宙


def test_genre_map_unseen_genre_global_fallback():
    """未见 genre（训练参考集从未出现）→ 回退全局均值，不崩、不 NaN。"""
    dataset = pd.DataFrame({"movie_id": [1, 2], "genres": [["A"], ["Z"]]})
    train = pd.DataFrame({"movie_id": [1, 1], "tag_id": [10, 11], "relevance": [0.3, 0.9]})
    df = genre_map(train, dataset, movie_ids=[2])
    got = dict(zip(df["tag_id"], df["score"]))
    assert got[10] == pytest.approx(0.3)
    assert got[11] == pytest.approx(0.9)


def test_genre_map_leakage_guard():
    """Review Focus 5：拟合参考集含测试电影 → assert_no_leakage 触发。"""
    dataset = pd.DataFrame({"movie_id": [1, 2], "genres": [["A"], ["B"]]})
    train = pd.DataFrame({"movie_id": [1, 2], "tag_id": [10, 10],
                          "relevance": [0.8, 0.4]})
    with pytest.raises(ValueError, match="泄露|重叠"):
        genre_map(train, dataset, movie_ids=[1], test_ids={2})


# ---------------------------------------------------------------------------
# random：确定性 + 稠密覆盖 + [0,1) + 哈希公式钉
# ---------------------------------------------------------------------------

def test_random_deterministic_dense_bounded():
    dataset = pd.DataFrame({"movie_id": [7, 3], "title": ["x", "y"]})
    tags = [10, 11, 12]
    df1 = random_baseline(dataset, tags, seed=7)
    df2 = random_baseline(dataset, tags, seed=7)
    pd.testing.assert_frame_equal(df1, df2)             # 同 seed 同帧（确定性）
    assert len(df1) == 6                                # 2 电影 × 3 tag 稠密
    assert set(zip(df1["movie_id"], df1["tag_id"])) == \
        {(m, t) for m in (7, 3) for t in tags}          # 每对恰好一行
    assert df1["score"].between(0, 1).all()
    assert not (df1["score"] == 1.0).any()              # [0,1) 开区间
    assert (df1["method"] == "random").all()
    assert list(df1.columns) == ["movie_id", "tag_id", "score", "method"]


def test_random_score_is_hash_of_triple():
    """分数公式钉（控制器裁定 1）：score = _hash_unit(f"rand|{movie}|{tag}|{seed}")，
    与 sample.py 同源副本逐位一致。"""
    dataset = pd.DataFrame({"movie_id": [5]})
    df = random_baseline(dataset, [9], seed=3)
    assert float(df.loc[0, "score"]) == _hash_unit("rand|5|9|3")


def test_random_seed_changes_scores():
    dataset = pd.DataFrame({"movie_id": [1, 2]})
    a = random_baseline(dataset, [1, 2, 3, 4, 5, 6, 7, 8], seed=1)
    b = random_baseline(dataset, [1, 2, 3, 4, 5, 6, 7, 8], seed=2)
    assert (a["score"].to_numpy() != b["score"].to_numpy()).any()


# ---------------------------------------------------------------------------
# bert：离线单元（懒导入纪律 + 读盘守卫 + 防泄露先于 torch）
# ---------------------------------------------------------------------------

def test_bert_movie_texts_reads_synthetic_layout(tmp_path):
    """_movie_texts：剧本优先、剧本缺失回退字幕、两者皆缺 → 空串；
    混合 str/缺失列的 pd.notna 守卫路径由 None 混排覆盖。"""
    make_synthetic_dataset(tmp_path / "data" / "synthetic")   # 与 _paths_for 布局一致
    paths = _paths_for(tmp_path)
    dataset = pd.DataFrame({
        "movie_id": [1, 2, 3],
        "script_path": ["scripts/mn01.txt", "scripts/missing.txt", None],
        "subtitles_path": ["subtitles/mn01.srt", "subtitles/mn02.srt", None]})
    texts = _movie_texts(dataset, paths)
    assert "CRIMSON HORIZON" in texts[1]        # 剧本正文优先
    assert "Neon Alley" in texts[2]             # 剧本缺失 → 回退字幕
    assert texts[3] == ""                       # 无剧本无字幕 → 空串


def test_bert_baseline_leakage_guard_before_any_torch(tmp_path):
    """bert_baseline 首语句防泄露断言：泄露分割在触及 torch/模型下载之前即抛
    （离线可测；网络/下载路径由 RUN_BERT_TEST 豁免用例覆盖）。"""
    dataset = pd.DataFrame({"movie_id": [1, 2], "script_path": [None, None],
                            "subtitles_path": [None, None]})
    scores = pd.DataFrame({"movie_id": [1], "tag_id": [10], "relevance": [0.5]})
    with pytest.raises(ValueError, match="泄露|重叠"):
        bert_baseline(AppConfig(), _paths_for(tmp_path), dataset, scores,
                      train_ids=[1, 2], test_ids={2})


@pytest.mark.skipif(os.environ.get("RUN_BERT_TEST") != "1",
                    reason="需要网络与模型下载，RUN_BERT_TEST=1 开启"
                           "（全局无网络约束的唯一豁免，照 A 的 test_bert_baseline）")
def test_bert_train_predict_structural():
    """adapt A 的 test_bert_baseline_learns_keyword_signal：文本→矩阵→训练→预测
    管道结构验证（微型随机模型 1 epoch 不保证学到关键词信号，故只做结构性
    断言；学习效果由真实模型人工验证）。"""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from movietaggen.baselines.bert_baseline import _train_predict

    train_texts = {1: "neon night city neon dreams", 2: "quiet harbor dawn"}
    train_matrix = pd.DataFrame({"movie_id": [1, 2], "tag_id": [1, 1],
                                 "relevance": [0.9, 0.1]})
    test_texts = {3: "all neon everything"}
    pred = _train_predict(train_texts, train_matrix, test_texts,
                          epochs=1, model_name="hf-internal-testing/tiny-random-BertModel")
    import math
    assert 3 in pred and set(pred[3]) == {1}
    assert math.isfinite(pred[3][1])


# ---------------------------------------------------------------------------
# 07 编排脚本：客户端工厂分支 + 端到端 mock 集成 + 未知方法拒绝
# ---------------------------------------------------------------------------

def test_make_scoring_client_mock_branch():
    """mock 配置 → MockScoringClient（不建 OpenAI 客户端、不读环境变量）。"""
    client = run_baselines_script.make_scoring_client(AppConfig(mock=True))
    assert isinstance(client, MockScoringClient)


def test_make_scoring_client_openai_branch():
    """非 mock → OpenAIClient：模型/温度/限速/extra_body 取生成配置（model 配置，
    非 judge 配置——direct_score 用主生成模型打分，同 A）；仅构造，transport
    惰性、无网络。"""
    cfg = AppConfig(mock=False)
    client = run_baselines_script.make_scoring_client(cfg)
    assert isinstance(client, OpenAIClient)
    assert client.model == cfg.model.model
    assert client.temperature == cfg.model.temperature
    assert client.base_url_env == cfg.model.base_url_env
    assert client.api_key_env == cfg.model.api_key_env
    assert client.request_interval == cfg.model.request_interval
    assert client.extra_body == cfg.model.extra_body


@pytest.fixture()
def data_ready(tmp_path):
    """00→02 就绪：合成数据 + dataset/genome_scores/genome_tags。基线不需要
    stage-gen/stage-align——direct_score 的输入是 dataset 本身。"""
    make_synthetic_dataset(tmp_path / "data" / "synthetic")   # 00（同 00 脚本布局）
    paths = _paths_for(tmp_path)
    build_dataset(paths)                      # 02
    return paths


def test_run_baselines_end_to_end_mock(data_ready, tmp_path, monkeypatch):
    """00→02→07 全链（mock）：baselines.parquet 含三 method 且各 method 稠密
    覆盖（movie × tag 全对）、manifest 恰 {"methods","rows"} 两键、配置快照落
    baselines_dir（结果目录自包含）。"""
    paths = data_ready
    # 打桩 project_root → main() 的 load_config/paths_for 落在 tmp 树
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    rc = run_baselines_script.main("configs/smoke.yaml")   # baselines.run = 离线三元组
    assert rc == 0

    dataset = pd.read_parquet(paths.dataset)
    genome_tags = pd.read_parquet(paths.genome_tags)
    n_pairs = len(dataset) * len(genome_tags)
    all_pairs = {(int(m), int(t)) for m in dataset["movie_id"]
                 for t in genome_tags["tag_id"]}

    df = pd.read_parquet(paths.baselines_dir / "baselines.parquet")
    assert list(df.columns) == ["movie_id", "tag_id", "score", "method"]
    assert set(df["method"]) == {"direct_score", "genre_map", "random"}
    for method, part in df.groupby("method"):
        assert len(part) == n_pairs                     # 每 method 稠密覆盖
        assert set(zip(part["movie_id"].astype(int),
                       part["tag_id"].astype(int))) == all_pairs
    assert df["score"].notna().all()                    # mock 全解析 + 其余方法无缺失

    manifest = json.loads(
        (paths.baselines_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"methods", "rows"}
    assert manifest["methods"] == ["direct_score", "genre_map", "random"]  # config 序
    assert manifest["rows"] == len(df) == 3 * n_pairs

    assert (paths.baselines_dir / "config.yaml").exists()   # 配置快照


def test_unknown_method_value_error(data_ready, tmp_path, monkeypatch):
    """未知方法名 → 中文 ValueError 点名合法值；报错前不落任何产物。"""
    paths = data_ready
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "exp_name: smoke\n"
        "mock: true\n"
        "data:\n"
        "  source: synthetic\n"
        "baselines:\n"
        "  run: [random, nope]\n"
        "  direct_score_source: own_run\n", encoding="utf-8")
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="未知基线方法"):
        run_baselines_script.main(str(bad))
    assert not paths.baselines_dir.exists()             # 失败不落半成品


def test_smoke_config_baselines_run_offline_trio():
    """控制器裁定 4：smoke.yaml 的 baselines.run 钉为离线三元组——schema 默认
    含 bert，而 bert 需要 HF 模型下载，冒烟必须免下载，故显式排除。"""
    assert load_config("configs/smoke.yaml").baselines.run == \
        ["direct_score", "genre_map", "random"]


def test_movie_limit_aligns_all_method_movie_sets(data_ready, tmp_path, monkeypatch):
    """I5：data.movie_limit 设置时全部基线方法在同一抽样电影集上产出（与生成
    管道完全相同的代码路径：sorted ids + random.Random(data.seed).sample）——
    试点规模下方法对比才有共同的电影轴口径。direct_score 内部抽样即同一路径。"""
    import random

    paths = data_ready
    cfg = tmp_path / "limit.yaml"
    cfg.write_text(
        "exp_name: smoke\n"
        "mock: true\n"
        "data:\n"
        "  source: synthetic\n"
        "  movie_limit: 2\n"
        "protocol:\n"
        "  temporal_split_year: 2019\n"
        "baselines:\n"
        "  run: [direct_score, genre_map, random]\n"
        "  direct_score_source: own_run\n", encoding="utf-8")
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    assert run_baselines_script.main(str(cfg)) == 0

    dataset = pd.read_parquet(paths.dataset)
    expected = set(random.Random(42).sample(
        sorted(dataset["movie_id"].astype(int).tolist()), 2))
    df = pd.read_parquet(paths.baselines_dir / "baselines.parquet")
    per_method = {m: set(s.astype(int))
                  for m, s in df.groupby("method")["movie_id"]}
    assert set(per_method) == {"direct_score", "genre_map", "random"}
    for method, ids in per_method.items():
        assert ids == expected, method                  # 三方法同一电影集
