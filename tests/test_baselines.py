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
- 07 接线（440 Task 7）：direct_score 按 direct_score_source 分流（a_borrow
  端到端读迁移物、不建 client、不受 movie_limit）；拟合面固定分割（eval_test
  清单电影不入 train_scores 的泄露探针）；movie_limit 时全部受限方法的电影集
  == sample_limited_movie_ids 输出（统一断言，T6 过渡态钉并回）。
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
from movietaggen.eval.protocol import sample_limited_movie_ids
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


def test_movie_limit_all_methods_on_sample_limited_movie_ids(data_ready, tmp_path, monkeypatch):
    """440 Task 7 统一抽样（T6 过渡态钉并回）：movie_limit 时全部受限方法
    （own_run direct_score / genre_map / random）在同一 sample_limited_movie_ids
    输出电影集上产出——抽样池 = fixed_split 训练侧，受限跑永不接触 eval_test
    清单电影（440 对 I5 的裁定扩展）。旧 I5"07 侧全集抽样"与 Task 6 过渡态
    （genre_map / random 暂走全集）就此废除。bert 需 HF 模型下载，不入断言
    （离线约束，同冒烟三元组）。"""
    paths = data_ready
    cfg = tmp_path / "limit.yaml"
    cfg.write_text(
        "exp_name: smoke\n"
        "mock: true\n"
        "data:\n"
        "  source: synthetic\n"
        "  movie_limit: 2\n"
        "baselines:\n"
        "  run: [direct_score, genre_map, random]\n"
        "  direct_score_source: own_run\n", encoding="utf-8")
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    assert run_baselines_script.main(str(cfg)) == 0

    dataset = pd.read_parquet(paths.dataset)
    df = pd.read_parquet(paths.baselines_dir / "baselines.parquet")
    per_method = {m: set(s.astype(int))
                  for m, s in df.groupby("method")["movie_id"]}
    assert set(per_method) == {"direct_score", "genre_map", "random"}

    expected = set(sample_limited_movie_ids(dataset, 2, 42, paths.eval_test))
    for method, got in per_method.items():
        assert got == expected, method


def test_run_baselines_a_borrow_branch(tmp_path, monkeypatch):
    """440 Task 7 dispatch：direct_score_source=a_borrow → load_direct_score_a
    读 A 迁移评分——不建 client、不受 movie_limit（固定工件恒 88 部）。

    a_borrow × synthetic 组合被 config 校验拦截，故走 real 分支：raw_dir 指
    tmp fixture 目录、tmp processed 下自建小 parquet（不读仓库真实迁移物）。
    movie_limit=1 而产出电影集仍是迁移物全体 3 部——钉住"不受 movie_limit"；
    make_scoring_client 打桩为即炸——钉住"不建 client"。"""
    raw_dir = tmp_path / "rawroot"
    raw_dir.mkdir()
    processed = tmp_path / "data" / "processed"    # real 分支的 processed 布局
    processed.mkdir(parents=True)
    pd.DataFrame({
        "movie_id": [101, 102, 103, 104],
        "title": ["Fixture One", "Fixture Two", "Fixture Three", "Fixture Four"],
        "year": [2018, 2019, 2020, 2021],
        "genres": [["Action"], ["Drama"], ["Comedy"], ["Drama"]],
    }).to_parquet(processed / "dataset.parquet")
    pd.DataFrame({"movie_id": [101, 102, 103], "tag_id": [11, 11, 11],
                  "relevance": [0.5, 0.6, 0.7]}).to_parquet(
        processed / "genome_scores.parquet")
    pd.DataFrame({"tag_id": [11, 22]}).to_parquet(processed / "genome_tags.parquet")
    (processed / "eval_test.txt").write_text("104\n", encoding="utf-8")

    # 自建迁移物 fixture：3 电影 × 2 tag × 2 run；(101, 22) 两轮 0.2/0.6 →
    # 逐对均值 0.4（均值口径经 07 落盘保持的探针）
    rows = []
    for m in (101, 102, 103):
        for t in (11, 22):
            for r, run_id in enumerate(("run-0", "run-1")):
                rows.append({"movie_id": m, "tag_id": t, "group": "视觉风格",
                             "modality_set": "video|script|dialogue",
                             "run_id": run_id,
                             "score": (0.2, 0.6)[r] if (m, t) == (101, 22) else 0.4})
    a_raw = pd.DataFrame(rows, columns=["movie_id", "tag_id", "group",
                                        "modality_set", "run_id", "score"])
    a_raw.to_parquet(processed / "direct_score_a.parquet")

    cfg = tmp_path / "a_borrow.yaml"
    cfg.write_text(
        "exp_name: smoke\n"
        "mock: true\n"
        "data:\n"
        "  source: real\n"
        f"  raw_dir: {raw_dir.as_posix()}\n"
        "  movie_limit: 1\n"
        "baselines:\n"
        "  run: [direct_score]\n"
        "  direct_score_source: a_borrow\n", encoding="utf-8")
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    def _no_client(config):
        raise AssertionError("a_borrow 分支不应建 client")
    monkeypatch.setattr(run_baselines_script, "make_scoring_client", _no_client)

    assert run_baselines_script.main(str(cfg)) == 0

    out_dir = tmp_path / "results" / "baselines"
    df = pd.read_parquet(out_dir / "baselines.parquet")
    assert list(df.columns) == ["movie_id", "tag_id", "score", "method"]
    assert set(df["method"]) == {"direct_score"}
    a_movies, a_tags = set(a_raw["movie_id"]), set(a_raw["tag_id"])
    # 迁移物电影全体产出（movie_limit=1 未生效于 a_borrow——固定工件不抽样）
    assert set(df["movie_id"].astype(int)) == a_movies
    assert len(df) == len(a_movies) * len(a_tags)      # 稠密：电影数 × tag 数
    got = {(int(m), int(t)): s for m, t, s
           in zip(df["movie_id"], df["tag_id"], df["score"])}
    assert got[(101, 22)] == pytest.approx(0.4)        # 逐对均值口径保持
    manifest = json.loads(
        (out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"methods": ["direct_score"], "rows": len(df)}


def test_run_baselines_fit_set_excludes_eval_test_movie(data_ready, tmp_path, monkeypatch):
    """07 侧防泄露契约（440 固定分割）：genre_map 的拟合参考集（train_scores）
    只来自 fixed_split 训练侧——eval_test 清单电影的 genome 相关度不得进入
    拟合集（否则基线"见过"测试电影真值，对比系统性虚高）。

    探针：给清单电影单独追加专属 tag 999 的分数；07 跑完后 genre_map 输出
    （其 tag 宇宙 = train_scores 的 tag 宇宙）不得出现 999，且原全 tag 覆盖
    保持（正对照：train_scores 非空、来源为训练侧全体电影）。旧时序分割下
    清单电影（合成清单命中 2018 年的电影 1）落入 year < 2019 训练侧、探针
    tag 随之泄入——本测试在旧实现下红。"""
    paths = data_ready
    eval_movie = int(paths.eval_test.read_text(encoding="utf-8").split()[0])
    probe_tag = 999
    scores = pd.read_parquet(paths.genome_scores)
    scores = pd.concat(
        [scores, pd.DataFrame(
            [{"movie_id": eval_movie, "tag_id": probe_tag, "relevance": 0.9}])],
        ignore_index=True)
    scores.to_parquet(paths.genome_scores)
    monkeypatch.setattr("movietaggen.config.project_root", lambda: tmp_path)

    assert run_baselines_script.main("configs/smoke.yaml") == 0

    df = pd.read_parquet(paths.baselines_dir / "baselines.parquet")
    gm = df[df["method"] == "genre_map"]
    assert not gm.empty
    assert probe_tag not in set(gm["tag_id"].astype(int))     # 泄露探针
    genome_tags = pd.read_parquet(paths.genome_tags)          # 正对照：tag 全覆盖
    assert set(gm["tag_id"].astype(int)) == set(genome_tags["tag_id"].astype(int))
