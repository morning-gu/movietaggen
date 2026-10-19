"""LLM-as-judge 审计（mapping + gen_quality 双任务）：prompt 上下文、解析降级、
过滤重聚合、抽样确定性、mock 确定性、缓存接线与空实验哨兵的响亮失败。

覆盖面（brief 五测试 + 控制器裁定钉）：
- prompt 上下文三变体：pair 无标题行 / pair+title 有 / pair+full 追加内容节
  （500 token 预算截断；无文件不渲染节）；
- 解析容错：清洁 JSON / ```json 围栏 / 前后缀文本 → 解析成功；垃圾 / 非法
  verdict / 缺键 / 越界分 → None（调用方计 parse_failed）；
- apply_filter：reject 的映射不进重聚合，scores_filtered 在被拒 (movie, tag_id)
  上与 scores_calibrated 不同（brief 核心断言）+ 测试内独立重算全帧比对；
- sample_rate：ceil(0.5 × 映射数)，seed 确定（两次运行逐帧一致）；
- mock 评审：确定性（同输入同输出）、信封形状、usage 常量、verdict/打分与
  哈希规则一致；
- 缓存接线：mock=false 时缓存写入 + 二跑零调用；mock=true 不读不写缓存；
- 哨兵：stage-align/calibration.json 为 {"fitted": false} 空实验哨兵时
  apply_filter 大声报错（点名文件与哨兵，不静默跳过过滤）。
"""
import hashlib
import importlib
import json
import math

import pandas as pd
import pytest

from movietaggen.audit.judge import (
    JUDGE_GEN_COLUMNS,
    JUDGE_MAPPING_COLUMNS,
    JUDGE_SYSTEM_PROMPT,
    build_gen_quality_messages,
    build_mapping_messages,
    parse_gen_quality,
    parse_mapping_verdict,
    run_audit,
)
from movietaggen.audit.mock import MockJudgeClient
from movietaggen.align.calibrate import Calibrator
from movietaggen.align.mapping import SCORE_COLUMNS, aggregate_scores
from movietaggen.config import Paths, load_config
from movietaggen.data.align import build_dataset
from movietaggen.data.sample import make_synthetic_dataset
from movietaggen.generate.cache import ResponseCache
from movietaggen.generate.mock import MockGenClient
from movietaggen.generate.pipeline import run_generation

# 数字开头的模块名不能用 import 语句，importlib 按字符串装载（同 test_align_script）
run_alignment = importlib.import_module("scripts.05_run_alignment").run_alignment


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
        baselines_dir=tmp_path / "results" / "baselines")


@pytest.fixture()
def fake_stage_align(tmp_path):
    """00→02→04(mock)→05 全链就绪：stage-gen + stage-align（含真校准）留给审计跑。"""
    make_synthetic_dataset(tmp_path)
    paths = _paths_for(tmp_path)
    build_dataset(paths)
    cfg = load_config("configs/smoke.yaml")
    run_generation(cfg, paths, MockGenClient(), cache=None)
    run_alignment(cfg, paths, "configs/smoke.yaml")
    return cfg, paths


def _text(messages) -> str:
    """拼接消息中的全部文本（judge prompt 均为纯文本 content）。"""
    return "\n".join(m["content"] for m in messages
                     if isinstance(m["content"], str))


def _hash_unit(key: str) -> float:
    """与 audit/mock.py、data/sample.py 同一实现（测试独立复算 mock 规则用）。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def _filtered_expected(mappings: pd.DataFrame, judge_df: pd.DataFrame,
                       agg: str, calib_dict: dict) -> pd.DataFrame:
    """测试内独立重算过滤重聚合（drop reject 键 → 聚合 → 校准），供全帧比对。"""
    rejected = {(int(r.movie_id), str(r.tag), int(r.tag_id))
                for r in judge_df.itertuples() if r.verdict == "reject"}
    keep_mask = [(int(m), str(t), int(g)) not in rejected
                 for m, t, g in zip(mappings["movie_id"], mappings["tag"],
                                    mappings["tag_id"])]
    expected = aggregate_scores(mappings[keep_mask], agg)
    calibrator = Calibrator.from_dict(calib_dict)
    expected["score"] = calibrator.transform(expected["score"].to_numpy(dtype=float))
    return expected


# ---------------------------------------------------------------------------
# prompt 组装
# ---------------------------------------------------------------------------

def test_mapping_prompt_contexts():
    """pair 无标题行；pair+title 有标题行 + 完整四行 + 指令；system 固定。"""
    args = ("Inception", "mind-bending", "thought-provoking", "发人深省", 0.83)
    p_title = build_mapping_messages(*args, "pair+title")
    p_pair = build_mapping_messages(*args, "pair")
    t_title, t_pair = _text(p_title), _text(p_pair)

    assert p_title[0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    # _text 拼接了 system+user，标题行断言取 user 内容（system 之后的首行）
    user_title = _text(p_title).split("\n", 1)[1]
    assert user_title.startswith("电影：《Inception》")
    assert "Inception" in t_title and "Inception" not in t_pair   # pair 无标题
    assert '自由标签: "mind-bending"' in t_title
    assert '候选 genome 标签: "thought-provoking"（定义: 发人深省）' in t_title
    assert "嵌入相似度: 0.8300" in t_title
    # 指令行收尾，verdict 合法值与 reason 键在指令中言明
    assert '"verdict"' in t_title and '"reason"' in t_title
    # pair 变体仍含判定所需的三要素
    assert '自由标签: "mind-bending"' in t_pair
    assert '候选 genome 标签: "thought-provoking"' in t_pair
    assert "嵌入相似度: 0.8300" in t_pair
    # 定义缺失时省略定义后缀（不渲染空括号）
    p_nodef = build_mapping_messages("T", "a", "b", "", 0.5, "pair+title")
    assert "定义" not in _text(p_nodef)
    # 非法 context 大声报错
    with pytest.raises(ValueError):
        build_mapping_messages(*args, "bogus")


def test_mapping_prompt_pair_full_content_sections():
    """pair+full 追加内容节：剧本/台词节选各自 500 token 截断；无文件不渲染节。"""
    args = ("Inception", "mind-bending", "thought-provoking", "发人深省", 0.83)
    p = build_mapping_messages(*args, "pair+full",
                               script="INT. NIGHT — rain.", subtitles="They move.")
    t = _text(p)
    assert "## 内容节选" in t
    assert "INT. NIGHT — rain." in t and "They move." in t
    assert "剧本节选" in t and "台词节选" in t

    # 无内容文件 → 不渲染内容节（文件缺失是合法情形）
    p_none = build_mapping_messages(*args, "pair+full")
    assert "## 内容节选" not in _text(p_none)

    # 超长剧本截断到 500 token（近似 4 字符/token → 2000 字符）
    long_script = "x" * 9999
    p_long = build_mapping_messages(*args, "pair+full", script=long_script)
    assert _text(p_long).count("x") == 2000


def test_gen_quality_prompt():
    """gen_quality prompt：标题行 + 两个 JSON 数组行 + 打分指令。"""
    p = build_gen_quality_messages("Neon Alley", ["neon visuals", "tense"],
                                   ["neon visual", "slow burn"])
    t = _text(p)
    assert p[0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert p[1]["content"].startswith("电影：《Neon Alley》")
    assert '生成标签: ["neon visuals", "tense"]' in t
    assert '真实用户标签: ["neon visual", "slow burn"]' in t
    assert '"relevance"' in t and '"coverage"' in t


# ---------------------------------------------------------------------------
# 解析降级
# ---------------------------------------------------------------------------

def test_parse_verdict_variants():
    ok = '{"verdict": "accept", "reason": "ok"}'
    assert parse_mapping_verdict(ok) == {"verdict": "accept", "reason": "ok"}
    assert parse_mapping_verdict("```json\n" + ok + "\n```") == {
        "verdict": "accept", "reason": "ok"}                       # 围栏容忍
    assert parse_mapping_verdict("前缀 " + ok + " 后缀") == {
        "verdict": "accept", "reason": "ok"}                       # 前后缀容忍
    assert parse_mapping_verdict("garbage") is None
    assert parse_mapping_verdict('{"verdict": "maybe", "reason": "x"}') is None
    assert parse_mapping_verdict('{"verdict": "accept"}') is None   # 缺 reason
    assert parse_mapping_verdict('{"verdict": "accept", "reason": 3}') is None
    assert parse_mapping_verdict('["verdict"]') is None             # 非 dict
    assert parse_mapping_verdict("") is None


def test_parse_gen_quality_variants():
    ok = '{"relevance": 4, "coverage": 2}'
    assert parse_gen_quality(ok) == {"relevance": 4, "coverage": 2}
    assert parse_gen_quality("```json\n" + ok + "\n```") == {
        "relevance": 4, "coverage": 2}
    assert parse_gen_quality('{"relevance": 1, "coverage": 5}') is not None
    assert parse_gen_quality('{"relevance": 0, "coverage": 3}') is None   # 越界
    assert parse_gen_quality('{"relevance": 6, "coverage": 3}') is None
    assert parse_gen_quality('{"relevance": true, "coverage": 3}') is None  # bool 拒绝
    assert parse_gen_quality('{"relevance": 4.0, "coverage": 3}') is None   # 非整数
    assert parse_gen_quality('{"relevance": "4", "coverage": 3}') is None
    assert parse_gen_quality('{"relevance": 4}') is None                    # 缺键
    assert parse_gen_quality("garbage") is None


# ---------------------------------------------------------------------------
# mock 评审确定性
# ---------------------------------------------------------------------------

def test_mock_judge_deterministic():
    """同输入同输出；信封形状；usage 常量；verdict/打分与哈希规则一致。"""
    client = MockJudgeClient()
    msgs = build_mapping_messages("Inception", "mind-bending",
                                  "thought-provoking", "发人深省", 0.83)
    r1 = client.complete(msgs, run_id="judge-mapping")
    r2 = client.complete(msgs, run_id="judge-mapping")
    assert r1.ok and r1.content == r2.content
    assert r1.usage == {"prompt_tokens": 100, "completion_tokens": 50}
    # 信封形状（缓存命中路径从 choices[0].message.content 还原文本的前提）
    assert r1.raw["choices"][0]["message"]["content"] == r1.content
    # verdict 与哈希规则一致：accept ⇔ hash(free|genome) > 0.2
    unit = _hash_unit("mind-bending|thought-provoking")
    parsed = parse_mapping_verdict(r1.content)
    assert parsed["verdict"] == ("accept" if unit > 0.2 else "reject")
    assert isinstance(parsed["reason"], str) and parsed["reason"]

    # gen_quality 同理：确定性 + 1-5 值域 + 哈希规则
    gmsgs = build_gen_quality_messages("Neon Alley", ["neon visuals"], ["tense"])
    g1 = client.complete(gmsgs, run_id="judge-gen")
    g2 = client.complete(gmsgs, run_id="judge-gen")
    assert g1.content == g2.content and g1.ok
    gp = parse_gen_quality(g1.content)
    assert gp is not None and 1 <= gp["relevance"] <= 5 and 1 <= gp["coverage"] <= 5

    # 无法识别的 prompt 大声报错（mock 不静默编造结果）
    with pytest.raises(ValueError):
        client.complete([{"role": "user", "content": "无关文本"}], run_id="x")


# ---------------------------------------------------------------------------
# run_audit 端到端（mock 全链）
# ---------------------------------------------------------------------------

def test_run_audit_end_to_end(fake_stage_align):
    """双任务全量 + apply_filter：五产物齐全、schema 固定、manifest 计数自洽。"""
    cfg, paths = fake_stage_align
    out = run_audit(cfg, paths, MockJudgeClient(), cache=None,
                    config_path="configs/smoke.yaml")
    assert out == paths.stage_audit
    for name in ("judge_mapping.parquet", "judge_gen_quality.parquet",
                 "scores_filtered.parquet", "run_manifest.json"):
        assert (paths.stage_audit / name).exists(), f"缺产物 {name}"
    assert (paths.results / "config.yaml").exists()   # 配置快照随结果落盘

    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    jm = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    assert list(jm.columns) == JUDGE_MAPPING_COLUMNS
    assert len(jm) == len(mappings)                   # sample_rate=1.0 → 全量
    assert set(jm["verdict"]) <= {"accept", "reject"}  # mock 不产生 parse_failed
    assert set(jm["judge_model"]) == {cfg.audit.judge_model}
    assert len(jm) > 0

    jg = pd.read_parquet(paths.stage_audit / "judge_gen_quality.parquet")
    assert list(jg.columns) == JUDGE_GEN_COLUMNS
    assert set(jg["movie_id"]) == {1, 2, 3, 4, 5}     # 合成 5 部全有 gen+用户标签
    assert jg["relevance"].between(1, 5).all()
    assert jg["coverage"].between(1, 5).all()

    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"n_mappings", "n_judged", "n_accept", "n_reject",
                             "n_parse_failed", "n_gen_quality_movies",
                             "judge_model", "sample_rate"}
    assert manifest["n_mappings"] == len(mappings)
    assert manifest["n_judged"] == len(jm)
    assert manifest["n_accept"] == int((jm["verdict"] == "accept").sum())
    assert manifest["n_reject"] == int((jm["verdict"] == "reject").sum())
    assert manifest["n_parse_failed"] == 0
    assert (manifest["n_accept"] + manifest["n_reject"]
            + manifest["n_parse_failed"]) == manifest["n_judged"]
    assert manifest["n_gen_quality_movies"] == len(jg) == 5
    assert manifest["judge_model"] == cfg.audit.judge_model
    assert manifest["sample_rate"] == 1.0


def test_apply_filter_changes_scores(fake_stage_align):
    """reject 的映射不进重聚合：被拒 (movie, tag_id) 的分数改变 + 全帧独立重算一致。"""
    cfg, paths = fake_stage_align
    run_audit(cfg, paths, MockJudgeClient(), cache=None)

    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    calib = pd.read_parquet(paths.stage_align / "scores_calibrated.parquet")
    jm = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    sf = pd.read_parquet(paths.stage_audit / "scores_filtered.parquet")
    assert list(sf.columns) == list(SCORE_COLUMNS)

    calib_dict = json.loads(
        (paths.stage_align / "calibration.json").read_text(encoding="utf-8"))
    expected = _filtered_expected(mappings, jm, cfg.align.agg, calib_dict)
    pd.testing.assert_frame_equal(sf, expected)

    # brief 核心断言：至少一个被拒 (movie, tag_id) 上 filtered ≠ calibrated
    rejected_pairs = {(int(r.movie_id), int(r.tag_id)) for r in jm.itertuples()
                      if r.verdict == "reject"}
    assert rejected_pairs, "mock 哈希规则下应有被拒映射"
    merged = sf.merge(calib, on=["movie_id", "tag_id"], suffixes=("_f", "_c"))
    changed = merged[merged.apply(
        lambda r: (int(r["movie_id"]), int(r["tag_id"])) in rejected_pairs, axis=1)]
    diff = changed[changed["score_f"] != changed["score_c"]]
    assert len(diff) > 0


def test_sample_rate_samples(fake_stage_align):
    """sample_rate=0.5 → 判定行数 = ceil(0.5 × 映射数)，同 seed 两次运行一致。"""
    cfg, paths = fake_stage_align
    n = len(pd.read_parquet(paths.stage_align / "mappings.parquet"))
    cfg.audit.sample_rate = 0.5

    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    jm1 = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    assert len(jm1) == math.ceil(0.5 * n)
    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_judged"] == len(jm1)
    assert manifest["n_mappings"] == n                # n_mappings 记全量口径

    # 同 seed 重跑：抽样与判定结果逐帧一致
    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    jm2 = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    pd.testing.assert_frame_equal(jm1, jm2)

    # 抽样行是 mappings 的子集（按 movie_id, tag, tag_id 键）
    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    keys_m = set(zip(mappings["movie_id"], mappings["tag"], mappings["tag_id"]))
    keys_j = set(zip(jm1["movie_id"], jm1["tag"], jm1["tag_id"]))
    assert keys_j <= keys_m


def test_apply_filter_false_writes_no_scores_filtered(fake_stage_align):
    """apply_filter=false → 不生成 scores_filtered.parquet，其余产物照常。"""
    cfg, paths = fake_stage_align
    cfg.audit.apply_filter = False
    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    assert not (paths.stage_audit / "scores_filtered.parquet").exists()
    assert (paths.stage_audit / "judge_mapping.parquet").exists()
    assert (paths.stage_audit / "run_manifest.json").exists()


def test_apply_filter_requires_mapping_task(fake_stage_align):
    """judge_tasks 仅 gen_quality 而 apply_filter=true → 大声报错（无判定依据可过滤）。"""
    cfg, paths = fake_stage_align
    cfg.audit.judge_tasks = ["gen_quality"]
    cfg.audit.apply_filter = True
    with pytest.raises(ValueError, match="mapping"):
        run_audit(cfg, paths, MockJudgeClient(), cache=None)

    # 关闭过滤后可单跑 gen_quality：不产 judge_mapping / scores_filtered
    cfg.audit.apply_filter = False
    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    assert (paths.stage_audit / "judge_gen_quality.parquet").exists()
    assert not (paths.stage_audit / "judge_mapping.parquet").exists()
    assert not (paths.stage_audit / "scores_filtered.parquet").exists()
    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_judged"] == 0 and manifest["n_gen_quality_movies"] == 5


def test_parse_failed_recorded_and_kept(fake_stage_align):
    """不可解析输出 → verdict='parse_failed' 记录不重试；过滤时保守保留（不视为坏）。"""
    class GarbageJudge:
        model = "garbage-1"

        def complete(self, messages, run_id):
            from movietaggen.generate.client import LLMResponse
            return LLMResponse(content="完全不是 JSON", usage={},
                               raw={"garbage": True}, ok=True)

    cfg, paths = fake_stage_align
    run_audit(cfg, paths, GarbageJudge(), cache=None)

    mappings = pd.read_parquet(paths.stage_align / "mappings.parquet")
    jm = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    assert set(jm["verdict"]) == {"parse_failed"}
    assert (jm["reason"] == "").all()
    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_parse_failed"] == manifest["n_judged"] == len(mappings)

    # parse_failed 保守保留：scores_filtered == 全量聚合再校准（无 reject 可剔）
    sf = pd.read_parquet(paths.stage_audit / "scores_filtered.parquet")
    calib_dict = json.loads(
        (paths.stage_align / "calibration.json").read_text(encoding="utf-8"))
    expected = _filtered_expected(mappings, jm, cfg.align.agg, calib_dict)
    pd.testing.assert_frame_equal(sf, expected)


def test_calibration_sentinel_loud_error(fake_stage_align):
    """空实验哨兵（fitted=false）+ apply_filter=true → 大声报错，点名文件与哨兵。"""
    cfg, paths = fake_stage_align
    calib_path = paths.stage_align / "calibration.json"
    calib_path.write_text(
        json.dumps({"method": "isotonic", "fitted": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="哨兵") as excinfo:
        run_audit(cfg, paths, MockJudgeClient(), cache=None)
    assert str(calib_path) in str(excinfo.value)      # 报错点名校准文件


def test_empty_experiment_degrades_with_filter_off(fake_stage_align):
    """空实验（空 stage-gen → 空映射 + 哨兵）：apply_filter=true 报哨兵错；
    false 时合法空审计（空 schema 产物 + 零 manifest）。"""
    cfg, paths = fake_stage_align
    pd.DataFrame(columns=["movie_id", "tag", "n_runs"]).to_parquet(
        paths.stage_gen / "tags.parquet", index=False)
    run_alignment(cfg, paths, "configs/smoke.yaml")   # 重跑 05 → 空映射 + 哨兵

    with pytest.raises(ValueError, match="哨兵"):
        run_audit(cfg, paths, MockJudgeClient(), cache=None)   # apply_filter 默认 true

    cfg.audit.apply_filter = False
    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    jm = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    jg = pd.read_parquet(paths.stage_audit / "judge_gen_quality.parquet")
    assert len(jm) == 0 and list(jm.columns) == JUDGE_MAPPING_COLUMNS
    assert len(jg) == 0 and list(jg.columns) == JUDGE_GEN_COLUMNS
    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_mappings"] == manifest["n_judged"] == 0
    assert manifest["n_gen_quality_movies"] == 0


def test_gen_quality_skips_movies_without_user_tags(fake_stage_align):
    """无用户标签的电影跳过（不进 judge_gen_quality，也不进 n_gen_quality_movies）。"""
    cfg, paths = fake_stage_align
    tags = pd.read_csv(paths.movielens / "tags.csv")
    tags[tags["movieId"] != 3].to_csv(paths.movielens / "tags.csv", index=False)

    run_audit(cfg, paths, MockJudgeClient(), cache=None)
    jg = pd.read_parquet(paths.stage_audit / "judge_gen_quality.parquet")
    assert set(jg["movie_id"]) == {1, 2, 4, 5}
    manifest = json.loads(
        (paths.stage_audit / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_gen_quality_movies"] == 4


def test_pair_full_reads_movienet_content(fake_stage_align):
    """judge_context=pair+full → 评审 prompt 确实带上了 movienet 剧本/台词节选。"""
    class RecordingJudge:
        model = "mock-judge-1"

        def __init__(self):
            self._inner = MockJudgeClient()
            self.messages = []

        def complete(self, messages, run_id):
            self.messages.append(messages)
            return self._inner.complete(messages, run_id)

    cfg, paths = fake_stage_align
    cfg.audit.judge_context = "pair+full"
    recorder = RecordingJudge()
    run_audit(cfg, paths, recorder, cache=None)

    texts = ["\n".join(m["content"] for m in msgs if isinstance(m["content"], str))
             for msgs in recorder.messages]
    assert any("## 内容节选" in t and "剧本节选" in t for t in texts)
    assert any("台词节选" in t for t in texts)


def test_missing_tag_groups_loud_error(fake_stage_align):
    """mapping 判定需要 genome 定义：tag_groups.json 缺失 → 大声报错（不静默降级）。"""
    cfg, paths = fake_stage_align
    paths.tag_groups.unlink()
    with pytest.raises(FileNotFoundError, match="tag_groups"):
        run_audit(cfg, paths, MockJudgeClient(), cache=None)


def test_cache_roundtrip_and_mock_skips_cache(fake_stage_align, tmp_path):
    """mock=false：缓存写入 + 二跑零调用且结果一致；mock=true 不读不写缓存。"""
    class CountingJudge(MockJudgeClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, messages, run_id):
            self.calls += 1
            return super().complete(messages, run_id)

    cfg, paths = fake_stage_align
    n_mappings = len(pd.read_parquet(paths.stage_align / "mappings.parquet"))

    cfg.mock = False
    cache = ResponseCache(tmp_path / "llm-cache")
    c1 = CountingJudge()
    run_audit(cfg, paths, c1, cache)
    assert c1.calls == n_mappings + 5                 # 逐映射 + 逐电影
    cached = list(cache.root.glob("*.json"))
    assert cached                                     # 缓存已写入
    jm1 = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")

    c2 = CountingJudge()
    run_audit(cfg, paths, c2, cache)
    assert c2.calls == 0                              # 全量缓存命中
    jm2 = pd.read_parquet(paths.stage_audit / "judge_mapping.parquet")
    pd.testing.assert_frame_equal(jm1, jm2)

    # mock 配置不读不写缓存（确定性 + 不污染真实缓存，与生成侧同规）
    cfg.mock = True
    mock_cache = ResponseCache(tmp_path / "mock-cache")
    run_audit(cfg, paths, MockJudgeClient(), mock_cache)
    assert not list(mock_cache.root.glob("*.json"))


def test_judge_cache_key_extra_body_sensitivity():
    """I3：评审缓存键的采样参数指纹与 client.sampling_params 同构——extra_body
    变化必须换键（开/关思考不得互相命中）；未设置时与旧版
    {"temperature": ...} 键逐字节一致（存量缓存不失效）。"""
    from movietaggen.audit.judge import _sampling_params
    from movietaggen.generate.cache import cache_key

    msgs = [{"role": "user", "content": "hi"}]
    # 未设置 extra_body：与旧版手写 dict 的键完全一致
    assert cache_key(msgs, "m", _sampling_params(0.0, None), "judge-mapping") == \
        cache_key(msgs, "m", {"temperature": 0.0}, "judge-mapping")
    assert cache_key(msgs, "m", _sampling_params(0.0, {}), "judge-gen") == \
        cache_key(msgs, "m", {"temperature": 0.0}, "judge-gen")
    # extra_body 进键：开/关思考不得互相命中
    assert cache_key(msgs, "m", _sampling_params(0.0, {"enable_thinking": False}),
                     "judge-mapping") != \
        cache_key(msgs, "m", _sampling_params(0.0, None), "judge-mapping")
    # run_id 仍参与键：mapping 与 gen 两任务互不命中
    assert cache_key(msgs, "m", _sampling_params(0.0, {"enable_thinking": False}),
                     "judge-mapping") != \
        cache_key(msgs, "m", _sampling_params(0.0, {"enable_thinking": False}),
                  "judge-gen")
