"""阶段产物装载与跨实验借用：物理复制 + PROVENANCE 溯源、缺失报错、配置快照。"""
import hashlib
import json

import pytest

from movietaggen.config import Paths
from movietaggen.stages import load_stage, save_config_snapshot


def _paths_for(tmp_path, exp="x") -> Paths:
    """指向 tmp_path 的 Paths（同 tests/test_align.py 的 _synthetic_paths 模式）。"""
    return Paths(
        root=tmp_path,
        movienet=tmp_path / "raw" / "movienet",
        movielens=tmp_path / "raw" / "movielens",
        interim=tmp_path / "interim", processed=tmp_path / "processed",
        cache_llm=tmp_path / ".cache" / "llm",
        cache_embed=tmp_path / ".cache" / "embed",
        results=tmp_path / "results" / exp,
        stage_gen=tmp_path / "results" / exp / "stage-gen",
        stage_align=tmp_path / "results" / exp / "stage-align",
        stage_audit=tmp_path / "results" / exp / "stage-audit",
        tag_groups=tmp_path / "processed" / "tag_groups.json",
        dataset=tmp_path / "processed" / "dataset.parquet",
        genome_scores=tmp_path / "processed" / "genome_scores.parquet",
        genome_tags=tmp_path / "processed" / "genome_tags.parquet",
        aligned=tmp_path / "interim" / "aligned.parquet",
        baselines_dir=tmp_path / "results" / "baselines")


def test_load_stage_local_missing_raises(tmp_path):
    """input_exp 为空且目录缺失 → 报错并给出补救命令。"""
    paths = _paths_for(tmp_path)
    with pytest.raises(FileNotFoundError) as ei:
        load_stage(paths, "stage-gen", None)
    assert "stage-gen" in str(ei.value)
    assert "04_run_generation" in str(ei.value)   # 信息里命名产出该阶段的脚本


def test_load_stage_local_exists_returns(tmp_path):
    paths = _paths_for(tmp_path)
    (paths.results / "stage-gen").mkdir(parents=True)
    (paths.results / "stage-gen" / "tags.parquet").write_bytes(b"x")
    out = load_stage(paths, "stage-gen", None)
    assert out == paths.results / "stage-gen"


def test_load_stage_borrow_copies_with_provenance(tmp_path):
    """借用：从其它实验物理复制 + 写 PROVENANCE.json（source_exp、逐文件 sha256）。"""
    paths = _paths_for(tmp_path, exp="consumer")
    src = tmp_path / "results" / "donor" / "stage-gen"
    src.mkdir(parents=True)
    (src / "responses.jsonl").write_text('{"task_key": "1|run-0"}\n', encoding="utf-8")
    (src / "tags.parquet").write_bytes(b"PARQUET-BYTES")
    (src / "nested").mkdir()                      # 子目录也一并复制、逐文件记入溯源
    (src / "nested" / "extra.txt").write_text("n", encoding="utf-8")

    out = load_stage(paths, "stage-gen", "donor")
    assert out == paths.results / "stage-gen"
    assert (out / "responses.jsonl").read_text(encoding="utf-8") == '{"task_key": "1|run-0"}\n'
    assert (out / "tags.parquet").read_bytes() == b"PARQUET-BYTES"
    assert (out / "nested" / "extra.txt").read_text(encoding="utf-8") == "n"

    prov = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["source_exp"] == "donor"
    assert prov["stage"] == "stage-gen"
    assert prov["copied_at"]
    by_name = {f["name"]: f["sha256"] for f in prov["files"]}
    assert by_name["responses.jsonl"] == hashlib.sha256(
        (src / "responses.jsonl").read_bytes()).hexdigest()
    assert by_name["tags.parquet"] == hashlib.sha256(b"PARQUET-BYTES").hexdigest()
    assert by_name["nested/extra.txt"] == hashlib.sha256(b"n").hexdigest()


def test_load_stage_borrow_replaces_stale_local(tmp_path):
    """借用前清空本地：上次借用的旧文件不残留（本地自包含 = 与源一致）。"""
    paths = _paths_for(tmp_path, exp="consumer")
    local = paths.results / "stage-gen"
    local.mkdir(parents=True)
    (local / "stale.txt").write_text("old", encoding="utf-8")
    src = tmp_path / "results" / "donor" / "stage-gen"
    src.mkdir(parents=True)
    (src / "fresh.txt").write_text("new", encoding="utf-8")

    out = load_stage(paths, "stage-gen", "donor")
    assert not (out / "stale.txt").exists()
    assert (out / "fresh.txt").exists()


def test_load_stage_self_borrow_rejected(tmp_path):
    """input_exp = 本实验名 → 拒绝（否则清空本地后从已被删的源复制，数据丢失）。"""
    paths = _paths_for(tmp_path, exp="x")
    with pytest.raises(ValueError):
        load_stage(paths, "stage-gen", "x")       # paths.results.name == "x"


def test_load_stage_borrow_source_missing(tmp_path):
    paths = _paths_for(tmp_path, exp="consumer")
    with pytest.raises(FileNotFoundError):
        load_stage(paths, "stage-gen", "nonexistent")


def test_save_config_snapshot(tmp_path):
    paths = _paths_for(tmp_path)
    cfg = tmp_path / "my.yaml"
    cfg.write_text("exp_name: x\n", encoding="utf-8")
    dest = save_config_snapshot(cfg, paths)
    assert dest == paths.results / "config.yaml"
    assert dest.read_text(encoding="utf-8") == "exp_name: x\n"


def test_save_config_snapshot_missing_source_noop(tmp_path):
    """A 的 04 行为：配置文件不存在（如测试内构造的 AppConfig）→ 静默跳过。"""
    paths = _paths_for(tmp_path)
    assert save_config_snapshot(tmp_path / "nope.yaml", paths) is None
    assert not (paths.results / "config.yaml").exists()


def test_load_stage_self_borrow_nested_name_rejected(tmp_path):
    """I4：嵌套命名的自借用必须被拒——exp_name="ablation/video-only" 时
    paths.results.name 只是 "video-only"，按名字符串比较会漏判并在 rmtree
    清空本地后毁掉本实验产物；按解析路径比较则正确拦截（守卫先于任何
    复制/删除动作触发，无需预先创建目录）。"""
    paths = _paths_for(tmp_path, exp="ablation/video-only")
    with pytest.raises(ValueError, match="借用本实验自己"):
        load_stage(paths, "stage-gen", "ablation/video-only")


def test_load_stage_nested_name_borrow_other_exp_works(tmp_path):
    """I4：嵌套命名下借用**不同的**实验仍正常（守卫不得误伤合法借用）。"""
    paths = _paths_for(tmp_path, exp="ablation/consumer")
    src = tmp_path / "results" / "ablation" / "donor" / "stage-gen"
    src.mkdir(parents=True)
    (src / "tags.parquet").write_bytes(b"PARQUET-BYTES")

    out = load_stage(paths, "stage-gen", "ablation/donor")
    assert (out / "tags.parquet").read_bytes() == b"PARQUET-BYTES"
    prov = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["source_exp"] == "ablation/donor"


def test_load_stage_provenance_excludes_itself(tmp_path):
    """M1a：链式借用时源目录带着上一棒的 PROVENANCE.json——它不得进入逐文件
    校验清单（本目录会写入自己的新 PROVENANCE.json，旧校验和必然自我失真）。"""
    paths = _paths_for(tmp_path, exp="consumer")
    src = tmp_path / "results" / "donor" / "stage-gen"
    src.mkdir(parents=True)
    (src / "tags.parquet").write_bytes(b"PARQUET-BYTES")
    (src / "PROVENANCE.json").write_text('{"stale": true}', encoding="utf-8")

    out = load_stage(paths, "stage-gen", "donor")
    prov = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in prov["files"]]
    assert "PROVENANCE.json" not in names               # 自身不入清单
    assert names == ["tags.parquet"]                    # 其余文件照常在册
    assert not (out / "PROVENANCE.json").read_text(encoding="utf-8") \
        == '{"stale": true}'                            # 落盘的是新溯源


def test_load_stage_provenance_config_sha256(tmp_path):
    """M1b（spec §1.3）：溯源携带源实验 config.yaml 的 sha256（可对账借用
    产物与源配置版本）；源无 config.yaml → null。"""
    paths = _paths_for(tmp_path, exp="consumer")
    src = tmp_path / "results" / "donor" / "stage-gen"
    src.mkdir(parents=True)
    (src / "tags.parquet").write_bytes(b"PARQUET-BYTES")

    out = load_stage(paths, "stage-gen", "donor")
    prov = json.loads((out / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov["config_sha256"] is None                # 源无 config.yaml

    (tmp_path / "results" / "donor" / "config.yaml").write_text(
        "exp_name: donor\n", encoding="utf-8")
    out2 = load_stage(paths, "stage-gen", "donor")
    prov2 = json.loads((out2 / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert prov2["config_sha256"] == hashlib.sha256(
        (tmp_path / "results" / "donor" / "config.yaml").read_bytes()).hexdigest()
