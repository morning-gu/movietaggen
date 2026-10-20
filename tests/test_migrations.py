"""A 仓库迁移物入库校验（440 宇宙协议升级 Task 1）。

三份迁移物物理入库后钉住其内容契约：eval_test.txt（88 部测试清单）、
direct_score_a.parquet（A 的 v3-full 实跑预测）、provenance JSON（sha256
与结构字段，后续 Task 6 加载器按此契约引用）。任何入库后漂移在此报警。
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eval_test_txt_format():
    """A 的测试清单：恰好 88 行、每行可解析 int、无重复 movieId。"""
    lines = (PROCESSED / "eval_test.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 88                     # 行数钉死 88
    ids = [int(line) for line in lines]         # 空行/非整数 → ValueError 报错
    assert len(set(ids)) == 88                  # 无重复


def test_direct_score_a_parquet_shape():
    """A v3-full 预测：297,792 行 = 88 电影 × 1,128 tag × 3 run；列名与顺序、
    modality_set 取值、每 (movie_id, tag_id) 恰 3 行、覆盖清单 == eval_test。"""
    df = pd.read_parquet(PROCESSED / "direct_score_a.parquet")
    assert list(df.columns) == ["movie_id", "tag_id", "group",
                                "modality_set", "run_id", "score"]
    assert len(df) == 88 * 1128 * 3
    assert df["movie_id"].nunique() == 88
    assert df["tag_id"].nunique() == 1128
    assert set(df["modality_set"].unique()) == {"video|script|dialogue"}
    counts = df.groupby(["movie_id", "tag_id"]).size()
    assert (counts == 3).all()                  # 每对 (movie, tag) 恰 3 个 run

    eval_ids = {int(line) for line
                in (PROCESSED / "eval_test.txt").read_text(encoding="utf-8").splitlines()}
    assert set(df["movie_id"].astype(int)) == eval_ids   # 覆盖清单与 eval_test 一致


def test_provenance_sha_matches():
    """provenance 的 sha256 与实际文件现算一致（防漂移）；Task 6 依赖的结构
    字段齐全，expected 与 raw_jsonl_sha256 钉住控制器给定的真值。"""
    text = (PROCESSED / "direct_score_a.provenance.json").read_text(encoding="utf-8")
    prov = json.loads(text)
    assert prov["direct_score_a_sha256"] == _sha256(PROCESSED / "direct_score_a.parquet")
    assert prov["eval_test_sha256"] == _sha256(PROCESSED / "eval_test.txt")

    # Task 6 加载器契约：字段齐全 + 期望形状 + 未迁移的 annotations.jsonl 真值
    assert set(prov) == {"source_repo", "copied_date", "eval_test_sha256",
                         "direct_score_a_sha256", "expected", "raw_jsonl_sha256",
                         "v3full_config", "v3full_run_manifest", "v3full_cost_report"}
    assert prov["expected"] == {"rows": 297792, "movies": 88,
                                "tags": 1128, "runs": 3}
    assert prov["raw_jsonl_sha256"] == \
        "2ccae7ff02ff2e72d280a15c07ce7324d282cc5ff952df12a86de8c99e668ae1"
    for key in ("v3full_config", "v3full_run_manifest", "v3full_cost_report"):
        assert isinstance(prov[key], str) and prov[key]   # 三段原文文本非空
