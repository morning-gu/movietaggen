"""审计校准抽样导出与回填配对：LLM-as-judge 人工标注桥。

职责边界（控制器裁定）：
- export_calibration_sample：从 judge_mapping 产物抽 n 行组成人工标注表。
  列固定 8 列（CALIBRATION_COLUMNS），human_verdict 全空字符串留给人填；
  title / genome_tag 分别从 dataset（movie_id）/ genome_tags（tag_id）连接。
  抽样先按 (movie_id, tag, tag_id) 稳定排序再 random.Random(seed).sample：
  同 seed 同样本，跨运行、跨 pandas 版本一致（不用 df.sample——其结果依赖
  实现行序与实现内部顺序）；输出行序即排序序（人工核对友好）。judge 行的
  movie_id / tag_id 在参考表中缺失 → 中文 ValueError 点名缺失 id（审计产物
  与数据集必须一致——注 sheet 里静默 NaN 比崩溃更糟；校验覆盖全部 judge 行，
  在抽样之前做，坏行即使不被抽中也大声报错）。
- refill_kappa：读回人工填好的 CSV（utf-8），取 human_verdict 非空（去首尾
  空白）且 judge_verdict 为 accept/reject 的行（parse_failed 不构成有效配对，
  跳过），返回逐行对齐的 (judge_list, human_list)。任一侧为空、或任一侧
  全同类 → 返回 None（kappa 无定义，调用方据此跳过；kappa 本体在 Task 17
  的 metrics 模块，不在本模块）。每格读取时去首尾空白（人工填表难免手滑
  多敲空格）。

CSV 往返字符串保真：写出用 pandas 默认 quoting（含逗号的标题/标签自动加
引号），读回 dtype=str + keep_default_na=False——空单元格还原为 ""（而非
NaN），"NA" 类字面量不被吞。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

# 标注表固定 8 列（顺序即列序；judge 的 reason / judge_model 不进注 sheet）
CALIBRATION_COLUMNS = ["movie_id", "title", "tag", "tag_id", "genome_tag",
                       "cosine", "judge_verdict", "human_verdict"]

# 构成有效配对的 judge 判定值（parse_failed 行跳过）
_KAPPA_JUDGE_VERDICTS = ("accept", "reject")


def export_calibration_sample(judge_df: pd.DataFrame, dataset: pd.DataFrame,
                              genome_tags: pd.DataFrame, n: int,
                              seed: int) -> pd.DataFrame:
    """从判定结果抽 n 行组成人工标注表（列见 CALIBRATION_COLUMNS）。

    n > 行数时全量导出；同 seed 抽样逐行一致。
    """
    title_by_movie = dict(zip(dataset["movie_id"].astype(int),
                              dataset["title"]))
    tag_by_id = dict(zip(genome_tags["tag_id"].astype(int),
                         genome_tags["tag"]))
    ordered = judge_df.sort_values(
        ["movie_id", "tag", "tag_id"], kind="mergesort").reset_index(drop=True)

    # 一致性校验覆盖全部 judge 行（抽样前）：审计产物与参考表必须一致，
    # 缺失 → 中文报错点名 id，绝不静默产出带 NaN 的标注行
    for mid, tid in zip(ordered["movie_id"], ordered["tag_id"]):
        mid, tid = int(mid), int(tid)
        if mid not in title_by_movie:
            raise ValueError(
                f"judge_df 的 movie_id {mid} 不在 dataset 中"
                "（审计产物与数据集不一致，请核对 stage-audit 与 dataset "
                "是否出自同一实验）")
        if tid not in tag_by_id:
            raise ValueError(
                f"judge_df 的 tag_id {tid} 不在 genome_tags 中"
                "（审计产物与 genome 标签表不一致，请核对 stage-audit "
                "与 02 产物）")

    k = min(n, len(ordered))
    chosen = sorted(random.Random(seed).sample(range(len(ordered)), k))
    rows = []
    for i in chosen:
        r = ordered.iloc[i]
        mid, tid = int(r["movie_id"]), int(r["tag_id"])
        rows.append({
            "movie_id": mid,
            "title": str(title_by_movie[mid]),
            "tag": str(r["tag"]),
            "tag_id": tid,
            "genome_tag": str(tag_by_id[tid]),
            "cosine": float(r["cosine"]),
            "judge_verdict": str(r["verdict"]),
            "human_verdict": "",
        })
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def refill_kappa(csv_path: Path) -> tuple[list[str], list[str]] | None:
    """读回填后的标注 CSV，返回逐行对齐的 (judge_list, human_list)。

    行入选条件：human_verdict 非空（去空白）且 judge_verdict ∈
    {accept, reject}（parse_failed 行不构成有效配对）。任一侧为空、或
    任一侧全同类 → None（kappa 无定义，调用方据此跳过）。

    T15：encoding 用 utf-8-sig——Windows 记事本/Excel 重存 CSV 会带 BOM，
    普通 utf-8 会把 BOM 并进首列表头（"movie_id" 变 "\\ufeffmovie_id"）而
    KeyError；utf-8-sig 对无 BOM 的普通 UTF-8 逐字节等价。
    """
    raw = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str,
                      keep_default_na=False).fillna("")
    judge_list: list[str] = []
    human_list: list[str] = []
    for _, row in raw.iterrows():
        judge = str(row["judge_verdict"]).strip()
        human = str(row["human_verdict"]).strip()
        if not human or judge not in _KAPPA_JUDGE_VERDICTS:
            continue
        judge_list.append(judge)
        human_list.append(human)
    if not judge_list or not human_list:
        return None
    if len(set(judge_list)) < 2 or len(set(human_list)) < 2:
        return None
    return judge_list, human_list
