"""tags.csv 自由标签分布分析（B 特有，无 A 对应物）。

三重用途：五维度提示词的离线设计先验、生成质量评估的参考分布、
论文 Intro 章节的证据。归一口径：仅 casefold（大小写折叠），不做
strip/去标点/拼写纠错——错拼变体（如 "neon visualz"）保留为独立
标签，因此与 genome 的字面重合是严格下界；嵌入口径的重叠留给评估层
（spec §3.2 注：嵌入重叠在 eval 报告中体现）。
"""
from __future__ import annotations

import json

import pandas as pd

from movietaggen.config import Paths
from movietaggen.data.movielens import load_user_tags

# 固定桶键：始终全量输出（空桶计 0），保证下游 schema 稳定。
YEAR_BUCKETS = ("<1980", "1980-1999", "2000-2009", "2010-2019", ">=2020", "unknown")


def _year_bucket(year) -> str:
    """年份 → 分桶键；空值 → unknown。边界：[1980,1999] / [2000,2009] / [2010,2019]。"""
    if year is None or pd.isna(year):
        return "unknown"
    y = int(year)
    if y < 1980:
        return "<1980"
    if y <= 1999:
        return "1980-1999"
    if y <= 2009:
        return "2000-2009"
    if y <= 2019:
        return "2010-2019"
    return ">=2020"


def _top_tag_counts(tags: pd.Series, top_n: int) -> list[tuple[str, int]]:
    """标签序列 → casefold 归一计数的前 top_n：行数降序、同频按标签升序。"""
    counts = tags.str.casefold().value_counts()
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def freetag_stats(user_tags: pd.DataFrame, dataset: pd.DataFrame,
                  top_n: int = 50) -> dict:
    """自由标签分布统计。user_tags 先过滤到 dataset 电影集合，再统计。

    返回键：
    - n_unique: 唯一标签数（casefold 归一）
    - n_rows: 实验集内标签行数
    - tags_per_movie_mean: 平均每部被标注电影的标签数（分母 = 过滤后
      至少有一条标签的电影；未被标注的电影不拉低均值）
    - top_tags: [[tag, count], ...]（tag 为 casefold 形式；行数降序、同频升序）
    - year_buckets: 各年代桶的标签行数（按 dataset.year 归桶；空年份 → unknown）
    """
    tags = user_tags[user_tags["movie_id"].isin(set(dataset["movie_id"]))]
    n_rows = len(tags)
    per_movie = tags.groupby("movie_id").size()
    year_by_movie = dict(zip(dataset["movie_id"], dataset["year"]))
    buckets = {k: 0 for k in YEAR_BUCKETS}
    for mid, cnt in per_movie.items():
        buckets[_year_bucket(year_by_movie[mid])] += int(cnt)
    return {
        "n_unique": int(tags["tag"].str.casefold().nunique()),
        "n_rows": n_rows,
        "tags_per_movie_mean": float(per_movie.mean()) if n_rows else 0.0,
        "top_tags": [[t, int(c)] for t, c in _top_tag_counts(tags["tag"], top_n)],
        "year_buckets": buckets,
    }


def literal_overlap(user_tags: pd.DataFrame, genome_tags: pd.DataFrame,
                    top_n: int = 50) -> dict:
    """高频自由标签（casefold）与 genome 标签集合（casefold）的严格字面重合。

    返回键：top_n_total（参与比较的高频标签数）、exact_match（字面一致数）、
    match_list: [[自由标签, genome 原文], ...]（自由标签为 casefold 形式；
    genome 侧同形多写时取首次出现的原文）。错拼近失（如 "neon visualz"）
    不计入——字面口径是重合下界，嵌入口径的重叠留给评估层。
    """
    top = _top_tag_counts(user_tags["tag"], top_n)
    genome_original: dict[str, str] = {}
    for folded, original in zip(genome_tags["tag"].str.casefold(), genome_tags["tag"]):
        genome_original.setdefault(folded, original)
    matches = [[t, genome_original[t]] for t, _ in top if t in genome_original]
    return {
        "top_n_total": len(top),
        "exact_match": len(matches),
        "match_list": matches,
    }


def build_report(stats: dict, overlap: dict, movie_count: int | None = None) -> str:
    """渲染中文 markdown 摘要：概况、Top-10 表、年份分桶表、字面重合摘要。"""
    lines = ["# 自由标签分布分析报告", "", "## 概况", ""]
    if movie_count is not None:
        lines.append(f"- 实验集电影数: {movie_count}")
    lines += [
        f"- 标签行数（实验集内）: {stats['n_rows']}",
        f"- 唯一标签数（casefold 归一）: {stats['n_unique']}",
        f"- 平均每部被标注电影标签数: {stats['tags_per_movie_mean']:.2f}",
        "",
        "## 高频标签 Top 10",
        "",
    ]
    if stats["top_tags"]:
        lines += ["| 排名 | 标签 | 行数 |", "|---:|---|---:|"]
        lines += [f"| {i} | {t} | {c} |"
                  for i, (t, c) in enumerate(stats["top_tags"][:10], 1)]
    else:
        lines.append("（无标签行）")
    lines += ["", "## 年份分桶（标签行数）", "", "| 年代 | 标签行数 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in stats["year_buckets"].items()]
    ratio = (overlap["exact_match"] / overlap["top_n_total"] * 100
             if overlap["top_n_total"] else 0.0)
    lines += [
        "",
        "## 与 Genome 标签的字面重合",
        "",
        "- 口径: tags.csv 全量行的高频自由标签（casefold 归一，未按实验集过滤）",
        f"- 参与比较的高频自由标签数: {overlap['top_n_total']}",
        f"- 字面一致数: {overlap['exact_match']}（{ratio:.1f}%）",
    ]
    if overlap["match_list"]:
        shown = overlap["match_list"][:10]
        more = f"（等共 {overlap['exact_match']} 个）" if overlap["exact_match"] > len(shown) else ""
        lines.append("- 一致清单: " + "；".join(f"{t} ↔ {g}" for t, g in shown) + more)
    else:
        lines.append("- 一致清单: 无")
    lines.append("")
    return "\n".join(lines)


def analyze_freetags(paths: Paths) -> dict | None:
    """读 dataset parquet + tags.csv + genome parquet → 统计、重合、写 interim 产物。

    产物：paths.interim/freetag_stats.json（两段式负载）与
    paths.interim/freetag_report.md。tags.csv 缺失 → 返回 None（调用方
    提示跳过）；dataset / genome parquet 缺失（未跑 02）→ FileNotFoundError。
    返回 {"freetag_stats": ..., "literal_overlap": ...}（与 JSON 负载同构）。
    """
    if not paths.dataset.exists():
        raise FileNotFoundError(f"未找到数据集 {paths.dataset}（请先运行 02_build_dataset.py）")
    if not paths.genome_tags.exists():
        raise FileNotFoundError(f"未找到 genome 标签 {paths.genome_tags}（请先运行 02_build_dataset.py）")
    tags_path = paths.movielens / "tags.csv"
    if not tags_path.exists():
        return None
    dataset = pd.read_parquet(paths.dataset)
    user_tags = load_user_tags(tags_path)
    genome_tags = pd.read_parquet(paths.genome_tags)

    stats = freetag_stats(user_tags, dataset)
    overlap = literal_overlap(user_tags, genome_tags)

    paths.interim.mkdir(parents=True, exist_ok=True)
    payload = {"freetag_stats": stats, "literal_overlap": overlap}
    (paths.interim / "freetag_stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (paths.interim / "freetag_report.md").write_text(
        build_report(stats, overlap, movie_count=len(dataset)), encoding="utf-8")
    return payload
