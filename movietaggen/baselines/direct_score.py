"""直接评分基线：A 的分组打分管道（annotate/pipeline.py）在 B 框架的完整复刻。

消融轴 9 的对照目标：与 B 的"生成→映射"两段式方法比较，A 的方法对
(movie × tag组) 任务直接产出逐 tag 关联度分。复刻要点：
- 任务矩阵 movie × group × run（baselines.n_runs，run_id = run-{r}）；
- prompt 组装 / 宽松解析 / 解析重试与 A 逐字一致（scoring_prompts.py 为
  A 的 prompts.py 原样复制，mock_scoring.py 为 A 的 mock.py 原样复制）；
- few-shot 从时序分割的训练侧（year < protocol.temporal_split_year，即 A
  fitting_pool_ids 的时序模式语义）选 top-k 分数和最高的电影，并以
  assert_no_leakage 收尾自检（eval.protocol 的全框架防泄露约束）；
- 缓存集成保留：键的 run 维度用 direct|{group}|{run_id} 命名空间；mock 模式
  不读不写缓存。

与 A 的差异（均已在行内注明）：jsonl 续跑简化为内存循环 + 直接返回 DataFrame
（落盘 baselines.parquet 由 07 脚本负责）；缓存仅收 ok 响应（同 generate
管道的既定裁定——失败信封入缓存会让续跑永久复播失败）。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from movietaggen.baselines.scoring_prompts import (
    FewShot,
    TagDef,
    build_messages,
    parse_scores_lenient,
)
from movietaggen.config import AppConfig
from movietaggen.data.grouping import load_tag_groups
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.eval.protocol import assert_no_leakage, temporal_split
from movietaggen.generate.cache import ResponseCache, cache_key
from movietaggen.generate.client import sampling_params

# 直接评分基线固定全模态（A 的默认语义）；不受 generate.modalities 配置影响
_BASELINE_MODALITIES = ["video", "script", "dialogue"]


def select_few_shots(dataset: pd.DataFrame, genome_scores: pd.DataFrame,
                     n: int, pool_ids, top_k: int,
                     tag_names: dict[int, str]) -> list[FewShot]:
    """从拟合池（pool_ids，防泄露约束）中选 top-k 分数和最高的 n 部。

    池语义见 eval.protocol.fitting_pool_ids（movie_list 补集或分割年前）。
    tag_names: tag_id → tag 文本（来自 tag_groups；few-shot 展示 tag 名而非 id）。
    """
    if n <= 0:
        return []
    pool = set(pool_ids)
    train = dataset[dataset["movie_id"].isin(pool)]
    scores = genome_scores[genome_scores["movie_id"].isin(pool)]
    if scores.empty:
        return []
    top_sums = (scores.groupby("movie_id")["relevance"]
                .apply(lambda s: s.nlargest(top_k).sum()))
    best = top_sums.sort_values(ascending=False).head(n).index.tolist()
    title_map = dict(zip(dataset["movie_id"], dataset["title"]))
    year_map = dict(zip(dataset["movie_id"], dataset["year"]))
    shots = []
    for movie_id in best:
        rows = genome_scores[genome_scores["movie_id"] == movie_id].nlargest(top_k, "relevance")
        shots.append(FewShot(
            title=title_map[movie_id], year=int(year_map[movie_id]),
            scores={tag_names[int(t)]: round(float(r), 2)
                    for t, r in zip(rows["tag_id"], rows["relevance"])}))
    return shots


def few_shot_movie_ids(dataset: pd.DataFrame, shots: list[FewShot]) -> set[int]:
    """few-shot 展示电影的 movie_id 集合（B 侧新增：assert_no_leakage 的输入）。

    A 的 FewShot 只携带 title/year（prompt 展示所需），不含 movie_id——
    此处按 (title, year) 回查 dataset 还原 id（与 select_few_shots 内部的
    title_map/year_map 同源；电影标题+年份在 MovieLens 中唯一）。
    """
    id_of = {(str(t), int(y)): int(m)
             for t, y, m in zip(dataset["title"], dataset["year"], dataset["movie_id"])
             if pd.notna(y)}
    return {id_of[(s.title, int(s.year))] for s in shots}


def _available_modalities(mm: MovieMultimodal, wanted: list[str]) -> list[str]:
    avail = []
    if "video" in wanted and mm.keyframes:
        avail.append("video")
    if "script" in wanted and mm.script:
        avail.append("script")
    if "dialogue" in wanted and mm.subtitles:
        avail.append("dialogue")
    return avail


def _to_multimodal(row, movienet_root: Path) -> MovieMultimodal:
    # 注意：混合 str/缺失列经 pandas3 string dtype + parquet 往返后缺失为 NaN（float）
    # —— NaN 为真值，必须用 pd.notna 守卫（真实数据 479/1035 有剧本的混合列实测触发）
    frames = [movienet_root / k for k in row["keyframes"]]
    script = None
    if pd.notna(row["script_path"]) and row["script_path"]:
        p = movienet_root / row["script_path"]
        script = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    subtitles = None
    if pd.notna(row["subtitles_path"]) and row["subtitles_path"]:
        p = movienet_root / row["subtitles_path"]
        subtitles = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    return MovieMultimodal(str(row["movienet_id"]), str(row["title"]),
                           int(row["year"]) if pd.notna(row["year"]) else None,
                           None, frames, script, subtitles)


def run_direct_score(config: AppConfig, paths, client,
                     cache: ResponseCache | None) -> pd.DataFrame:
    """跑完直接评分基线，返回稠密 (movie, tag) 分数表（列 movie_id, tag_id, score,
    method="direct_score"）；落盘 parquet 由 07 脚本负责。

    每部电影的某 (tag) 的分数 = 各轮宽松解析出分的均值；所有轮都未解析出
    （含无内容模态、解析失败）→ NaN（稠密行保留，交由评估侧处理）。
    """
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups)

    movie_ids = sorted(dataset["movie_id"].tolist())
    if config.data.movie_limit is not None and config.data.movie_limit < len(movie_ids):
        movie_ids = random.Random(config.data.seed).sample(movie_ids, config.data.movie_limit)

    # few-shot：只从时序分割的训练侧选取（A fitting_pool_ids 时序模式语义），
    # 选取后以防泄露断言收尾自检（全框架约束）
    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    train_ids, test_ids = temporal_split(dataset, config.protocol.temporal_split_year)
    few_shots = select_few_shots(
        dataset, genome_scores, config.baselines.few_shot_examples,
        sorted(train_ids), config.protocol.top_k, tag_names)
    assert_no_leakage(few_shot_movie_ids(dataset, few_shots), set(test_ids))

    groups_df = {g: df for g, df in tag_groups.groupby("group")}
    group_names = sorted(groups_df)
    movie_rows = {int(row["movie_id"]): row for _, row in dataset.iterrows()}

    tasks_total = len(movie_ids) * len(group_names) * config.baselines.n_runs
    print(f"直接评分基线：待跑 {tasks_total} 个任务"
          f"（movie × group × run, few_shot={len(few_shots)}）", flush=True)

    # 分数累积：(movie_id, tag_id) → 各轮解析出的分数列表（缺失轮不计入均值）
    acc: dict[tuple[int, int], list[float]] = {}
    processed = 0
    for movie_id in movie_ids:
        mm = _to_multimodal(movie_rows[movie_id], paths.movienet)
        available = _available_modalities(mm, _BASELINE_MODALITIES)
        # 仅当内容模态均不可得才跳过调用（A 的 no_content 语义）——该电影全部
        # (tag) 无任何轮解析出分数 → 稠密表中记 NaN。基线固定全模态，无 title-only 探针语义
        for group in group_names:
            tag_defs = [TagDef(int(r.tag_id), r.tag, r.definition)
                        for r in groups_df[group].itertuples()]
            expected = [t.tag for t in tag_defs]
            for r in range(config.baselines.n_runs):
                run_id = f"run-{r}"
                scores: dict[str, float] = {}
                if available:
                    messages = build_messages(mm, tag_defs, available, few_shots, config.model)
                    # 宽松解析重试（同 A）：仅全空才重试，换 eff_run 即换缓存键
                    for attempt in range(3):
                        eff_run = run_id if attempt == 0 else f"{run_id}#r{attempt}"
                        raw = None
                        if cache is not None and not config.mock:
                            key = cache_key(messages, config.model.model,
                                            sampling_params(config.model),
                                            f"direct|{group}|{eff_run}")
                            raw = cache.get(key)
                        if raw is None:
                            resp = client.complete(messages, run_id=eff_run)
                            raw = resp.raw
                            content = resp.content
                            # 与 A 的差异：仅缓存 ok 响应——失败信封（{"error": ...}）入缓存
                            # 会让续跑永久复播失败（同 generate 管道的既定裁定）
                            if cache is not None and not config.mock and resp.ok:
                                cache.put(key, raw)
                        else:
                            # 缓存命中：存的是完整原始响应信封，从信封还原文本
                            content = raw.get("choices", [{}])[0].get(
                                "message", {}).get("content", "")
                        scores = parse_scores_lenient(content, expected)
                        if scores:
                            break
                for td in tag_defs:
                    s = scores.get(td.tag)
                    if s is not None:
                        acc.setdefault((movie_id, int(td.tag_id)), []).append(float(s))
                processed += 1
                if processed % 50 == 0:
                    print(f"直接评分基线 进度 {processed}/{tasks_total}", flush=True)

    all_tag_ids = sorted(int(t) for t in tag_groups["tag_id"])
    rows = [{"movie_id": m, "tag_id": tid,
             "score": sum(acc[(m, tid)]) / len(acc[(m, tid)]) if (m, tid) in acc else float("nan"),
             "method": "direct_score"}
            for m in movie_ids for tid in all_tag_ids]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
