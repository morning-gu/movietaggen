"""直接评分基线：A 的分组打分管道（annotate/pipeline.py）在 B 框架的完整复刻。

消融轴 9 的对照目标：与 B 的"生成→映射"两段式方法比较，A 的方法对
(movie × tag组) 任务直接产出逐 tag 关联度分。复刻要点：
- 任务矩阵 movie × group × run（baselines.n_runs，run_id = run-{r}）；
- prompt 组装 / 宽松解析 / 解析重试与 A 逐字一致（scoring_prompts.py 为
  A 的 prompts.py 原样复制，mock_scoring.py 为 A 的 mock.py 原样复制）；
- few-shot 从固定分割的训练侧（train = dataset − processed/eval_test.txt 清单，
  440 协议；真实数据为 A 迁移的 88 部固定测试集）选 top-k 分数和最高的电影，
  并以 assert_no_leakage 收尾自检（eval.protocol 的全框架防泄露约束）；
- 缓存集成保留：键的 run 维度用 direct|{group}|{run_id} 命名空间；mock 模式
  不读不写缓存。

与 A 的差异（均已在行内注明）：jsonl 续跑简化为内存循环 + 直接返回 DataFrame
（落盘 baselines.parquet 由 07 脚本负责）；缓存仅收 ok 响应（同 generate
管道的既定裁定——失败信封入缓存会让续跑永久复播失败）。
"""
from __future__ import annotations

import pandas as pd

from movietaggen.baselines.scoring_prompts import (
    FewShot,
    TagDef,
    build_messages,
    parse_scores_lenient,
)
from movietaggen.config import AppConfig
from movietaggen.data.grouping import load_tag_groups
from movietaggen.eval.protocol import (
    assert_no_leakage,
    fixed_split,
    sample_limited_movie_ids,
)
from movietaggen.generate.cache import ResponseCache, cache_key
from movietaggen.generate.client import sampling_params
from movietaggen.generate.content import available_modalities, to_multimodal

# 直接评分基线固定全模态（A 的默认语义）；不受 generate.modalities 配置影响
_BASELINE_MODALITIES = ["video", "script", "dialogue"]


def select_few_shots(dataset: pd.DataFrame, genome_scores: pd.DataFrame,
                     n: int, pool_ids, top_k: int,
                     tag_names: dict[int, str]) -> list[FewShot]:
    """从拟合池（pool_ids，防泄露约束）中选 top-k 分数和最高的 n 部。

    池语义由调用方给定（own_run 为 fixed_split 的训练侧；选取后由
    assert_no_leakage 收尾自检）。
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


def load_direct_score_a(paths, dataset: pd.DataFrame) -> pd.DataFrame:
    """读 A 的 direct_score 评分迁移物，聚合为稠密 (movie, tag) 分数表。

    a_borrow 分支的数据入口（07 按配置分流；own_run 不经此函数）。输入
    parquet 列 movie_id, tag_id, group, modality_set, run_id, score（逐轮
    原始分）。守门与口径：
    - 文件缺失 → ValueError 点名路径与成因（config 层已拦 synthetic+a_borrow，
      但 direct 调用方也可能踩，此处再拦一道）；
    - 电影全集校验（Review Focus 1）：迁移物 movie_id 必须 ⊆
      dataset["movie_id"]，越界点名缺失电影并提示清单/数据集版本错配；
    - modality_set 唯一值必须是 video|script|dialogue（A 的全模态默认），
      其它值点名实际值集；
    - 均值口径：groupby(["movie_id","tag_id"])["score"].mean()——pandas mean
      默认 skipna，天然实现"仅计已解析轮均值、全空 → NaN"（B 既有裁定，
      与 own_run 的逐轮累计-均值语义一致）；
    - 返回列 movie_id, tag_id, score, method="direct_score"，与
      run_direct_score 输出同构（07 落盘 baselines.parquet 无感知）。
    """
    if not paths.direct_score_a.is_file():
        raise ValueError(
            f"direct_score_a 迁移物不存在: {paths.direct_score_a}"
            "（baselines.direct_score_source=a_borrow 需要 A 的评分迁移物；"
            "synthetic 数据没有对应产物，请改用 own_run 或核对 processed 目录）")
    raw = pd.read_parquet(paths.direct_score_a)

    missing = sorted(set(raw["movie_id"]) - set(dataset["movie_id"]))
    if missing:
        raise ValueError(
            f"direct_score_a 含数据集外的电影（前 {min(5, len(missing))} 个）: "
            f"{missing[:5]}——迁移物清单与数据集版本错配，"
            "请核对两者 movie_id 宇宙是否同批生成")
    got_modalities = set(raw["modality_set"].unique())
    expected_modalities = {"video|script|dialogue"}
    if got_modalities != expected_modalities:
        raise ValueError(
            f"direct_score_a 的 modality_set 应只有 {sorted(expected_modalities)}，"
            f"实际值集: {sorted(map(str, got_modalities))}")

    agg = (raw.groupby(["movie_id", "tag_id"], as_index=False)["score"].mean())
    agg["method"] = "direct_score"
    return agg[["movie_id", "tag_id", "score", "method"]]


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

    # 受限跑抽样走 sample_limited_movie_ids（440 协议）：从固定分割训练侧
    # 抽样，测试清单电影绝不入样；movie_limit=None → 全量升序（不读清单）
    movie_ids = sample_limited_movie_ids(
        dataset, config.data.movie_limit, config.data.seed, paths.eval_test)

    # few-shot：只从固定分割的训练侧选取（train = dataset − eval_test 清单；
    # 真实数据为 A 迁移的 88 部固定测试集），选取后以防泄露断言收尾自检
    # （全框架约束）
    tag_names = dict(zip(tag_groups["tag_id"], tag_groups["tag"]))
    train_ids, test_ids = fixed_split(dataset, paths.eval_test)
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
        mm = to_multimodal(movie_rows[movie_id], paths.movienet)
        available = available_modalities(mm, _BASELINE_MODALITIES)
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
