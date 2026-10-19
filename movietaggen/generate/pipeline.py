"""阶段一生成编排：电影 × 轮次 任务矩阵，断点续跑，可选并行。

改编自 A（genometag）的 annotate/pipeline.py：打分管道 → 自由标签生成管道。
responses.jsonl 是续跑的事实来源（append-only，每任务一行，行级 flush——
看门狗以 jsonl 增长判活）；结束时跨轮并集去重压缩为 stage-gen/tags.parquet
供对齐阶段使用。任务相互独立，workers>1 时线程池并行与串行结果等价（有测试钉死）。
mock 模式不读不写缓存（确定性 + 不污染真实缓存）。

与 A 的结构差异：任务矩阵无 group 维度——自由标签生成不预设词表，任务 =
movie × run-{r}；解析产出 list[str] 标签（三级降级解析）而非逐 tag 分数。
few-shot 不在本任务接线（默认 []），参数已贯通（后续任务传示例即可）。
"""
from __future__ import annotations

import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movietaggen.config import AppConfig
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.generate.cache import ResponseCache, cache_key
from movietaggen.generate.client import sampling_params
from movietaggen.generate.prompts import (
    FewShotTags,
    build_gen_messages,
    normalize_tag,
    parse_tags,
)


@dataclass(frozen=True)
class TaskSpec:
    movie_id: int
    run_id: str


def build_task_matrix(movie_ids: list[int], n_runs: int) -> list[TaskSpec]:
    return [TaskSpec(m, f"run-{r}") for m in movie_ids for r in range(n_runs)]


def _read_jsonl_lines(jsonl_path: Path) -> tuple[list[dict], int]:
    """逐行解析 jsonl，容忍中断残留的截断/损坏行（跳过并计数）。"""
    good: list[dict] = []
    bad = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"[04] 跳过损坏的 jsonl 行: {bad}（上次运行中断残留，相关任务将重跑）")
    return good, bad


def run_generation(config: AppConfig, paths, client, cache: ResponseCache | None,
                   dry_run: bool = False,
                   few_shots: list[FewShotTags] | None = None) -> Path | None:
    """跑完阶段一生成，返回 tags.parquet 路径（dry-run 返回 None 且不写文件）。"""
    dataset = pd.read_parquet(paths.dataset)

    movie_ids = sorted(dataset["movie_id"].tolist())
    if config.data.movie_limit is not None and config.data.movie_limit < len(movie_ids):
        movie_ids = random.Random(config.data.seed).sample(movie_ids, config.data.movie_limit)

    tasks = build_task_matrix(movie_ids, config.generate.n_runs)
    movie_rows = {int(row["movie_id"]): row for _, row in dataset.iterrows()}
    few_shots = few_shots or []

    if dry_run:
        # spec 的 --dry-run：打印前 3 个任务的 prompt 文本，不调用、不写文件
        for task in tasks[:3]:
            mm = _to_multimodal(movie_rows[task.movie_id], paths.movienet)
            available = _available_modalities(mm, config.generate.modalities)
            messages = build_gen_messages(
                mm, config.generate.dimensions, available, few_shots, config.model)
            texts = []
            for m in messages:
                c = m["content"]
                if isinstance(c, str):
                    texts.append(c)
                else:
                    texts.extend(p.get("text", "") for p in c if p.get("type") == "text")
            print(f"[dry-run] 任务 {task.movie_id}|{task.run_id}\n"
                  + "\n".join(texts)[:1500] + "\n")
        return None

    paths.stage_gen.mkdir(parents=True, exist_ok=True)
    jsonl_path = paths.stage_gen / "responses.jsonl"
    done_keys: set[str] = set()
    if jsonl_path.exists():
        records, _ = _read_jsonl_lines(jsonl_path)
        done_keys = {r["task_key"] for r in records}
    todo = [t for t in tasks if f"{t.movie_id}|{t.run_id}" not in done_keys]
    total = len(todo)
    n_workers = max(1, config.generate.workers)
    if total:
        print(f"[04] 待跑 {total} 个任务（续跑跳过 {len(done_keys)}），"
              f"workers={n_workers}", flush=True)

    cost = {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    counts = {"done": 0, "parse_failed": 0, "no_content": 0}
    lock = threading.Lock()

    with jsonl_path.open("a", encoding="utf-8") as sink:
        def process(task: TaskSpec) -> None:
            """单任务：构建 prompt → 调用（缓存优先）→ 解析；落盘在锁内。"""
            task_key = f"{task.movie_id}|{task.run_id}"
            mm = _to_multimodal(movie_rows[task.movie_id], paths.movienet)
            available = _available_modalities(mm, config.generate.modalities)
            call_cost = {"total_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
            # 仅当"请求了内容模态但均不可得"才跳过；modalities=[] 是 title-only 探针的显式意图
            if not available and config.generate.modalities:
                status, tags, usage = "no_content", [], {}
                line = json.dumps({"task_key": task_key, "status": status,
                                   "tags": tags, "usage": usage}, ensure_ascii=False)
            else:
                messages = build_gen_messages(
                    mm, config.generate.dimensions, available, few_shots, config.model)

                # 解析重试（mock/真实皆可能偶发输出不可解析——重试换 run 后缀换缓存键）
                tags: list[str] = []
                usage: dict = {}
                for attempt in range(3):
                    eff_run = task.run_id if attempt == 0 else f"{task.run_id}#r{attempt}"
                    raw = None
                    if cache is not None and not config.mock:
                        key = cache_key(messages, config.model.model,
                                        sampling_params(config.model), eff_run)
                        raw = cache.get(key)
                    if raw is None:
                        resp = client.complete(messages, run_id=eff_run)
                        raw = resp.raw
                        content = resp.content
                        usage = resp.usage
                        call_cost["total_calls"] += 1
                        call_cost["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        call_cost["completion_tokens"] += usage.get("completion_tokens", 0)
                        # 与 A 的差异：仅缓存 ok 响应——失败信封（{"error": ...}）入缓存
                        # 会让续跑永久复播失败，违背缓存"断点自愈"的设计意图
                        if cache is not None and not config.mock and resp.ok:
                            cache.put(key, raw)
                    else:
                        # 缓存命中：存的是完整原始响应信封，从信封还原文本
                        content = raw.get("choices", [{}])[0].get(
                            "message", {}).get("content", "")
                    tags = parse_tags(content)
                    if tags:
                        break

                status = "done" if tags else "parse_failed"
                line = json.dumps({"task_key": task_key, "status": status,
                                   "tags": tags, "usage": usage}, ensure_ascii=False)

            with lock:
                sink.write(line + "\n")
                sink.flush()   # 行级落盘：看门狗以 jsonl 增长判活（并行长跑必需）
                counts[status] += 1
                for k in call_cost:
                    cost[k] += call_cost[k]
                processed = counts["done"] + counts["parse_failed"] + counts["no_content"]
                if processed % 25 == 0 or processed == total:
                    print(f"[04] 进度 {processed}/{total}"
                          f"（done={counts['done']} failed={counts['parse_failed']}）",
                          flush=True)

        if n_workers == 1:
            for task in todo:
                process(task)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(process, t) for t in todo]
                for f in futures:
                    f.result()

    # 压缩 jsonl → tags.parquet（含历史行，防丢；损坏行跳过，不阻塞续跑）
    records, _ = _read_jsonl_lines(jsonl_path)
    rows = _union_tags(records)
    parquet_path = paths.stage_gen / "tags.parquet"
    pd.DataFrame(rows, columns=["movie_id", "tag", "n_runs"]).to_parquet(
        parquet_path, index=False)

    (paths.stage_gen / "run_manifest.json").write_text(
        json.dumps({"tasks": counts, **cost, "n_tags_total": len(rows)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return parquet_path


def _union_tags(records: list[dict]) -> list[dict]:
    """跨轮并集去重：(movie, normalize_tag) 唯一，n_runs = 产出该标签的轮数。

    保留首次出现的原形（按 jsonl 行序 = 并行时的任务完成序）——实验产物
    固化后不漂移，这是有意为之；归一化键才跨轮稳定，原形只取代表。
    """
    by_key: dict[tuple[int, str], dict] = {}
    for r in records:
        if r.get("status") != "done":
            continue
        movie_id, run_id = str(r["task_key"]).split("|", 1)
        for tg in r.get("tags") or []:
            entry = by_key.setdefault((int(movie_id), normalize_tag(tg)),
                                      {"tag": tg, "runs": set()})
            entry["runs"].add(run_id)
    return [{"movie_id": m, "tag": e["tag"], "n_runs": len(e["runs"])}
            for (m, _), e in sorted(by_key.items())]


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
