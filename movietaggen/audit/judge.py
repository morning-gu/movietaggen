"""LLM-as-judge 审计（双任务）：映射对齐判定 + 生成质量打分 + 过滤重聚合。

两条任务线（cfg.audit.judge_tasks 子集）：
- mapping：逐 (自由标签, genome 候选) 映射评审 prompt（三档上下文 pair /
  pair+title / pair+full，定义来自 tag_groups.json）→ parse_mapping_verdict；
  解析失败记 verdict="parse_failed"（不重试，调用方口径如实在册）。
- gen_quality：逐电影 stage-gen 标签 vs 用户自由标签（tags.csv）→ 1-5 双项
  打分；无用户标签的电影跳过（打分无从谈起），数量可由
  (有生成标签的电影数 − n_gen_quality_movies) 从 manifest 推得。

控制器裁定在本模块固化：
- 过滤语义：apply_filter=true 时只剔除 verdict=="reject" 的映射（键 =
  movie_id+tag+tag_id）；accept 与 parse_failed 保守保留（未证明是坏的）；
  未被抽样到的映射同样保守保留——抽样是审计成本控制，不应放大为数据删除。
  过滤后按 cfg.align.agg 重聚合，再复用 stage-align/calibration.json 的
  Calibrator 换算（与 scores_calibrated 同量纲，可比）。
- 校准哨兵：calibration.json 是 {"method": ..., "fitted": false} 空实验哨兵
  （Task 13 产物）而 apply_filter=true → 大声报错点名文件与哨兵，绝不静默
  跳过校准（静默产物会以"已校准"的假象污染评估对比）。
- 缓存接线与生成管道同规：mock 配置不读不写缓存；仅 ok 响应入缓存；
  cache_key 的 run_id 固定 "judge-mapping" / "judge-gen"（多轮一致性分析
  可换 run_id 后缀发新请求）。

JSON 解析复刻 A（genometag）的 _extract_json_object 模式：直接 json.loads
失败则正则截取首个 {...} 再解析——实测教训：模型可能用 ```json 围栏或前后
缀文本包裹 JSON，直接 loads 会把整批好结果当解析失败报废。
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from movietaggen.align.calibrate import Calibrator
from movietaggen.align.mapping import aggregate_scores
from movietaggen.config import VALID_JUDGE_CONTEXTS
from movietaggen.data.align import DATASET_COLUMNS
from movietaggen.data.grouping import load_tag_groups
from movietaggen.data.movielens import load_user_tags
from movietaggen.generate.cache import ResponseCache, cache_key
from movietaggen.generate.content import truncate_to_budget
from movietaggen.stages import load_stage, save_config_snapshot

JUDGE_SYSTEM_PROMPT = "你是标签质量审计专家，输出严格的 JSON 对象。"

# 缓存键 run_id（多轮/重试语义：换后缀即换键，与生成管道 #r{n} 同法）
JUDGE_MAPPING_RUN_ID = "judge-mapping"
JUDGE_GEN_RUN_ID = "judge-gen"

# pair+full 内容节选的逐模态 token 预算（控制器裁定）
FULL_CONTEXT_TOKEN_BUDGET = 500

MAPPING_VERDICTS = ("accept", "reject")
VERDICT_PARSE_FAILED = "parse_failed"

JUDGE_MAPPING_COLUMNS = ["movie_id", "tag", "tag_id", "cosine",
                         "verdict", "reason", "judge_model"]
JUDGE_GEN_COLUMNS = ["movie_id", "relevance", "coverage", "judge_model"]

_MAPPING_INSTRUCTION = (
    '判定该映射的语义是否对齐且适用于该电影。输出 {"verdict": "accept" 或 '
    '"reject", "reason": "一句话理由"}，不要输出其他内容。')
_GEN_INSTRUCTION = (
    "对照真实用户标签，为生成标签的相关性与覆盖度打分（各 1-5 整数）。"
    '输出 {"relevance": n, "coverage": n}，不要输出其他内容。')

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_mapping_messages(title: str, free_tag: str, genome_tag: str,
                           genome_def: str, cosine: float,
                           context: str = "pair+title",
                           *, script: str | None = None,
                           subtitles: str | None = None) -> list[dict]:
    """组装映射判定 prompt（三档上下文，控制器规定的固定格式）。

    - pair：无标题行（纯标签对判断）；pair+title（默认）：含 电影：《title》；
      pair+full：在 嵌入相似度 行后追加 ## 内容节选（剧本/台词各按
      FULL_CONTEXT_TOKEN_BUDGET 截断；script/subtitles 文本缺省时不渲染节，
      对应文件缺失的合法情形）。
    - genome_def 为空时省略（定义: ...）后缀（不渲染空括号）。
    """
    if context not in VALID_JUDGE_CONTEXTS:
        raise ValueError(
            f"context 必须是 {VALID_JUDGE_CONTEXTS}，得到 {context!r}")
    parts: list[str] = []
    if context != "pair":
        parts.append(f"电影：《{title}》")
    parts.append(f'自由标签: "{free_tag}"')
    genome_line = f'候选 genome 标签: "{genome_tag}"'
    if genome_def:
        genome_line += f"（定义: {genome_def}）"
    parts.append(genome_line)
    parts.append(f"嵌入相似度: {cosine:.4f}")
    if context == "pair+full" and (script or subtitles):
        sections = ["## 内容节选"]
        if script:
            sections.append("剧本节选：\n"
                            + truncate_to_budget(script, FULL_CONTEXT_TOKEN_BUDGET))
        if subtitles:
            sections.append("台词节选：\n"
                            + truncate_to_budget(subtitles, FULL_CONTEXT_TOKEN_BUDGET))
        parts.append("\n".join(sections))
    parts.append(_MAPPING_INSTRUCTION)
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)}]


def build_gen_quality_messages(title: str, gen_tags: list[str],
                               user_tags: list[str]) -> list[dict]:
    """组装生成质量打分 prompt（生成标签数组 vs 真实用户标签数组 → 1-5 双项）。"""
    parts = [
        f"电影：《{title}》",
        f"生成标签: {json.dumps(list(gen_tags), ensure_ascii=False)}",
        f"真实用户标签: {json.dumps(list(user_tags), ensure_ascii=False)}",
        _GEN_INSTRUCTION,
    ]
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(parts)}]


def _extract_json_object(text: str):
    """直接 json.loads；失败则正则截取首个 {...} 再解析（复刻 A 的
    genometag/annotate/prompts.py::_extract_json_object——围栏/前后缀容忍）。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return None
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return None


def parse_mapping_verdict(text: str) -> dict | None:
    """解析映射判定输出为 {"verdict": "accept"|"reject", "reason": str}。

    verdict 合法值 + reason 为字符串（可为空）方通过；其余一律 None
    （调用方记 parse_failed，不抛异常不重试）。
    """
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return None
    verdict = data.get("verdict")
    reason = data.get("reason")
    if verdict not in MAPPING_VERDICTS or not isinstance(reason, str):
        return None
    return {"verdict": verdict, "reason": reason}


def _valid_score(v) -> bool:
    """1-5 整数校验：bool 是 int 子类，显式拒绝；浮点/字符串数值不收。"""
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 5


def parse_gen_quality(text: str) -> dict | None:
    """解析生成质量打分为 {"relevance": 1-5, "coverage": 1-5}（整数严格校验）。"""
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return None
    relevance = data.get("relevance")
    coverage = data.get("coverage")
    if not _valid_score(relevance) or not _valid_score(coverage):
        return None
    return {"relevance": relevance, "coverage": coverage}


def _read_excerpts(row, movienet_root: Path) -> tuple[str | None, str | None]:
    """按 dataset 行读 (剧本全文, 台词全文)；文件缺失返回 None（合法情形）。

    NaN 守卫照搬 pipeline._to_multimodal 的实测教训：混合 str/缺失列经
    pandas3 string dtype + parquet 往返后缺失为 NaN（float），必须 pd.notna。
    """
    script = subtitles = None
    if pd.notna(row["script_path"]) and row["script_path"]:
        p = Path(movienet_root) / row["script_path"]
        script = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    if pd.notna(row["subtitles_path"]) and row["subtitles_path"]:
        p = Path(movienet_root) / row["subtitles_path"]
        subtitles = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
    return script, subtitles


def _sampling_params(temperature: float,
                     extra_body: dict | None) -> dict:
    """评审采样参数指纹：与 generate.client.sampling_params 输出同构
    （temperature + extra_body（truthy 时）进缓存键，I3——换端点专有参数
    即换键；未设置 extra_body 时与旧版 {"temperature": ...} 键逐字节一致，
    存量缓存不失效）。不直接复用 sampling_params：评审的温度取 audit 配置、
    extra_body 取 model 配置，无单一 model_cfg 承载二者，故按同构 dict 自建。"""
    params: dict = {"temperature": temperature}
    if extra_body:
        params["extra_body"] = dict(extra_body)
    return params


def _judge_content(client, cache: ResponseCache | None, messages: list[dict],
                   run_id: str, model: str, temperature: float,
                   extra_body: dict | None = None) -> str:
    """缓存优先的单次评审调用；仅 ok 响应入缓存（失败信封入缓存会让续跑
    永久复播失败，与生成管道同规）。返回模型文本。"""
    key = None
    if cache is not None:
        key = cache_key(messages, model, _sampling_params(temperature, extra_body),
                        run_id)
        raw = cache.get(key)
        if raw is not None:
            return raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    resp = client.complete(messages, run_id=run_id)
    if cache is not None and resp.ok:
        cache.put(key, resp.raw)
    return resp.content


def _read_or_empty(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    """读 parquet；文件缺失按空表处理并打印降级警告（照 05 脚本同法）。"""
    if not path.exists():
        print(f"[audit] 警告: {label} 缺失（{path}），按空表处理")
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def run_audit(config, paths, client, cache: ResponseCache | None,
              config_path: str | Path | None = None) -> Path:
    """跑完阶段三审计，返回 stage-audit 目录。

    产物：judge_mapping.parquet（mapping 任务）、judge_gen_quality.parquet
    （gen_quality 任务）、scores_filtered.parquet（apply_filter=true）、
    run_manifest.json（8 键固定 schema），外加 results 根的 config.yaml
    快照（config_path 非空时；05/06 脚本接线传入，库调用可省）。

    client/cache 由调用方构造注入（judge 用 OpenAIClient(judge_model) 或
    MockJudgeClient，见 06 脚本）；mock 配置不读不写缓存。
    """
    cfg = config.audit
    do_mapping = "mapping" in cfg.judge_tasks
    do_gen = "gen_quality" in cfg.judge_tasks
    # mock 不走缓存与生成管道同规；缓存对象由调用方注入
    use_cache = cache if not config.mock else None

    align_dir = load_stage(paths, "stage-align", cfg.input_exp)
    mappings = pd.read_parquet(align_dir / "mappings.parquet")
    dataset = _read_or_empty(paths.dataset, DATASET_COLUMNS, "dataset")
    movie_rows = ({int(r["movie_id"]): r for _, r in dataset.iterrows()}
                  if len(dataset) else {})

    judge_rows: list[dict] = []
    gen_rows: list[dict] = []
    n_accept = n_reject = n_parse_failed = 0

    if do_mapping and len(mappings) > 0:
        if len(dataset) == 0:
            raise ValueError(
                f"dataset 为空或缺失（{paths.dataset}），但 mappings 含 "
                f"{len(mappings)} 条映射：评审 prompt 需要电影标题。"
                "请先运行 02 构建数据集。")
        genome_tags = _read_or_empty(paths.genome_tags, ["tag_id", "tag"],
                                     "genome_tags")
        if len(genome_tags) == 0:
            raise ValueError(
                f"genome_tags 为空或缺失（{paths.genome_tags}），"
                "无法将映射的 tag_id 还原为 genome 标签文本。")
        if not paths.tag_groups.exists():
            raise FileNotFoundError(
                f"tag_groups.json 缺失: {paths.tag_groups}。映射判定 prompt "
                "需要 genome 标签的一句话定义；请先运行数据准备（00/02 脚本）"
                "产出该文件。")
        tag_groups = load_tag_groups(paths.tag_groups)
        # 定义缺失的标签宽松降级为省略定义后缀（prompt 仍可判定）；
        # genome 标签文本缺失则大声失败（无文本的判定 prompt 无意义）
        defs = dict(zip(tag_groups["tag_id"].astype(int),
                        tag_groups["definition"]))
        tag_text = dict(zip(genome_tags["tag_id"].astype(int),
                            genome_tags["tag"]))

        ordered = mappings.sort_values(
            ["movie_id", "tag", "tag_id"], kind="mergesort").reset_index(drop=True)
        n = len(ordered)
        k = min(n, math.ceil(cfg.sample_rate * n))
        chosen = sorted(random.Random(cfg.seed).sample(range(n), k))
        print(f"[audit] 映射判定 {k}/{n} 条"
              f"（sample_rate={cfg.sample_rate}, context={cfg.judge_context}, "
              f"workers={cfg.workers}）", flush=True)

        # pair+full 节选池外预读：并行池内惰性读文件会竞态共享 dict，预读
        # 既是并行前提也顺带保住"逐电影只读一次"的原语义
        excerpts: dict[int, tuple[str | None, str | None]] = {}
        if cfg.judge_context == "pair+full":
            for mid in {int(ordered.iloc[i]["movie_id"]) for i in chosen}:
                excerpts[mid] = _read_excerpts(movie_rows[mid], paths.movienet)

        lock = threading.Lock()
        done = 0

        def judge_mapping_one(i: int) -> dict:
            """单条映射评审（独立任务，串行/并行同一闭包）。"""
            nonlocal done
            row = ordered.iloc[i]
            mid = int(row["movie_id"])
            if mid not in movie_rows:
                raise ValueError(
                    f"mapping 的 movie_id {mid} 不在 dataset 中"
                    "（输入产物不一致，疑似 stage-align 与 dataset 错配）")
            tid = int(row["tag_id"])
            if tid not in tag_text:
                raise ValueError(
                    f"mapping 的 tag_id {tid} 不在 genome_tags 中"
                    "（输入产物不一致，疑似 stage-align 与 02 产物错配）")
            script = subtitles = None
            if cfg.judge_context == "pair+full":
                script, subtitles = excerpts[mid]
            messages = build_mapping_messages(
                str(movie_rows[mid]["title"]), str(row["tag"]),
                tag_text[tid], defs.get(tid, ""), float(row["cosine"]),
                cfg.judge_context, script=script, subtitles=subtitles)
            content = _judge_content(client, use_cache, messages,
                                     JUDGE_MAPPING_RUN_ID, cfg.judge_model,
                                     cfg.judge_temperature,
                                     config.model.extra_body)
            parsed = parse_mapping_verdict(content)
            if parsed is None:
                verdict, reason = VERDICT_PARSE_FAILED, ""
            else:
                verdict, reason = parsed["verdict"], parsed["reason"]
            with lock:
                done += 1
                if done % 500 == 0 or done == k:
                    print(f"[audit] 映射判定进度 {done}/{k}", flush=True)
            return {
                "movie_id": mid, "tag": str(row["tag"]), "tag_id": tid,
                "cosine": float(row["cosine"]), "verdict": verdict,
                "reason": reason, "judge_model": cfg.judge_model}

        # pool.map 保序：输出与串行逐位一致（parquet 写前另有排序，双保险）
        if cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                judge_rows = list(pool.map(judge_mapping_one, chosen))
        else:
            judge_rows = [judge_mapping_one(i) for i in chosen]
        # 计数从结果事后统计（池内累加计数器需加锁，事后统计免锁且两路径同源）
        n_accept = sum(1 for r in judge_rows if r["verdict"] == "accept")
        n_reject = sum(1 for r in judge_rows if r["verdict"] == "reject")
        n_parse_failed = sum(1 for r in judge_rows
                             if r["verdict"] == VERDICT_PARSE_FAILED)

    if do_gen:
        gen_dir = load_stage(paths, "stage-gen", cfg.input_exp)
        tags = pd.read_parquet(gen_dir / "tags.parquet")
        user_tags = load_user_tags(paths.movielens / "tags.csv")
        gen_by_movie = {int(m): [str(t) for t in g["tag"]]
                        for m, g in tags.groupby("movie_id", sort=True)}
        # sorted(set(...))：同一标签多用户重复打标去重，顺序确定
        user_by_movie = {int(m): sorted(set(g["tag"].astype(str)))
                         for m, g in user_tags.groupby("movie_id")}
        candidates = sorted(set(gen_by_movie) & set(user_by_movie))
        if candidates and len(dataset) == 0:
            raise ValueError(
                f"dataset 为空或缺失（{paths.dataset}），但生成质量判定有 "
                f"{len(candidates)} 部候选电影：评审 prompt 需要电影标题。")
        skipped = len(gen_by_movie) - len(candidates)
        if skipped:
            print(f"[audit] 生成质量判定跳过 {skipped} 部无用户标签的电影",
                  flush=True)

        def judge_gen_one(mid: int) -> dict | None:
            """单电影生成质量评审（不可解析返回 None，调用方过滤计数）。"""
            if mid not in movie_rows:
                raise ValueError(
                    f"movie_id {mid} 不在 dataset 中（输入产物不一致，"
                    "疑似 stage-gen 与 dataset 错配）")
            messages = build_gen_quality_messages(
                str(movie_rows[mid]["title"]), gen_by_movie[mid],
                user_by_movie[mid])
            content = _judge_content(client, use_cache, messages,
                                     JUDGE_GEN_RUN_ID, cfg.judge_model,
                                     cfg.judge_temperature,
                                     config.model.extra_body)
            parsed = parse_gen_quality(content)
            if parsed is None:
                # gen_quality 无 verdict 列可记录，缺失行不进产物；计数可见
                print(f"[audit] 警告: 电影 {mid} 的生成质量输出不可解析，已跳过")
                return None
            return {"movie_id": mid, "relevance": parsed["relevance"],
                    "coverage": parsed["coverage"],
                    "judge_model": cfg.judge_model}

        if cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                gen_rows = [r for r in pool.map(judge_gen_one, candidates)
                            if r is not None]
        else:
            gen_rows = [r for r in (judge_gen_one(m) for m in candidates)
                        if r is not None]

    scores_filtered: pd.DataFrame | None = None
    if cfg.apply_filter:
        if not do_mapping:
            raise ValueError(
                "audit.apply_filter=true 需要 judge_tasks 包含 'mapping'："
                "过滤依据映射判定结果（剔除 reject 后重聚合）。请在 "
                "judge_tasks 中加入 'mapping'，或将 apply_filter 设为 false。")
        rejected = {(int(r["movie_id"]), str(r["tag"]), int(r["tag_id"]))
                    for r in judge_rows if r["verdict"] == "reject"}
        keep_mask = [(int(m), str(t), int(g)) not in rejected
                     for m, t, g in zip(mappings["movie_id"], mappings["tag"],
                                        mappings["tag_id"])]
        scores_filtered = aggregate_scores(mappings[keep_mask], config.align.agg)
        calib_path = align_dir / "calibration.json"
        calib_dict = json.loads(calib_path.read_text(encoding="utf-8"))
        if calib_dict.get("fitted") is False:
            raise ValueError(
                f"校准文件为空实验哨兵（fitted=false）: {calib_path}。"
                "其来源 stage-align 未拟合校准器（空实验），复用校准无从谈起；"
                "请先在非空实验上运行 05 对齐，或将 audit.apply_filter 设为 "
                "false。绝不静默跳过校准（未校准的过滤分与 scores_calibrated "
                "量纲不同，会污染评估对比）。")
        calibrator = Calibrator.from_dict(calib_dict)
        scores_filtered["score"] = calibrator.transform(
            scores_filtered["score"].to_numpy(dtype=float))

    paths.stage_audit.mkdir(parents=True, exist_ok=True)
    if do_mapping:
        pd.DataFrame(judge_rows, columns=JUDGE_MAPPING_COLUMNS).sort_values(
            ["movie_id", "tag", "tag_id"], kind="mergesort").to_parquet(
            paths.stage_audit / "judge_mapping.parquet", index=False)
    if do_gen:
        pd.DataFrame(gen_rows, columns=JUDGE_GEN_COLUMNS).sort_values(
            "movie_id", kind="mergesort").to_parquet(
            paths.stage_audit / "judge_gen_quality.parquet", index=False)
    if scores_filtered is not None:
        scores_filtered.to_parquet(
            paths.stage_audit / "scores_filtered.parquet", index=False)

    manifest = {
        "n_mappings": int(len(mappings)),
        "n_judged": int(len(judge_rows)),
        "n_accept": int(n_accept),
        "n_reject": int(n_reject),
        "n_parse_failed": int(n_parse_failed),
        "n_gen_quality_movies": int(len(gen_rows)),
        "judge_model": cfg.judge_model,
        "sample_rate": cfg.sample_rate,
    }
    (paths.stage_audit / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # 配置快照随结果落盘（结果目录自包含、可复现；脚本接线传入 config_path）
    if config_path is not None:
        save_config_snapshot(config_path, paths)
    return paths.stage_audit
