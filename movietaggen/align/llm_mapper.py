"""LLM-as-mapper（B② 映射层改进实验）：嵌入召回 top-K 候选 → MLLM 语义选择。

与 embed 映射器（align/mapping.py 的余弦 top-k）的分工：嵌入负责召回
（top-K 候选池），LLM 负责选择（语义最匹配的一个，或 0=无匹配）——
精确隔离"选择机制"这一被测变量，召回基础与 embed 模式完全一致。

设计裁定（控制器，2026-10-24）：
- **上下文无关 tag↔tag**：映射判语义等价，不含电影标题——电影适用性是
  审计（audit/judge.py）的关注点，两层不混。副产品：同一标签字符串跨
  电影可去重调用（嵌入缓存逐文本确定性 → 同串候选恒同），~19k (movie,
  tag) 行去重到唯一字符串数次调用；
- choice 0 = 无合理匹配：映射期就拒（审计发现的 57% reject 率前移到
  映射决策），被拒标签不产映射行；
- temperature 固定 0.0（决策组件非生成）；run_id 固定 "llm-mapper"
  （换后缀即换缓存键发新请求，与生成/审计管道同法）；
- 解析容错与 audit.judge._extract_json_object 同模式（```json 围栏/
  前后缀文本容忍）——跨层 import 私有函数破坏分层，此处独立实现同构
  逻辑并注明出处；
- 并行：mapper_workers 线程池（audit.workers 同款模式），pool.map 保序。
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd

from movietaggen.align.mapping import MAPPING_COLUMNS
from movietaggen.generate.cache import ResponseCache, cache_key

MAPPER_SYSTEM_PROMPT = "你是标签语义匹配专家，输出严格的 JSON 对象。"
MAPPER_RUN_ID = "llm-mapper"
MAPPER_TEMPERATURE = 0.0

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

# 选择结果哨兵：-1 = 解析失败/越界（与 0=无匹配区分，manifest 可分别计数）
_CHOICE_PARSE_FAILED = -1


def build_mapper_messages(free_tag: str,
                          candidates: list[dict]) -> list[dict]:
    """组装映射选择 prompt：自由标签 + 编号候选（带定义时附）+ 选择指令。

    candidates: [{"tag_id", "tag", "definition"(可缺)}, ...]（余弦降序，
    编号 1..K）；指令含 0=无匹配出口与 {"choice": n} 输出格式。
    """
    lines = [f'自由标签: "{free_tag}"', "候选 genome 标签:"]
    for i, c in enumerate(candidates, 1):
        line = f'{i}. "{c["tag"]}"'
        if c.get("definition"):
            line += f"（定义: {c['definition']}）"
        lines.append(line)
    lines.append("选出与自由标签语义最匹配的一个候选（编号）；若没有合理匹配，输出 0。")
    lines.append('输出 {"choice": 编号}，不要输出其他内容。')
    return [{"role": "system", "content": MAPPER_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def parse_choice(text: str) -> int | None:
    """解析 {"choice": n}：非负整数方通过（0=无匹配）；其余 None。

    JSON 直载失败则正则截取首个 {...} 再解析（audit.judge 同模式：模型
    可能用 ```json 围栏或前后缀文本包裹）。bool 是 int 子类，显式拒绝。
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    c = data.get("choice")
    if not isinstance(c, int) or isinstance(c, bool) or c < 0:
        return None
    return c


def _mapper_params(extra_body: dict | None) -> dict:
    """缓存键采样参数：temperature 固定 0.0 + extra_body（truthy 时进键）。"""
    params: dict = {"temperature": MAPPER_TEMPERATURE}
    if extra_body:
        params["extra_body"] = dict(extra_body)
    return params


def _mapper_content(client, cache: ResponseCache | None, messages: list[dict],
                    model_name: str, extra_body: dict | None) -> str:
    """缓存优先的单次选择调用；仅 ok 响应入缓存（与生成/审计管道同规）。"""
    key = None
    if cache is not None:
        key = cache_key(messages, model_name, _mapper_params(extra_body),
                        MAPPER_RUN_ID)
        raw = cache.get(key)
        if raw is not None:
            return raw.get("choices", [{}])[0].get("message", {}).get("content", "")
    resp = client.complete(messages, run_id=MAPPER_RUN_ID)
    if cache is not None and resp.ok:
        cache.put(key, resp.raw)
    return resp.content


def llm_select_mappings(candidates: pd.DataFrame,
                        genome_tag_text: dict[int, str],
                        tag_defs: dict[int, str] | None,
                        client, cache: ResponseCache | None,
                        model_name: str, extra_body: dict | None,
                        workers: int = 1) -> pd.DataFrame:
    """候选池（map_free_tags 的 top-K 输出）→ LLM 选择 → MAPPING_COLUMNS 子集。

    genome_tag_text: tag_id → genome 标签文本（候选池只带 tag_id，prompt 需要
    文本；缺映射的 tag_id 大声失败——无文本的选择 prompt 无意义）。

    - 按 tag 字符串去重调用：同串跨电影的候选序列须一致（嵌入缓存逐文本
      确定性保证；不一致即大声失败——错位候选会静默产出垃圾选择）；候选
      列表取**单部电影**的 K 行（跨电影行按 rank 排序会交错成 2K 条）；
    - choice 0（无匹配）与解析失败/越界：该标签的全部 (movie, tag) 行剔除
      （映射期拒绝，与审计的 reject-drop 语义对齐）；
    - 选中行的 rank/cosine/low_conf 原样保留（low_conf 由 map_free_tags
      按阈值算好——LLM 选中的低余弦候选恰是"嵌入排位失误"的分析素材）；
    - workers > 1 时线程池并行（任务独立、pool.map 保序）；进度每 500 打印。
    """
    if candidates.empty:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    tag_defs = tag_defs or {}

    # 唯一 tag 字符串 → 候选列表（单电影的 K 行，rank 升序）+ 一致性守卫
    by_tag: dict[str, list[dict]] = {}
    for tag, g in candidates.groupby("tag", sort=True):
        seqs = g.groupby("movie_id")["tag_id"].apply(tuple).unique()
        if len(seqs) > 1:
            raise ValueError(
                f"自由标签 {tag!r} 跨电影的候选 tag_id 序列不一致"
                f"（{len(seqs)} 种）——嵌入缓存应保证同串同候选，"
                "疑似缓存污染或矩阵行序错位")
        g_one = g[g["movie_id"] == g["movie_id"].iloc[0]].sort_values(
            "rank", kind="mergesort")
        cands = []
        for r in g_one.itertuples():
            tid = int(r.tag_id)
            if tid not in genome_tag_text:
                raise ValueError(
                    f"候选 tag_id {tid} 不在 genome_tag_text 中——候选池与 "
                    "genome 标签表错配，无文本的选择 prompt 无意义")
            cands.append({"tag_id": tid, "tag": genome_tag_text[tid],
                          "definition": tag_defs.get(tid)})
        by_tag[str(tag)] = cands

    unique_tags = sorted(by_tag)
    print(f"[llm-mapper] 选择 {len(unique_tags)} 个唯一标签"
          f"（候选池 {len(candidates)} 行去重，workers={workers}）", flush=True)
    lock = threading.Lock()
    done = 0

    def select_one(tag: str) -> int:
        """单标签选择 → 选中 tag_id（0=无匹配；-1=解析失败/越界）。"""
        nonlocal done
        cands = by_tag[tag]
        messages = build_mapper_messages(tag, cands)
        content = _mapper_content(client, cache, messages, model_name,
                                  extra_body)
        choice = parse_choice(content)
        if choice is None or choice > len(cands):
            result = _CHOICE_PARSE_FAILED
        else:
            result = cands[choice - 1]["tag_id"] if choice > 0 else 0
        with lock:
            done += 1
            if done % 500 == 0 or done == len(unique_tags):
                print(f"[llm-mapper] 进度 {done}/{len(unique_tags)}", flush=True)
        return result

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            picks = list(pool.map(select_one, unique_tags))
    else:
        picks = [select_one(t) for t in unique_tags]
    selection = dict(zip(unique_tags, picks))

    keep = []
    for r in candidates.itertuples():
        pick = selection.get(str(r.tag))
        if pick is not None and pick > 0 and int(r.tag_id) == pick:
            keep.append(r)
    n_drop = sum(1 for p in picks if p == 0)
    n_fail = sum(1 for p in picks if p == _CHOICE_PARSE_FAILED)
    print(f"[llm-mapper] 选择完成: 命中 {len(keep)} 行（无匹配 {n_drop} / "
          f"解析失败 {n_fail} 个唯一标签）", flush=True)
    if not keep:
        return pd.DataFrame(columns=MAPPING_COLUMNS)
    return pd.DataFrame([{
        "movie_id": int(r.movie_id), "tag": str(r.tag), "tag_id": int(r.tag_id),
        "cosine": float(r.cosine), "rank": int(r.rank),
        "low_conf": bool(r.low_conf)} for r in keep], columns=MAPPING_COLUMNS)
