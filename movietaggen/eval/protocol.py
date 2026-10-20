"""评估协议：时序分割、冷启动子集、tag 类型分解、防泄露断言。

四个函数逐字复制自 genometag-mllm 项目的 genometag/eval/protocol.py
（A/B 共用同一评估协议语义；assert_no_leakage 错误文案中提及的
"movie_list 补集" 为 A 的拟合池语境，按逐字复制约定保留）。
防泄露是全框架约束：校准拟合（05 的内联时序过滤）与基线 few-shot 选取
（Task 19/20）以本协议的时序语义为准，并以 assert_no_leakage 收尾自检。
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


def temporal_split(dataset: pd.DataFrame, year: int) -> tuple[pd.Index, pd.Index]:
    train = dataset.loc[dataset["year"] < year, "movie_id"]
    test = dataset.loc[dataset["year"] >= year, "movie_id"]
    return pd.Index(train), pd.Index(test)


def cold_start_subset(dataset: pd.DataFrame, user_tags: pd.DataFrame,
                      max_tags: int) -> pd.Index:
    counts = user_tags.groupby("movie_id").size()
    n = dataset["movie_id"].map(lambda m: counts.get(m, 0))
    return pd.Index(dataset.loc[n <= max_tags, "movie_id"])


def tag_type_breakdown(tag_groups: pd.DataFrame) -> dict[str, list[int]]:
    return {t: g["tag_id"].tolist() for t, g in tag_groups.groupby("tag_type")}


def assert_no_leakage(fit_movie_ids, test_movie_ids) -> None:
    leak = set(fit_movie_ids) & set(test_movie_ids)
    if leak:
        raise ValueError(
            f"拟合池与测试集重叠（数据泄露）: {sorted(leak)[:5]}... "
            "拟合组件只允许使用评估集之外的数据（movie_list 补集或 temporal_split_year 之前）")


def fixed_split(dataset: pd.DataFrame, eval_test_path: Path) -> tuple[pd.Index, pd.Index]:
    """固定分割：按测试清单切分（440 协议新增，非 A 迁移物）。

    读 eval_test 清单（每行一个 movie_id 整数，空行跳过），test = 清单命中
    的全集行，train = 全集 − test；替代时序分割成为各拟合面（校准 / few-shot
    / genre 统计 / BERT 训练集）的统一分割源。清单坏输入大声失败：
    - 文件缺失 → ValueError 点名路径；
    - 非整数行 → ValueError 点名行号与内容；
    - 清单为空（无有效 id 行）→ ValueError 点名路径与成因；
    - 清单 id 不在 dataset["movie_id"] → ValueError 点名缺失电影。
    返回 (train movie_id Index, test movie_id Index)，两侧均按 dataset 出现序
    （确定性；与 temporal_split 的行序语义一致）。
    """
    eval_test_path = Path(eval_test_path)
    if not eval_test_path.is_file():
        raise ValueError(f"eval_test 清单文件不存在: {eval_test_path}")
    test_ids: list[int] = []
    for lineno, raw in enumerate(
            eval_test_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            test_ids.append(int(line))
        except ValueError:
            raise ValueError(
                f"eval_test 清单第 {lineno} 行不是整数: {line!r}（文件: {eval_test_path}）"
            ) from None
    if not test_ids:
        raise ValueError(f"eval_test 清单为空（无有效 id 行）: {eval_test_path}")
    known = set(dataset["movie_id"])
    missing = sorted(set(test_ids) - known)
    if missing:
        raise ValueError(
            f"eval_test 清单含数据集外的电影（前 {min(5, len(missing))} 个）: "
            f"{missing[:5]}（文件: {eval_test_path}）")
    mask = dataset["movie_id"].isin(set(test_ids))
    return (pd.Index(dataset.loc[~mask, "movie_id"]),
            pd.Index(dataset.loc[mask, "movie_id"]))


def sample_limited_movie_ids(dataset: pd.DataFrame, movie_limit: int | None, seed: int,
                             eval_test_path: Path) -> list[int]:
    """受限跑的电影抽样（440 协议新增，非 A 迁移物）。

    movie_limit 为 None → sorted(dataset["movie_id"]) 全量（不读清单）；
    否则训练池 = fixed_split 的 train 侧（测试清单之外的全体），从
    sorted(池) 以 random.Random(seed).sample 抽 min(movie_limit, len(pool))
    部——limit ≥ 池 → 全池（不崩溃、不空集）。返回 list[int]：None 分支
    为升序全量，抽样分支为 sample 的抽样序（同 seed 确定）。
    """
    if movie_limit is None:
        return sorted(dataset["movie_id"].tolist())
    train_ids, _ = fixed_split(dataset, eval_test_path)
    pool = sorted(train_ids.tolist())
    return random.Random(seed).sample(pool, min(movie_limit, len(pool)))
