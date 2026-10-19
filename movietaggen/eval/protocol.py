"""评估协议：时序分割、冷启动子集、tag 类型分解、防泄露断言。

四个函数逐字复制自 genometag-mllm 项目的 genometag/eval/protocol.py
（A/B 共用同一评估协议语义；assert_no_leakage 错误文案中提及的
"movie_list 补集" 为 A 的拟合池语境，按逐字复制约定保留）。
防泄露是全框架约束：校准拟合（05 的内联时序过滤）与基线 few-shot 选取
（Task 19/20）以本协议的时序语义为准，并以 assert_no_leakage 收尾自检。
"""
from __future__ import annotations

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
