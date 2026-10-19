"""随机基线：movie × tag 稠密覆盖，score = 哈希均匀值（确定性、可复现）。

与 A 的 random_baseline（np.random 顺序采样）不同：分数绑定
(movie_id, tag_id, seed) 三元组、不依赖行序/调用顺序——同 seed 重跑逐字节
同帧，增删电影不影响其余电影的分数。
"""
from __future__ import annotations

import hashlib

import pandas as pd


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值（sample.py 的同源副本，第四份为约定）。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def random_baseline(dataset: pd.DataFrame, all_tag_ids, seed: int) -> pd.DataFrame:
    """每电影 × 每 tag 一行；score = _hash_unit(f"rand|{movie_id}|{tag_id}|{seed}") ∈ [0,1)。"""
    rows = [{"movie_id": int(m), "tag_id": int(t),
             "score": _hash_unit(f"rand|{int(m)}|{int(t)}|{seed}"), "method": "random"}
            for m in dataset["movie_id"] for t in all_tag_ids]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
