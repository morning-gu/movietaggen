"""剧本 + BERT 多标签回归基线（可选依赖 torch/transformers）。

只用时序训练参考集训练（防泄露），在全部电影上输出预测。
懒导入：torch/transformers 只在训练函数内部导入——本模块自身无 torch 也可导入；
缺依赖时在调用点抛 ImportError，由 07 脚本的 except 捕获后优雅跳过。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _movie_texts(dataset: pd.DataFrame, paths) -> dict[int, str]:
    texts = {}
    for _, r in dataset.iterrows():
        # 混合 str/缺失列经 parquet 往返后缺失为 NaN（float，为真值）——pd.notna 守卫
        text = None
        if pd.notna(r["script_path"]) and r["script_path"]:
            p = paths.movienet / r["script_path"]
            text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        if not text and pd.notna(r["subtitles_path"]) and r["subtitles_path"]:
            p = paths.movienet / r["subtitles_path"]
            text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        texts[int(r["movie_id"])] = (text or "")[:2000]
    return texts


def _train_predict(train_texts: dict[int, str], train_matrix: pd.DataFrame,
                   test_texts: dict[int, str], epochs: int = 3,
                   model_name: str = "distilbert-base-uncased",
                   batch_size: int = 8, max_length: int = 256,
                   lr: float = 2e-5) -> dict[int, dict[int, float]]:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tag_ids = sorted(train_matrix["tag_id"].unique())
    tag_col = {t: i for i, t in enumerate(tag_ids)}
    movies = sorted(train_texts)
    mat = np.zeros((len(movies), len(tag_ids)), dtype=np.float32)
    movie_row = {m: i for i, m in enumerate(movies)}
    for _, r in train_matrix.iterrows():
        mat[movie_row[int(r["movie_id"])], tag_col[int(r["tag_id"])]] = r["relevance"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.out = nn.Linear(self.encoder.config.hidden_size, len(tag_ids))

        def forward(self, ids, mask):
            return self.out(self.encoder(input_ids=ids, attention_mask=mask)
                            .last_hidden_state[:, 0])

    torch.manual_seed(0)
    model = Head()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    enc = tokenizer([train_texts[m] for m in movies], truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt")
    X = {"ids": enc["input_ids"], "mask": enc["attention_mask"]}
    Y = torch.from_numpy(mat)
    n = len(movies)
    model.train()
    for _ in range(epochs):
        # 小批量训练：全量 forward 一次吃下整个训练集，真实规模（数千电影）显存不可行
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            opt.zero_grad()
            loss = loss_fn(model(X["ids"][lo:hi], X["mask"][lo:hi]), Y[lo:hi])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        tenc = tokenizer([test_texts[m] for m in sorted(test_texts)], truncation=True,
                         max_length=max_length, padding=True, return_tensors="pt")
        preds = model(tenc["input_ids"], tenc["attention_mask"]).numpy()
    return {m: {tag_ids[i]: float(v) for i, v in enumerate(row)}
            for m, row in zip(sorted(test_texts), preds)}


def bert_baseline(config, paths, dataset: pd.DataFrame, scores: pd.DataFrame,
                  train_ids, test_ids, predict_ids=None) -> pd.DataFrame:
    from movietaggen.eval.protocol import assert_no_leakage
    assert_no_leakage(set(train_ids), set(test_ids))

    texts = _movie_texts(dataset, paths)
    train_scores = scores[scores["movie_id"].isin(set(train_ids))]
    # 电影级 tag 矩阵（训练参考集）
    train_matrix = train_scores[["movie_id", "tag_id", "relevance"]]
    train_texts = {m: t for m, t in texts.items() if m in set(train_ids)}
    # I5（440）：预测电影集可受限（07 传 sample_limited_movie_ids 的训练池抽样
    # 集，对齐其余受限基线的电影集；拟合参考集仍用全部训练侧电影，不受限），
    # None = 全部电影（原语义）
    if predict_ids is not None:
        texts = {m: t for m, t in texts.items() if m in set(predict_ids)}
    preds = _train_predict(train_texts, train_matrix, texts)
    rows = [{"movie_id": m, "tag_id": t, "score": float(np.clip(v, 0.0, 1.0)),
             "method": "bert"}
            for m, tag_scores in preds.items() for t, v in tag_scores.items()]
    return pd.DataFrame(rows, columns=["movie_id", "tag_id", "score", "method"])
