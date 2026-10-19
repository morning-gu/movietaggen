"""评估报告：双口径指标表（CSV + markdown）+ 生成质量双口径（嵌入 + judge）。

骨架镜像 genometag-mllm 的 genometag/eval/report.py（predictions dict →
metrics_table → 子集切片 → concat），差异：
- 预测来源：stage-align/scores_raw.parquet、scores_calibrated.parquet、
  stage-audit/scores_filtered.parquet（存在时）+ baselines/baselines.parquet
  （存在时，method 名透传）；
- 指标集合换 Task 17：dual_rmse 双口径 + spearman + 逐电影 P@K/NDCG@K；
- 生成质量 gen_quality.json：嵌入口径（stage-gen tags vs 用户 top-K 标签）
  + judge 口径（judge_gen_quality.parquet 均值）；
- 不复制 A 的模态×标签热力图（B 无此需求，YAGNI）。

metrics.csv 行 = method × subset；subset 五视图（spec §7.3：描述性分桶，
不作协议）：全部 / 时序训练(<y) / 时序测试(≥y) / tag:content / tag:preference
（冷启动是方法属性不是子集行）。多实验对比：第一个实验保持裸方法名，其余
实验以 `two-stage[raw|<实验目录名>]` 后缀消歧。

嵌入口径的池级约定：生成侧 = stage-gen tags.parquet 全部行（跨电影并集，
不去重）；用户侧 = 按出现电影数取 top-K（K = eval.coverage_top_k）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from movietaggen.align.embedding import embed_texts, make_embedder
from movietaggen.config import AppConfig, ProtocolConfig
from movietaggen.eval.metrics import (
    dual_rmse, gen_coverage, gen_redundancy, gen_relevance, ndcg_at_k,
    precision_at_k, spearman_corr,
)
from movietaggen.eval.protocol import tag_type_breakdown, temporal_split

_METRIC_COLUMNS = ["method", "subset", "rmse_full", "rmse_pred_only",
                   "spearman", "precision@k", "ndcg@k"]


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """(movie_id, tag_id, score) → movie × tag 宽矩阵。"""
    return df.pivot(index="movie_id", columns="tag_id", values="score")


def _evaluate_method(pred: pd.DataFrame, truth: pd.DataFrame,
                     all_tag_ids: set[int],
                     cfg: ProtocolConfig) -> dict[str, float]:
    """单方法指标：dual_rmse 双口径 + spearman（pred ⋈ truth 预测对）+
    逐电影 P@K/NDCG@K（pred/truth 各自透视宽矩阵后逐电影计算，电影间取均值）。

    P@K/NDCG@K 沿用 A 的宽透视做法：相关集 = truth ≥ relevance_threshold，
    top-k 按 pred 降序。"""
    dual = dual_rmse(pred, truth, all_tag_ids)
    # pred 的 score 先重命名为 score_p：truth 侧列名为 relevance，合并无列冲突时
    # pandas 不会追加后缀，直接引用 score_p 会 KeyError（同 A）
    joined = pred.rename(columns={"score": "score_p"}).merge(
        truth, on=["movie_id", "tag_id"])
    rho = spearman_corr(joined["score_p"].to_numpy(dtype=float),
                        joined["relevance"].to_numpy(dtype=float))
    pk: list[float] = []
    nk: list[float] = []
    if not joined.empty:
        pw = _wide(joined.rename(columns={"score_p": "score"}))
        tw = _wide(joined.rename(columns={"relevance": "score"}))
        for movie in pw.index:
            if movie in tw.index:
                pk.append(precision_at_k(
                    pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                    cfg.top_k, cfg.relevance_threshold))
                nk.append(ndcg_at_k(
                    pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                    cfg.top_k, cfg.relevance_threshold))
    return {"rmse_full": dual["rmse_full"],
            "rmse_pred_only": dual["rmse_pred_only"],
            "spearman": rho,
            "precision@k": float(np.mean(pk)) if pk else 0.0,
            "ndcg@k": float(np.mean(nk)) if nk else 0.0}


def metrics_table(predictions: dict[str, pd.DataFrame], truth: pd.DataFrame,
                  cfg: ProtocolConfig, tag_ids: list[int] | None = None,
                  movie_ids: list[int] | None = None) -> pd.DataFrame:
    """method × 指标表。tag_ids / movie_ids 给定时 pred 与 truth 同步过滤——
    子集的标签宇宙 all_tag_ids = 过滤后 truth 的 tag_id 集合（tag 类型子集的
    rmse_full 只在该类型的标签宇宙上补 0 对齐，不掺入其它类型）。"""
    if tag_ids is not None:
        truth = truth[truth["tag_id"].isin(set(tag_ids))]
    if movie_ids is not None:
        truth = truth[truth["movie_id"].isin(set(movie_ids))]
    all_tag_ids = set(truth["tag_id"].astype(int))
    rows = []
    for method, pred in predictions.items():
        p = pred
        if tag_ids is not None:
            p = p[p["tag_id"].isin(set(tag_ids))]
        if movie_ids is not None:
            p = p[p["movie_id"].isin(set(movie_ids))]
        rows.append({"method": method, **_evaluate_method(p, truth, all_tag_ids, cfg)})
    return pd.DataFrame(rows)


def _top_user_tags(user_tags: pd.DataFrame, k: int) -> list[str]:
    """top-K 用户标签：按出现电影数（per-movie frequency）降序；groupby 键序
    + 稳定排序使并列标签按字典序，跨进程确定。"""
    counts = user_tags.groupby("tag")["movie_id"].nunique()
    return counts.sort_values(ascending=False, kind="mergesort").index.tolist()[:k]


def _embed(embedder, texts: list[str], cache_root: Path | None) -> np.ndarray:
    """cache_root 给定走 embed_texts 逐文本磁盘缓存；None 直嵌（测试路径）。"""
    if cache_root is not None:
        return embed_texts(embedder, texts, cache_root)
    return embedder.embed(texts)


def _judge_means(exp_dir: Path) -> dict[str, float]:
    """judge 口径：stage-audit/judge_gen_quality.parquet（未跑 gen_quality
    任务则文件不存在）逐电影 relevance/coverage 均值，键前缀 judge_。"""
    path = exp_dir / "stage-audit" / "judge_gen_quality.parquet"
    if not path.exists():
        return {}
    jg = pd.read_parquet(path)
    if jg.empty:
        return {}
    return {"judge_relevance": float(jg["relevance"].mean()),
            "judge_coverage": float(jg["coverage"].mean())}


def gen_quality_summary(exp_dir: Path, user_tags: pd.DataFrame | None,
                        cfg: AppConfig, cache_root: Path | None) -> dict:
    """生成质量双口径（对第一个实验目录）：
    - 嵌入口径：relevance（双向 top-1 余弦均值）/ coverage（用户 top-K 标签
      被生成近邻覆盖比例，τ = eval.coverage_tau）/ redundancy（生成侧两两
      余弦绝对值均值），嵌入用 eval.embed_model（mock 配置 → MockEmbedder）；
    - judge 口径：judge_gen_quality.parquet 均值（键前缀 judge_）。
    user_tags 为 None → 嵌入口径以 note 占位，judge 口径照常（若可得）。"""
    out: dict = {}
    if user_tags is not None:
        gen = pd.read_parquet(exp_dir / "stage-gen" / "tags.parquet")
        user_top = _top_user_tags(user_tags, cfg.eval.coverage_top_k)
        embedder = make_embedder(cfg.eval.embed_model, mock=cfg.mock)
        gen_vecs = _embed(embedder, gen["tag"].tolist(), cache_root)
        user_vecs = _embed(embedder, user_top, cache_root)
        out.update({
            "relevance": gen_relevance(gen_vecs, user_vecs),
            "coverage": gen_coverage(user_vecs, gen_vecs, cfg.eval.coverage_tau),
            "redundancy": gen_redundancy(gen_vecs),
            "embed_model": embedder.name,
            "tau": cfg.eval.coverage_tau,
            "n_user_tags": len(user_top),
            "n_gen_tags": int(len(gen)),
        })
    else:
        out["note"] = "无用户标签，嵌入口径跳过"
    out.update(_judge_means(exp_dir))
    return out


def write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path,
                 cfg: AppConfig, dataset: pd.DataFrame,
                 truth: pd.DataFrame, tag_groups: pd.DataFrame | None,
                 user_tags: pd.DataFrame | None,
                 cache_root: Path | None = None) -> Path:
    """汇总多实验 + 基线 → evaluation/ 目录（metrics.csv、report.md、
    gen_quality.json）。

    cache_root：嵌入磁盘缓存根（paths.cache_embed）；None = 不落缓存直嵌
    （测试路径）。baseline_dir 恒可传（baselines.parquet 未跑基线时不存在，
    此处自行降级）。

    metrics.csv 的 subset 视图（spec §7.3：描述性分桶，不作协议）：
    - 全部 / 时序训练(<year) / 时序测试(≥year)：与各方法自身电影集取交
      （pred/truth 同步过滤即交集语义）——训练集上近记忆式复述、测试集上
      才见真泛化；
    - tag:<类型>（有 tag_groups 时）：内容型 vs 偏好型标签的逐类型对比。
    """
    if not exp_dirs:
        raise ValueError("write_report：exp_dirs 为空——至少需要一个含 "
                         "stage-align/scores_raw.parquet 的实验目录")
    protocol = cfg.protocol
    predictions: dict[str, pd.DataFrame] = {}
    for i, exp in enumerate(exp_dirs):
        label = "" if i == 0 else f"|{exp.name}"
        predictions[f"two-stage[raw{label}]"] = pd.read_parquet(
            exp / "stage-align" / "scores_raw.parquet")
        predictions[f"two-stage[calibrated{label}]"] = pd.read_parquet(
            exp / "stage-align" / "scores_calibrated.parquet")
        filtered = exp / "stage-audit" / "scores_filtered.parquet"
        if filtered.exists():
            predictions[f"two-stage[filtered{label}]"] = pd.read_parquet(filtered)
    if baseline_dir is not None and (baseline_dir / "baselines.parquet").exists():
        base = pd.read_parquet(baseline_dir / "baselines.parquet")
        for method, df in base.groupby("method"):
            predictions[str(method)] = df[["movie_id", "tag_id", "score"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    y = protocol.temporal_split_year
    train_ids, test_ids = temporal_split(dataset, y)

    sections: list[pd.DataFrame] = [
        metrics_table(predictions, truth, protocol).assign(subset="全部")]
    sections.append(metrics_table(predictions, truth, protocol,
                                  movie_ids=list(train_ids)
                                  ).assign(subset=f"时序训练(<{y})"))
    sections.append(metrics_table(predictions, truth, protocol,
                                  movie_ids=list(test_ids)
                                  ).assign(subset=f"时序测试(≥{y})"))
    # 逐 tag 类型子集：内容型 vs 偏好型（有 tag_groups 时产出）
    if tag_groups is not None:
        for tag_type, tids in tag_type_breakdown(tag_groups).items():
            sections.append(metrics_table(predictions, truth, protocol,
                                          tag_ids=tids).assign(subset=f"tag:{tag_type}"))

    table = pd.concat(sections, ignore_index=True)[_METRIC_COLUMNS]
    table.to_csv(out_dir / "metrics.csv", index=False)

    gq = gen_quality_summary(exp_dirs[0], user_tags, cfg, cache_root)
    (out_dir / "gen_quality.json").write_text(
        json.dumps(gq, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["# 评估报告\n", table.to_markdown(index=False), "",
          "\n## 生成质量（嵌入口径 + judge 口径）\n",
          f"```json\n{json.dumps(gq, ensure_ascii=False, indent=1)}\n```", ""]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return out_dir
