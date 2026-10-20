"""评估报告：双口径指标表（CSV + markdown）+ 生成质量双口径（嵌入 + judge）。

骨架镜像 genometag-mllm 的 genometag/eval/report.py（predictions dict →
metrics_table → 子集切片 → concat），差异：
- 预测来源：stage-align/scores_raw.parquet、scores_calibrated.parquet、
  stage-audit/scores_filtered.parquet（存在时）+ two-stage[threshold-filtered]
  （第一个实验的 mappings 剔除 low_conf 后重聚合校准，I6 阈值消融曲线）+
  baselines/baselines.parquet（存在时，method 名透传）；
- 指标集合换 Task 17：dual_rmse 双口径 + spearman + 逐电影 P@K/NDCG@K，
  另附 n_pred（方法×子集实际参与评估的预测对数，空集以 NaN 指标行 +
  n_pred=0 亮明，不以 0.0 伪装满分——I1）；direct_score 的合法 NaN 分数行
  在 metrics_table 收口剔除（C1）；NDCG 的 IDCG 分母取完整 truth 行的相关
  总数，稀疏方法的 unretrieved relevant 计入惩罚（I2）；
- 生成质量 gen_quality.json：嵌入口径（stage-gen tags vs 用户 top-K 标签）
  + judge 口径（judge_gen_quality.parquet 均值）；
- 不复制 A 的模态×标签热力图（B 无此需求，YAGNI）。

metrics.csv 行 = method × subset；subset 视图（spec §7.3：描述性分桶，
不作协议）：全部 / 时序训练(<y) / 时序测试(≥y) / tag:content / tag:preference
（冷启动是方法属性不是子集行）；eval_test_ids 给定时另加固定两行——
固定测试(88)=论文主表口径（spec §2.4）：所有方法在同一 88 部上对比；
固定训练(352) 为调试口径；全部/时序/tag:* 保留为描述性视图，n_pred 列
诚实标注各方法实际预测数（direct_score 在 440 行 n_pred=88 属预期）。
多实验对比：第一个实验保持裸方法名，其余
实验以 `two-stage[raw|<实验目录名>]` 后缀消歧。

嵌入口径的逐电影约定（spec §7.1「生成标签 ↔ 该电影真实用户标签」）：
按 movie_id 分组 stage-gen tags.parquet，对每部有用户标签的电影取**该电影
自己的** top-K 用户标签（K = eval.coverage_top_k，频次为该电影内的出现次数，
并列按字典序稳定），逐电影嵌入并计算三件套后对电影取均值；无用户标签的
生成电影不入嵌入口径，以 n_movies_no_user_tags 计数。跨电影记账（一部电影
的生成标签替另一部电影的用户标签得分）因此不可得。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from movietaggen.align.calibrate import Calibrator
from movietaggen.align.embedding import embed_texts, make_embedder
from movietaggen.align.mapping import aggregate_scores
from movietaggen.config import AppConfig, ProtocolConfig
from movietaggen.eval.metrics import (
    dual_rmse, gen_coverage, gen_redundancy, gen_relevance, ndcg_at_k,
    precision_at_k, spearman_corr,
)
from movietaggen.eval.protocol import tag_type_breakdown, temporal_split

_METRIC_COLUMNS = ["method", "subset", "rmse_full", "rmse_pred_only",
                   "spearman", "precision@k", "ndcg@k", "n_pred"]

# I1：空预测集的方法×子集行——NaN 指标（0.0 会被误读为"完美"），n_pred=0 亮明空集
_EMPTY_ROW_METRICS = {"rmse_full": np.nan, "rmse_pred_only": np.nan,
                      "spearman": np.nan, "precision@k": np.nan, "ndcg@k": np.nan}


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """(movie_id, tag_id, score) → movie × tag 宽矩阵。"""
    return df.pivot(index="movie_id", columns="tag_id", values="score")


def _evaluate_method(pred: pd.DataFrame, truth: pd.DataFrame,
                     all_tag_ids: set[int],
                     cfg: ProtocolConfig) -> dict[str, float]:
    """单方法指标：dual_rmse 双口径 + spearman（pred ⋈ truth 预测对）+
    逐电影 P@K/NDCG@K（pred/truth 各自透视宽矩阵后逐电影计算，电影间取均值）。

    P@K/NDCG@K 沿用 A 的宽透视做法：相关集 = truth ≥ relevance_threshold，
    top-k 按 pred 降序。NDCG 的 IDCG 分母取该电影**完整 truth 行**的相关总数
    （I2：只按 pred ⋈ truth 的受限宇宙计数会让稀疏方法的 unretrieved relevant
    不扣分，NDCG 结构性虚高偏向稀疏方法）；P@K 两宇宙数学等价（分母恒 = k），
    按受限宇宙照旧。"""
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
        # 完整 truth 行（同一评估子集的标签宇宙）：I2 的相关总数来源
        truth_full = _wide(truth.rename(columns={"relevance": "score"}))
        for movie in pw.index:
            if movie in tw.index:
                pk.append(precision_at_k(
                    pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                    cfg.top_k, cfg.relevance_threshold))
                # movie 在 tw.index ⇒ 其 truth 行存在 ⇒ 必在 truth_full.index
                n_rel_full = int(
                    (truth_full.loc[movie].to_numpy()
                     >= cfg.relevance_threshold).sum())
                nk.append(ndcg_at_k(
                    pw.loc[movie].to_numpy(), tw.loc[movie].to_numpy(),
                    cfg.top_k, cfg.relevance_threshold, n_relevant=n_rel_full))
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
    rmse_full 只在该类型的标签宇宙上补 0 对齐，不掺入其它类型）。

    每方法行附带 n_pred = 该方法×子集实际参与评估的预测对数（子集过滤 +
    NaN 剔除后），空集在 metrics.csv 中可见而非伪装成满分。"""
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
        # C1：direct_score 合法产出 NaN 分数行（无内容电影 / 全轮解析失败——
        # 其 docstring 明言"交由评估侧处理"），必须先剔除：NaN 入 dual_rmse
        # 会令整个方法的 rmse_full/rmse_pred_only 变 NaN，spearman 的 NaN 守卫
        # 还会静默返回 0.0 伪装成真实分（镜像 A 的 ann.dropna(subset=["score"])
        # 模式，一处收口覆盖全部预测来源）。
        p = p.dropna(subset=["score"])
        if p.empty:
            # I1：空预测集 ≠ 满分。dual_rmse 的全 0 退化返回是其纯函数契约，
            # 报告层在此裁定上报口径：NaN 指标行 + n_pred=0（空集可见）。
            rows.append({"method": method, **_EMPTY_ROW_METRICS, "n_pred": 0})
        else:
            rows.append({"method": method,
                         **_evaluate_method(p, truth, all_tag_ids, cfg),
                         "n_pred": int(len(p))})
    return pd.DataFrame(rows)


def _movie_top_user_tags(movie_tags: pd.Series, k: int) -> list[str]:
    """单部电影的 top-K 用户标签：按该电影内的出现次数（多用户同标签重复
    计频）降序；先按标签名字典序再稳定降序，使并列标签次序跨进程确定。"""
    counts = movie_tags.value_counts()
    return (counts.sort_index()
                  .sort_values(ascending=False, kind="mergesort")
                  .index.tolist()[:k])


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
    - 嵌入口径（spec §7.1 逐电影语义）：按 movie_id 分组 stage-gen tags，每部
      有用户标签的电影取**该电影自己的** top-K 用户标签（K = eval.coverage_top_k，
      频次为该电影内出现次数、并列按字典序稳定），逐电影嵌入后计算 relevance
      （双向 top-1 余弦均值）/ coverage（该电影用户 top-K 被该电影生成近邻覆盖
      的比例，τ = eval.coverage_tau）/ redundancy（该电影生成侧两两余弦绝对值
      均值），再对有非空双侧的电影取均值——跨电影记账不可得。嵌入用
      eval.embed_model（mock 配置 → MockEmbedder）。无用户标签的生成电影不入
      嵌入口径，计入 n_movies_no_user_tags；无可平均电影时三件套按 0.0 上报
      （T17 空侧退化约定，n_movies=0 提示口径）。
    - judge 口径：judge_gen_quality.parquet 均值（键前缀 judge_）。
    user_tags 为 None → 嵌入口径以 note 占位，judge 口径照常（若可得）。"""
    out: dict = {}
    if user_tags is None:
        out["note"] = "无用户标签，嵌入口径跳过"
    else:
        gen = pd.read_parquet(exp_dir / "stage-gen" / "tags.parquet")
        embedder = make_embedder(cfg.eval.embed_model, mock=cfg.mock)
        user_by_movie = {m: s for m, s in user_tags.groupby("movie_id")["tag"]}
        per: dict[str, list[float]] = {"relevance": [], "coverage": [], "redundancy": []}
        n_no_user = 0
        for movie, g in gen.groupby("movie_id"):
            movie_tags = user_by_movie.get(movie)
            if movie_tags is None or len(movie_tags) == 0:
                n_no_user += 1          # 生成电影无用户标签：不入嵌入口径
                continue
            user_top = _movie_top_user_tags(movie_tags, cfg.eval.coverage_top_k)
            gen_vecs = _embed(embedder, g["tag"].tolist(), cache_root)
            user_vecs = _embed(embedder, user_top, cache_root)
            per["relevance"].append(gen_relevance(gen_vecs, user_vecs))
            per["coverage"].append(gen_coverage(user_vecs, gen_vecs, cfg.eval.coverage_tau))
            per["redundancy"].append(gen_redundancy(gen_vecs))
        out.update({
            "relevance": float(np.mean(per["relevance"])) if per["relevance"] else 0.0,
            "coverage": float(np.mean(per["coverage"])) if per["coverage"] else 0.0,
            "redundancy": float(np.mean(per["redundancy"])) if per["redundancy"] else 0.0,
            "embed_model": embedder.name,
            "tau": cfg.eval.coverage_tau,
            "n_movies": len(per["relevance"]),
            "n_movies_no_user_tags": n_no_user,
            "n_gen_tags": int(len(gen)),
        })
    out.update(_judge_means(exp_dir))
    return out


def threshold_filtered_predictions(exp_dir: Path, agg: str) -> pd.DataFrame | None:
    """阈值筛选变体（I6，spec §8 轴 6 的低置信过滤曲线；仅对第一个实验目录计算）：
    stage-align/mappings.parquet 剔除 low_conf 行后按 cfg.align.agg 重聚合，
    calibration.json 为真实 Calibrator 时再复用其换算（与 scores_calibrated
    同量纲可比）。数据已在盘上，零重跑成本——使阈值 0.4/0.6 消融臂产出真实
    不同的分数，而非同分假曲线。

    - mappings.parquet 缺失 → None（变体不可得，调用方跳过）；
    - calibration.json 为空实验哨兵（fitted=false，Task 13 产物）→ None +
      打印说明（无拟合参数可复用，绝不静默以未校准分数混入对比）；
    - 全部映射 low_conf → 空聚合返回（评估层按 I1 以 NaN 行上报空集）。"""
    mappings_path = exp_dir / "stage-align" / "mappings.parquet"
    if not mappings_path.exists():
        return None
    mappings = pd.read_parquet(mappings_path)
    kept = mappings[~mappings["low_conf"].astype(bool)]
    variant = aggregate_scores(kept, agg)
    calib_path = exp_dir / "stage-align" / "calibration.json"
    calib_dict = json.loads(calib_path.read_text(encoding="utf-8"))
    if calib_dict.get("fitted") is False:
        print(f"[report] 提示: {calib_path} 为空实验哨兵（fitted=false），"
              "two-stage[threshold-filtered] 变体跳过（无拟合校准器可复用）")
        return None
    calibrator = Calibrator.from_dict(calib_dict)
    variant["score"] = calibrator.transform(
        variant["score"].to_numpy(dtype=float))
    return variant


def write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path,
                 cfg: AppConfig, dataset: pd.DataFrame,
                 truth: pd.DataFrame, tag_groups: pd.DataFrame | None,
                 user_tags: pd.DataFrame | None,
                 cache_root: Path | None = None,
                 eval_test_ids: list[int] | None = None) -> Path:
    """汇总多实验 + 基线 → evaluation/ 目录（metrics.csv、report.md、
    gen_quality.json）。

    cache_root：嵌入磁盘缓存根（paths.cache_embed）；None = 不落缓存直嵌
    （测试路径）。baseline_dir 恒可传（baselines.parquet 未跑基线时不存在，
    此处自行降级）。eval_test_ids：固定测试清单的 movie_id 侧（08 恒传
    fixed_split(dataset, paths.eval_test) 的 test 侧）；None = 不产固定行
    （向后兼容，既有调用不变）。

    metrics.csv 的 subset 视图（spec §7.3：描述性分桶，不作协议）：
    - 全部 / 时序训练(<year) / 时序测试(≥year)：与各方法自身电影集取交
      （pred/truth 同步过滤即交集语义）——训练集上近记忆式复述、测试集上
      才见真泛化；
    - 固定测试(88) / 固定训练(352)（eval_test_ids 给定时）：论文主表口径
      （spec §2.4）——所有方法在同一 88 部固定测试集上对比（同集严格可比）；
      固定训练(352) 为调试口径；全部/时序/tag:* 保留为描述性视图，n_pred
      列诚实标注各方法实际预测数（direct_score 在 440 行 n_pred=88 属预期）；
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
    # I6：阈值筛选曲线仅对第一个实验目录计算（metrics.csv 新增
    # two-stage[threshold-filtered] 方法行；变体不可得/哨兵时自然缺席）
    tf = threshold_filtered_predictions(exp_dirs[0], cfg.align.agg)
    if tf is not None:
        predictions["two-stage[threshold-filtered]"] = tf
    if baseline_dir is not None and (baseline_dir / "baselines.parquet").exists():
        base = pd.read_parquet(baseline_dir / "baselines.parquet")
        for method, df in base.groupby("method"):
            predictions[str(method)] = df[["movie_id", "tag_id", "score"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    y = protocol.temporal_split_year
    train_ids, test_ids = temporal_split(dataset, y)

    sections: list[pd.DataFrame] = [
        metrics_table(predictions, truth, protocol).assign(subset="全部")]
    # 固定子集行（440 协议，spec §2.4）：eval_test_ids 给定时所有方法在同一
    # 固定测试集上对比——严格可比的论文主表口径；固定训练为补集侧（用
    # dataset 的 movie_id 列推导，不动时序分割）调试口径。行序：紧跟"全部"、
    # 时序行之前（主表口径紧邻全量视图）。
    if eval_test_ids is not None:
        eval_ids = list(eval_test_ids)
        sections.append(metrics_table(predictions, truth, protocol,
                                      movie_ids=eval_ids
                                      ).assign(subset=f"固定测试({len(eval_ids)})"))
        eval_id_set = set(eval_ids)
        train_side = [m for m in dataset["movie_id"] if m not in eval_id_set]
        sections.append(metrics_table(predictions, truth, protocol,
                                      movie_ids=train_side
                                      ).assign(subset=f"固定训练({len(dataset) - len(eval_ids)})"))
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
