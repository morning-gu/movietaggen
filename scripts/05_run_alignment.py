"""05 · 阶段二对齐主跑（嵌入 → 映射 → 聚合 → 训练分割校准；mock 配置零成本）。

管线（复用 T9-T12 的既有部件）：
load_stage（input_exp 非空时跨实验物理借用 stage-gen，PROVENANCE 溯源）→
genome 标签与自由标签各嵌一次（embed_texts 逐文本磁盘缓存，cache_embed
跨实验共享；per-experiment 缓存命中自动生效）→ map_free_tags top-k 余弦
映射 → aggregate_scores 全量聚合 → 校准（scores_raw join genome_scores
真值、fixed_split 按 processed/eval_test.txt 清单取训练侧电影得训练对 →
fit_calibration → transform 全量）→ 五产物落盘：mappings.parquet /
scores_raw.parquet / scores_calibrated.parquet / calibration.json /
run_manifest.json，外加 results 根的 config.yaml 快照（结果目录自包含）。

两条控制器裁定在此固化：
- 聚合语义：scores_raw = aggregate_scores 对 mappings 全部行聚合，low_conf
  仅是旗标，本脚本任何环节不过滤（阈值筛选曲线是评估期 Task 18 的事）；
- 训练对防泄露：校准拟合集只允许 fixed_split 的训练侧电影（train =
  dataset − test，测试侧由 processed/eval_test.txt 清单钉死——真实数据为
  A 迁移的 88 部固定测试集；接线处 assert_no_leakage 收尾自检，出处
  spec §2.2 固定测试分割）。

空实验降级：无自由标签或空数据集 → 空 schema 产物 + 零输出 manifest，
返回 0（合法空实验可表示，校准无从拟合故以 fitted:false 哨兵占位）；
但数据非空而固定分割后训练对为空 → fit_calibration 的 ValueError 传播
（配置问题，如清单覆盖全部电影、或产物只含测试清单电影）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.align.calibrate import fit_calibration  # noqa: E402
from movietaggen.align.embedding import embed_texts, make_embedder  # noqa: E402
from movietaggen.align.mapping import (  # noqa: E402
    MAPPING_COLUMNS,
    SCORE_COLUMNS,
    aggregate_scores,
    map_free_tags,
)
from movietaggen.config import AppConfig, Paths, load_config, paths_for  # noqa: E402
from movietaggen.eval.protocol import assert_no_leakage, fixed_split  # noqa: E402
from movietaggen.stages import load_stage, save_config_snapshot  # noqa: E402

# processed 产物（02 产出）缺失时的空 schema 退化（列与 data.align.DATASET_COLUMNS 一致）
_DATASET_COLUMNS = ["movie_id", "title", "year", "genres", "imdb_id",
                    "movienet_id", "keyframes", "script_path", "subtitles_path"]
_CALIB_PAIR_COLUMNS = ["pred", "truth"]


def _read_or_empty(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    """读 parquet；文件缺失按空表处理（合法空实验可表示）并打印降级警告。"""
    if not path.exists():
        print(f"[05] 警告: {label} 缺失（{path}），按空表处理（空实验降级）")
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def run_alignment(config: AppConfig, paths: Paths, config_path: str | Path) -> Path:
    """跑完阶段二对齐，返回 stage-align 目录（五产物 + 配置快照）。

    config.mock=true 时嵌入走 MockEmbedder（零网络零依赖）；config_path
    仅用于配置快照落盘（save_config_snapshot 源不存在时静默跳过）。
    """
    # 1. 装载/借用阶段一生成产物（借用时本实验 stage-gen/ 带 PROVENANCE.json）
    gen_dir = load_stage(paths, "stage-gen", config.align.input_exp)
    tags = pd.read_parquet(gen_dir / "tags.parquet")
    free_tags = tags[["movie_id", "tag"]]       # 行序即映射矩阵的行序（mapping.py 有行数守卫）
    dataset = _read_or_empty(paths.dataset, _DATASET_COLUMNS, "dataset")
    genome_tags = _read_or_empty(paths.genome_tags, ["tag_id", "tag"], "genome_tags")
    genome_scores = _read_or_empty(
        paths.genome_scores, ["movie_id", "tag_id", "relevance"], "genome_scores")

    # embedder 构造是惰性的（Local/Api 均不在此加载模型/建客户端），空实验路径也安全
    embedder = make_embedder(config.align.embed_model, mock=config.mock,
                             device=config.align.embed_device)

    # 空实验降级：无自由标签或空数据集 → 整体零输出（对齐/校准均无从谈起）
    mappings = pd.DataFrame(columns=MAPPING_COLUMNS)
    scores_raw = pd.DataFrame(columns=SCORE_COLUMNS)
    scores_calibrated = pd.DataFrame(columns=SCORE_COLUMNS)
    train_pairs = pd.DataFrame(columns=_CALIB_PAIR_COLUMNS)
    calib_dict = {"method": config.align.calib_method, "fitted": False}

    if len(free_tags) > 0 and len(dataset) > 0:
        # 2. 嵌入：genome 标签一次、自由标签一次（embed_texts 逐文本缓存 + 同文去重，
        #    矩阵行序与对应 DataFrame 行序逐行对齐）
        free_matrix = embed_texts(embedder, free_tags["tag"].tolist(), paths.cache_embed)
        genome_matrix = embed_texts(embedder, genome_tags["tag"].tolist(), paths.cache_embed)

        # 3. 映射 + 聚合：top-k 余弦候选；聚合作用于全部行（low_conf 仅旗标不过滤）
        mappings = map_free_tags(free_tags, genome_tags, free_matrix, genome_matrix,
                                 config.align.top_k, config.align.threshold)
        scores_raw = aggregate_scores(mappings, config.align.agg)

        if len(scores_raw) > 0:
            # 4. 校准：scores_raw join genome 相关度真值得 (pred, truth) 对
            #    （year 列不 join：其唯一消费者是已被固定分割替换的时序过滤）
            joined = scores_raw.merge(genome_scores, on=["movie_id", "tag_id"], how="inner")
            # 【防泄露契约】校准拟合集只允许固定分割的训练侧电影：测试侧由
            # processed/eval_test.txt 清单钉死（真实数据为 A 迁移的 88 部固定
            # 测试集），train = dataset − test。测试清单电影的真值一旦进入拟合
            # 集，校准器等于"见过"测试电影的 ground truth，评估期校准前后的
            # 对比会系统性虚高。清单缺失/坏行/越界由 fixed_split 大声失败；
            # 接线以 assert_no_leakage 收尾自检（spec §2.2 固定测试分割）。
            train_ids, test_ids = fixed_split(dataset, paths.eval_test)
            assert_no_leakage(train_ids, test_ids)
            train = joined[joined["movie_id"].isin(set(train_ids))]
            train_pairs = pd.DataFrame({"pred": train["score"].astype(float),
                                        "truth": train["relevance"].astype(float)})
            # 数据非空而训练对为空 → ValueError 传播（配置问题，不静默降级）
            calibrator = fit_calibration(train_pairs, config.align.calib_method)
            calib_dict = calibrator.to_dict()
            # 5. transform 全量行（含测试侧电影；评估期才做分割内对比）
            scores_calibrated = scores_raw.copy()
            scores_calibrated["score"] = calibrator.transform(
                scores_raw["score"].to_numpy(dtype=float))

    manifest = {
        "n_movies": int(len(dataset)),
        "n_free_tags": int(len(free_tags)),
        "n_mappings": int(len(mappings)),
        "n_scores": int(len(scores_raw)),
        "calib_method": config.align.calib_method,
        "train_pairs": int(len(train_pairs)),
        "embed_model": embedder.name,
    }

    paths.stage_align.mkdir(parents=True, exist_ok=True)
    mappings.to_parquet(paths.stage_align / "mappings.parquet", index=False)
    scores_raw.to_parquet(paths.stage_align / "scores_raw.parquet", index=False)
    scores_calibrated.to_parquet(paths.stage_align / "scores_calibrated.parquet", index=False)
    (paths.stage_align / "calibration.json").write_text(
        json.dumps(calib_dict, ensure_ascii=False, indent=1), encoding="utf-8")
    (paths.stage_align / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # 配置快照随结果落盘（结果目录自包含、可复现）
    save_config_snapshot(config_path, paths)
    return paths.stage_align


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    out = run_alignment(config, paths, config_path)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"[05] 对齐完成 → {out}（mappings={manifest['n_mappings']}, "
          f"scores={manifest['n_scores']}, train_pairs={manifest['train_pairs']}）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
