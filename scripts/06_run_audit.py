"""06 · 阶段三审计主跑（LLM-as-judge 双任务 + 过滤重聚合 + 校准抽样导出）。

管线（复用 T14/T15 的既有部件，接线模式照 04/05 脚本）：
load_config → paths_for → judge 客户端构造（config.mock → MockJudgeClient，
零网络零缓存；否则 OpenAIClient，judge 的模型/温度取 audit 配置——可与生成
模型不同，端点接线沿用 model 配置）→ 缓存（mock 不读不写，与生成/审计管道
同规；否则 ResponseCache(cache_llm)）→ run_audit 产出 judge_mapping.parquet /
judge_gen_quality.parquet / scores_filtered.parquet（apply_filter=true）/
run_manifest.json + 配置快照（结果目录自包含）→ judge_mapping.parquet 存在
（judge_tasks 含 mapping）时追加导出 stage-audit/calibration_sample.csv：
人工标注桥，行数 = min(audit.calibration_sample, 判定行数)，seed =
audit.seed，title/genome_tag 从 dataset/genome_tags 连接（读自 paths）。

空实验与配置问题不静默：数据非空而校准哨兵未拟合（apply_filter=true）、
数据集与审计产物错配等一律由 run_audit / export_calibration_sample 的
ValueError 传播——脚本只接线，不吞错。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.audit.calibration import export_calibration_sample  # noqa: E402
from movietaggen.audit.judge import run_audit  # noqa: E402
from movietaggen.audit.mock import MockJudgeClient  # noqa: E402
from movietaggen.config import AppConfig, load_config, paths_for  # noqa: E402
from movietaggen.data.align import DATASET_COLUMNS  # noqa: E402
from movietaggen.generate.cache import ResponseCache  # noqa: E402
from movietaggen.generate.client import OpenAIClient  # noqa: E402

_GENOME_TAG_COLUMNS = ["tag_id", "tag"]


def make_judge_client(config: AppConfig):
    """judge 客户端工厂：mock 配置 → MockJudgeClient；否则 OpenAIClient(judge_model)。

    与 make_client（generate/client.py）同构：load_dotenv 只在非 mock 路径
    调用（保持 import 无副作用）；judge 的模型/温度取 audit 配置（可与生成
    模型/温度不同），端点接线（base_url/api_key 环境变量名、限速、
    extra_body）沿用 model 配置。OpenAIClient 构造是惰性的（transport 在
    首次 complete 时才建），此处无网络。
    """
    if config.mock:
        return MockJudgeClient()
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAIClient(
        model=config.audit.judge_model,
        base_url_env=config.model.base_url_env,
        api_key_env=config.model.api_key_env,
        temperature=config.audit.judge_temperature,
        request_interval=config.model.request_interval,
        extra_body=config.model.extra_body)


def _read_or_empty(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    """读 parquet；文件缺失按空表处理（合法空实验可表示）并打印降级警告（照 05）。"""
    if not path.exists():
        print(f"[06] 警告: {label} 缺失（{path}），按空表处理（空实验降级）")
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    client = make_judge_client(config)
    cache = None if config.mock else ResponseCache(paths.cache_llm)
    out = run_audit(config, paths, client, cache, config_path=config_path)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"[06] 审计完成 → {out}（judged={manifest['n_judged']}, "
          f"accept={manifest['n_accept']}, reject={manifest['n_reject']}, "
          f"gen_quality_movies={manifest['n_gen_quality_movies']}）")

    # 校准抽样导出：judge_mapping.parquet 存在（judge_tasks 含 mapping）时追加；
    # 行数 min(cfg.audit.calibration_sample, 判定行数)，seed 取 audit.seed
    mapping_path = out / "judge_mapping.parquet"
    if mapping_path.exists():
        judge_df = pd.read_parquet(mapping_path)
        dataset = _read_or_empty(paths.dataset, DATASET_COLUMNS, "dataset")
        genome_tags = _read_or_empty(
            paths.genome_tags, _GENOME_TAG_COLUMNS, "genome_tags")
        sample = export_calibration_sample(
            judge_df, dataset, genome_tags,
            min(config.audit.calibration_sample, len(judge_df)),
            config.audit.seed)
        csv_path = out / "calibration_sample.csv"
        sample.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[06] 校准抽样导出 → {csv_path}（rows={len(sample)}）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
