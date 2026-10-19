"""08 · 评估主跑（两阶段各产物 + 基线 → 指标表 / 报告 / 生成质量双口径）。

管线（复用 Task 18 的 write_report，接线模式照 06 脚本）：
load_config → paths_for → 读 dataset / truth（genome_scores）/ tag_groups
（文件可缺 → None，tag:* 子集随之跳过）/ user_tags（tags.csv 可缺 → None，
嵌入口径随之跳过）→ 实验目录集合 = 本实验 results 目录 + --extra 指定的
其它实验目录（保留含 stage-align/scores_raw.parquet 者）→ write_report 产出
results/<exp_name>/evaluation/{metrics.csv, report.md, gen_quality.json}。

baseline_dir 恒传 paths.baselines_dir：baselines.parquet 未跑基线时不存在，
write_report 自行降级（指标表只含 two-stage 各口径）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.data.grouping import load_tag_groups  # noqa: E402
from movietaggen.data.movielens import load_user_tags  # noqa: E402
from movietaggen.eval.report import write_report  # noqa: E402


def main(config_path: str = "configs/default.yaml",
         extra_exps: list[str] | None = None) -> int:
    config = load_config(config_path)
    paths = paths_for(config)

    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups) if paths.tag_groups.exists() else None
    user_tags_path = paths.movielens / "tags.csv"
    user_tags = load_user_tags(user_tags_path) if user_tags_path.exists() else None

    exps = [paths.results] + [paths.root / "results" / e for e in (extra_exps or [])]
    exp_dirs = [e for e in exps if (e / "stage-align" / "scores_raw.parquet").exists()]
    out = write_report(exp_dirs, paths.baselines_dir, paths.results / "evaluation",
                       config, dataset, truth, tag_groups, user_tags,
                       cache_root=paths.cache_embed)
    print(f"[08] 评估完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="参与对比的其他实验名（如 ablation/video-only）")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.extra))
