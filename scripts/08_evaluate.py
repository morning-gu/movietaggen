"""08 · 评估主跑（两阶段各产物 + 基线 → 指标表 / 报告 / 生成质量双口径）。

管线（复用 Task 18 的 write_report，接线模式照 06 脚本）：
load_config → paths_for → 读 dataset / truth（genome_scores）/ tag_groups
（文件可缺 → None，tag:* 子集随之跳过）/ user_tags（tags.csv 可缺 → None，
嵌入口径随之跳过）→ 实验目录集合 = 本实验 results 目录 + --extra 指定的
其它实验目录（保留含 stage-align/scores_raw.parquet 者）→ write_report 产出
results/<exp_name>/evaluation/{metrics.csv, report.md, gen_quality.json}。

主实验缺 stage-align/scores_raw.parquet 时大声警告点名目录（M6）——评估仍
继续（--extra 的对比实验可照常出表），但绝不让 --extra 静默顶替主实验。

--kappa <csv_path>（I7，M4-M5 人工校准回路的出数口）：读回人工填好的标注
CSV → refill_kappa 配对（T15）→ cohen_kappa（T17），中文摘要行打印 +
kappa.json 落入 evaluation/（{"cohen_kappa", "n"}）；refill_kappa 返回
None（无有效配对 / 单侧全同类）时打印原因说明并跳过（不落 kappa.json）。

baseline_dir 恒传 paths.baselines_dir：baselines.parquet 未跑基线时不存在，
write_report 自行降级（指标表只含 two-stage 各口径）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.audit.calibration import refill_kappa  # noqa: E402
from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.data.grouping import load_tag_groups  # noqa: E402
from movietaggen.data.movielens import load_user_tags  # noqa: E402
from movietaggen.eval.metrics import cohen_kappa  # noqa: E402
from movietaggen.eval.report import write_report  # noqa: E402


def run_kappa(csv_path: Path, out_dir: Path) -> None:
    """--kappa 接线（I7）：refill_kappa（T15）配对 → cohen_kappa（T17）→
    中文摘要 + kappa.json；无有效配对（未填 / 单类）时打印原因并跳过。"""
    pairs = refill_kappa(csv_path)
    if pairs is None:
        print(f"[08] kappa：{csv_path} 无有效配对行（human_verdict 未填、"
              "仅有 parse_failed 行、或任一侧全同类），跳过 kappa 计算")
        return
    judge_list, human_list = pairs
    kappa = cohen_kappa(judge_list, human_list)
    print(f"[08] Cohen's kappa（judge vs 人工）= {kappa:.4f}"
          f"（n={len(judge_list)} 有效配对）")
    (out_dir / "kappa.json").write_text(
        json.dumps({"cohen_kappa": kappa, "n": len(judge_list)},
                   ensure_ascii=False, indent=1), encoding="utf-8")


def main(config_path: str = "configs/default.yaml",
         extra_exps: list[str] | None = None,
         kappa_csv: str | None = None) -> int:
    config = load_config(config_path)
    paths = paths_for(config)

    dataset = pd.read_parquet(paths.dataset)
    truth = pd.read_parquet(paths.genome_scores)
    tag_groups = load_tag_groups(paths.tag_groups) if paths.tag_groups.exists() else None
    user_tags_path = paths.movielens / "tags.csv"
    user_tags = load_user_tags(user_tags_path) if user_tags_path.exists() else None

    # M6：主实验缺 stage-align 评分 → 大声警告点名（评估可继续，但绝不静默
    # 换掉主对象——只有 --extra 实验出表会让人误把对比臂当主结果读）
    if not (paths.results / "stage-align" / "scores_raw.parquet").exists():
        print(f"[08] 警告: 主实验 {paths.results} 缺少 "
              "stage-align/scores_raw.parquet——指标表将不含本实验的 "
              "two-stage 口径（先跑 05，或检查 exp_name 是否写对）")

    exps = [paths.results] + [paths.root / "results" / e for e in (extra_exps or [])]
    exp_dirs = [e for e in exps if (e / "stage-align" / "scores_raw.parquet").exists()]
    out = write_report(exp_dirs, paths.baselines_dir, paths.results / "evaluation",
                       config, dataset, truth, tag_groups, user_tags,
                       cache_root=paths.cache_embed)
    # I7：人工校准回路出数口（--kappa 给定时）
    if kappa_csv:
        run_kappa(Path(kappa_csv), out)
    print(f"[08] 评估完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="参与对比的其他实验名（如 ablation/video-only）")
    parser.add_argument("--kappa", default=None,
                        help="人工标注 CSV 路径（refill_kappa 配对 → cohen_kappa"
                             " → evaluation/kappa.json）")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.extra, args.kappa))
