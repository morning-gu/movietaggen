"""端到端冒烟：configs/smoke.yaml（合成样本 + mock）串联 00→08。

用途：日常回归 + pytest 集成测试。幂等：先清空 results/smoke/（基线目录
results/baselines/ 为跨实验共享产物，不清理——重跑直接覆盖同名文件）。
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movietaggen.config import load_config, paths_for  # noqa: E402

STEPS = ["00_make_synthetic", "01_verify_data", "02_build_dataset",
         "03_analyze_freetags", "04_run_generation", "05_run_alignment",
         "06_run_audit", "07_run_baselines", "08_evaluate"]


def main() -> int:
    # Windows 中文控制台默认 GBK(CP936)，本脚本的 ✓ (U+2713) 无法编码会直接
    # UnicodeEncodeError 崩溃；降级为 replace 后中文仍可读，✓ 显示为 ?。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cfg_path = "configs/smoke.yaml"
    config = load_config(cfg_path)
    paths = paths_for(config)
    if paths.results.exists():
        shutil.rmtree(paths.results)          # 幂等

    for step in STEPS:
        mod = importlib.import_module(f"scripts.{step}")
        rc = mod.main(cfg_path)
        if rc != 0:
            print(f"[smoke] {step} 失败 (rc={rc})")
            return 1
        print(f"[smoke] {step} ✓")

    # 关键产物断言
    checks = [
        paths.tag_groups,                                   # data/synthetic/processed/tag_groups.json
        paths.interim / "freetag_stats.json",               # 03 自由标签统计（合成源 → data/synthetic/interim/）
        paths.results / "config.yaml",                      # 配置快照（结果目录自包含，I1）
        paths.stage_gen / "tags.parquet",
        paths.stage_gen / "run_manifest.json",
        paths.stage_align / "scores_calibrated.parquet",
        paths.stage_align / "calibration.json",
        paths.stage_audit / "judge_mapping.parquet",
        paths.stage_audit / "judge_gen_quality.parquet",
        paths.stage_audit / "scores_filtered.parquet",
        paths.stage_audit / "calibration_sample.csv",
        paths.results / "evaluation" / "metrics.csv",
        paths.results / "evaluation" / "gen_quality.json",
        paths.baselines_dir / "baselines.parquet",
        paths.baselines_dir / "config.yaml",                # 基线配置快照（I1）
    ]
    for c in checks:
        if not c.exists():
            print(f"[smoke] 缺少产物: {c}")
            return 1
    manifest = json.loads(
        (paths.stage_gen / "run_manifest.json").read_text(encoding="utf-8"))
    print(f"[smoke] PASS — stage-gen tasks={manifest.get('tasks')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
