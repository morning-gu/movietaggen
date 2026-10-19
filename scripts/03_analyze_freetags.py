"""03 · tags.csv 自由标签分布分析（B 特有）：分布统计 + genome 字面重合报告。

逻辑在 movietaggen.data.freetag.analyze_freetags；产物为
paths.interim/freetag_stats.json 与 freetag_report.md。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.data.freetag import analyze_freetags  # noqa: E402


def main(config_path: str = "configs/default.yaml") -> int:
    paths = paths_for(load_config(config_path))
    try:
        result = analyze_freetags(paths)
    except FileNotFoundError as e:
        print(f"[03] {e}")
        return 1
    if result is None:
        print(f"[03] 未找到用户标签 {paths.movielens / 'tags.csv'}，跳过自由标签分析")
        return 0
    stats, overlap = result["freetag_stats"], result["literal_overlap"]
    print(f"[03] 自由标签分析完成: {stats['n_rows']} 行 / {stats['n_unique']} 个唯一标签"
          f"（实验集内），genome 字面重合 {overlap['exact_match']}/{overlap['top_n_total']}")
    print(f"     报告: {paths.interim / 'freetag_report.md'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
