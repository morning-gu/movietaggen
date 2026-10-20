"""02 · 对齐 + 产出实验数据集（逻辑在 movietaggen.data.align.build_dataset）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.data.align import build_dataset  # noqa: E402


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    stats = build_dataset(paths_for(config), universe=config.data.universe)
    print(f"[02] 数据集构建完成: {stats['movies']} 部电影 "
          f"(universe={stats['universe']}, 剔除 {stats['filtered_out']} 部), "
          f"{stats['genome_rows']} 行 genome scores")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
