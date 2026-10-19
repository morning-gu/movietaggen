"""00 · 生成合成样本到 data/synthetic/。幂等：重新生成覆盖旧文件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.data.sample import make_synthetic_dataset  # noqa: E402


def main(config_path: str = "configs/smoke.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    out_root = paths.movienet.parent.parent  # data/synthetic/
    make_synthetic_dataset(out_root)
    print(f"[00] 合成样本已生成: {out_root}")
    print(f"     MovieNet 镜像: {paths.movienet}")
    print(f"     MovieLens 镜像: {paths.movielens}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()
    raise SystemExit(main(args.config))
