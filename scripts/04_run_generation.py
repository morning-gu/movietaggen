"""04 · 阶段一生成主跑（mock 配置零成本；--dry-run 打印 prompt 不调用不写文件）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from movietaggen.config import load_config, paths_for  # noqa: E402
from movietaggen.generate.cache import ResponseCache  # noqa: E402
from movietaggen.generate.client import make_client  # noqa: E402
from movietaggen.generate.pipeline import run_generation  # noqa: E402
from movietaggen.stages import save_config_snapshot  # noqa: E402


def main(config_path: str = "configs/default.yaml", dry_run: bool = False) -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    client = make_client(config)
    cache = None if config.mock else ResponseCache(paths.cache_llm)
    out = run_generation(config, paths, client, cache, dry_run=dry_run)
    # 配置快照随结果落盘（结果目录自包含、可复现）；dry-run 承诺不写文件，故跳过
    if out is not None:
        save_config_snapshot(config_path, paths)
    if out is None:
        print("[04] dry-run 完成（未调用、未写文件）")
    else:
        print(f"[04] 生成完成 → {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="打印 prompt，不调用不写文件")
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.dry_run))
