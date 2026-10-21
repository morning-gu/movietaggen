"""01 · 数据完整性校验：必需文件存在、行数、年份分布（spec §9 风险的检查点）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.config import load_config, paths_for  # noqa: E402

REQUIRED_FILES = [
    "genome-scores.csv", "genome-tags.csv", "movies.csv", "links.csv", "tags.csv",
]


def verify_data(paths) -> list[str]:
    """返回问题描述清单；空列表 = 通过。"""
    problems: list[str] = []
    for name in REQUIRED_FILES:
        if not (paths.movielens / name).exists():
            problems.append(f"缺少 MovieLens 文件: {paths.movielens / name}")
    if not (paths.movienet / "movies.csv").exists():
        problems.append(f"缺少 MovieNet 元数据: {paths.movienet / 'movies.csv'}")
    if problems:
        return problems
    # 年份分布（对齐可行性的参考信息；temporal_split_year 仅驱动 metrics.csv
    # 的年代分桶描述性视图，协议分割已固定为 eval_test 清单，与此处无关）
    try:
        mn = pd.read_csv(paths.movienet / "movies.csv")
        if "year" in mn.columns:
            dist = mn["year"].dropna().astype(int).value_counts().sort_index()
            print("[01] MovieNet 年份分布（前 10 / 后 10）：")
            print(dist.head(10).to_string())
            print("...")
            print(dist.tail(10).to_string())
            latest = int(dist.index.max())
            print(f"[01] 最新年份: {latest}（描述性参考：协议分割已固定为"
                  " eval_test 清单，不依赖该值）")
    except Exception as e:  # noqa: BLE001 —— 校验脚本要报告而非崩溃
        problems.append(f"MovieNet movies.csv 解析失败: {e}")
    return problems


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    problems = verify_data(paths)
    if problems:
        print("[01] 数据校验失败：")
        for p in problems:
            print(f"  - {p}")
        return 1
    # 宇宙口径（440 协议）：dataset 与对齐产物均在时报告数据集规模。K = M − N 是
    # "对齐→数据集共减"数（genome 覆盖剔除 + 宇宙剔除两层合计），与 02 打印的
    # filtered_out（仅宇宙层）含义不同——措辞用"共减"而非裸"剔除"，避免日志歧义。
    # 任一产物缺失（尚未跑 02）则跳过该行不报错。
    if paths.dataset.exists() and paths.aligned.exists():
        n_ds = len(pd.read_parquet(paths.dataset))
        n_al = len(pd.read_parquet(paths.aligned))
        print(f"[01] 宇宙: {config.data.universe} → dataset {n_ds} 部 / "
              f"对齐 {n_al} 部（对齐→数据集共减 {n_al - n_ds} 部："
              f"genome 覆盖剔除 + 宇宙剔除两层合计）")
    print("[01] 数据校验通过")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
