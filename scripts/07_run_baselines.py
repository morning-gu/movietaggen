"""07 · 基线主跑（direct_score / genre_map / random / bert → baselines.parquet）。

管线（按 config.baselines.run 顺序执行、逐方法拼接，接线模式照 05/06 脚本）：
load_config → paths_for → 读 dataset / genome_scores / genome_tags → 逐方法
dispatch：
- direct_score：A 分组打分管道复刻（T19）。客户端工厂 mock 配置 →
  MockScoringClient（零网络零缓存）；否则 OpenAIClient——取生成配置（model
  配置的模型/温度/限速/extra_body），非 judge 配置（direct_score 用主生成
  模型打分，同 A）；缓存 mock 不读不写，否则 ResponseCache(cache_llm)。
- genre_map / bert：拟合参考集只允许 year < protocol.temporal_split_year 的
  电影（内联过滤，防泄露契约见 main），测试集 = year ≥ split；genre_map 与
  bert_baseline 内部均以 assert_no_leakage 收尾自检。
- random：dataset 全电影 × genome_tags 全 tag 的哈希均匀分（确定性）。
- 未知方法名 → 中文 ValueError 点名合法值（配置拼错不静默）。
- 电影集口径（I5）：data.movie_limit 设置时，全部方法在同一抽样电影集上
  产出（与生成管道同代码路径：sorted ids + random.Random(seed).sample）；
  direct_score 内部抽样即同一路径，无需传参自然对齐；拟合参考集（时序
  训练侧）不受 movie_limit 限制——受限的是产出/评估的电影轴。

bert 懒导入（torch/transformers 在训练函数内部导入）：模块自身缺 torch 也能
导入，缺依赖在调用点抛 ImportError，被本脚本捕获后优雅跳过（照 A 的 05
语义），其余基线照常产出。

产物（baselines_dir 固定 results/baselines，不随 exp_name）：baselines.parquet
（列 movie_id, tag_id, score, method）+ run_manifest.json（{"methods": 实际
产出方法的有序列表, "rows": 总行数}）+ config.yaml 快照（结果目录自包含）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.baselines.direct_score import run_direct_score  # noqa: E402
from movietaggen.baselines.genre_map import genre_map  # noqa: E402
from movietaggen.baselines.mock_scoring import MockScoringClient  # noqa: E402
from movietaggen.baselines.random_baseline import random_baseline  # noqa: E402
from movietaggen.config import AppConfig, load_config, paths_for  # noqa: E402
from movietaggen.generate.cache import ResponseCache  # noqa: E402
from movietaggen.generate.client import OpenAIClient  # noqa: E402
from movietaggen.stages import save_config_snapshot  # noqa: E402

# 合法方法名（config.baselines.run 的元素域；未知值在 dispatch 处报错）
VALID_BASELINE_METHODS = ["direct_score", "genre_map", "random", "bert"]

_BASELINE_COLUMNS = ["movie_id", "tag_id", "score", "method"]


def make_scoring_client(config: AppConfig):
    """direct_score 客户端工厂：mock 配置 → MockScoringClient；否则 OpenAIClient。

    与 make_client / make_judge_client 同构：load_dotenv 只在非 mock 路径调用
    （保持 import 无副作用）；模型/温度/限速/extra_body 取生成配置（model 配置）
    ——direct_score 用主生成模型打分（同 A 的打分管道），不是 judge 配置。
    OpenAIClient 构造是惰性的（transport 在首次 complete 时才建），此处无网络。
    """
    if config.mock:
        return MockScoringClient()
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAIClient(
        model=config.model.model,
        base_url_env=config.model.base_url_env,
        api_key_env=config.model.api_key_env,
        temperature=config.model.temperature,
        request_interval=config.model.request_interval,
        extra_body=config.model.extra_body)


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    genome_tags = pd.read_parquet(paths.genome_tags)

    # 【防泄露契约】genre_map / bert 的拟合参考集只允许 year < temporal_split_year
    # 的电影：测试分割（year ≥ split）的 genome 相关度一旦进入拟合集，基线等于
    # "见过"测试电影的 ground truth，基线对比会系统性虚高。year 缺失/无法解析
    # 的电影按不可信处理，绝不进拟合集。此处为内联过滤，与 T18
    # protocol.temporal_split 语义一致（同 05 校准的内联模式），genre_map /
    # bert_baseline 内部再以 assert_no_leakage 收尾自检。
    year = pd.to_numeric(dataset["year"], errors="coerce")
    train_ids = dataset.loc[year < config.protocol.temporal_split_year,
                            "movie_id"].tolist()
    test_ids = set(dataset.loc[year >= config.protocol.temporal_split_year,
                               "movie_id"].tolist())
    train_scores = genome_scores[genome_scores["movie_id"].isin(set(train_ids))]

    # 【I5 · 试点规模可比性】genre_map / random / bert 与 direct_score 必须在
    # 同一电影集上产出，方法对比才有意义：movie_limit 设置时按生成管道
    # （generate/pipeline.py）完全相同的代码路径抽样——sorted id 列表 +
    # random.Random(data.seed).sample，选定集合与 04/直接评分基线逐元素一致。
    # 抽样后排序仅定行序、不改集合成员。direct_score 的内部抽样即这两行代码
    # （movietaggen/baselines/direct_score.py run_direct_score），无需传参即
    # 落在同一集合，此处不再重复传入。
    all_movie_ids = sorted(dataset["movie_id"].tolist())
    if (config.data.movie_limit is not None
            and config.data.movie_limit < len(all_movie_ids)):
        movie_ids = sorted(random.Random(config.data.seed).sample(
            all_movie_ids, config.data.movie_limit))
    else:
        movie_ids = all_movie_ids
    movie_set = set(movie_ids)
    dataset_for_random = dataset[dataset["movie_id"].isin(movie_set)]

    frames = []
    methods = []
    for name in config.baselines.run:
        if name == "direct_score":
            client = make_scoring_client(config)
            cache = None if config.mock else ResponseCache(paths.cache_llm)
            frames.append(run_direct_score(config, paths, client, cache))
        elif name == "genre_map":
            frames.append(genre_map(train_scores, dataset, movie_ids,
                                    test_ids=test_ids))
        elif name == "random":
            frames.append(random_baseline(
                dataset_for_random,
                sorted(int(t) for t in genome_tags["tag_id"]),
                config.data.seed))
        elif name == "bert":
            try:
                from movietaggen.baselines.bert_baseline import bert_baseline
                frames.append(bert_baseline(config, paths, dataset, genome_scores,
                                            train_ids, test_ids,
                                            predict_ids=movie_ids))
            except ImportError:
                # 懒导入：模块自身无 torch 也能导入，缺依赖在调用点才抛出
                print("[07] torch/transformers 未安装，跳过 bert 基线"
                      "（pip install torch transformers）")
                continue
        else:
            raise ValueError(
                f"未知基线方法: {name!r}（合法值: {VALID_BASELINE_METHODS}）")
        methods.append(name)

    # 空运行列表（或 bert 全被跳过）可表示：空 schema 产物 + 零行 manifest
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=_BASELINE_COLUMNS))
    paths.baselines_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(paths.baselines_dir / "baselines.parquet", index=False)
    manifest = {"methods": methods, "rows": int(len(out))}
    (paths.baselines_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # 配置快照随结果落盘（结果目录自包含、可复现）。save_config_snapshot 的
    # 落点由 paths.results 决定，07 的结果目录是 baselines_dir（固定
    # results/baselines、不随 exp_name），以最小 shim 复用同一函数（落点
    # baselines_dir/config.yaml）。
    save_config_snapshot(config_path, SimpleNamespace(results=paths.baselines_dir))
    print(f"[07] 基线完成 → {paths.baselines_dir / 'baselines.parquet'}"
          f"（methods={len(methods)}, rows={len(out)}）")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    raise SystemExit(main(parser.parse_args().config))
