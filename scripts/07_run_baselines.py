"""07 · 基线主跑（direct_score / genre_map / random / bert → baselines.parquet）。

管线（按 config.baselines.run 顺序执行、逐方法拼接，接线模式照 05/06 脚本）：
load_config → paths_for → 读 dataset / genome_scores / genome_tags → 逐方法
dispatch：
- direct_score 按 config.baselines.direct_score_source 分流：
  a_borrow → load_direct_score_a 读 A 的评分迁移物（groupby 逐对均值聚合）——
  固定工件恒 88 部，不建 client / cache、不受 movie_limit 限制；
  own_run → A 分组打分管道复刻（T19）。客户端工厂 mock 配置 →
  MockScoringClient（零网络零缓存）；否则 OpenAIClient——取生成配置（model
  配置的模型/温度/限速/extra_body），非 judge 配置（direct_score 用主生成
  模型打分，同 A）；缓存 mock 不读不写，否则 ResponseCache(cache_llm)。
- 分割源（440 协议）：train / test 一律来自 fixed_split——test = eval_test
  清单命中的电影，train = 全集 − 清单（替代时序分割，成为校准 / few-shot /
  基线拟合的统一分割源；清单坏输入大声失败）。
- genre_map / bert：拟合参考集只允许 fixed_split 训练侧电影，测试侧 = 清单
  电影；genre_map 与 bert_baseline 内部均以 assert_no_leakage 收尾自检。
- random：产出电影集 × genome_tags 全 tag 的哈希均匀分（确定性）。
- 未知方法名 → 中文 ValueError 点名合法值（配置拼错不静默）。
- 电影集口径（I5 · 440 裁定扩展）：data.movie_limit 设置时，全部受限方法
  （own_run direct_score / genre_map / random / bert）在同一
  sample_limited_movie_ids 输出电影集上产出——抽样池 = fixed_split 训练侧，
  受限跑永不接触 eval_test 清单电影；movie_limit=None → 全量升序（不读
  清单）。a_borrow 的 direct_score 不受 movie_limit 限制（固定工件恒 88 部）。
  拟合参考集（训练侧全体）不受 movie_limit 限制——受限的是产出/评估的电影轴。

bert 懒导入（torch/transformers 在训练函数内部导入）：模块自身缺 torch 也能
导入，缺依赖在调用点抛 ImportError，被本脚本捕获后优雅跳过（照 A 的 05
语义），其余基线照常产出。

产物（baselines_dir 固定 results/baselines，不随 exp_name）：baselines.parquet
（列 movie_id, tag_id, score, method）+ run_manifest.json（{"methods": 实际
产出方法的有序列表, "rows": 总行数}；a_borrow 的 direct_score 另附
direct_score_provenance 键——A 迁移物溯源引用：provenance JSON 的
source_repo 与 direct_score_a_sha256 两字段，文件缺失记
{"file": 路径, "missing": true} 不崩，溯源引用不是守门）+ config.yaml 快照
（结果目录自包含）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# 仓库根加入 sys.path，使脚本在未安装包的情况下也能直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from movietaggen.baselines.direct_score import (  # noqa: E402
    load_direct_score_a,
    run_direct_score,
)
from movietaggen.baselines.genre_map import genre_map  # noqa: E402
from movietaggen.baselines.mock_scoring import MockScoringClient  # noqa: E402
from movietaggen.baselines.random_baseline import random_baseline  # noqa: E402
from movietaggen.config import AppConfig, load_config, paths_for  # noqa: E402
from movietaggen.eval.protocol import (  # noqa: E402
    fixed_split,
    sample_limited_movie_ids,
)
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


def load_direct_score_provenance(processed_dir) -> dict:
    """A 迁移物的 PROVENANCE 溯源引用（spec §2.3，终审修复波 M2）。

    读 processed/direct_score_a.provenance.json，嵌入 source_repo 与
    direct_score_a_sha256 两字段进 run_manifest——a_borrow 产物可追溯到 A 的
    源仓库与文件校验和。溯源是引用不是守门：文件缺失不崩，键值降级为
    {"file": 路径, "missing": true} 留痕（守门在 load_direct_score_a 内部）。
    """
    prov_path = Path(processed_dir) / "direct_score_a.provenance.json"
    if not prov_path.is_file():
        return {"file": str(prov_path), "missing": True}
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    return {"source_repo": prov.get("source_repo"),
            "direct_score_a_sha256": prov.get("direct_score_a_sha256")}


def main(config_path: str = "configs/default.yaml") -> int:
    config = load_config(config_path)
    paths = paths_for(config)
    dataset = pd.read_parquet(paths.dataset)
    genome_scores = pd.read_parquet(paths.genome_scores)
    genome_tags = pd.read_parquet(paths.genome_tags)

    # 【防泄露契约】genre_map / bert 的拟合参考集只允许固定分割训练侧电影
    # （dataset − eval_test 清单；440 协议）：测试清单电影的 genome 相关度一旦
    # 进入拟合集，基线等于"见过"测试电影的 ground truth，基线对比会系统性
    # 虚高。分割源统一为 fixed_split（eval_test 清单逐行整数，缺失/坏行/越界
    # 大声失败），与 05 校准接线同款；genre_map / bert_baseline 内部再以
    # assert_no_leakage 收尾自检。
    train_ids, test_ids = fixed_split(dataset, paths.eval_test)
    test_ids = set(test_ids)   # set 契约（genre_map / bert_baseline 的入参类型）
    train_scores = genome_scores[genome_scores["movie_id"].isin(set(train_ids))]

    # 【I5 · 受限跑训练池抽样（440 裁定扩展）】movie_limit 设置时，全部受限
    # 方法（own_run direct_score / genre_map / random / bert）在同一抽样电影集
    # 上产出：sample_limited_movie_ids 从 fixed_split 训练侧（dataset −
    # eval_test 清单）抽 min(limit, 池) 部——受限跑永不接触测试清单电影。
    # movie_limit=None → 全量升序（不读清单）。a_borrow 的 direct_score 是
    # 固定工件（恒 88 部），不参与抽样、不受 movie_limit 限制。拟合参考集
    # （训练侧全体）不受 movie_limit 限制——受限的是产出/评估的电影轴。
    movie_ids = sample_limited_movie_ids(
        dataset, config.data.movie_limit, config.data.seed, paths.eval_test)
    movie_set = set(movie_ids)
    dataset_for_random = dataset[dataset["movie_id"].isin(movie_set)]

    frames = []
    methods = []
    # a_borrow 跑 direct_score 时的溯源引用（spec §2.3；own_run / 未跑
    # direct_score 时为 None，manifest 不带该键）
    direct_score_provenance = None
    for name in config.baselines.run:
        if name == "direct_score":
            if config.baselines.direct_score_source == "a_borrow":
                # a_borrow：读 A 的评分迁移物（固定工件恒 88 部）——不建
                # client / cache、不受 movie_limit；电影宇宙 / 测试集相等 /
                # 网格完整 / modality_set 守门在 load_direct_score_a 内部。
                # 另取 PROVENANCE 溯源引用进 manifest（spec §2.3）
                frames.append(load_direct_score_a(paths, dataset))
                direct_score_provenance = load_direct_score_provenance(
                    paths.processed)
            else:
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
    if direct_score_provenance is not None:
        # a_borrow 的 direct_score 附溯源引用（spec §2.3）：产物可追溯到 A 的
        # 源仓库与迁移物 sha256（provenance 文件缺失时为 missing 标记留痕）
        manifest["direct_score_provenance"] = direct_score_provenance
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
