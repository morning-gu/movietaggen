"""配置加载、校验与项目路径推导。

约定优于配置：数据/缓存/结果目录布局固定（见 Paths），YAML 只描述实验参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_DIMENSIONS = ("style", "mood", "theme", "narrative", "character")
VALID_MODALITIES = ("video", "script", "dialogue")
VALID_SOURCES = ("real", "synthetic")
VALID_UNIVERSES = ("three_modality", "all")   # 440 协议评测宇宙
VALID_DS_SOURCES = ("a_borrow", "own_run")    # direct_score 基线的分数来源
VALID_AGG = ("max", "mean")
VALID_CALIB_METHODS = ("isotonic", "linear")
VALID_JUDGE_TASKS = ("mapping", "gen_quality")
VALID_JUDGE_CONTEXTS = ("pair", "pair+title", "pair+full")


@dataclass
class DataConfig:
    source: str = "real"            # real | synthetic
    universe: str = "three_modality"  # three_modality | all：评测宇宙（440 协议）
    movie_limit: int | None = None  # None = 全量
    seed: int = 42
    # 仅 source: real 生效；相对路径按仓库根解析；None = 仓库内 data/raw
    raw_dir: str | None = "../genometag-mllm/data/raw"


@dataclass
class ModelConfig:
    base_url_env: str = "OPENAI_BASE_URL"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "qwen3.7-plus"
    temperature: float = 0.7
    max_frames: int = 12
    script_token_budget: int = 4000
    dialogue_token_budget: int = 3000
    request_interval: float = 0.5   # 两次请求间隔秒数（简单限速）
    extra_body: dict = field(default_factory=dict)  # 端点专有参数透传（如 DashScope {enable_thinking: false}）


@dataclass
class GenerateConfig:
    n_runs: int = 3
    # 空列表合法：仅 title 的探针实验
    modalities: list[str] = field(default_factory=lambda: list(VALID_MODALITIES))
    dimensions: list[str] = field(default_factory=lambda: list(VALID_DIMENSIONS))
    few_shot_examples: int = 0
    workers: int = 1   # 并行线程数（1=串行）。任务相互独立，结果与串行等价


@dataclass
class AlignConfig:
    embed_model: str = "intfloat/multilingual-e5-base"
    embed_device: str = "cpu"
    top_k: int = 3
    threshold: float = 0.5
    agg: str = "max"                # max | mean：top-k 相似度聚合方式
    input_exp: str | None = None    # None = 用本配置的 generate 产物；字符串 = 复用其它实验
    calib_method: str = "isotonic"  # isotonic | linear：分数校准


@dataclass
class AuditConfig:
    input_exp: str | None = None
    judge_tasks: list[str] = field(default_factory=lambda: ["mapping", "gen_quality"])
    judge_model: str = "qwen3.7-plus"
    judge_temperature: float = 0.0
    judge_context: str = "pair+title"  # pair | pair+title | pair+full：评审上下文
    sample_rate: float = 1.0           # 抽样比例 (0,1]
    apply_filter: bool = True
    calibration_sample: int = 500
    seed: int = 42
    workers: int = 1                   # judge 并行线程数（1=串行）；04 同款线程池模式


@dataclass
class EvalConfig:
    embed_model: str = "intfloat/multilingual-e5-base"
    coverage_top_k: int = 10
    # E5 余弦分布压缩偏高（映射对实测 0.80+），0.7 会让 coverage 平凡=1.0——
    # pilot 2026-10-22 定标 0.85
    coverage_tau: float = 0.85


@dataclass
class BaselinesConfig:
    run: list[str] = field(default_factory=lambda: ["direct_score", "genre_map", "random", "bert"])
    n_runs: int = 3
    few_shot_examples: int = 2
    # direct_score 的分数来源：a_borrow 读 A 迁移评分（真实数据工件，仅 source: real
    # 合法，validate_config 拦截）；own_run 用本仓库 direct_score 管道自跑
    direct_score_source: str = "a_borrow"


@dataclass
class ProtocolConfig:
    temporal_split_year: int = 2015
    relevance_threshold: float = 0.5
    top_k: int = 10
    cold_start_max_tags: int = 0


# 顶层合法键：未知键一律报错（防止拼错/过时配置被静默忽略）。
KNOWN_TOP_LEVEL_KEYS = {
    "exp_name", "mock", "data", "model", "generate", "align", "audit",
    "eval", "baselines", "protocol",
}


@dataclass
class AppConfig:
    exp_name: str = "default"
    mock: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)


@dataclass
class Paths:
    root: Path
    movienet: Path      # MovieNet 原始数据根（real 或 synthetic）
    movielens: Path     # MovieLens 数据根（含 genome-*.csv 的 ml-latest 目录）
    interim: Path
    processed: Path
    cache_llm: Path
    cache_embed: Path
    results: Path        # results/<exp_name>
    stage_gen: Path      # 阶段产物：生成
    stage_align: Path    # 阶段产物：对齐
    stage_audit: Path    # 阶段产物：审计
    tag_groups: Path
    dataset: Path
    genome_scores: Path
    genome_tags: Path
    aligned: Path
    baselines_dir: Path  # 固定 results/baselines，不随 exp_name
    eval_test: Path      # 固定测试清单：real 为 A 迁入提交入库，synthetic 由 00 生成
    direct_score_a: Path  # A 评分迁移物：仅 source: real 有文件，a_borrow 已被 config 校验拦在 synthetic 之外


def project_root() -> Path:
    """仓库根 = movietaggen/ 的上一级。"""
    return Path(__file__).resolve().parent.parent


def paths_for(config: AppConfig) -> Paths:
    root = project_root()
    if config.data.source == "synthetic":
        base = root / "data" / "synthetic"
        movienet = base / "raw" / "movienet"
        movielens = base / "raw" / "movielens"
        interim = base / "interim"
        processed = base / "processed"
    else:
        # raw_dir 仅 real 生效；相对路径按仓库根解析；None 退回仓库内 data/raw
        raw = Path(config.data.raw_dir) if config.data.raw_dir else root / "data" / "raw"
        if not raw.is_absolute():
            raw = root / raw
        movienet = raw / "movienet"
        movielens = raw / "movielens" / "ml-latest"
        interim = root / "data" / "interim"
        processed = root / "data" / "processed"
    results = root / "results" / config.exp_name
    return Paths(
        root=root, movienet=movienet, movielens=movielens,
        interim=interim, processed=processed,
        cache_llm=root / ".cache" / "llm",
        cache_embed=root / ".cache" / "embed",
        results=results,
        stage_gen=results / "stage-gen",
        stage_align=results / "stage-align",
        stage_audit=results / "stage-audit",
        tag_groups=processed / "tag_groups.json",
        dataset=processed / "dataset.parquet",
        genome_scores=processed / "genome_scores.parquet",
        genome_tags=processed / "genome_tags.parquet",
        aligned=interim / "aligned.parquet",
        baselines_dir=root / "results" / "baselines",
        # 两个分支同路径规则：固定测试清单与 A 评分迁移物都落在各自 processed 下
        # （eval_test：real 为 A 迁入提交入库、synthetic 由 00 生成；
        #   direct_score_a：仅 real 有文件，a_borrow 已被校验拦在 synthetic 之外）
        eval_test=processed / "eval_test.txt",
        direct_score_a=processed / "direct_score_a.parquet",
    )


def validate_config(cfg: AppConfig) -> None:
    if cfg.data.source not in VALID_SOURCES:
        raise ValueError(f"data.source 必须是 {VALID_SOURCES}，得到 {cfg.data.source!r}")
    if cfg.data.universe not in VALID_UNIVERSES:
        raise ValueError(f"data.universe 必须是 {VALID_UNIVERSES}，得到 {cfg.data.universe!r}")
    bad = [m for m in cfg.generate.modalities if m not in VALID_MODALITIES]
    if bad:
        raise ValueError(f"generate.modalities 含未知模态 {bad}，可选 {VALID_MODALITIES}")
    bad = [d for d in cfg.generate.dimensions if d not in VALID_DIMENSIONS]
    if bad:
        raise ValueError(f"generate.dimensions 含未知维度 {bad}，可选 {VALID_DIMENSIONS}")
    if not cfg.generate.dimensions:
        raise ValueError("generate.dimensions 不能为空（生成至少需要一个维度；modalities 可空）")
    if cfg.generate.n_runs < 1:
        raise ValueError("generate.n_runs 必须 ≥ 1")
    if cfg.generate.few_shot_examples < 0:
        raise ValueError("generate.few_shot_examples 必须 ≥ 0")
    if cfg.generate.workers < 1:
        raise ValueError("generate.workers 必须 ≥ 1（1=串行）")
    if cfg.align.agg not in VALID_AGG:
        raise ValueError(f"align.agg 必须是 {VALID_AGG}，得到 {cfg.align.agg!r}")
    if cfg.align.top_k < 1:
        raise ValueError("align.top_k 必须 ≥ 1（每自由标签至少取 1 个候选）")
    if not 0.0 < cfg.align.threshold < 1.0:
        raise ValueError("align.threshold 必须在 (0,1) 开区间")
    if cfg.align.calib_method not in VALID_CALIB_METHODS:
        raise ValueError(f"align.calib_method 必须是 {VALID_CALIB_METHODS}，得到 {cfg.align.calib_method!r}")
    bad = [t for t in cfg.audit.judge_tasks if t not in VALID_JUDGE_TASKS]
    if bad:
        raise ValueError(f"audit.judge_tasks 含未知任务 {bad}，可选 {VALID_JUDGE_TASKS}")
    if cfg.audit.judge_context not in VALID_JUDGE_CONTEXTS:
        raise ValueError(f"audit.judge_context 必须是 {VALID_JUDGE_CONTEXTS}，得到 {cfg.audit.judge_context!r}")
    if not 0.0 < cfg.audit.sample_rate <= 1.0:
        raise ValueError("audit.sample_rate 必须在 (0,1] 区间")
    if cfg.audit.workers < 1:
        raise ValueError("audit.workers 必须 ≥ 1（1=串行）")
    if cfg.baselines.n_runs < 1:
        raise ValueError("baselines.n_runs 必须 ≥ 1")
    if cfg.baselines.few_shot_examples < 0:
        raise ValueError("baselines.few_shot_examples 必须 ≥ 0")
    if cfg.baselines.direct_score_source not in VALID_DS_SOURCES:
        raise ValueError(
            f"baselines.direct_score_source 必须是 {VALID_DS_SOURCES}，"
            f"得到 {cfg.baselines.direct_score_source!r}")
    if cfg.baselines.direct_score_source == "a_borrow" and cfg.data.source != "real":
        raise ValueError(
            "baselines.direct_score_source=a_borrow 读取的迁移物"
            "（direct_score_a.parquet）是真实数据工件，synthetic 数据没有对应产物，"
            "请改用 own_run")
    if cfg.data.movie_limit is not None and cfg.data.movie_limit < 1:
        raise ValueError("data.movie_limit 为 None 或 ≥ 1")
    if not isinstance(cfg.model.extra_body, dict):
        raise ValueError("model.extra_body 必须是字典，如 {enable_thinking: false}")
    if not 0.0 < cfg.protocol.relevance_threshold < 1.0:
        raise ValueError("protocol.relevance_threshold 必须在 (0,1) 开区间")
    if cfg.protocol.top_k < 1:
        raise ValueError("protocol.top_k 必须 ≥ 1")


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(raw) - KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"配置含未知顶层键: {sorted(unknown)}（合法键: {sorted(KNOWN_TOP_LEVEL_KEYS)}）")
    # M3：mock 只收布尔——YAML 的 mock: "false" 是非空字符串，bool() 化为
    # True 会把想关 mock 的运行静默打成 mock（方向反转的静默失败比崩溃糟）。
    # 引号包住的 false/true 一律大声报错，提示用户去掉引号。
    mock_raw = raw.get("mock", False)
    if not isinstance(mock_raw, bool):
        raise ValueError(
            f"mock 必须是布尔值（true/false），得到 {mock_raw!r}"
            f"（{type(mock_raw).__name__}）——YAML 中请写 mock: false，"
            "去掉引号（mock: \"false\" 是字符串，会被当作真值）")
    cfg = AppConfig(
        exp_name=raw.get("exp_name", "default"),
        mock=mock_raw,
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        generate=GenerateConfig(**raw.get("generate", {})),
        align=AlignConfig(**raw.get("align", {})),
        audit=AuditConfig(**raw.get("audit", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        baselines=BaselinesConfig(**raw.get("baselines", {})),
        protocol=ProtocolConfig(**raw.get("protocol", {})),
    )
    validate_config(cfg)
    return cfg
