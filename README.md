# MovieTagGen

MLLM 两阶段生成式电影标签提取与 genome 标签对齐的研究实验框架（自由生成 →
嵌入映射 → genome 对齐）。研究问题、章节计划与文献定位见
`docs/research-plan.md`；框架设计见 `docs/superpowers/specs/2026-10-19-movietaggen-framework-design.md`。

## 安装

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
# 注：sentence-transformers 会带入 torch（bert 基线所需），无需单独安装
cp .env.example .env   # 填入 OPENAI_API_KEY / OPENAI_BASE_URL
```

## 验证安装（零成本）

```bash
.venv/Scripts/python -m pytest tests/ -v      # 全部测试：无网络、无 key、无模型下载
.venv/Scripts/python scripts/run_smoke.py     # 端到端冒烟（合成样本 + mock）串联 00→08
```

## 真实数据流程

1. 按 `docs/download-guide.md` 下载数据（默认复用方向 A 的 `data/raw/`，
   由 `data.raw_dir` 指向 `../genometag-mllm/data/raw`）
2. `.venv/Scripts/python scripts/01_verify_data.py`（检查文件与**年份分布**）
3. `.venv/Scripts/python scripts/02_build_dataset.py` → 检查 `data/interim/align_report.md`
4. `.venv/Scripts/python scripts/03_analyze_freetags.py`（一次性分析，产物
   `data/interim/freetag_stats.json` + `freetag_report.md`）
5. 小规模试跑：`configs/pilot.yaml` 已设 `data.movie_limit: 50`，
   `.venv/Scripts/python scripts/04_run_generation.py --config configs/pilot.yaml`
6. `.venv/Scripts/python scripts/05_run_alignment.py --config configs/pilot.yaml`
7. `.venv/Scripts/python scripts/06_run_audit.py --config configs/pilot.yaml`
8. `.venv/Scripts/python scripts/07_run_baselines.py --config configs/pilot.yaml`
9. `.venv/Scripts/python scripts/08_evaluate.py --config configs/pilot.yaml`

全量实验直接用 `configs/default.yaml`（`movie_limit: null` = 全量；换实验名时
复制另存改 `exp_name` 即可）。长跑 04 建议用 `scripts/watchdog.sh`
（断点续跑 + 卡死自动重启，见脚本头注释）。

## 消融实验

`configs/ablation/` 下 16 个配置覆盖九轴消融矩阵（模态 3 + title-only、维度
5、以及 top_k / threshold / agg / embed_model / calib_method / n_runs 各轴）。
模态/维度/n_runs 轴臂生成产物不同，需各自跑 04→05；对齐参数轴臂可把
`align.input_exp`（需要审计口径时连同 `audit.input_exp`）指向主实验、复用
stage-gen 只跑 05。最后统一评估：

```bash
.venv/Scripts/python scripts/08_evaluate.py --config configs/default.yaml \
    --extra ablation/video-only ablation/script-only ablation/dialogue-only ...
```

`--extra` 接受多个实验名（对应 `results/` 下的目录名），评估表含
two-stage[raw/calibrated] 与各基线逐口径对比，产物入 `evaluation/`。

## 目录约定与产物 schema

见 `docs/data-format.md`。API 调用缓存在 `.cache/llm/`、嵌入缓存在
`.cache/embed/`（重跑零成本、可断点续跑）。
