# MovieTagGen 项目设计文档（方向 B 实验框架）

> 日期：2026-10-19
> 状态：待用户终审（builder-check 已完成并合入决议：两项方法学决策经用户确认，六项代定待复核）
> 上游文档：`方向B_MovieTagGen_研究计划.md`（构建项目时复制入 `docs/research-plan.md`，同 A 的模式）
> 姊妹项目：`genometag-mllm/`（方向 A，代码与基础设施的复制来源）、`modaltag-decomp/`（方向 C）

---

## 1. 项目背景与目标

**研究问题**：MLLM 能否生成与 MovieLens 用户标签（tags.csv）语义一致的自由形式标签，并通过嵌入映射层对齐到 genome tag 体系（1,128 个标准化标签 + 关联度评分）？

**核心论点**（研究计划 B-Step 1）："MLLM 自由生成 → 嵌入映射 → genome 对齐"两阶段框架同时产出开放词表标签（对应 tags.csv）与标准化评分（对应 genome-scores）；LLM-as-judge 审计机制为自动化标签提供可量化的质量保证。

**本项目定位**：支撑方向 B 从数据准备到论文投稿（WWW / SIGIR 2027）全周期的实验代码库。本文档定义项目结构与实验框架的设计。

**完成度目标（本次构建）**：完整实验框架——数据准备管道（复制适配自 A）+ tags.csv 自由标签分析（B 特有）+ 阶段一生成 + 阶段二对齐 + LLM-as-judge 审计 + 评估 + 全部基线，在合成小样本上端到端可跑通。真实数据由用户手动下载到位后，同一套代码经配置切换即可处理（同 A 的模式）。

**反目标**（以下任一成立即视为失败，即使技术实现完好）：

1. **消融实验无法通过配置实现**——任何论文要消融的变量若需要改代码才能切换，违背本框架的核心目的（用户在需求确认中明确的最怕点）。
2. **阶段结果不可归因**——无法区分"生成阶段的错"与"对齐阶段的错"，论文核心风险"两阶段误差累积"就无从实证分析。
3. **API 成本失控**——同参数重复调用不命中缓存；消融下游阶段时重跑上游。
4. **仓库不独立自包含**——运行时硬依赖 A 的代码或绝对路径（`data.raw_dir` 默认相对指向 A 的原始数据除外：必须可配置覆盖，且 download-guide 说明独立获取方式）。

### 1.1 已确认的需求决策（用户已拍板）

| 决策点 | 结论 |
|--------|------|
| 仓库形态 | 平级独立 git 仓库，与 A/C 相同模式 |
| A 资产复用 | **复制适配**：代码裁剪复制进 B 后按需改造；原始数据不复制，`data.raw_dir` 配置默认指向 A 的 `data/raw` |
| 建设范围 | **全框架**：数据 / 生成 / 对齐 / 审计 / 评估 / 基线六模块；端到端推荐评估（M5-M6）只定义接口不实现 |
| 消融保证 | 实验变量全部配置化 + 阶段产物隔离落盘（§2 两条硬保证） |
| 消融轴 | 九轴清单已确认（§8），另有一项 title-only 候选臂待终审定夺 |
| 对齐评估口径 | **校准 + 双口径**（2026-10-19 builder-check 后确认）：余弦相似度在训练分割上校准到 genome [0,1] 量纲（防泄露）；RMSE 同时报全谱（未映射 tag 补 0）与仅已预测对两个口径；Spearman / Precision@K 作量纲无关指标；direct_score 保持稠密形态 |
| direct_score 复刻度 | **完整复刻 A**（2026-10-19 builder-check 后确认）：按 A 的实际机制——约 20 个 tag 组逐组打分、每组带 tag 定义、few-shot 2 部参考电影（取自训练分割）、JSON 输出 {tag: score}；`tag_groups.json` 作为 baseline 基础设施迁入 B（明确不在 §14"genome 评分专有逻辑剔除"之列） |
| 语言约定 | README / 注释 / 文档中文，代码标识符英文（沿用 A/C） |

### 1.2 设计者代定的默认选择（初始，终审时可推翻）

| 项 | 默认 | 理由 |
|----|------|------|
| 目录 / 包名 | `movietaggen` | 对齐计划名 MovieTagGen 与 A/C 命名模式 |
| 生成模型 | DashScope qwen3.7-plus（OpenAI 兼容抽象：base_url + model 可配置，同 A） | 沿用 A 已验证的技术栈 |
| LLM-as-judge | 默认 DashScope qwen3.7-plus；`audit.judge_model` 可配置（qwen3.7-max / GPT-4 等） | 计划文档写的 GPT-4 按当前实际栈改为可配置 |
| 嵌入模型 | 本地 sentence-transformers（E5 系列默认）；`align.embed_model` 可切 SBERT / API（Ada、DashScope embedding） | 本地免费可离线；计划要求对比多种嵌入模型 |
| 生成次数 | `generate.n_runs: 3`（计划规定） | B-Step 2，可消融 1/2/3 |
| pilot 规模 | `movie_limit: 50` | 对齐 A 的试点体量 |
| 合成数据 | 扩展 A 的合成器：合成电影额外携带"用户自由标签"字段 | B 的评估需要 tags.csv 的模拟物 |

### 1.3 builder-check 决议（2026-10-19，代定，终审时请逐条复核）

| 决议 | 内容 |
|------|------|
| 评估协议继承 A 修订版 | 研究计划"时序分割 + 冷启动（同方向 A）"写于 A 的 2026-10-16 协议修订**之前**。按修订后语义执行：时序分割降级为按年代描述性分桶（`temporal_split_year` 代码保留，同 A 用 2015 切描述性视图）；冷启动转为方法属性（输入纯内容驱动、天然冷启动可用）+ limitation 说明，不再作为实验协议。**注意：这是从 A 传播过来的论文叙事变化，请确认接受** |
| 防泄露扩展 | A 的 `assert_no_leakage` 模式扩展到 B 特有泄露面：genre_map 统计集、BERT 训练集、few-shot 参考池（direct_score 与生成层预留）、judge 校准抽样、**校准拟合集**（对齐评估口径引入）——全部限训练分割，逐组件接线自检 |
| 生成质量 judge 任务 | 新增第二个 judge 任务（audit 层，`audit.judge_tasks: [mapping, gen_quality]` 配置）：对照该电影真实用户标签给生成标签的相关性 / 覆盖度打分。兑现 §12"嵌入相似度 + LLM judge 双口径"承诺；与计划 §3.5 的人工评估对齐，另支持人工标注导出 |
| 阶段产物跨实验复用 | 物理复制 + PROVENANCE 元数据（源实验名 + 配置快照哈希 + 文件校验和），不用 symlink（Windows 需特权）；阶段产物为小体量 JSON，磁盘代价可忽略；`align.input_exp` 配置指定源实验。兼顾自包含与"只重跑受影响阶段" |
| 生成 prompt 形态 | 零-shot 默认（计划的示例 prompt 即零-shot，且天然消除一个泄露面）；`prompt.few_shot_examples` 预留配置（默认 0），启用时参考电影限训练分割；§3.2 自由标签统计是**离线设计参考**（设计 prompt 模板时人看统计），不运行时注入 |
| title-only 与探针预留 | `modalities: []`（title-only）为合法配置（继承 A 修订后的管道语义：空模态列表是显式意图，不判 no_content）；`movietaggen/probes/` 扩展位预留，mismatch 探针推迟到真实数据阶段再建（A 的后见之明前移为预留位）；消融矩阵新增 title-only 候选臂（§8 第 10 行，可删） |
| judge 上下文口径 | `audit.judge_context: pair+title` 默认（可配 pair / pair+title / pair+full）：judge 看（自由标签, genome tag）对 + 电影标题，判"语义等价 + 对该电影适用"；人工校准标注指引同步（标注者看标题 + 映射对），保证 kappa 双方口径一致 |

---

## 2. 总体架构

**架构风格**：可导入包 + 编号阶段脚本 + 配置驱动（同 A）。

**三条横切设计**：

1. **API 调用磁盘缓存**（复制 A 的缓存模块并泛化）：以 `(prompt + model + 参数 + run_id)` 哈希为键落盘（`.cache/llm/`）。在 B 中覆盖四类调用：`n_runs` 多次生成（run_id 区分）、judge 审计判定（mapping 与 gen_quality 两个任务）、direct_score 基线打分、API 嵌入（如用 Ada）。这是成本控制、断点续跑、可复现的共同基础。
2. **Mock 模式 + 合成样本**（复制 A）：确定性假客户端 + 合成数据集，让全管道在无 key、无真实数据、无网络下可运行、可测试。
3. **阶段产物隔离**（B 新增）：`results/<exp>/stage-gen/`、`stage-align/`、`stage-audit/` 三级子目录，各阶段独立落盘、独立读取；实验目录自包含（配置快照 + 全部阶段产物 + 报告）。**消融某阶段只重跑该阶段**——嵌入模型消融复用生成产物，prompt 消融复用建库产物。跨实验借用 = 物理复制 + PROVENANCE 元数据（§1.3），`align.input_exp` / `audit.input_exp` 配置指定源实验。

### 目录结构

```
movietaggen/
├── README.md                      # 快速开始：冒烟 → 下载 → 真实数据（同 A 的叙事结构）
├── requirements.txt               # 核心依赖（含 sentence-transformers，见 §11）
├── requirements-baselines.txt     # torch/transformers（BERT 基线可选）
├── .env.example                   # OPENAI_API_KEY / OPENAI_BASE_URL（DashScope 兼容，同 A）
├── configs/
│   ├── default.yaml               # 全字段基线配置（消融配置只写差异字段，继承覆盖）
│   ├── pilot.yaml                 # movie_limit 50 的试点配置
│   ├── smoke.yaml                 # 合成 + mock 冒烟
│   ├── baselines.yaml
│   └── ablation/                  # 模态、五维度、n_runs、embed_model、top_k 等消融配置族
├── movietaggen/
│   ├── config.py                  # 配置加载 + 校验 + 快照
│   ├── data/                      # 复制适配自 A：解析、对齐、合成样本；+ freetag.py（B 特有）
│   ├── generate/                  # 阶段一：prompts / client / cache / mock / pipeline
│   ├── align/                     # 阶段二：embedding / mapping / aggregate / calibrate
│   ├── audit/                     # judge（mapping + gen_quality 双任务）/ calibration
│   ├── baselines/                 # direct_score / genre_map / bert / random
│   ├── probes/                    # 扩展位预留（mismatch 探针推迟到真实数据阶段）
│   └── eval/                      # metrics / protocol / report
├── scripts/
│   ├── 00_make_synthetic.py
│   ├── 01_verify_data.py
│   ├── 02_build_dataset.py
│   ├── 03_analyze_freetags.py     # B 特有
│   ├── 04_run_generation.py
│   ├── 05_run_alignment.py
│   ├── 06_run_audit.py
│   ├── 07_run_baselines.py
│   ├── 08_evaluate.py             # 支持 --extra 跨实验对比（复制 A 模式）
│   └── run_smoke.py
├── tests/
├── notebooks/
├── data/                          # raw（默认指向 A）/ interim / synthetic / processed（tag_groups.json）
└── results/                       # <exp_name>/ 自包含实验目录
```

---

## 3. 数据层（`movietaggen/data/`）

### 3.1 数据流

MovieNet（关键帧 / 剧本 / 字幕）+ MovieLens ml-latest（movies / tags / genome-tags / genome-scores）→ 对齐建库（复制 A 的对齐模块与 `scripts/02`）→ 每电影多模态输入包（关键帧 + 剧本片段 + 台词片段）。`data.raw_dir` 默认 `../genometag-mllm/data/raw`，可配置覆盖。

### 3.2 自由标签分析（B 特有，`data/freetag.py` + `scripts/03`）

- tags.csv 分布统计：唯一标签数、频次谱、每电影标签数分布、top-N 标签、按年代 / 热度分桶
- 词汇层重叠：高频自由标签与 1,128 genome tags 的字面与嵌入重叠
- 产物：`data/interim/freetag_stats.json` + 摘要报告
- 用途：(a) 五维度 prompt 设计的**离线**高频标签先验（设计模板时人看统计，不运行时注入，见 §1.3）(b) 生成质量评估的参照分布 (c) 论文 Intro 中 tags.csv vs genome 双体系动机的实证数据

### 3.3 合成样本（扩展 A 的合成器）

合成电影 = 多模态内容 + genome 评分 + **合成用户自由标签**（从内容相关词表采样，注入别名 / 噪声 / 错拼，模拟 tags.csv 的真实噪声特征）。自由标签字段是 B 生成质量指标（相关性 / 覆盖度）的模拟 ground truth。

---

## 4. 阶段一：生成层（`movietaggen/generate/`）

- `prompts.py`：五维度引导模板（visual style / emotional tone / narrative theme / storytelling technique / character traits）；`prompt.dimensions` 配置子集；**零-shot 默认**，`prompt.few_shot_examples` 预留（默认 0，参考电影限训练分割，防泄露接线）；`modalities` 支持子集与**空列表（title-only，合法显式意图）**；要求输出 15-20 个标签；输出解析容错（编号 / 逗号 / 引号变体、去重归一）
- `client.py` / `cache.py` / `mock.py`：复制 A 的 OpenAI 兼容客户端 + 磁盘缓存 + 确定性 mock 客户端
- `pipeline.py` + `scripts/04_run_generation.py`：`n_runs` 次生成（缓存键含 run_id）→ 并集去重 → 落盘 `stage-gen/`（per-movie JSON：标签列表 + 来源 run + 原始响应留存，供案例分析）
- **mock 设计约束**：mock 客户端基于合成电影内容确定性产出伪标签；不同 `dimensions` / `modalities`（含空模态）配置必须产生可区分的输出（否则消融测试假绿）；但不断言语义质量

---

## 5. 阶段二：对齐层（`movietaggen/align/`）

- `embedding.py`：sentence-transformers 本地加载（E5 默认）+ OpenAI 兼容 API 嵌入可选 + mock 嵌入（确定性哈希向量，冒烟零依赖）；**嵌入缓存**（`.cache/embed/`）：genome tags 1,128 条嵌入全实验只算一次，自由标签嵌入按 (model, text) 缓存；sentence-transformers 惰性导入
- `mapping.py`：每自由标签 → top-k genome tag 候选（余弦相似度）；低于 `threshold` 标记 low-confidence；`top_k` 可配 1 / 3
- `aggregate.py`：多自由标签映射到同一 genome tag → `max` / `mean` 聚合为预测关联度（原始余弦量纲）
- `calibrate.py`（对齐评估口径决议引入）：在**训练分割**的（聚合分, genome 真值）对上拟合保序回归 / 线性映射，把余弦量纲校准到 genome [0,1]；拟合集防泄露 assert；校准参数随 `stage-align/` 产物落盘
- `scripts/05_run_alignment.py`：从 `stage-gen/` 产物读取（本实验产物或 `align.input_exp` 指定的源实验产物，物理复制 + PROVENANCE），不触发生成

---

## 6. 审计层（`movietaggen/audit/`，LLMGenQC-adapted）

- `judge.py`：**双任务**，`audit.judge_tasks` 配置——
  - `mapping`（默认）：对每个（自由标签 → genome tag）映射做 accept / reject 二元判定，审计维度（语义对齐度 / 一致性 / 覆盖度 / 安全性）写入 judge prompt
  - `gen_quality`（默认开，pilot 体量下成本可控）：对照该电影真实用户标签，给生成标签的相关性 / 覆盖度打分——生成质量的 LLM judge 口径（§7.1 双口径承诺）
  - judge 上下文：`audit.judge_context: pair+title` 默认（可配 pair / pair+title / pair+full），判"语义等价 + 对该电影适用"
  - `audit.judge_model` 可配置；judge 调用走统一缓存；`audit.sample_rate` 抽样控制成本
- `calibration.py`：随机抽样（默认 500）映射导出人工标注表（CSV：movie、自由标签、候选 genome tag、相似度、judge 判定、人工判定空列；标注者看标题 + 映射对，与 judge 口径一致）；gen_quality 任务同样支持人工标注导出；回填后由 eval 层计算 Cohen's kappa
- `audit.apply_filter`：on / off——过滤低质量映射后重新聚合 vs 原始聚合，直接支撑 LLMGenQC 价值论证
- `scripts/06_run_audit.py`：读 `stage-align/`（或 `audit.input_exp` 指定源），产 `stage-audit/`（逐映射判定 + gen_quality 评分 + 过滤后重聚合评分矩阵）

---

## 7. 评估层与基线（`movietaggen/eval/` + `baselines/`）

### 7.1 指标（操作化定义在设计期固定）

- **生成质量**（vs tags.csv，双口径）：
  - 嵌入口径：相关性 relevance（生成标签 ↔ 真实用户标签的嵌入相似度，双向 top-1 均值）；覆盖度 coverage（真实 top-K 用户标签中被生成标签近邻覆盖的比例，K / τ 配置化）；冗余度 redundancy（生成标签内部平均两两嵌入相似度）
  - judge 口径：`gen_quality` 任务的 judge 评分（相关性 / 覆盖度），另有 LLM-人工混合路径（校准表导出）
  - 评估用嵌入模型独立配置（`eval.embed_model`），避免"对齐用什么嵌入、评估就用什么嵌入"的循环论证
- **对齐质量**（校准后预测评分 vs genome-scores.csv）：
  - RMSE **双口径**：全谱（未映射 tag 补 0）与仅已预测对——前者反映稀疏选择能力，后者排除补 0 影响，报告必须同时给出
  - Spearman（已预测对）与 Precision@K 作量纲无关指标
  - 校准在训练分割拟合（防泄露），见 §5 `calibrate.py`
- **审计质量**：judge vs 人工标注的 Cohen's kappa（校准表回填后）
- **跨实验对比**：`08_evaluate.py --extra` 模式（对比表 + 热力图，复制 A）

### 7.2 基线（`scripts/07_run_baselines.py`）

1. `direct_score`（**完整复刻 A**）：约 20 个 tag 组逐组打分，每组 prompt 含 tag 定义 + few-shot 2 部参考电影（训练分割采样，防泄露 assert）+ JSON 输出 {tag: score}；`data/processed/tag_groups.json` 产物随仓库提交迁入（同 A 模式），分组逻辑代码作为 baseline 基础设施携带、可重新生成
2. `genre_map`：metadata genre → genome tag 静态映射表（统计式实现可复制 A 的 genre baseline；统计集限训练分割）
3. `bert`：剧本文本 + BERT 多标签分类（可选依赖；训练集限训练分割）
4. `random`：随机标签 / 评分下界

### 7.3 评估协议与防泄露（继承 A 修订版，见 §1.3）

- 时序分割降级为按年代描述性分桶（`temporal_split_year` 代码保留，默认 2015 切描述性视图）；冷启动转为方法属性 + limitation，不作实验协议
- `eval/protocol.py` 的 `assert_no_leakage` 覆盖 B 全部拟合面：genre_map 统计集、BERT 训练集、few-shot 参考池（direct_score 与生成层预留）、judge 校准抽样、校准拟合集

### 7.4 端到端推荐（接口预留，不实现）

仅定义最终产物 schema：(a) movie × genome_tag 评分矩阵（校准后）(b) movie × 开放词表标签表。推荐器与 Recall@K / NDCG@K 评估 M5-M6 届时作为独立模块扩展。

---

## 8. 消融矩阵（九轴已确认 + 一项候选）

| # | 消融轴 | 配置项 | 支撑的论文论证 |
|---|--------|--------|----------------|
| 1 | 输入模态 | `modalities` 子集 | 继承 A 的消融配置模式 |
| 2 | 五维度 prompt | `prompt.dimensions` 子集 | 五维度引导的价值 |
| 3 | 生成次数 | `generate.n_runs: 1/2/3` | 多次并集去重的价值 |
| 4 | 嵌入模型 | `align.embed_model: E5/SBERT/Ada` | 计划明确要求对比 |
| 5 | 映射深度 | `align.top_k: 1 vs 3` | "保留不确定性"防御 |
| 6 | 阈值扫描 | `align.threshold` | 低置信度过滤曲线 |
| 7 | 聚合策略 | `align.agg: max/mean` | 多标签聚合选择 |
| 8 | 审计开关 | `audit.apply_filter: on/off` | LLMGenQC 价值的直接证据 |
| 9 | 两阶段 vs 直接评分 | baseline 1（direct_score，完整复刻 A） | 反转测试的核心对比（校准 + 双口径保证公平） |
| 10 | 无内容盲测（**候选**） | `modalities: []`（title-only） | 记忆性排查（A 探针 1 前移预留；builder-check 新增，终审可删） |

另有 `audit.judge_model` 等配置项可用，不在已确认消融清单内。

**每轴消融只重跑受影响阶段**（§2 横切设计三）；消融配置采用"继承 default.yaml、只写差异字段"的模式（A 的 ablation 目录模式）。

---

## 9. 运行约定与冒烟路径

- **冒烟**：`run_smoke.py` = 合成样本 + mock 全管道（00→08 缩样端到端，零 key、零网络、零模型下载）
- **真实数据**：README 编号步骤；download-guide 说明 MovieLens（指向 A 的 `data/raw` 或独立下载）与 MovieNet（尚未落地，与 A 共同的前置条件）的获取与校验方式
- **实验目录**：`results/<exp_name>/` 自包含——配置快照、stage-gen / align / audit 产物（借用的为物理副本 + PROVENANCE）、评估报告

---

## 10. 测试策略（TDD）

- **单元**：prompt 组装（dimensions 子集、空模态、few-shot 开关）、输出解析容错、映射 / 聚合 / 校准数值正确性、指标公式（含 kappa、双口径 RMSE）、缓存键与命中、校准表导出 / 回填、防泄露 assert 触发
- **集成**：合成数据 + mock 全管道端到端；阶段隔离验证（改 align 配置不触碰 stage-gen 产物；input_exp 借用产出带 PROVENANCE 的副本）
- **mock 双约束**（沿用 C 的经验教训）：(a) 不断言语义质量——mock 输出无语义保证，测试只断言结构 / schema / 流控正确性；(b) 消融可区分——不同配置（含空模态）的 mock 输出必须统计可区分，否则"消融轴生效"无法被测试
- 全部测试无网络、无 key

---

## 11. 依赖

- **核心**（requirements.txt）：pyyaml、openai（兼容协议）、pandas、numpy、scikit-learn（指标 + 保序回归校准）、sentence-transformers（对齐是 B 的核心阶段而非可选基线；惰性导入，mock / 缓存命中路径不加载）
- **可选基线**（requirements-baselines.txt）：torch、transformers（BERT 基线）
- **测试**：pytest

---

## 12. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 两阶段误差累积（论文核心风险） | 阶段产物隔离 + 阶段级指标：生成质量独立于对齐可测，误差可归因到阶段 |
| 量纲 / 稀疏性不可比（审稿人视角） | 校准 + 双口径 RMSE + 量纲无关指标（Spearman / P@K）；direct_score 完整复刻不作弱化稻草人 |
| MovieNet 未落地 | 与 A 共同前置；开发期合成数据主干；download-guide + 校验脚本先行 |
| judge 审计调用量大（15-20 标签 × top-3 × 电影数 + gen_quality 任务） | judge 缓存 + `audit.sample_rate` 分批 + 校准抽样复用 |
| tags.csv 评估操作化主观 | 指标公式在设计期固定（§7.1）；嵌入 + LLM judge 双口径，另有人工标注导出路径 |
| 消融配置组合爆炸 | 配置继承 + 差异字段模式；每轴消融只重跑受影响阶段 |
| 记忆性虚高（tags.csv 真值比 A 更怕背标签） | title-only 合法配置 + 候选消融臂；probes/ 扩展位预留，真实数据阶段建 mismatch 探针 |

---

## 13. 研究计划里程碑对应

| 里程碑 | 框架支撑 |
|--------|---------|
| M1-M2 数据 + prompt 设计 + 预跑 | `00-03` + `04` + `pilot.yaml` |
| M2-M4 生成 + 对齐实验 | `04` / `05` + ablation 配置族 |
| M4-M5 审计 + 人工校准 | `06` + calibration 导出 / kappa |
| M5-M6 端到端推荐 | 接口预留，届时扩展 |
| M6-M8 论文写作 | 评估报告 + 案例分析素材（stage 产物留存原始响应） |

---

## 14. 留给建造者的事

- 从 A 复制文件的精确清单与裁剪点（需读 A 源码定夺；原则：数据层与客户端 / 缓存 / mock 整体迁移，genome 评分专有逻辑剔除——**例外：tag 分组产物与逻辑作为 direct_score baseline 基础设施迁入，见 §1.1**）
- 模块内函数签名与文件内组织
- 合成自由标签生成器的具体词表与噪声模型
- 校准组件的具体拟合方法（保序回归 vs 线性，留配置）
- 测试用例逐条设计（在 §10 双约束下）
- README / download-guide / data-format 成文
