# MovieTagGen 实验框架实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零建成 `movietaggen/` 独立实验框架仓库——两阶段（自由标签生成 → genome 对齐）+ LLM-as-judge 审计 + 评估 + 基线，合成数据上端到端可跑通、全部消融轴配置化。

**Architecture:** 可导入包 + 编号阶段脚本 + 配置驱动（复制适配姊妹项目 A `genometag-mllm`）。三条横切设计：LLM 调用磁盘缓存、mock+合成样本零成本路径、阶段产物隔离（`results/<exp>/stage-{gen,align,audit}/`，跨实验借用 = 物理复制 + PROVENANCE）。

**Tech Stack:** Python 3.11+ / pandas / pyarrow / scikit-learn（保序回归校准）/ scipy / sentence-transformers（E5，惰性导入）/ openai 兼容协议（DashScope qwen3.7-plus）/ pytest。

**Spec:** `方向B_MovieTagGen_框架设计.md`（父目录草稿；Task 1 迁入仓库 `docs/superpowers/specs/2026-10-19-movietaggen-framework-design.md`，此后以仓库内版本为准）。

**复制来源：** A 仓库位于 `../genometag-mllm/`（相对 B 仓库根）。所有"复制"均指从该路径复制后按任务所述适配（改包名 import、删除 genome 评分专有逻辑），**逐字复制的文件不在本计划中重抄**——执行者读 A 源码。

**对 spec 的一处偏离（建造者决议）：** spec §11 的 `requirements-baselines.txt`（torch/transformers 可选）不再单独存在——核心依赖 sentence-transformers 本身携带 torch 与 transformers，BERT 基线无需额外安装。单一 `requirements.txt`。

## Global Constraints

- 工作目录：`C:\code\research-topics\movie-tag-research\movietaggen\`（Task 1 创建）。所有相对路径以此为根。
- Windows + bash：venv 解释器为 `.venv/Scripts/python`；pytest 用 `.venv/Scripts/python -m pytest`。
- 语言约定：README / 注释 / 文档中文，代码标识符英文（spec §1.1）。
- 全部测试无网络、无 API key、无模型下载；mock 路径不读不写 LLM 缓存（spec §10）。
- 配置未知顶层键一律报错；全部字段校验（spec §2，沿用 A 的 `KNOWN_TOP_LEVEL_KEYS` 模式）。
- 重依赖（sentence-transformers / torch）惰性导入——mock 与缓存命中路径不得触发加载（spec §11）。
- 结果目录自包含：配置快照 + 各阶段产物 + 报告；借用产物必须带 `PROVENANCE.json`（spec §1.3）。
- 每任务一次提交，conventional commits（`feat:` / `test:` / `chore:` / `docs:`）。
- 复制 A 代码时保留其原始注释中的实测教训标注（它们是踩坑记录，不是废话）。

## Review Focus

实现时最可能咬人的五类输入/故障（每行的测试已钉进对应任务）：

1. **中断残留的截断 jsonl 行**——续跑须跳过并计数，不得崩溃或静默吞掉整文件（Task 9，复制 A 的 `_read_jsonl_lines` 容忍逻辑 + 测试）。
2. **`modalities: []` 被误判 no_content**——空模态列表是 title-only 探针的显式意图，不是缺数据（Task 7/9，A 修过的 bug，测试钉死）。
3. **缓存键对 run_id / 采样参数不敏感**——n_runs 并集会塌缩成单次、换 temperature 不换键（Task 3，键含两者 + 测试）。
4. **parquet 往返后混合列 NaN 为 float 真值**——`pd.notna` 守卫缺失会把 NaN 当路径用（Task 9，A 实测 479/1035 行触发，复制守卫 + 测试）。
5. **拟合组件吃到测试分割**——校准 / genre_map / few-shot / BERT 任何一处泄露都让消融轴 9 的对比失效（Task 12/13/19/20，`assert_no_leakage` 触发测试）。

---

### Task 1: 仓库骨架与文档迁移

**Files:**
- Create: `movietaggen/` 仓库根（`.gitignore`, `.env.example`, `requirements.txt`, `README.md`, `movietaggen/__init__.py`, `tests/__init__.py`, `tests/test_scaffold.py`, `docs/research-plan.md`, `docs/superpowers/specs/2026-10-19-movietaggen-framework-design.md`, `docs/superpowers/plans/2026-10-19-movietaggen-framework.md`）

**Interfaces:**
- Produces: 仓库根路径（后续所有任务的 cwd）；`.venv` 可用。

- [ ] **Step 1: 创建目录与骨架文件**

`git init`；`.gitignore`：`.venv/`, `__pycache__/`, `.cache/`, `data/interim/`, `data/synthetic/`, `results/`, `.env`, `*.pyc`。注意 `data/processed/` **不入** ignore（tag_groups.json 要提交）。`.env.example`（复制 A 的，内容 `OPENAI_BASE_URL=` / `OPENAI_API_KEY=`）。`requirements.txt`：

```
pandas>=2.0
pyarrow>=14.0
numpy>=1.26
PyYAML>=6.0
Pillow>=10.0
openai>=1.40
python-dotenv>=1.0
matplotlib>=3.8
tabulate>=0.9
scipy>=1.11
scikit-learn>=1.3
sentence-transformers>=2.7
pytest>=8.0
```

`README.md` 先写标题 + 一句话定位（Task 22 补全）。`movietaggen/__init__.py` 空文件。

- [ ] **Step 2: 迁移文档**

复制 `../方向B_MovieTagGen_研究计划.md` → `docs/research-plan.md`；`../方向B_MovieTagGen_框架设计.md` → `docs/superpowers/specs/2026-10-19-movietaggen-framework-design.md`；本计划 → `docs/superpowers/plans/2026-10-19-movietaggen-framework.md`。

- [ ] **Step 3: venv 与依赖**

`python -m venv .venv`；`.venv/Scripts/python -m pip install -r requirements.txt`（sentence-transformers 首次拉取 torch 较大，一次性成本）。

- [ ] **Step 4: 脚手架测试**

`tests/test_scaffold.py`：

```python
def test_package_importable():
    import movietaggen
    assert movietaggen.__file__ is not None
```

- [ ] **Step 5: 验证**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: 1 passed。

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: 仓库骨架、依赖与文档迁移（spec + plan 入库）"
```

---

### Task 2: 配置系统

**Files:**
- Create: `movietaggen/config.py`, `configs/default.yaml`, `configs/pilot.yaml`, `configs/smoke.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无（首个包模块）。
- Produces: `load_config(path) -> AppConfig`；`paths_for(cfg) -> Paths`；`validate_config(cfg)`；`VALID_DIMENSIONS = ("style", "mood", "theme", "narrative", "character")`；`VALID_MODALITIES = ("video", "script", "dialogue")`。`AppConfig` 字段：`exp_name, mock, data, model, generate, align, audit, eval, baselines, protocol`。`Paths` 字段：`root, movienet, movielens, interim, processed, cache_llm, cache_embed, results, stage_gen, stage_align, stage_audit, tag_groups, dataset, genome_scores, genome_tags, aligned, baselines_dir`（`results` = `results/<exp_name>`；`stage_*` 为其下 `stage-gen|stage-align|stage-audit`；`baselines_dir` = `results/baselines` 固定不随 exp_name）。

- [ ] **Step 1: 写失败测试**

按 A 的 `tests/test_config.py` 模式（`_write` helper + MINIMAL dict）。关键断言：

```python
MINIMAL = {"exp_name": "t", "data": {"source": "synthetic"}}

def test_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, MINIMAL))
    assert cfg.generate.n_runs == 3 and cfg.generate.few_shot_examples == 0
    assert cfg.generate.dimensions == list(VALID_DIMENSIONS)
    assert cfg.align.top_k == 3 and cfg.align.threshold == 0.5 and cfg.align.agg == "max"
    assert cfg.audit.judge_tasks == ["mapping", "gen_quality"]
    assert cfg.audit.judge_context == "pair+title" and cfg.audit.apply_filter is True
    assert cfg.eval.coverage_top_k == 10 and cfg.eval.coverage_tau == 0.7

def test_empty_modalities_legal(tmp_path):   # Review Focus 2 的配置面
    cfg = load_config(_write(tmp_path, {**MINIMAL, "generate": {"modalities": []}}))
    assert cfg.generate.modalities == []

def test_invalid_dimension_rejected / test_invalid_agg_rejected /
    test_invalid_judge_context_rejected / test_unknown_top_level_key_rejected /
    test_threshold_range / test_raw_dir_override(tmp_path):
    # raw_dir 指向后：paths_for 的 movielens = raw_dir/"movielens"/"ml-latest"
```

- [ ] **Step 2: 跑测试确认失败** — Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`，Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现 `movietaggen/config.py`**

复制 A 的 `genometag/config.py` 骨架（dataclass 分节 + KNOWN_TOP_LEVEL_KEYS + validate + load），分节改为：

- `DataConfig`: `source ("real"|"synthetic"), movie_limit, seed, raw_dir: str | None = "../genometag-mllm/data/raw"`（仅 real 生效，相对仓库根解析）
- `ModelConfig`: 同 A（base_url_env, api_key_env, model="qwen3.7-plus", temperature, max_frames=12, script_token_budget=4000, dialogue_token_budget=3000, request_interval, extra_body）
- `GenerateConfig`: `n_runs=3, modalities=list(VALID_MODALITIES), dimensions=list(VALID_DIMENSIONS), few_shot_examples=0, workers=1`
- `AlignConfig`: `embed_model="intfloat/multilingual-e5-base", embed_device="cpu", top_k=3, threshold=0.5, agg=("max"|"mean"), input_exp=None, calib_method=("isotonic"|"linear")`
- `AuditConfig`: `input_exp=None, judge_tasks=["mapping","gen_quality"], judge_model="qwen3.7-plus", judge_temperature=0.0, judge_context=("pair"|"pair+title"|"pair+full"), sample_rate=1.0, apply_filter=True, calibration_sample=500, seed=42`
- `EvalConfig`: `embed_model="intfloat/multilingual-e5-base", coverage_top_k=10, coverage_tau=0.7`
- `BaselinesConfig`: `run=["direct_score","genre_map","random","bert"], n_runs=3, few_shot_examples=2`
- `ProtocolConfig`: 同 A（temporal_split_year=2015, relevance_threshold=0.5, top_k=10, cold_start_max_tags=0）

校验规则：dimensions/modalities 子集合法且**空列表合法**；agg、judge_context、calib_method 枚举校验；0<threshold<1；0<sample_rate≤1；n_runs≥1；few_shot_examples≥0。

- [ ] **Step 4: 跑测试确认通过** — `.venv/Scripts/python -m pytest tests/test_config.py -v`，Expected: PASS。

- [ ] **Step 5: 写三个配置文件**

`default.yaml`（全字段、`source: real`、`movie_limit: null`）；`pilot.yaml`（`exp_name: pilot`, `movie_limit: 50`，其余同 default）；`smoke.yaml`（`exp_name: smoke`, `mock: true`, `source: synthetic`, `generate.n_runs: 2`, `workers: 1`）。

- [ ] **Step 6: Commit** — `git add movietaggen/config.py configs/ tests/test_config.py`，`chore: 配置系统（九节 schema + 校验 + 阶段路径）`。

---

### Task 3: MLLM 客户端与磁盘缓存

**Files:**
- Create: `movietaggen/generate/client.py`, `movietaggen/generate/cache.py`
- Test: `tests/test_client.py`, `tests/test_cache.py`

**Interfaces:**
- Produces: `OpenAIClient`（同 A：transport 注入、json_object 降级、指数退避）；`LLMResponse(content, usage, raw, ok)`；`sampling_params(model_cfg) -> dict`；`make_client(config) -> OpenAIClient | MockGenClient`（mock 分支惰性 import `movietaggen.generate.mock`——该模块 Task 8 才存在，本任务不测 mock 分支）；`cache_key(messages, model, params, run_id) -> str`；`ResponseCache(root).get/put`（原子替换 + 截断自愈）。

- [ ] **Step 1: 写失败测试**

复制 A 的 `tests/test_cache.py` 全部用例 + `tests/test_client.py` 的传输注入/降级/退避用例（改 import 为 `movietaggen.generate.*`）。新增 Review Focus 3 用例：

```python
def test_cache_key_sensitive_to_run_id_and_params():
    m = [{"role": "user", "content": "x"}]
    assert cache_key(m, "mo", {"temperature": 0.7}, "run-0") != cache_key(m, "mo", {"temperature": 0.7}, "run-1")
    assert cache_key(m, "mo", {"temperature": 0.7}, "r") != cache_key(m, "mo", {"temperature": 0.3}, "r")
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 复制适配**

`genometag/annotate/client.py` → `movietaggen/generate/client.py`，`genometag/annotate/cache.py` → `movietaggen/generate/cache.py`。改动仅：import 路径；`make_client` 的 mock 分支 import `from movietaggen.generate.mock import MockGenClient`。其余逐字保留（含注释里的实测教训）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: OpenAI 兼容客户端与磁盘缓存（复制适配自 A）`。

---

### Task 4: 合成数据生成器（扩展 A：自由标签 + 五维度）

**Files:**
- Create: `movietaggen/data/sample.py`, `scripts/00_make_synthetic.py`, `scripts/__init__.py`
- Test: `tests/test_sample.py`

**Interfaces:**
- Produces: `make_synthetic_dataset(out_root: Path) -> None`；`SYNTHETIC_MOVIES`（A 的 5 部）；`SYNTHETIC_TAGS`（A 的 24 个 + 新增两组，见 Step 3）；`DIMENSION_GROUPS: dict[str, str]`（组名→维度：视觉风格→style、情绪氛围→mood、叙事主题→theme、受众偏好→无）；`base_score(title, tag) -> float`；`user_tags_for(title: str) -> list[str]`（确定性，4-8 个带噪声自由标签）。
- 产物布局同 A：`data/synthetic/raw/{movienet,movielens}/` + `data/synthetic/processed/tag_groups.json`；tags.csv 列 `userId,movieId,tag,timestamp`。

- [ ] **Step 1: 写失败测试**

```python
def test_synthetic_layout(tmp_path):
    make_synthetic_dataset(tmp_path)
    assert (tmp_path / "raw/movielens/tags.csv").exists()
    tags = pd.read_csv(tmp_path / "raw/movielens/tags.csv")
    assert len(tags) >= 20            # 每电影 4-8 个自由标签

def test_user_tags_deterministic_and_noisy():
    a, b = user_tags_for("Crimson Horizon"), user_tags_for("Crimson Horizon")
    assert a == b and 4 <= len(a) <= 8
    # 噪声注入：至少存在一个非规范形式（大小写/错拼/别名变体）

def test_new_groups_present(tmp_path):
    make_synthetic_dataset(tmp_path)
    groups = json.loads((tmp_path/"processed/tag_groups.json").read_text(encoding="utf-8"))
    names = {g["group"] for g in groups}
    assert {"叙事手法", "角色特征"} <= names       # narrative / character 两组
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

复制 A 的 `genometag/data/sample.py` 为底，扩展：

1. `SYNTHETIC_TAGS` 追加两组（`tag_type: "content"`）：叙事手法（narrative）6 个（如 "nonlinear timeline", "twist ending", "voice-over narration", "slow pacing", "framing story", "open ending"）；角色特征（character）6 个（如 "flawed protagonist", "morally gray hero", "strong female lead", "mentor figure", "unlikely duo", "ensemble cast"）。每个带中文一句话 definition（风格同 A）。
2. `DIMENSION_GROUPS = {"视觉风格": "style", "情绪氛围": "mood", "叙事主题": "theme", "叙事手法": "narrative", "角色特征": "character"}`（受众偏好不在其中——偏好型标签不由内容维度生成）。
3. `user_tags_for(title)`：从该电影的维度组词表哈希采样 4-8 个规范标签，再以哈希概率注入噪声变体（小写化 / 去尾 s / 常见错拼替换）——自由标签与生成 mock 共享词表，使相关性/覆盖度指标非退化。
4. tags.csv 写入：每电影 `user_tags_for(title)` 各一行（userId 轮转 1-3）。

`scripts/00_make_synthetic.py`：复制 A 的（改 import + 调 B 的 `make_synthetic_dataset`）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 合成数据生成器（A 扩展：五维度组 + 自由标签 ground truth）`。

---

### Task 5: 数据适配器与对齐建库

**Files:**
- Create: `movietaggen/data/movielens.py`, `movietaggen/data/movienet.py`, `movietaggen/data/align.py`, `movietaggen/data/grouping.py`, `scripts/01_verify_data.py`, `scripts/02_build_dataset.py`
- Test: `tests/test_movielens.py`, `tests/test_movienet.py`, `tests/test_align.py`, `tests/test_grouping.py`

**Interfaces:**
- Produces: A 同名函数逐一保留：`load_genome_scores/load_genome_tags/load_movies/load_links/load_user_tags`；`parse_title_year`；`MovieMultimodal`；`load_movienet`；`load_srt`；`align(movies_ml, links, movienet_movies) -> AlignResult`；`build_dataset(paths) -> dict`；`normalize_imdb`；`load_tag_groups(path) -> pd.DataFrame`（A 的 `tags/grouping.py` 逐字复制到 `movietaggen/data/grouping.py`，含 `build_grouping_messages` / `parse_grouping_response` / `group_tags_via_llm` 供再生成）。

- [ ] **Step 1: 写失败测试**

复制 A 的 `tests/test_movielens.py`、`tests/test_movienet.py`、`tests/test_align.py` 全部用例（改 import）；从 A 的 `tests/test_grouping.py` 摘取 `load_tag_groups` 与 `parse_grouping_response` 相关用例。新增：

```python
def test_raw_dir_override(tmp_path, monkeypatch):
    # 写一份最小 fake raw 到 tmp，data.raw_dir 指向它 → paths_for 的 movielens/movienet 落在 fake 根下
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 复制适配**

四个文件从 A 逐字复制（`movielens.py`、`movienet.py`、`align.py`、`tags/grouping.py`→`data/grouping.py`），仅改：import 路径；`align.py` 的 `build_dataset` 中 `paths` 类型引用。`paths_for`（Task 2 已实现）负责 raw_dir 解析——本任务只消费。脚本 01/02 复制 A 的同名脚本（改 import；01 保留年份分布检查输出）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: MovieLens/MovieNet 适配器与对齐建库（复制适配自 A）`。

---

### Task 6: 自由标签分布分析（B 特有）

**Files:**
- Create: `movietaggen/data/freetag.py`, `scripts/03_analyze_freetags.py`
- Test: `tests/test_freetag.py`

**Interfaces:**
- Produces: `freetag_stats(user_tags: pd.DataFrame, dataset: pd.DataFrame, top_n: int = 50) -> dict`（键：`n_unique, n_rows, tags_per_movie_mean, top_tags: list[[tag, count]], year_buckets: dict[str, int]`）；`literal_overlap(user_tags: pd.DataFrame, genome_tags: pd.DataFrame, top_n: int = 50) -> dict`（键：`top_n_total, exact_match, match_list`——高频自由标签与 genome tag 字面一致的数目与清单）。嵌入口径的重叠留给评估层（spec §3.2 注：嵌入重叠在 eval 报告中体现）。

- [ ] **Step 1: 写失败测试**

手造 10 行 user_tags + 3 部电影 dataset：断言 `n_unique`、`top_tags` 排序、年份分桶计数；`literal_overlap` 中 "neon visuals"（同时在 genome_tags）被计入 exact_match，"neon visualz"（错拼）不计入。

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

`freetag_stats`：`user_tags` 过滤到 dataset 电影集合后分组统计；年份分桶用 dataset 的 year（<1980/1980-1999/2000-2009/2010-2019/≥2020）。`literal_overlap`：top_n 高频自由标签（casefold 后）与 genome tag 集合（casefold）求交。`scripts/03`：读 tags.csv + dataset → 写 `paths.interim/freetag_stats.json` 与 `freetag_report.md`（摘要 markdown）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: tags.csv 自由标签分布分析（03 脚本 + 报告）`。

---

### Task 7: 生成 prompt 组装与输出解析

**Files:**
- Create: `movietaggen/generate/content.py`, `movietaggen/generate/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `MovieMultimodal`（Task 5）、`ModelConfig`（Task 2）。
- Produces: `approx_tokens/truncate_to_budget/sample_keyframes/encode_image`（content.py，自 A prompts.py 抽出的共享助手）；`FewShotTags(title: str, year: int, tags: list[str])`；`SYSTEM_PROMPT`；`MOVIE_TITLE_LINE = "电影：《{title}》（{year}）"`；`DIMENSIONS_BLOCK = "## 生成维度"`；`build_gen_messages(movie, dimensions: list[str], modalities: list[str], few_shots: list[FewShotTags], model_cfg) -> list[dict]`；`parse_tags(text: str) -> list[str]`；`normalize_tag(tag: str) -> str`（casefold + 压缩空白）。内容节标题沿用 A：`## 关键帧` / `## 剧本节选` / `## 台词/字幕`（mock 解析契约）。

- [ ] **Step 1: 写失败测试**

```python
def test_dimension_subset_changes_instruction():
    full = build_gen_messages(mm, ["style", "mood"], ["video"], [], cfg)
    text = _text(full)
    assert "视觉风格" in text and "角色特征" not in text

def test_empty_modalities_is_title_only():     # Review Focus 2
    msgs = build_gen_messages(mm, list(VALID_DIMENSIONS), [], [], cfg)
    text = _text(msgs)
    assert "电影：《" in text and "## 关键帧" not in text and "## 剧本节选" not in text

def test_parse_tags_variants():
    assert parse_tags('["neon visuals", "slow burn"]') == ["neon visuals", "slow burn"]
    assert parse_tags("1. neon visuals\n2. slow burn") == ["neon visuals", "slow burn"]
    assert parse_tags("- Neon Visuals\n- neon visuals") == ["Neon Visuals"]   # 归一去重

def test_few_shot_block_rendered / test_budget_truncation / test_images_attached_when_video()
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

结构照 A 的 `build_messages`（标题行 → 指令 → few-shot 块 → 维度块 → 内容节），差异：指令为"基于提供的电影内容，围绕上述维度生成 15-20 个英文标签（单词或短语）。只输出 JSON 数组，如 [\"tag one\"]，不要输出其他内容。"；维度块每行 `- style：视觉风格（画面、色彩、镜头语言）`（mood 情绪基调 / theme 叙事主题 / narrative 叙事手法 / character 角色特征），只渲染配置的维度。`parse_tags`：先 `json.loads`（失败则正则截取 `[...]`），再降级逐行解析 `- `/`数字.` 前缀；全部经 `normalize_tag` 去重（保留首个原形）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 五维度生成 prompt 组装与标签解析`。

---

### Task 8: Mock 生成客户端

**Files:**
- Create: `movietaggen/generate/mock.py`
- Test: `tests/test_mock.py`

**Interfaces:**
- Consumes: prompt 契约（Task 7 的标题行 / 维度块 / 内容节标题）。
- Produces: `MockGenClient(n_tags_per_dimension: int = 4)`，`.complete(messages, run_id) -> LLMResponse`（content 为 JSON 数组字符串，raw 为 OpenAI 信封形状——缓存测试契约同 A）；`DIMENSION_MODALITIES = {"style": {"video"}, "mood": {"video", "dialogue"}, "theme": {"script", "dialogue"}, "narrative": {"script"}, "character": {"script", "dialogue"}}`。

- [ ] **Step 1: 写失败测试**

```python
def test_deterministic_and_run_noise():
    a = MockGenClient().complete(msgs, "run-0"); b = MockGenClient().complete(msgs, "run-0")
    assert a.content == b.content
    c = MockGenClient().complete(msgs, "run-1")
    assert a.content != c.content            # 轮次噪声：n_runs 并集非平凡

def test_dimension_subset_changes_output():   # 消融可区分（spec §10 mock 约束 b）
    full = _tags(msgs_all_dims); part = _tags(msgs_style_only)
    assert set(part) < set(full) and len(part) < len(full)

def test_modality_gating():                   # 消融可区分
    # 同维度、video-only prompt：narrative/character 维度不产标签（DIMENSION_MODALITIES 门控）
    # title-only（无内容节）：只产 1 个标题派生标签

def test_prompt_contract_parsed():            # 从 prompt 文本解析标题/维度/可用内容节
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

照 A 的 `MockClient` 结构（`_extract_text` + 正则解析 + `_hash_unit`）。规则：解析标题与维度列表及出现的 `## 关键帧/## 剧本节选/## 台词/字幕` 节；对每个维度 d：若 `DIMENSION_MODALITIES[d] ∩ 可用内容节 ≠ ∅`，从词表 `VOCAB[d]`（SYNTHETIC_TAGS 对应组的英文标签）按 `hash(title, d, run_id)` 采样 `n_tags_per_dimension` 个并加 run 噪声（哈希决定偶尔换词）；title-only 时输出 `[normalize(title) + " movie"]` 单标签。

- [ ] **Step 4: 确认通过** — pytest PASS；补 `tests/test_client.py` 中 `make_client(mock=True)` 返回 `MockGenClient` 的用例。
- [ ] **Step 5: Commit** — `feat: 确定性 mock 生成客户端（维度/模态门控可区分）`。

---

### Task 9: 生成管道与阶段产物

**Files:**
- Create: `movietaggen/stages.py`, `movietaggen/generate/pipeline.py`, `scripts/04_run_generation.py`
- Test: `tests/test_pipeline.py`, `tests/test_stages.py`

**Interfaces:**
- Consumes: `make_client` / `ResponseCache` / `cache_key` / `sampling_params`（Task 3）；`build_gen_messages` / `parse_tags` / `normalize_tag`（Task 7）；`MockGenClient`（Task 8）。
- Produces: `run_generation(config, paths, client, cache, dry_run=False) -> Path | None`；`stages.py`：`save_config_snapshot(config_path, paths)`；`load_stage(paths, stage: str, input_exp: str | None) -> Path`（input_exp 非空 → 从 `results/<input_exp>/<stage>/` 物理复制到本实验该 stage 目录 + 写 `PROVENANCE.json`；为空 → 本实验目录存在即返回，不存在报错）；`_read_jsonl_lines`（A 的容忍逻辑）。`stage-gen/` 产物：`tags.parquet`（列 `movie_id, tag, n_runs`——并集去重后每 (movie, tag) 一行，`n_runs` = 出现轮数）、`responses.jsonl`（行：`task_key, status, tags, usage`）、`run_manifest.json`、`config.yaml` 快照。

- [ ] **Step 1: 写失败测试**

```python
def test_resume_skips_done(tmp_path, fake_setup):
    # 预写 responses.jsonl 一行 task_key → 二次运行不再调用 client（计数 transport 调用数）

def test_truncated_jsonl_line_tolerated():    # Review Focus 1
    # 预写一行截断 JSON + 一行完好 → 运行成功且 manifest 计数正确

def test_union_dedupe_across_runs():
    # mock 两次 run：tags.parquet 的 (movie, tag) 唯一，n_runs ∈ {1,2}

def test_nan_mixed_column_guard():            # Review Focus 4
    # dataset 行 script_path 为 NaN → _to_multimodal 不把 NaN 当路径（复制 A 的 pd.notna 守卫）

def test_empty_modalities_not_no_content():   # Review Focus 2
    # modalities=[] 运行产出 tags（title-only 标签），status 无 no_content

def test_load_stage_borrow_copies_with_provenance(tmp_path):
    # input_exp 指向已存在的其他实验 → 复制 + PROVENANCE.json（source_exp、files+sha256）
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

照 A 的 `pipeline.py` 骨架（任务矩阵 movie × run、jsonl append + 行级 flush、锁内计数、`_to_multimodal` 连 `pd.notna` 守卫逐字复制、`_available_modalities` 含空列表语义）。任务矩阵无 group 维度：`TaskSpec(movie_id, run_id)`。缓存集成同 A（mock 不读写缓存；缓存命中从信封还原 content）。收尾：union 去重 → `tags.parquet`；`run_manifest.json`。`scripts/04`：照 A 的 04（dry-run、配置快照）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 阶段一生成管道（续跑/并集去重/阶段产物/借用溯源）`。

---

### Task 10: 嵌入层（本地 / API / mock + 缓存）

**Files:**
- Create: `movietaggen/align/embedding.py`
- Test: `tests/test_embedding.py`

**Interfaces:**
- Produces: `class MockEmbedder`（`.name = "mock"`；`.embed(texts: list[str]) -> np.ndarray (n, 64)` 单位向量；规范化：casefold + 仅留字母数字与空格后哈希——同形文本向量相同，不同文本近似正交）；`class LocalEmbedder`（`.name = model_name`；惰性 `from sentence_transformers import SentenceTransformer`；`embed` 返回 L2 归一化矩阵）；`class ApiEmbedder`（`.name = f"api:{model}"`；openai SDK `embeddings.create`，env 同 ModelConfig）；`make_embedder(model_name: str, mock: bool, device: str = "cpu")`；`embed_texts(embedder, texts: list[str], cache_root: Path) -> np.ndarray`（键 = sha256(embedder.name + text)，值 = 向量 JSON；命中不调用 embedder）。

- [ ] **Step 1: 写失败测试**

```python
def test_mock_deterministic_and_canonical():
    a = MockEmbedder().embed(["Neon Visuals", "neon visuals"])
    assert np.allclose(a[0], a[1])                       # 规范化后同形
    assert abs(cos(MockEmbedder().embed(["aaa"])[0], MockEmbedder().embed(["bbb"])[0])) < 0.5

def test_cache_hit_skips_embedder(tmp_path):
    # 计数 embedder 调用：第二次 embed_texts 同文本 0 次调用

def test_lazy_import_guard():
    # make_embedder("mock", mock=True) 全程不触发 sentence_transformers 加载
    assert "sentence_transformers" not in sys.modules    #（用子进程或 importlib 卫生处理）
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

MockEmbedder：`_hash_unit(canonical)` 生成 64 维 → L2 归一化。LocalEmbedder/ApiEmbedder 惰性导入；ApiEmbedder 批量调用（一次请求全部 texts，端点限批则 64/批）。`embed_texts` 缓存读写同 `ResponseCache` 的原子替换模式。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 嵌入层（E5 本地 / API / mock + 磁盘缓存）`。

---

### Task 11: 映射与聚合

**Files:**
- Create: `movietaggen/align/mapping.py`
- Test: `tests/test_mapping.py`

**Interfaces:**
- Consumes: numpy 矩阵（由调用方用 Task 10 的 embedder 生成；本模块纯函数便于数值测试）。
- Produces: `map_free_tags(free_tags: pd.DataFrame, genome_tags: pd.DataFrame, free_matrix: np.ndarray, genome_matrix: np.ndarray, top_k: int, threshold: float) -> pd.DataFrame`（列 `movie_id, tag, tag_id, cosine, rank, low_conf`；free_tags 行序与 free_matrix 对齐，genome_tags 行序与 genome_matrix 对齐）；`aggregate_scores(mappings: pd.DataFrame, agg: str) -> pd.DataFrame`（列 `movie_id, tag_id, score`；同 (movie, tag_id) 多映射 → max 或 mean）。

- [ ] **Step 1: 写失败测试**

构造 4 维正交向量手工算：tag A 与 genome t1 余弦 0.9、t2 余弦 0.6，threshold=0.5, top_k=3 → 两行均保留、rank 1/2、low_conf 均 False；threshold=0.7 → t2 行 low_conf=True。两个自由标签都映射 t1（0.9 / 0.7）→ agg=max 得 0.9、agg=mean 得 0.8。top_k=1 → 只留 rank=1 行。

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

`free_matrix @ genome_matrix.T`（均 L2 归一化）逐自由标签 `argsort` 取 top_k；`low_conf = cosine < threshold`。聚合用 groupby。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 自由标签→genome 映射与聚合（top-k/阈值/聚合策略）`。

---

### Task 12: 校准组件

**Files:**
- Create: `movietaggen/align/calibrate.py`
- Test: `tests/test_calibrate.py`

**Interfaces:**
- Produces: `fit_calibration(pairs: pd.DataFrame, method: str) -> Calibrator`（pairs 列 `pred, truth`——调用方保证仅训练分割，本模块不管分割）；`Calibrator.transform(scores: np.ndarray) -> np.ndarray`（单调、输出 [0,1]）；`Calibrator.to_dict() / Calibrator.from_dict()`（JSON 往返）。`method="isotonic"` → sklearn `IsotonicRegression(out_of_bounds="clip")`；`"linear"` → 一元线性回归 clip。

- [ ] **Step 1: 写失败测试**

```python
def test_isotonic_monotone_and_bounded():
    pairs = pd.DataFrame({"pred": [0.6, 0.7, 0.8, 0.9], "truth": [0.02, 0.10, 0.35, 0.60]})
    cal = fit_calibration(pairs, "isotonic")
    out = cal.transform(np.array([0.55, 0.65, 0.75, 0.85, 0.95]))
    assert np.all(np.diff(out) >= 0) and out.min() >= 0 and out.max() <= 1
    assert abs(cal.transform(np.array([0.8]))[0] - 0.35) < 1e-6   # 训练点过拟合检查

def test_roundtrip(tmp_path)   # to_dict/from_dict 数值一致
def test_linear_method()
```

- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — 如上；`to_dict` 存 method + 拟合参数（isotonic 存 X_thresholds/y_thresholds）。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 余弦→genome 量纲校准（保序/线性 + 序列化）`。

---

### Task 13: 阶段二对齐脚本

**Files:**
- Create: `scripts/05_run_alignment.py`
- Test: `tests/test_align_script.py`

**Interfaces:**
- Consumes: `load_stage`（Task 9）；`make_embedder` / `embed_texts`（Task 10）；`map_free_tags` / `aggregate_scores`（Task 11）；`fit_calibration`（Task 12）；训练分割以 `dataset[dataset.year < cfg.protocol.temporal_split_year]` 内联实现（与 T18 `protocol.temporal_split` 语义一致，独立保留不回改）。
- Produces: `stage-align/` 产物：`mappings.parquet`（Task 11 schema）、`scores_raw.parquet`、`scores_calibrated.parquet`（同 schema，score 为校准后）、`calibration.json`、`run_manifest.json`、`config.yaml` 快照。

- [ ] **Step 1: 写失败测试**

集成测试（合成 + mock，tmp 目录）：跑 00→02→04（mock）→ 05 后断言：

```python
def test_align_end_to_end(tmp_path):
    # stage-align 四个产物存在；scores_calibrated 的 score ∈ [0,1]
    # mappings 的 cosine ∈ [-1, 1]；scores_raw 与 mappings 的 max 聚合一致

def test_borrow_does_not_touch_source(tmp_path):
    # 05 用 align.input_exp 借用 stage-gen → 源实验 stage-gen 目录 mtime/内容不变，
    # 本实验 stage-gen/ 带 PROVENANCE.json

def test_stage_gen_untouched_by_align(tmp_path):
    # 跑 05 前后哈希 stage-gen/tags.parquet 一致（阶段隔离，spec 反目标 2/3）
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

脚本主体：load_config → `load_stage(paths, "stage-gen", cfg.align.input_exp)` → embed genome tags（`embed_texts`，全实验缓存）→ embed 自由标签 → `map_free_tags` → `aggregate_scores` → 校准：`scores_raw` join `genome_scores` 真值、过滤 `year < temporal_split_year` 的电影得训练对 → `fit_calibration` → transform 全量 → 写产物 + manifest + 快照。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 阶段二对齐脚本（嵌入/映射/聚合/训练分割校准）`。

---

### Task 14: LLM-as-judge（mapping + gen_quality 双任务）

**Files:**
- Create: `movietaggen/audit/judge.py`, `movietaggen/audit/mock.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `OpenAIClient`（judge 用 `OpenAIClient(model=cfg.audit.judge_model, temperature=cfg.audit.judge_temperature, ...)`）；`ResponseCache` / `cache_key`。
- Produces: `build_mapping_messages(title, free_tag, genome_tag, genome_def, cosine, context: str) -> list[dict]`；`parse_mapping_verdict(text) -> dict | None`（`{"verdict": "accept"|"reject", "reason": str}`）；`build_gen_quality_messages(title, gen_tags: list[str], user_tags: list[str]) -> list[dict]`；`parse_gen_quality(text) -> dict | None`（`{"relevance": 1-5, "coverage": 1-5}`）；`run_audit(config, paths, client, cache) -> Path`；`MockJudgeClient`（verdict = `hash(free_tag, genome_tag) > 0.2`，gen_quality 1-5 由 hash 决定——确定性）。`stage-audit/` 产物：`judge_mapping.parquet`（`movie_id, tag, tag_id, cosine, verdict, reason, judge_model`）、`judge_gen_quality.parquet`（`movie_id, relevance, coverage, judge_model`）、`scores_filtered.parquet`（apply_filter=true 时：剔除 reject 映射后重聚合 + 复用 stage-align 校准；false 时不生成此文件）、`run_manifest.json`、`config.yaml` 快照。

- [ ] **Step 1: 写失败测试**

```python
def test_mapping_prompt_contexts():
    p = build_mapping_messages("Inception", "mind-bending", "thought-provoking", "发人深省", 0.83, "pair+title")
    assert "Inception" in _text(p) and "mind-bending" in _text(p)
    assert "Inception" not in _text(build_mapping_messages(..., "pair"))          # pair 无标题

def test_parse_verdict_variants():
    assert parse_mapping_verdict('{"verdict": "accept", "reason": "ok"}')["verdict"] == "accept"
    assert parse_mapping_verdict("```json\n{...}\n```") is not None              # 围栏
    assert parse_mapping_verdict("garbage") is None

def test_apply_filter_changes_scores(tmp_path, fake_stage_align):
    # reject 的映射不进重聚合：scores_filtered 与 scores_calibrated 在被拒 (movie, tag_id) 上不同

def test_sample_rate_samples(tmp_path, fake_stage_align):
    # sample_rate=0.5 → judge_mapping 行数 = ceil(0.5 × 映射数)，seed 确定

def test_mock_judge_deterministic()
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

judge prompt（pair+title 默认）：system "你是标签质量审计专家，输出严格的 JSON 对象。"；user 含 `电影：《title》`、`自由标签: "X"`、`候选 genome 标签: "Y"（定义: ...）`、`嵌入相似度: 0.83`、指令"判定该映射语义是否对齐且适用于该电影，输出 {\"verdict\": \"accept|reject\", \"reason\": \"一句话\"}"。context="pair" 去标题行；"pair+full" 追加剧本/台词截断节（budget 500 token）。gen_quality prompt：生成标签数组 + 真实用户标签数组 → 1-5 两项打分。`run_audit`：读 stage-align（`load_stage(paths, "stage-align", cfg.audit.input_exp)`）→ 逐映射判（缓存键 run_id="judge-mapping"，mock 不走缓存）→ sample_rate 抽样 → gen_quality 逐电影 → apply_filter 重聚合（复用 stage-align/calibration.json 的 Calibrator）。JSON 解析复用 A 的 `_extract_json_object` 模式（围栏/前后缀容忍）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: LLM-as-judge 双任务审计 + 过滤重聚合`。

---

### Task 15: 审计校准抽样导出

**Files:**
- Create: `movietaggen/audit/calibration.py`
- Test: `tests/test_calibration_export.py`

**Interfaces:**
- Produces: `export_calibration_sample(judge_df: pd.DataFrame, dataset: pd.DataFrame, n: int, seed: int) -> pd.DataFrame`（列 `movie_id, title, tag, tag_id, genome_tag, cosine, judge_verdict, human_verdict`——human_verdict 全空字符串）；`refill_kappa(csv_path: Path) -> tuple[list, list] | None`（读回填后的 CSV，human_verdict 非空行配对；任一侧全同类时返回 None——kappa 无定义）。

- [ ] **Step 1: 写失败测试**

抽样确定性（同 seed 同样本）、`n > 行数` 时全量、CSV 回填往返（写 → 手填几行 → `refill_kappa` 返回配对列表；全空 human_verdict → None）。

- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — `dataset.sample`? 不——固定 `random.Random(seed).sample`（无依赖顺序问题：先 sort by (movie_id, tag_id) 再抽）。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 审计校准抽样导出与回填配对`。

---

### Task 16: 审计脚本

**Files:**
- Create: `scripts/06_run_audit.py`
- Test: `tests/test_audit_script.py`

**Interfaces:**
- Consumes: `run_audit`（Task 14）、`export_calibration_sample`（Task 15）。
- Produces: `scripts/06_run_audit.py: main(config_path) -> int`（judge 客户端：mock → MockJudgeClient，否则 OpenAIClient(judge_model)；跑完追加导出 `stage-audit/calibration_sample.csv`——行数 `min(cfg.audit.calibration_sample, 映射数)`）。

- [ ] **Step 1: 写失败测试** — 合成 + mock 集成：00→02→04→05→06 全链，断言 `stage-audit/` 五个产物存在、`calibration_sample.csv` 行数 ≤500 且 schema 正确、`scores_filtered.parquet` 存在（apply_filter=true）。
- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — 照 A 的 04 脚本模式（load_config → 客户端 → run → 快照）。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 06 审计脚本（判审 + 过滤 + 校准抽样导出）`。

---

### Task 17: 评估指标

**Files:**
- Create: `movietaggen/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: 复制 A 的 `rmse, mae, precision_at_k, ndcg_at_k, paired_bootstrap`（不复制 `consistency_std`/`icc_2_1`——B 无轮次一致性论证，YAGNI）；新增 `spearman_corr(pred, truth) -> float`（scipy 包装，处理常数列 → nan 安全返回 0.0）；`cohen_kappa(a: list[str], b: list[str]) -> float`（手写，含 po/pe）；`gen_relevance(gen_vecs, user_vecs) -> float`（双向 top-1 余弦均值：`mean(avg_i max_j cos(g_i, u_j), avg_j max_i cos(g_i, u_j))`）；`gen_coverage(user_vecs_topk, gen_vecs, tau) -> float`（user 侧每行 max cos ≥ tau 的比例）；`gen_redundancy(gen_vecs) -> float`（两两余弦绝对值均值，n<2 → 0.0）；`dual_rmse(pred_df, truth_df, all_tag_ids: set[int]) -> dict`（`{"rmse_full": …, "rmse_pred_only": …, "n_pairs_full": …}`——full 口径把 (movie, tag) 缺失预测补 0 对齐全部 `all_tag_ids × pred 电影`）。

- [ ] **Step 1: 写失败测试**

手算锁值：`rmse/precision_at_k/ndcg_at_k` 抄 A 的测试值；`spearman_corr([1,2,3],[3,1,2])` ≈ 0.5；`cohen_kappa(["a","a","b","b"], ["a","a","b","b"]) == 1.0`、完全不一致 == 0.0（或负）、`cohen_kappa(["a","a"], ["a","a"])` 全同类 → 返回 0.0（无定义保护）；`gen_relevance` 用正交+相同向量构造断言界；`dual_rmse` 手造 2 电影 × 3 tag、pred 只覆盖 2 对：full 口径含补 0 项、pred_only 不含。

- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — 如上；`dual_rmse` 的 full 口径：truth 限制在 pred 的电影集合上，补全 `all_tag_ids` 维度。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 评估指标（双口径 RMSE/Spearman/kappa/生成质量三件套）`。

---

### Task 18: 评估协议与报告

**Files:**
- Create: `movietaggen/eval/protocol.py`, `movietaggen/eval/report.py`, `scripts/08_evaluate.py`
- Test: `tests/test_protocol.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `load_tag_groups`（`movietaggen/data/grouping.py`，Task 5）；Task 17 指标；各 stage 产物 schema。
- Produces: `protocol.py` 复制 A 的 `temporal_split, cold_start_subset, tag_type_breakdown, assert_no_leakage`（逐字，含错误文案）；`report.py`：`write_report(exp_dirs: list[Path], baseline_dir: Path | None, out_dir: Path, cfg, dataset, truth, tag_groups, user_tags) -> Path`，产出 `evaluation/{metrics.csv, report.md, gen_quality.json}`。`scripts/08_evaluate.py`：`main(config_path, extra_exps)`，`--extra` 参数同 A 的 06。
- `metrics.csv` 行 = method × subset，列含 `rmse_full, rmse_pred_only, spearman, precision@k, ndcg@k`；method 集：`two-stage[raw]`、`two-stage[calibrated]`、`two-stage[filtered]`（有 `scores_filtered.parquet` 时）+ baselines.parquet 各 method；subset：`全部 / 时序训练(<y) / 时序测试(≥y) / tag:content / tag:preference`（spec §7.3：描述性分桶，不作协议）。

- [ ] **Step 1: 写失败测试**

```python
def test_report_on_synthetic(fake_exp_tree, tmp_path):
    out = write_report(...)
    df = pd.read_csv(out / "metrics.csv")
    assert {"two-stage[raw]", "two-stage[calibrated]", "genre_map"} <= set(df["method"])
    assert {"rmse_full", "rmse_pred_only", "spearman"} <= set(df.columns)
    assert df["rmse_full"].notna().all()

def test_gen_quality_json(fake_exp_tree, tmp_path):
    gq = json.loads((out / "gen_quality.json").read_text(encoding="utf-8"))
    # 键：relevance/coverage/redundancy（嵌入口径，mock embedder 下非退化）+ judge_relevance/judge_coverage（有 judge 产物时）

def test_extra_exps_merged(fake_exp_tree, tmp_path):
    # 两个实验目录 → metrics.csv 含两者的 method 行
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

照 A 的 `report.py` 骨架（predictions dict → metrics_table → 子集切片 → concat）。差异：pred 来源为 `stage-align/scores_raw.parquet`、`scores_calibrated.parquet`、`stage-audit/scores_filtered.parquet` 与 `baselines/baselines.parquet`；指标换 Task 17 集合（`dual_rmse` + spearman + P@K/NDCG@K）；`gen_quality.json`：对 stage-gen tags vs tags.csv（合成路径下即 synthetic tags.csv）用 `eval.embed_model`（mock → MockEmbedder）算三件套 + judge 产物均值。fixture `fake_exp_tree`：直接调 00→02→04→05→06（mock）在 tmp 树上生成（测试内复用脚本 main，参照 A 的 test_pipeline 集成模式）。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 评估报告（双口径指标表 + 生成质量双口径 + --extra 跨实验对比）`。

---

### Task 19: direct_score 基线（完整复刻 A）

**Files:**
- Create: `movietaggen/baselines/scoring_prompts.py`, `movietaggen/baselines/direct_score.py`, `movietaggen/baselines/mock_scoring.py`
- Copy: `../genometag-mllm/data/processed/tag_groups.json` → `data/processed/tag_groups.json`（真实 1,128 tag 分组产物，入 git）
- Test: `tests/test_direct_score.py`

**Interfaces:**
- Consumes: `OpenAIClient` / `ResponseCache` / `cache_key`；`MovieMultimodal`；`load_tag_groups`（Task 5 已复制到 `movietaggen/data/grouping.py`，直接导入）。
- Produces: `scoring_prompts.py` = A 的 `prompts.py` 逐字复制（`TagDef, FewShot, build_messages, parse_scores, parse_scores_lenient`）；`direct_score.py`: `run_direct_score(config, paths, client, cache) -> pd.DataFrame`（列 `movie_id, tag_id, score, method="direct_score"`）；`select_few_shots`（A 的实现复制，含训练分割约束）；`mock_scoring.py` = A 的 `MockClient` 复制（`MockScoringClient`，base_score 相关性与合成真值天然相关）。

- [ ] **Step 1: 写失败测试**

```python
def test_prompt_mirrors_a_structure():
    # build_messages 产物含 MOVIE_TITLE_LINE、TAGS_BLOCK_START、组内 tag 定义行、few-shot 块

def test_few_shot_leakage_guard():             # Review Focus 5
    # select_few_shots 后 assert_no_leakage(few_shot 电影, 测试集) 通过；
    # 人为把测试电影塞进参考池 → 断言触发

def test_direct_score_end_to_end_mock(tmp_path):
    # 合成 + MockScoringClient：产物行数 = 电影×tag 全覆盖（稠密）；
    # mock 分数与真值相关性 > 0.3（base_score 机制，非退化检查）

def test_real_tag_groups_migrated():
    assert len(load_tag_groups(repo_root/"data/processed/tag_groups.json")) == 1128
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

`run_direct_score`：任务矩阵 movie × group × run（`baselines.n_runs`，默认 3）——照 A 的 pipeline 骨架（jsonl 续跑可简化为内存循环 + 直接落 parquet，但缓存集成保留：键含 run_id）；`parse_scores_lenient` → 轮次均值聚合 → 稠密 (movie, tag) 表。few-shot：`select_few_shots(dataset, genome_scores, cfg.baselines.few_shot_examples, temporal_split_year, top_k, tag_names)` + `assert_no_leakage`。

- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: direct_score 基线（A 分组评分完整复刻 + tag_groups 迁移）`。

---

### Task 20: genre_map / random / bert 基线与 07 脚本

**Files:**
- Create: `movietaggen/baselines/genre_map.py`, `movietaggen/baselines/random_baseline.py`, `movietaggen/baselines/bert_baseline.py`, `scripts/07_run_baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Produces: `genre_map(train_scores, dataset, movie_ids, test_ids=None) -> pd.DataFrame`（A 的 `genre_baseline` 复制，method 列值改 `"genre_map"`）；`random_baseline(dataset, all_tag_ids, seed) -> pd.DataFrame`（每电影 × 全 tag，score = `hash_unit(f"rand|{movie}|{tag}|{seed}")`，method `"random"`）；`bert_baseline` 复制 A 的（torch 已随核心依赖在场）；`scripts/07`：按 `cfg.baselines.run` 顺序执行 → 合并写 `results/baselines/baselines.parquet` + `config.yaml` 快照（direct_score 需要 stage-gen 借用？否——direct_score 独立调用模型，不依赖 stage-gen；其输入是 dataset 本身）。

- [ ] **Step 1: 写失败测试**

`genre_map`：训练集统计 + 未见 genre 回退全局均值 + `test_ids` 泄露触发（Review Focus 5）；`random`：确定性、稠密覆盖、score ∈ [0,1)；07 集成：合成 + mock（direct_score 用 MockScoringClient）→ `baselines.parquet` 含三个 method、行数一致、配置快照存在。

- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — 如上；`07` 的 direct_score 分支客户端：mock → MockScoringClient。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: genre_map/random/bert 基线与 07 编排脚本`。

---

### Task 21: 消融配置族

**Files:**
- Create: `configs/ablation/`（`video-only.yaml, script-only.yaml, dialogue-only.yaml, title-only.yaml, dim-style.yaml, dim-mood.yaml, dim-theme.yaml, dim-narrative.yaml, dim-character.yaml, nruns-1.yaml, topk-1.yaml, threshold-0.4.yaml, threshold-0.6.yaml, agg-mean.yaml, audit-off.yaml, embed-sbert.yaml`）
- Test: `tests/test_configs_load.py`

**Interfaces:**
- Consumes: `load_config`（Task 2）。
- Produces: 16 个消融配置——**完整配置文件**（非 overlay；沿用 A 的 ablation 模式：整份复制 default 改一个字段），`exp_name` 与文件名一致（如 `ablation/video-only`）。`embed-sbert.yaml` 的 `align.embed_model: "sentence-transformers/all-MiniLM-L6-v2"`（仅验证可加载，冒烟不跑它——真模型下载留真实实验）。

- [ ] **Step 1: 写失败测试**

```python
def test_all_configs_load(repo_root):
    for p in (repo_root / "configs").rglob("*.yaml"):
        load_config(p)                                     # 全部合法

def test_ablation_differs_as_intended(repo_root):
    base = load_config(repo_root / "configs/default.yaml")
    v = load_config(repo_root / "configs/ablation/video-only.yaml")
    assert v.generate.modalities == ["video"] and v.exp_name == "ablation/video-only"
    t = load_config(repo_root / "configs/ablation/title-only.yaml")
    assert t.generate.modalities == []                     # 消融臂 10（spec §8）
    a = load_config(repo_root / "configs/ablation/audit-off.yaml")
    assert a.audit.apply_filter is False
```

- [ ] **Step 2: 确认失败** — pytest FAIL。
- [ ] **Step 3: 实现** — 逐个生成（default 为模板改字段）。
- [ ] **Step 4: 确认通过** — pytest PASS。
- [ ] **Step 5: Commit** — `feat: 16 个消融配置（九轴 + title-only 候选臂全覆盖）`。

---

### Task 22: 冒烟、文档与终验

**Files:**
- Create: `scripts/run_smoke.py`, `docs/download-guide.md`, `docs/data-format.md`
- Modify: `README.md`（补全）
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 全部脚本（00–08）。
- Produces: `run_smoke.py: main() -> int`（STEPS = 00→08，幂等清空 `results/smoke/`）；README（安装 → 冒烟 → 真实数据编号步骤 → 消融指南，照 A 的 README 叙事）；download-guide（MovieLens：默认 `data.raw_dir` 指向 A 或独立下载；MovieNet：获取 + `convert_movienet.py` 说明——该脚本从 A 复制到 `scripts/`）；data-format（镜像 A 的，补 tags.csv 自由标签字段说明）。

- [ ] **Step 1: 写失败测试**

```python
def test_smoke_end_to_end(repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    from scripts.run_smoke import main
    assert main() == 0
```

- [ ] **Step 2: 确认失败** — pytest FAIL。

- [ ] **Step 3: 实现**

`run_smoke.py` 照 A（importlib 串 STEPS + 产物断言）。断言清单：`data/synthetic/processed/tag_groups.json`、`results/smoke/{config.yaml, stage-gen/tags.parquet, stage-gen/run_manifest.json, stage-align/scores_calibrated.parquet, stage-align/calibration.json, stage-audit/judge_mapping.parquet, stage-audit/judge_gen_quality.parquet, stage-audit/scores_filtered.parquet, stage-audit/calibration_sample.csv, evaluation/metrics.csv, evaluation/gen_quality.json}`、`results/baselines/baselines.parquet`、`paths.interim / "freetag_stats.json"`（smoke 为 synthetic 源，即 `data/synthetic/interim/freetag_stats.json`）。复制 A 的 `scripts/convert_movienet.py` + `watchdog.sh`（改包名 import）。补全 README / download-guide / data-format。

- [ ] **Step 4: 终验**

Run: `.venv/Scripts/python -m pytest tests/ -v` → 全绿；`.venv/Scripts/python scripts/run_smoke.py` → `PASS`。

- [ ] **Step 5: Commit** — `feat: 端到端冒烟 + README/download-guide/data-format，框架完成`。

---

## 自审记录

- **Spec 覆盖**：§2 三横切（缓存 T3、mock T4/8、阶段隔离 T9/13）、§3 数据层（T4/5/6）、§4 生成（T7/8/9）、§5 对齐（T10/11/12/13）、§6 审计（T14/15/16）、§7 评估基线（T17/18/19/20）、§8 消融矩阵（T21）、§9 运行（T22）、§1.3 决议（raw_dir T2/T5、防泄露 T12/13/19/20、零-shot T2、title-only T7/21、judge 上下文 T14、双口径 T17/18、PROVENANCE T9）。端到端推荐接口（spec §7.4）= `scores_calibrated/scores_filtered.parquet` schema 本身，无代码任务——符合"只定义不实现"。
- **类型一致性**：`load_stage(paths, stage, input_exp)` 在 T9 定义、T13/T14 消费；`Calibrator.to_dict/from_dict` T12 定义、T14 复用（stage-align/calibration.json）；`dual_rmse` 列名在 T17 定义、T18 断言一致。
- **偏离说明**：requirements-baselines.txt 取消（torch 随 sentence-transformers 进入核心，见计划头）。
