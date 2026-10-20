# 440 三模态宇宙 + A 预测迁移设计（方向 B 协议升级）

> 日期：2026-10-21
> 状态：待用户终审
> 上游事实：服务器输入核查（2026-10-21）——1,035 部对齐集中三模态全有 440 部（42.5%），
> theme/character 维度门控覆盖仅 42.5%；A 的协议 v3 宇宙即同一 440；A 的 `v3-full`
> 实跑已在 88 部测试集上完整完成（88 × 1,128 tags × 3 runs = 297,792 行，88/88 覆盖）。

---

## 1. 背景与意图

输入核查暴露的约束：剧本仅 479/1,035、字幕 804/1,035，theme/character 两维度只有 42.5%
的电影有完整模态输入。用户决定（2026-10-21）：

| 决策点 | 结论 |
|--------|------|
| 实验宇宙 | **总数据集 = 440 部三模态全有宇宙**（与 A 评估宇宙同集）；595 部部分模态电影完全退出（生成/对齐/审计/评估/vs tags.csv 生成质量分析全部在 440 上） |
| 测试分割 | **复用 A 的 88 部测试清单**（`eval_test.txt`，年代分层，A 仓库已入 git）——direct_score 基线与两阶段管道在同一批测试电影上严格可比 |
| direct_score 基线 | **拷贝 A 的 `v3-full` 实跑预测**（物理迁移 + PROVENANCE），不在 B 内重跑 |

**收益**：主实验生成量 15,525 → 6,600 次调用（−57% API 成本）；五维度全部完整输入；
模态消融各臂评估集严格同构；校准机制从时序分割升级为固定分割（A 修订版语义的彻底落实）。

## 2. 设计明细

### 2.1 宇宙过滤（`data.universe`）

- `DataConfig` 新增 `universe: str = "three_modality"`，合法值 `three_modality | all`；
  非法值配置校验大声失败。
- `data/align.py` 的 `build_dataset`：`three_modality` 时在 genome 覆盖过滤**之后**追加
  过滤——`keyframes` 列表非空 且 `script_path` 非空 且 `subtitles_path` 非空。
  过滤条件与输入核查脚本（2026-10-21）的磁盘实测口径一致。
- 合成样本每部电影三模态完整（`sample.py` 逐部写帧/剧本/字幕）→ 过滤对 synthetic
  是 no-op，冒烟路径行为不变。
- `01_verify_data` 的统计输出追加宇宙口径行（过滤前后部数）。

### 2.2 固定测试分割（`processed/eval_test.txt`）

- **迁移**：A 的 `data/processed/eval_test.txt`（88 行，每行一个 MovieLens movieId）
  复制入 B 的 `data/processed/eval_test.txt`，同 tag_groups.json 模式——提交入 git，
  迁移时 sha256 与 A 仓库版本核对；`.gitignore` 的 processed 白名单加此文件。
- **机制**：`eval/protocol.py` 新增 `fixed_split(dataset, eval_test_path) -> (train_ids,
  test_ids)`——读清单，断言 88 ⊆ dataset（缺失即大声失败，指明清单与数据集版本错配）；
  train = 440 − 88 = 352。`temporal_split` 保留但**仅作描述性分桶视图**（年代子集），
  不再作为任何拟合面的分割依据。
- **接线替换**（五处，原时序语义全部改为固定分割；88 清单含 2015 前影片，任何保留时序
  语义的拟合池都会与固定测试集重叠，`assert_no_leakage` 将直接炸——改接是必做项）：
  1. `05_run_alignment` 的校准拟合：pairs 仅取 train-352，接线处
     `assert_no_leakage(train_ids, test_ids)` 收尾；
  2. `baselines/direct_score.py`（own_run 模式）的 few-shot 参考池：限 train-352；
  3. `movietaggen/generate/prompts.py` few-shot（默认关，预留路径）：参考电影限 train-352；
  4. `baselines/genre_map.py` 的 genre 统计集：限 train-352；
  5. `baselines/bert_baseline.py` 的训练集：限 train-352。
- **synthetic 路径**：`00_make_synthetic` 追加写 `data/synthetic/processed/eval_test.txt`
  （种子固定的 20% 抽样）；真实/合成统一走 `processed/eval_test.txt` 相对路径，
  `paths_for` 已按 source 切换 processed 目录，零特判。清单缺失 = 校准接线大声失败。
- **pilot 抽样**：`04` 的 movie_limit 抽样改为**从 train-352 抽**（pilot 是 prompt 迭代
  预跑，不接触测试集，杜绝 prompt 调优期的测试污染）；`pilot.yaml` 的
  `baselines.run` **移除 direct_score**——pilot 期做方法对比无意义（a_borrow 恒定产出
  88 部测试电影，与 train 抽样零交集），direct_score 属主实验评估期的对比。

### 2.3 direct_score 基线 = A 预测迁移

- **迁移物**：A 服务器 `results/v3-full/annotations.parquet`（297,792 行；
  列 `movie_id, tag_id, group, modality_set, run_id, score`）复制入
  `data/processed/direct_score_a.parquet`，提交入 git（460KB）。随附 PROVENANCE：
  - `data/processed/direct_score_a.provenance.json`：源仓库/路径、A 的 config 快照与
    run_manifest/cost_report 原文（三份小文本一并迁入）、各文件 sha256、行数/部数/
    轮数校验值（88 × 1,128 × 3）、迁移日期；
  - A 的 `annotations.jsonl`（45.9MB 原始响应）**不迁移**，仅记录其 sha256 备查。
- **加载器**：`baselines/direct_score.py` 新增 `source: a_borrow | own_run` 配置
  （默认 `a_borrow`）。`a_borrow`：读 parquet → 按 `(movie_id, tag_id)` 对 `run_id`
  取均值（B 既有裁定"仅计已解析轮"——逐对 dropna 均值、全空记 NaN 交评估层 C1 收口；
  本数据 3 轮全在）→ 断言 88 × 1,128 覆盖完整、`modality_set` 全为
  `video|script|dialogue` → 产出与其他基线同构的评分矩阵。
  `own_run`：保留现有复刻管道作逃生口（few-shot 池已改 train-352）。
- **a_borrow 的边界裁定**：
  - 仅 `data.source: real` 合法——synthetic + a_borrow 是配置错误，加载器大声失败
    （迁移物是真实数据工件，合成路径没有对应物）；
  - **不响应 `movie_limit`**：恒定产出全部 88 部（固定工件，加载器断言清单完整）；
  - `smoke.yaml` 显式钉 `source: own_run`（冒烟继续走 mock_scoring 复刻管道，
    合成路径行为不变）；`a_borrow` 加载器由**单测 fixture parquet** 覆盖，不进冒烟。
- `07_run_baselines` 对 `a_borrow` 直接落盘基线产物（带 PROVENANCE 引用），不发生
  任何 API 调用。

### 2.4 评估层落地（08 / report）

"同一批测试电影严格可比"在 metrics 上的兑现：

- **新增子集行"固定测试(88)"** = 论文主表口径：所有方法（两阶段 / direct_score(a_borrow)
  / genre_map / random / bert）在该行同一 88 部上对比。另加"固定训练(352)"描述行
  （校准拟合质量的调试口径）。
- **产出电影轴**：各基线与两阶段沿用 I5 语义（无 movie_limit 时产出全 440，受限跑时
  按种子抽样），报告层按固定 88 清单**切片**出主表行——非 MLLM 基线逐电影独立预测，
  切片无损；a_borrow 恒定 88 部。
- **既有行全部保留为描述性视图**：全部(440)、时序训练/测试分桶、tag:content /
  tag:preference——`n_pred` 列继续诚实标注各方法实际预测数（direct_score 在 440 行
  上 n_pred=88 属预期，非对比口径）。
- A 仓库先例（评估电影集 = 第一个实验实际标注到的电影）不照搬——B 有固定清单，
  口径更干净。

### 2.5 一致性验证

- **一次性（实施期，服务器上）**：重建 B-440 后与 A 的 440 宇宙做集合相等性比对
  （A 宇宙 = A 的 dataset 三模态过滤，服务器 `genometag-mllm/data/processed/` 可导出）；
  不等即停，逐部 diff。**运行时不依赖 A 仓库**（自包含反目标不破）。
- **运行时（入库断言）**：`fixed_split` 断言 88 ⊆ dataset；`a_borrow` 加载器断言
  覆盖 88 × 1,128。

### 2.6 服务器侧执行链（建造者执行，非用户交接点）

本会话已具备服务器访问权（mtuser → su root）与部署配方（git archive tarball +
SFTP + 分离执行脚本），以下全部由建造者执行：

1. 服务器取 `results/v3-full/annotations.parquet` + 三份 provenance 小文本 → SFTP
   回本地 → 本地校验（行数 297,792 / 88 部 / 3 轮 / sha256）→ 落盘
   `data/processed/` 后 git 提交；
2. 代码合入后重传服务器 → 重跑 02（440 数据集）+ 03（440 口径自由标签分析）；
3. B-440 vs A-440 集合相等性一次性验证（服务器脚本，逐部 diff 报告）。

## 3. 变更面

| 文件 | 变更 |
|------|------|
| `movietaggen/config.py` | `DataConfig.universe` + 校验；baselines 配置 `direct_score.source` |
| `movietaggen/data/align.py` | 三模态过滤 + 统计返回 |
| `movietaggen/data/sample.py` | synthetic 写 `processed/eval_test.txt` |
| `movietaggen/eval/protocol.py` | `fixed_split`；docstring 时序语义降级说明 |
| `movietaggen/eval/report.py` | 新增"固定测试(88)"/"固定训练(352)"子集行（§2.4） |
| `scripts/01_verify_data.py` | 宇宙口径统计 |
| `scripts/05_run_alignment.py` | 校准拟合改固定分割 + 防泄露接线 |
| `movietaggen/baselines/direct_score.py` | `a_borrow` 加载器（含 source/边界断言）+ own_run few-shot 池改固定分割 |
| `movietaggen/baselines/genre_map.py`、`bert_baseline.py` | 统计集/训练集改接 `fixed_split`（防泄露必做，§2.2） |
| `movietaggen/generate/prompts.py` | few-shot 池（预留路径）改固定分割 |
| `scripts/04_run_generation.py` | pilot 抽样限 train |
| `scripts/08_evaluate.py` | 固定分割行接线（如 report 层外还需传清单） |
| `data/processed/` | 新增 `eval_test.txt` + `direct_score_a.parquet` + provenance（git 提交） |
| `.gitignore` | 新建 `data/processed/*` 忽略 + `!` 白名单（tag_groups.json / eval_test.txt / direct_score_a.parquet / provenance）——现状 processed 整目录未被忽略，无既有白名单可追加 |
| `configs/default.yaml`、`configs/pilot.yaml`、`configs/smoke.yaml` + ablation 16 个 | `universe` 显式化；pilot `baselines.run` 移除 direct_score；smoke 钉 `direct_score.source: own_run`、`temporal_split_year: 2019` 裁定废止说明 |
| `docs/research-plan.md` | §2 数据表（440 宇宙 + 88 test）、§4 协议（宇宙与划分 + 子集视图 + direct_score = A 实跑迁移） |
| `docs/data-format.md`、`docs/download-guide.md` | 两个新迁入文件 + universe/source 字段说明 |
| 测试 | 过滤口径、fixed_split（含缺清单/越界大声失败）、a_borrow 加载（fixture parquet：覆盖断言/均值/synthetic 拒绝）、五处防泄露接线、synthetic eval_test 生成、pilot 抽样不含测试集、report 固定行 |

**服务器侧收尾**（§2.6，建造者执行）：取 parquet 入库 → 重传代码 → 重跑 02（440）+
03（440 口径）→ B-440 vs A-440 集合相等性验证。

## 4. 不变量（反目标沿用）

1. 消融即配置——宇宙与基线来源均为配置项，无代码改路径；
2. 仓库自包含——运行时零依赖 A（迁移物全部物理入库 + sha256/PROVENANCE）；
3. 防泄露——所有拟合面限 train-352，`assert_no_leakage` 逐接线收尾；
4. 冒烟全绿——合成路径行为不变（smoke 钉 `own_run`，`universe` 过滤为 no-op）。

## 5. 留给建造者

- 测试用例逐条设计（TDD）；
- `a_borrow` 评分矩阵与 08 评估输入的确切对齐（列名/量纲断言）；
- research-plan 措辞的最小改动范围。
