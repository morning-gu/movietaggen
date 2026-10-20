# 方向 B: MovieTagGen — 两阶段生成式标签提取与 Genome 标签对齐

> 研究计划（2026-10-20 整理版：2026-10-19 设计决策——对齐评估校准+双口径、direct_score 完整复刻、评估协议继承 A 修订版——与 2026-10-20 实验框架建成已并入正文；旧版见 git 历史）
> 目标场所：WWW 2027 / SIGIR 2027 · Novel Method（IMRaD）· ACM 格式 · 正文 7,000-9,000 words

## 1. 研究问题与核心论点

**研究问题**：MLLM 能否生成与 MovieLens 用户标签（tags.csv）语义一致的自由形式标签，并通过映射层对齐到 genome tag 体系（1,128 个标准化标签 + 关联度评分）？

**核心论点**：通过"MLLM 自由生成 → 嵌入映射 → genome 对齐"的两阶段框架，可同时获得开放词表的生成式标签（对应 tags.csv）和标准化标签评分（对应 genome-scores），且管道质量通过 LLM-as-judge 审计机制量化保证。

**反驳者视角**：两阶段误差累积——生成阶段漏掉的重要标签对齐阶段无法补救；自由标签到 genome tag 的语义鸿沟（"mind-bending" → "thought-provoking"）导致映射错误。

**防御**：
1. **五维度 prompt 引导**（风格/情绪/主题/叙事/角色）+ 3 轮生成取并集去重——降低遗漏；
2. **top-3 候选 + 阈值标记低置信**——保留不确定性，阈值过滤曲线直接量化其价值；
3. **LLM-as-judge 逐映射审计**（accept/reject）+ 500 条人工抽样校准 Cohen's kappa——质量可量化，审计开关消融直接证明过滤收益。

**论证反转预案**：
- 若端到端推荐不及人工标签 → 冷启动场景（无人工标签）"有 vs 无"压倒"好 vs 更好"；
- 若方向 A（直接评分）在 RMSE 上优于两阶段 → 两阶段的独有价值 = **同时产出**开放词表与标准化评分（单阶段无法提供）；阶段产物隔离使误差可归因到具体阶段，这是"误差累积"质疑的实证回应路径。

## 2. 数据

| 数据 | 角色 | 关键事实 |
|------|------|----------|
| MovieLens ml-latest tags.csv | 生成质量参照 | 用户自由标签（开放词表、噪声、个性化）；分布与词汇重叠分析已入框架（03 脚本） |
| MovieLens genome（tags + scores） | 对齐 ground truth | 1,128 tags；relevance 长尾 |
| MovieNet movie1K | 多模态内容 | 关键帧/剧本/字幕；**已落地**（服务器部署 + 与 A 共享转换产物） |
| 对齐实验集 | 评估全集 | **440 部三模态宇宙**（评估宇宙与 A 协议 v3 同集）+ 88 部测试清单（A 迁入）；复用 A 的对齐管道（IMDb 优先 + 标题年份 fallback） |

**设计事实**：
- 评估协议继承 A 的 2026-10-16 修订结论（同一数据约束 B）：时序分割降级为按年代描述性分桶；冷启动转为方法属性 + limitation；
- **440/88/352 口径**：评估宇宙 = 440 部三模态全有电影（与 A 协议 v3 同集）；固定测试 = A 迁入的 88 部清单（`data/processed/eval_test.txt`）；训练/拟合池 = 352 部（440 − 88）；
- 原始数据经 `data.raw_dir` 默认共享 A 的 `data/raw`（ml-latest 1.5G 已在）；1,128-tag 语义分组产物（A 生成）已迁入本仓库供 direct_score 基线使用。

## 3. 方法

**3.1 两阶段管道**
- **阶段一 · 自由生成**：五维度 prompt（visual style / emotional tone / narrative theme / storytelling technique / character traits），输入 12 关键帧 + 剧本 4,000 tok + 台词 3,000 tok，输出 15-20 个英文标签；3 轮取并集去重；零-shot 为主（few-shot 预留，参考池限训练侧防泄露）。
- **阶段二 · genome 对齐**：嵌入（本地 E5 默认，SBERT / API 可配）→ 每标签 top-3 候选（余弦）→ 低于阈值标记低置信 → 多标签聚合（max/mean）→ **保序回归校准**（训练分割拟合，余弦量纲 → genome [0,1]，防泄露接线）。
- **审计（LLMGenQC-adapted）**：双任务 judge——映射审计（语义对齐 + 该电影适用性，accept/reject）与生成质量评分（对照真实用户标签，相关性/覆盖度 1-5）；judge 模型可配置（DashScope 默认；原计划的 GPT-4 改为可配置项）。
- **工程基础**：API 磁盘缓存（生成/双 judge/direct_score/嵌入四用户键隔离）+ 断点续跑 + **阶段产物隔离**（消融只重跑受影响阶段；跨实验借用 = 物理复制 + PROVENANCE 溯源）。

**3.2 方法属性**：输入纯内容驱动，天然冷启动可用（同 A）；额外产出开放词表标签——genome 体系外维度（如 "mind-bending" 这类用户真实语汇）的载体。

**方法防御**：
- "为什么不直接让 MLLM 输出 genome 评分（方向 A）？" → 自由生成捕捉对电影的自然理解、不受固定词表约束、能发现 genome 未覆盖维度；映射层提供标准化保证。两阶段分离"理解"与"对齐"两个关注点。基线 1 = A v3-full 实跑预测迁移（a_borrow：tag 分组评分 + few-shot + JSON 稠密输出的实跑产物），对比公平性由校准 + 双口径 RMSE 保证。
- "嵌入相似度如何保证语义对齐？" → LLM-as-judge 审计 + 人工校准 kappa；嵌入模型对比（E5/SBERT/Ada）为消融轴。
- 最大局限：两阶段误差累积 → 阶段产物隔离 + 阶段级指标使误差可归因（生成质量独立于对齐可测）。

## 4. 评估协议

**宇宙与划分**：评估宇宙 = 440 三模态全有（与 A 协议 v3 同集）；测试 = A 迁入的 88 部固定清单（`data/processed/eval_test.txt`）；训练/拟合池 = 352 部——校准、few-shot、genre 统计、BERT 全部限此，`assert_no_leakage` 逐接线收尾；时序分割仅为描述性分桶视图。

**对齐质量口径（核心对比公平性）**：预测评分经训练分割校准到 genome [0,1] 量纲；RMSE **双口径**（全谱补 0 / 仅已预测对，附 n_pred 列）+ Spearman（已预测对）+ Precision@10 / NDCG@10（IDCG 按该电影全量真值计，不偏袒稀疏方法）；空预测子集报 NaN 不报 0。

**生成质量（vs tags.csv，双口径）**：相关性 / 覆盖度 / 冗余度（嵌入口径，逐电影计算取均值；评估嵌入独立配置，避免循环论证）+ LLM judge 评分（相关性/覆盖度 1-5）。

**审计质量**：judge vs 人工标注 Cohen's kappa——校准表导出（500 条抽样 CSV）→ 人工回填 → `08 --kappa` 直接计算。

**子集视图**：**固定测试(88) = 论文主表口径**（所有方法同集可比）+ 固定训练(352) 调试口径；全部 · 时序训练/测试（描述性分桶）· tag:content · tag:preference；n_pred 诚实标注各方法实际预测数（direct_score 在全部(440) 行 n_pred=88 属预期，非对比口径）。

**消融矩阵（九轴 + 候选臂，全部配置化——`configs/ablation/` 16 个配置即改即跑）**：模态（video/script/dialogue/title-only*）· 五维度子集 · 生成次数 1/2/3 · 嵌入模型 · top_k 1/3 · 阈值 0.4/0.5/0.6（threshold-filtered 变体直接产出过滤曲线）· 聚合 max/mean · 审计过滤 on/off · 两阶段 vs direct_score。
（*title-only 兼作记忆排查臂——B 以 tags.csv 为生成真值，记忆风险高于 A；mismatch 探针已留 `probes/` 扩展位，真实数据阶段再建。）

**基线**：direct_score（A v3-full 实跑预测迁移 a_borrow：`data/processed/direct_score_a.parquet`，297,792 行 = 88×1,128×3 轮，PROVENANCE 溯源；own_run 复刻管道保留为逃生口）/ genre_map（拟合池统计，未见 genre 回退全局均值）/ random / 剧本 + BERT 多标签。

**防泄露原则**：一切拟合面（校准、genre 统计、BERT、few-shot）仅用固定分割训练侧（352 部）数据，逐组件 `assert_no_leakage` 接线；生成层 few-shot（预留路径，`generate/prompts.py` 现无池选择逻辑故本计划不改其代码）将来接线时必须用固定分割训练侧。

**端到端推荐（接口预留）**：movie × genome 评分矩阵 + 开放词表标签表 schema 已定义；Recall@K / NDCG@K 评估 M5-M6 届时扩展。

## 5. 预期结果与叙事

**最重要发现（预期）**：两阶段管道在自由标签生成上达到与用户标签高语义一致性；映射审计准确率超 85%；偏好驱动型标签存在语义鸿沟（与 A 的 content/preference 不对称发现互为呼应）。

**结果结构**：生成质量（双口径）→ 对齐质量（双口径 RMSE）→ 审计质量（kappa）+ 审计开关消融 → 九轴消融 → vs direct_score（反转测试核心）→ 阈值过滤曲线 → 案例分析（映射成功/失败对 + 自由标签发现 genome 外维度的实例）。

## 6. 章节计划

- **Introduction (800-1,000w)**：双体系动机（tags.csv 自由但噪声大 / genome 标准化但依赖用户行为，Harper & Konstan 2015；Deldjoo 2017）→ MLLM 生成能力恰好弥合 → Gap：无工作将 MLLM 生成式输出与双标签体系同时对齐，且缺可审计保证 → 贡献三句：① 首个"自由生成 → genome 对齐"两阶段框架 ② LLM-as-judge 可量化质量保证 ③ tags.csv 与 genome 双层面评估。
- **Related Work (1,200-1,500w)**：三故事线——自由标签生成（Folksonomication → NoteLLM → 本工作 MLLM 多模态）/ 标签映射与对齐（genome SVD++ 起源 → 嵌入空间映射路径）/ 生成质量审计（RAG-VisualRec LLMGenQC → LLM-as-judge 标注审计）。综述结论：自由生成与标准化分类是两条平行线，缺统一两者的框架。
- **Methodology (2,500-3,000w)**：§3 + §4 内容。
- **Results (1,800-2,200w)**：§5 结构。
- **Discussion (800-1,000w)**：vs Folksonomication（多模态输入 vs 纯文本；MLLM vs 传统 NN）/ vs VLM4Rec（生成标签并对齐体系 vs 生成描述做推荐特征）/ vs RAG-VisualRec（LLMGenQC 的迁移与适配）。核心记忆点：两阶段比直接评分更能捕捉 MLLM 对电影的自然理解——开放词表不受固定词表约束。
- **Conclusion (400-500w)**：框架同时产出开放词表与标准化评分；审计提供可量化保证；偏好型标签语义鸿沟为边界。

## 7. 论证压力测试

1. MovieLens 双标签体系各有优劣 → Harper & Konstan (2015)
2. MLLM 生成能力可产出自然语言标签 → Technology Cycle 探针
3. 自由标签经嵌入映射可对齐 genome 体系 → 嵌入相似度机制论证
4. LLM-as-judge 可审计映射质量 → RAG-VisualRec LLMGenQC
5. 两阶段管道在推荐下游不劣于人工标签 → 端到端评估（待数据；接口已预留）

**最弱点**：第 5 步——若 MLLM 标签推荐显著不及人工标签，端到端价值存疑 → 防御：冷启动"有 vs 无"论证（§1 反转预案）。

## 8. 关键文献定位表

| 位置 | 引用文献 | 角色 |
|------|---------|------|
| Intro 背景 | Harper & Konstan (2015) | MovieLens 数据集官方引用 |
| Intro 动机 | Deldjoo et al. (2017) | 视觉特征替代人工标签的先驱 |
| Related 故事1 | Folksonomication (2018) | 文本标签预测先驱 |
| Related 故事1 | NoteLLM (2024) arXiv:2403.01744 | 生成 hashtag 增强嵌入 |
| Related 故事2 | VLM4Rec (2026) | VLM 生成语义描述做推荐特征 |
| Related 故事3 | RAG-VisualRec (2025) | LLMGenQC 质量审计机制来源 |
| Method 数据 | MovieNet (2019) arXiv:1910.09368 | 电影多模态数据集 |
| Method 对齐 | StoryMovie (2026) | 剧本-字幕-视频对齐 |
| Eval 警示 | ML-1M Memorization (2026) | 数据泄露警示 |
| Eval 基线 | TrUMAn (2021) | trope 分类最近邻基线 |

## 9. 时间线

| 阶段 | 内容 | 时点 | 产出 |
|------|------|------|------|
| ✅ 实验框架 | 六模块 + 16 消融配置 + 审计与校准回路；296 tests + 冒烟绿；终审修复波闭环 | 2026-10 | 框架（已推送 origin/main） |
| 数据落地 | MovieNet 获取 + 转换 + 对齐建库 | ✅ 已完成（2026-10-21：服务器部署 + MovieNet 校验 + 440 宇宙数据集） | 真实实验数据集 |
| pilot | 50 部五维度 prompt 预跑 + 校准验证 | 数据后 ~2 周 | prompt 模板定稿 + 首批标签样例 |
| 生成 + 对齐实验 | 主实验 + 九轴消融 | ~2 月 | 两阶段结果 |
| 审计 + 人工校准 | judge 全量 + 500 条人工标注 → kappa | ~1 月 | 审计报告 |
| 端到端推荐 | 接口扩展 + Recall@K / NDCG@K | ~1 月 | 端到端结果 |
| 论文写作 | 初稿 → 内审 → 终稿 | ~3 月 | 投稿 WWW / SIGIR 2027 |

## 10. 目标场所

WWW 2027（The Web Conference）或 SIGIR 2027
论文类型：Novel Method（IMRaD 结构）· 引用格式：ACM 会议标准 · 预估字数：7,000-9,000 words（正文）
