# 方向 B: MovieTagGen

> academic-paper plan 模式 | 2026-10-15

---

 # ============================================================
 # 方向 B: MovieTagGen
 # 两阶段生成式标签提取与 Genome 标签对齐
 # ============================================================

 ## B-Step 0: 研究基础确认

 **已有材料**: 同方向 A，额外关注 tags.csv 的自由标签分布特征。
 **研究问题**: "MLLM 能否生成与 MovieLens 用户标签语义一致的自由形式标签，并通过映射层对齐到 genome tag 体系？"
 **研究就绪度**: 充分，但评估方案需更多设计（自由标签评估比固定分类更复杂）。

 ## B-Step 1: 核心论点

 > **[INSIGHT: thesis_statement]**
 > 通过"MLLM 自由生成 → 嵌入映射 → genome 对齐"的两阶段框架，可以同时获得开放词表的生成式标签（对应 tags.csv）和标准化标签评分（对应 genome-scores），且两阶段管道的质量可通过 LLM-as-judge 审计机制量化保证。

 **反驳者视角**: 两阶段管道的误差会累积——如果生成阶段漏掉了重要标签，对齐阶段无法补救。且自由标签到 genome tag 的语义鸿沟（如 "mind-bending" → "thought-provoking"）可能导致映射错误。

 **防御**: (1) 在生成阶段设计多维度 prompt 引导（风格/情绪/主题/叙事/角色五个维度），降低遗漏风险。(2) 对齐阶段保留 top-3 候选 genome tag 而非硬选 top-1，保留不确定性。(3) LLM-as-judge 审计每个映射的质量，低质量映射标记为"需人工复核"。

 ## B-Step 2: 章节计划

 ### B-Introduction (800-1000 words)

 **紧迫感**: MovieLens 有两套标签系统——tags.csv（用户自由输入，开放词表，高度个性化但噪声大）和 genome tag（1,128 个标准化标签 + 关联度评分，由 SVD++ 训练，质量高但依赖用户行为）。理想方案是能同时获得开放词表的灵活性和标准化体系的一致性。MLLM 的生成能力恰好可以弥合两者：先自由生成，再对齐标准化。

 **研究 Gap（一句话）**: 没有工作将 MLLM 的生成式标签输出与 MovieLens 既有标签体系（tags.csv + genome tags）同时对齐，且缺少对生成质量的可审计保证。

 **预期贡献声明**:
 1. 首个"MLLM 自由生成 → genome 对齐"的两阶段标签生成框架
 2. LLM-as-judge 质量审计机制，为自动化标签提供可量化质量保证
 3. 在 tags.csv 和 genome-scores 两个层面同时评估，填补现有评估空白

 ### B-Related Work (1200-1500 words)

 **讲故事数**: 3 个故事线
 1. **自由标签生成**: Folksonomication (2018) 从摘要预测标签 → NoteLLM (2024) 学习生成 hashtag → 本工作的 MLLM 多模态生成
 2. **标签映射与对齐**: MovieLens genome tag 的 SVD++ 训练方法 → 嵌入空间映射的技术路径
 3. **生成质量审计**: RAG-VisualRec (2025) 的 LLMGenQC 质量控制模块 → LLM-as-judge 在标注质量审计中的应用

 **文献综述结论**: 自由标签生成和标准化标签分类是两条平行线，缺少将两者统一的两阶段框架。

 ### B-Methodology (2500-3000 words)

 **方法被挑战时的防御**:
 - 挑战: "为什么不直接让 MLLM 输出 genome tag 评分（方向 A），而要先自由生成再映射？" → 回应: 自由生成阶段捕捉 MLLM 对电影的自然理解，不受固定词表约束；映射阶段提供标准化保证。两阶段分离了"理解"和"对齐"两个关注点。
 - 挑战: "映射层用嵌入相似度，如何保证语义对齐？" → 回应: 用 LLM-as-judge 审计每个映射，低质量映射标记并人工复核；同时对比多种嵌入模型（Ada / Sentence-BERT / E5）的映射质量。

 **最大局限**: 两阶段误差累积。

 **方法描述结构**:

 3.1 数据准备（同方向 A）

 3.2 阶段一：MLLM 自由标签生成
 - Prompt 设计：五维度引导（风格/情绪/主题/叙事手法/角色特征）
   - 示例 prompt: "Based on the movie keyframes, script, and dialogue provided, generate 15-20 natural language tags that describe this movie across five dimensions: visual style, emotional tone, narrative theme, storytelling technique, and character traits."
 - 输入: 关键帧 + 剧本片段 + 台词片段
 - 输出: 15-20 个自由形式标签（短语或单词）
 - 多次生成 (3 次) 取并集，去重

 3.3 阶段二：Genome 标签对齐
 - 嵌入模型: 对每个自由标签和 1,128 个 genome tag 计算嵌入
 - 映射策略: 每个自由标签 → top-3 候选 genome tag（余弦相似度）
 - 关联度聚合: 多个自由标签映射到同一 genome tag → 取最大相似度作为关联度
 - 阈值过滤: 相似度低于阈值 (如 0.5) 的映射标记为"低置信度"

 3.4 质量审计模块 (LLMGenQC-adapted)
 - 审计维度: 语义对齐度、一致性、覆盖度、安全性
 - 审计方式: 用强 LLM (GPT-4) 对每个自由标签 → genome tag 映射做 binary judgment (accept/reject)
 - 输出: 每个映射的 critic score + pass/fail 标签
 - 人工验证: 随机抽样 500 个映射，人工标注正确性，校准 LLM-as-judge

 3.5 评估协议
 - 生成质量: 生成标签 vs tags.csv 的人工评估（相关性、覆盖度、冗余度）
 - 对齐质量: 映射后 genome scores vs genome-scores.csv 的 RMSE
 - 审计质量: LLM-as-judge 与人工标注的一致率 (Cohen's kappa)
 - 端到端: 用 MLLM 标签做推荐，对比人工标签推荐 (Recall@K, NDCG@K)
 - 时序分割 + 冷启动 (同方向 A)

 3.6 对比基线
 - 基线 1: 方向 A 的直接评分方法（无自由生成阶段）
 - 基线 2: 仅用 genre → genome tag 映射表
 - 基线 3: 剧本文本 + BERT 多标签分类
 - 基线 4: 随机标签

 ### B-Results (1800-2200 words)

 **最重要发现（一句话）**: 两阶段管道在自由标签生成阶段达到了与 MovieLens 用户标签高语义一致性的水平，对齐阶段的 LLM-as-judge 审计准确率超过 85%，但在偏好驱动型标签上仍有语义鸿沟。

 **预期结果结构**:
 1. 生成质量: 自由标签 vs tags.csv 的相关性/覆盖度/冗余度指标
 2. 对齐质量: 映射后 genome scores 的 RMSE / Precision@K
 3. 审计质量: LLM-as-judge vs 人工标注的 kappa 一致率
 4. 端到端推荐: MLLM 标签推荐 vs 人工标签推荐的 Recall@K / NDCG@K
 5. 与方向 A 基线对比: 两阶段 vs 直接评分的质量差异
 6. 案例分析: 自由标签 → genome tag 映射的成功/失败案例

 ### B-Discussion (800-1000 words)

 **与文献对话**:
 - vs Folksonomication (2018): 多模态输入 vs 纯文本；MLLM vs 传统 NN
 - vs VLM4Rec (2026): 生成标签并对齐体系 vs 生成描述做推荐特征
 - vs RAG-VisualRec (2025): LLMGenQC 审计机制的迁移和适配

 **最想让读者记住的**: 两阶段"自由生成+对齐"框架比直接评分（方向 A）更能捕捉 MLLM 对电影的自然理解，因为自由生成不受固定词表约束，能发现 genome 体系未覆盖的维度。

 ### B-Conclusion (400-500 words)

 **如果只能留一段话**: 两阶段"MLLM 自由生成 → genome 对齐"框架同时产出开放词表标签和标准化评分，LLM-as-judge 审计提供了可量化的质量保证。自由标签生成能捕捉 genome 体系外的维度，但偏好驱动型标签的语义鸿沟仍是挑战。

 ## B-Step 2.5: 贡献锐化

 > **[INSIGHT: contribution_claim]**
 > 本文贡献在方法层面：提出两阶段标签生成框架（自由生成→对齐），首次将 MLLM 生成式标签输出与 MovieLens 双标签体系（tags.csv + genome tags）同时对齐，并引入 LLM-as-judge 审计机制为自动化标签提供可量化质量保证。

 ## B-Step 3: 论证压力测试

 **核心论证链**:
 1. MovieLens 双标签体系（自由 tag + genome tag）各有优劣 → 引 Harper & Konstan (2015)
 2. MLLM 生成能力可产出自然语言标签 → Technology Cycle 探针
 3. 自由标签通过嵌入映射可对齐到 genome 体系 → 嵌入相似度机制论证
 4. LLM-as-judge 可审计映射质量 → 参考 RAG-VisualRec LLMGenQC
 5. 两阶段管道在推荐下游任务上不劣于人工标签 → 端到端评估（待数据确认）

 **最弱点**: 论证链第 5 步——如果 MLLM 标签推荐显著不如人工标签，端到端价值存疑。防御: 即使推荐质量略低，在冷启动场景下（无人工标签）MLLM 标签仍是唯一选择，"有 vs 无"比"好 vs 更好"更重要。

 **反转测试**: 如果方向 A（直接评分）在 RMSE 上优于两阶段管道，则两阶段的价值仅在于自由标签的生成。防御: 两阶段的价值是"同时获得开放词表和标准化评分"，这是单阶段方法无法提供的。

 ## B-关键文献定位表

 | 位置 | 引用文献 | 角色 |
 |------|---------|------|
 | Intro 背景 | Harper & Konstan (2015) | MovieLens 数据集 |
 | Intro 动机 | Deldjoo et al. (2017) | 视觉特征替代标签 |
 | Related 故事1 | Folksonomication (2018) | 标签预测先驱 |
 | Related 故事1 | NoteLLM (2024) arXiv:2403.01744 | 生成 hashtag 增强嵌入 |
 | Related 故事2 | VLM4Rec (2026) | VLM 生成语义描述 |
 | Related 故事3 | RAG-VisualRec (2025) | LLMGenQC 质量审计 |
 | Method 对齐 | StoryMovie (2026) | 剧本-字幕对齐 |
 | Method 数据 | MovieNet (2019) | 多模态数据集 |
 | Eval 警示 | ML-1M Memorization (2026) | 数据泄露 |
 | Eval 基线 | TrUMAn (2021) | trope 分类基线 |

 ## B-时间线

 | 阶段 | 内容 | 时长 | 产出 |
 |------|------|------|------|
 | M1-M2 | 数据准备 + 五维度 prompt 设计 + 预跑 | 2 月 | prompt 模板 + 标签样例 |
 | M2-M4 | 阶段一生成实验 + 阶段二对齐实验 | 2 月 | 两阶段结果 |
 | M4-M5 | LLM-as-judge 审计 + 人工标注校准 | 1 月 | 审计报告 + kappa |
 | M5-M6 | 端到端推荐实验 + 时序分割 + 冷启动 | 1 月 | 推荐评估结果 |
 | M6-M8 | 论文写作 | 2 月 | 完整初稿 |
 | M8-M9 | 内部评审 + 修改 | 1 月 | 终稿 |
 | **总计** | | **8 月** | |

 ## B-目标场所

 WWW 2027 (The Web Conference) 或 SIGIR 2027
 论文类型: Novel Method
 引用格式: ACM
 预估字数: 7000-9000 words

 ---

---

 # 三方向横向对比

 | 维度 | A: GenomeTag-MLLM | B: MovieTagGen | C: ModalTag-Decomp |
 |------|-------------------|---------------|-------------------|
 | 论文类型 | Novel Method | Novel Method | 分析型 |
 | 核心贡献 | 方法（MLLM→genome 对齐） | 方法（两阶段+审计） | 分析（模态贡献分解） |
 | 论证风险 | 时序分割后性能下降 | 两阶段误差累积 | 发现太直觉性 |
 | Ground Truth | genome-scores.csv (干净) | tags.csv + genome (复杂) | genome-scores.csv (干净) |
 | 评估复杂度 | 中 | 高（含人工评估） | 低（消融即评估） |
 | 预估周期 | 6.5 月 | 8 月 | 5.5 月 |
 | 目标场所 | RecSys / CIKM | WWW / SIGIR | MM / RecSys |
 | 字数 | 6-8K | 7-9K | 5-7K |
 | 独立性 | 可独立 | 可独立（但最好在 A 之后） | 可独立（但最好在 A 之后） |
 | 组合建议 | 先做 A | A 完成后做 B 扩展 | A 或 C 先做均可 |

 **推荐执行顺序**: C → A → B
 - C 最快（5.5 月），先做建立模态理解基础
 - A 基于 C 的理解设计最优模态组合（6.5 月）
 - B 在 A 的基础上扩展为两阶段（8 月，但 A 的实验可复用）
 - 总计 ~20 月发 3 篇，第二年起每年 1-2 篇产出

 ---

 *以上三个研究计划基于 academic-paper plan 模式框架生成，已包含核心论点、章节计划、论证压力测试、文献定位表和时间线。可直接用于启动论文写作（切换到 full 模式）或提交导师审阅。*
