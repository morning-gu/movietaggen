# 数据产物 Schema

约定优于配置：目录布局由 `movietaggen/config.py` 的 `Paths` 固定。真实数据
interim/processed 在 `data/`；合成数据在 `data/synthetic/` 的同构镜像
（raw/interim/processed，由 `00_make_synthetic.py` 生成）。每个实验目录
`results/<exp_name>/`（如 `results/pilot/`、`results/ablation/video-only/`）
内分 stage-gen / stage-align / stage-audit / evaluation 四段，根下另有
`config.yaml` 配置快照（结果目录自包含、可复现）。

## 数据侧（真实：data/，合成：data/synthetic/）

### data/interim/aligned.parquet
| 列 | 类型 | 说明 |
|----|------|------|
| movie_id | int | MovieLens movieId（全管道主键） |
| title | str | 去年份的标题 |
| year | int | 上映年 |
| genres | list[str] | |
| imdb_id | int | 规范化 IMDb 数字 ID |
| movienet_id | str | |
| match_method | str | imdb / title |

### data/interim/align_report.md
匹配率、歧义清单（同名同年）、未匹配清单。

### data/interim/freetag_stats.json（03 一次性分析，B 特有）
两段式负载 `{"freetag_stats": {...}, "literal_overlap": {...}}`：
- freetag_stats：`n_unique`（casefold 归一唯一标签数）、`n_rows`（实验集内
  标签行数）、`tags_per_movie_mean`、`top_tags`（[[tag, count], ...] 前 50）、
  `year_buckets`（各年代桶标签行数）；
- literal_overlap：高频自由标签与 genome 标签的严格字面重合（`top_n_total`、
  `exact_match`、`match_list`）——字面口径是重合下界，嵌入口径留给评估层。
人读版摘要：同目录 `freetag_report.md`。

### data/processed/dataset.parquet
aligned 列去掉 match_method，加 `keyframes: list[str]`（相对 movienet 根）、
`script_path`、`subtitles_path`（相对路径或 null）。

### data/processed/genome_scores.parquet
`movie_id, tag_id, relevance`（仅对齐电影；ground truth）。

### data/processed/genome_tags.parquet
`tag_id, tag`。

### data/processed/tag_groups.json（入 git）
`[{tag_id, tag, group, tag_type: content|preference, definition}]`。

## results/<exp>/stage-gen/（阶段一：自由标签生成）

### responses.jsonl（append-only，续跑事实来源，看门狗以增长判活）
每行 `{"task_key": "<movie_id>|<run_id>", "status": "done|parse_failed|no_content",
"tags": [...], "usage": {...}}`——单任务一次生成全部维度，故键只含电影与轮次。

### tags.parquet
`movie_id, tag, n_runs`——jsonl 跨轮并集去重压缩（(movie, normalize_tag) 唯一，
n_runs = 产出该标签的轮数）。

### run_manifest.json
任务计数（done / parse_failed / no_content）、token 用量、`n_tags_total`。

## results/<exp>/stage-align/（阶段二：嵌入映射 + 校准）

### mappings.parquet
`movie_id, tag, tag_id, cosine, rank, low_conf`——每自由标签取 genome 余弦
top-k 候选各一行；low_conf 仅旗标（cosine < threshold），不丢行。

### scores_raw.parquet / scores_calibrated.parquet
均 `movie_id, tag_id, score`——前者对 mappings 全量聚合（agg: max|mean），
后者经校准（isotonic / linear）后的同构表。评估的推荐接口（spec §7.4）即
这两个文件本身，无额外实现。

### calibration.json
校准器 `to_dict` 序列化（仅 str/float/list，无 numpy 类型）：isotonic 为
`{method, x_thresholds, y_thresholds}`，linear 为 `{method, slope, intercept}`；
空实验无从拟合，以 `{method: ..., fitted: false}` 哨兵占位。配套
`run_manifest.json`（n_movies / n_free_tags / n_mappings / n_scores /
calib_method / train_pairs / embed_model）。

## results/<exp>/stage-audit/（阶段三：LLM-as-judge 审计）

### judge_mapping.parquet
`movie_id, tag, tag_id, cosine, verdict, reason, judge_model`——映射逐条判定
（verdict: accept / reject，解析失败记 parse_failed）。

### judge_gen_quality.parquet
`movie_id, relevance, coverage, judge_model`——逐电影生成质量判定。

### scores_filtered.parquet
`movie_id, tag_id, score`——judge reject 剔除后重聚合 + 重校准
（audit.apply_filter: false 时不产出）。

### calibration_sample.csv
人工标注桥，固定 8 列：`movie_id, title, tag, tag_id, genome_tag, cosine,
judge_verdict, human_verdict`（human_verdict 全空留给人填）。行数 =
min(audit.calibration_sample, 判定行数)，抽样种子 audit.seed。

## results/baselines/（跨实验共享，不随 exp_name）

### baselines.parquet
`movie_id, tag_id, score, method`——method ∈ direct_score / genre_map /
random / bert（按 config.baselines.run 子集）。配套 `run_manifest.json`
（实际产出方法与总行数）与 `config.yaml` 快照。

## results/<exp>/evaluation/（08 评估）

- `metrics.csv`：行 = 预测方法 × 子集视图，列固定 `method, subset, rmse_full,
  rmse_pred_only, spearman, precision@k, ndcg@k`（RMSE 双口径：内连接只评
  敢打分对 / 全叉积缺失补 0）。方法含 two-stage[raw / calibrated / filtered]
  与各基线；subset 五视图 = 全部 / 时序训练(<y) / 时序测试(≥y) /
  tag:content / tag:preference（后两个需 tag_groups）。`report.md` 为人读汇总。
- `gen_quality.json`：生成质量双口径——嵌入口径（relevance / coverage /
  redundancy，逐电影语义对用户 top-K 标签）+ judge 口径（judge_gen_quality
  均值，键前缀 `judge_`）。
