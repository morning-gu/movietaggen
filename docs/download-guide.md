# 数据下载指南

本框架的原始数据默认**与方向 A（GenomeTag-MLLM）共享**：`data.raw_dir`
默认指向 `../genometag-mllm/data/raw`（相对仓库根解析）——若方向 A 已下载
完毕，本框架无需任何下载，直接从 `01_verify_data.py` 开始。独立下载则把
数据放到本仓库 `data/raw/` 并在各配置把 `data.raw_dir` 设为 `null`。

## MovieLens（含 genome 与用户自由标签）

1. 访问 https://grouplens.org/datasets/movielens/ ，下载 **ml-latest.zip**
   （或 ml-25m.zip，genome 文件结构相同）
2. 解压到 `<raw_dir>/movielens/ml-latest/`（默认即
   `../genometag-mllm/data/raw/movielens/ml-latest/`），确保以下文件在位：
   - `genome-scores.csv`（movieId, tagId, relevance）
   - `genome-tags.csv`（tagId, tag；共 1,128 个）
   - `movies.csv`、`links.csv`
   - `tags.csv`（userId, movieId, tag, timestamp）——**本框架必需**：用户
     自由标签是阶段一生成的对照源（03 统计）与生成质量嵌入口径的参考集（08）
3. 校验：`.venv/Scripts/python scripts/01_verify_data.py`

## MovieNet（已落地：服务器部署 + 与 A 共享转换产物）

1. 访问 https://movienet.site/ ，按其流程申请/下载（通常含元数据 CSV 与各
   模态数据的分块下载；关键帧体量约数百 GB，建议在服务器磁盘上操作）
2. 官方分发格式转换为适配器假定格式：

   ```bash
   .venv/Scripts/python scripts/convert_movienet.py \
       --movienet-root /cache/MovieNet \
       --out ../genometag-mllm/data/raw/movienet   # 独立下载时用 data/raw/movienet
   ```

   该脚本从方向 A 复制（假设的官方分发布局一致，见脚本头注释：list.v1.txt、
   meta_data/meta/<tt>.json、script1K.v1.zip、subtitle1K.v1.zip、frames/）。
3. 转换后的目标结构（适配器 `movietaggen/data/movienet.py` 假定的格式，
   与实际发布有出入时改该模块顶部的常量与 `load_movienet`）：

   ```
   <raw_dir>/movienet/
   ├── movies.csv            # movie_id,title,year,imdb_id
   ├── keyframes/<id>/*.jpg
   ├── scripts/<id>.txt      # 覆盖不足时字幕可作台词模态代理
   └── subtitles/<id>.srt
   ```

4. 校验：`01_verify_data.py` 会报告**年份分布**（对齐可行性的参考信息）。
   MovieNet 可能不含较新电影——这只影响 metrics.csv 的年代分桶视图
   （`protocol.temporal_split_year` 仅驱动该描述性视图）；协议分割已固定为
   `data/processed/eval_test.txt` 清单（440 协议），无需随数据年份重定分割。

## A 迁移物（已提交入 git，正常使用无需自行获取）

两个跨仓库迁移物随代码提交入库，日常使用无需任何获取步骤；来源备查：

- `data/processed/eval_test.txt`：固定测试清单（88 行 movieId），自方向 A
  仓库 `data/processed/eval_test.txt` 复制迁入，sha256 与 A 侧核对一致
  （钉在 `data/processed/direct_score_a.provenance.json`）。
- `data/processed/direct_score_a.parquet`：方向 A `v3-full` 实跑预测
  （297,792 行 = 88 × 1,128 × 3 轮），自 A 服务器
  `results/v3-full/annotations.parquet` 迁入（随附
  `direct_score_a.provenance.json` 溯源：源仓库/路径、sha256、行数校验值）。
