"""合成小样本：结构镜像真实原始数据格式，用于开发、测试与冒烟。

产出四块：
- raw/movienet/   （镜像 MovieNet 假定格式）
- raw/movielens/  （镜像 MovieLens ml-latest 格式）
- processed/tag_groups.json（合成样本的分组是已知的，直接生成，不走 LLM）
- processed/eval_test.txt（合成等价的固定测试清单，与真实 88 部清单同格式，
  供 fixed_split / 冒烟消费；Random(42) 确定性采样，重跑幂等）

相对 A（genometag）的扩展：
- 标签组从 4 组扩到 6 组：新增叙事手法（narrative）与角色特征（character），
  内容组与 config.VALID_DIMENSIONS 的五个维度一一对应（受众偏好除外——
  偏好型标签表达观众主观评价，不由内容维度机制生成）；
- tags.csv 从常量占位（"fun"）改为确定性自由标签：user_tags_for(title)
  哈希采样 4-8 个规范标签并按哈希注入噪声变体（小写化 / 去尾 s / 常见
  错拼），模拟真实用户自由标签的不规范书写；
- 规范词表（SYNTHETIC_TAGS / DIMENSION_GROUPS / base_score）与
  MockGenClient（movietaggen/generate/mock.py）共享，使生成伪标签与
  用户自由标签天然重叠——相关性 / 覆盖度指标在 mock 路径下因此非退化。

ground truth 生成规则：relevance = 0.85 * base_score(title, tag) + 0.15 * 独立噪声。
MockScoringClient（baselines）预测 base_score + 小幅轮次噪声，
两者天然相关——这是 mock"可控相关性"的实现基础。
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

SYNTHETIC_MOVIES = [
    # movie_id(ML), movienet_id, title, year, genres, imdb_id(不带 tt 前缀的数字)
    {"movie_id": 1, "movienet_id": "mn01", "title": "Crimson Horizon", "year": 2018,
     "genres": ["Action", "Thriller"], "imdb_id": 1},
    {"movie_id": 2, "movienet_id": "mn02", "title": "Neon Alley", "year": 2019,
     "genres": ["Crime", "Drama"], "imdb_id": 2},
    {"movie_id": 3, "movienet_id": "mn03", "title": "Quiet Harbor", "year": 2020,
     "genres": ["Drama", "Romance"], "imdb_id": 3},
    {"movie_id": 4, "movienet_id": "mn04", "title": "Last Ember", "year": 2021,
     "genres": ["Sci-Fi", "Action"], "imdb_id": 4},
    {"movie_id": 5, "movienet_id": "mn05", "title": "Paper Skies", "year": 2023,
     "genres": ["Drama", "Mystery"], "imdb_id": 5},
]

SYNTHETIC_TAGS = [
    # group: 视觉风格 / 情绪氛围 / 叙事主题 / 受众偏好 / 叙事手法 / 角色特征
    {"tag_id": 1,  "tag": "neon visuals",    "group": "视觉风格", "tag_type": "content",
     "definition": "画面以霓虹灯光与高饱和夜景为主"},
    {"tag_id": 2,  "tag": "dark palette",    "group": "视觉风格", "tag_type": "content",
     "definition": "整体色调偏暗、低照明场景占主导"},
    {"tag_id": 3,  "tag": "vivid colors",    "group": "视觉风格", "tag_type": "content",
     "definition": "色彩鲜艳明亮、对比强烈"},
    {"tag_id": 4,  "tag": "grainy texture",  "group": "视觉风格", "tag_type": "content",
     "definition": "胶片颗粒感或做旧质感"},
    {"tag_id": 5,  "tag": "static camera",   "group": "视觉风格", "tag_type": "content",
     "definition": "大量固定机位长镜头"},
    {"tag_id": 6,  "tag": "slow pans",       "group": "视觉风格", "tag_type": "content",
     "definition": "缓慢的横摇/横移镜头"},
    {"tag_id": 7,  "tag": "melancholic",     "group": "情绪氛围", "tag_type": "content",
     "definition": "整体情绪忧郁、低落"},
    {"tag_id": 8,  "tag": "tense",           "group": "情绪氛围", "tag_type": "content",
     "definition": "持续紧张、压迫感"},
    {"tag_id": 9,  "tag": "hopeful",         "group": "情绪氛围", "tag_type": "content",
     "definition": "充满希望与治愈感"},
    {"tag_id": 10, "tag": "bleak",           "group": "情绪氛围", "tag_type": "content",
     "definition": "荒凉、绝望的世界观"},
    {"tag_id": 11, "tag": "dreamlike",       "group": "情绪氛围", "tag_type": "content",
     "definition": "梦幻、超现实的氛围"},
    {"tag_id": 12, "tag": "nostalgic",       "group": "情绪氛围", "tag_type": "content",
     "definition": "怀旧、追忆过去的情绪"},
    {"tag_id": 13, "tag": "revenge plot",    "group": "叙事主题", "tag_type": "content",
     "definition": "以复仇为核心驱动力"},
    {"tag_id": 14, "tag": "family bonds",    "group": "叙事主题", "tag_type": "content",
     "definition": "家庭关系与亲情羁绊"},
    {"tag_id": 15, "tag": "identity crisis", "group": "叙事主题", "tag_type": "content",
     "definition": "主角经历身份认同危机"},
    {"tag_id": 16, "tag": "heist plan",      "group": "叙事主题", "tag_type": "content",
     "definition": "围绕精心策划的抢劫/盗窃行动"},
    {"tag_id": 17, "tag": "lost love",       "group": "叙事主题", "tag_type": "content",
     "definition": "失去的爱情或错过的恋人"},
    {"tag_id": 18, "tag": "survival",        "group": "叙事主题", "tag_type": "content",
     "definition": "绝境求生主线"},
    {"tag_id": 19, "tag": "overrated",       "group": "受众偏好", "tag_type": "preference",
     "definition": "名过其实、评价高于实际质量"},
    {"tag_id": 20, "tag": "cult classic",    "group": "受众偏好", "tag_type": "preference",
     "definition": "小众狂热追捧的经典"},
    {"tag_id": 21, "tag": "feel-good",       "group": "受众偏好", "tag_type": "preference",
     "definition": "看完心情变好的轻松作品"},
    {"tag_id": 22, "tag": "slow burn",       "group": "受众偏好", "tag_type": "preference",
     "definition": "节奏缓慢、渐入佳境"},
    {"tag_id": 23, "tag": "rewatchable",     "group": "受众偏好", "tag_type": "preference",
     "definition": "值得反复观看"},
    {"tag_id": 24, "tag": "niche appeal",    "group": "受众偏好", "tag_type": "preference",
     "definition": "只对特定人群有吸引力"},
    # —— B 新增两组（tag_id 接续 A 的编号 25-36）——
    {"tag_id": 25, "tag": "nonlinear timeline", "group": "叙事手法", "tag_type": "content",
     "definition": "叙事时间线被打乱、以非线性顺序推进"},
    {"tag_id": 26, "tag": "twist ending",       "group": "叙事手法", "tag_type": "content",
     "definition": "结局出现颠覆预期的重大反转"},
    {"tag_id": 27, "tag": "voice-over narration", "group": "叙事手法", "tag_type": "content",
     "definition": "以画外音旁白贯穿并推进叙事"},
    {"tag_id": 28, "tag": "slow pacing",        "group": "叙事手法", "tag_type": "content",
     "definition": "整体叙事节奏缓慢、铺垫绵长"},
    {"tag_id": 29, "tag": "framing story",      "group": "叙事手法", "tag_type": "content",
     "definition": "套层叙事：故事之中再套故事"},
    {"tag_id": 30, "tag": "open ending",        "group": "叙事手法", "tag_type": "content",
     "definition": "开放式结尾、不给出明确答案"},
    {"tag_id": 31, "tag": "flawed protagonist", "group": "角色特征", "tag_type": "content",
     "definition": "主角带有明显缺陷与内在弱点"},
    {"tag_id": 32, "tag": "morally gray hero",  "group": "角色特征", "tag_type": "content",
     "definition": "主人公道德立场模糊、亦正亦邪"},
    {"tag_id": 33, "tag": "strong female lead", "group": "角色特征", "tag_type": "content",
     "definition": "以独立有力的女性角色为核心"},
    {"tag_id": 34, "tag": "mentor figure",      "group": "角色特征", "tag_type": "content",
     "definition": "存在引导主角成长的导师型人物"},
    {"tag_id": 35, "tag": "unlikely duo",       "group": "角色特征", "tag_type": "content",
     "definition": "身份悬殊或格格不入的双人搭档"},
    {"tag_id": 36, "tag": "ensemble cast",      "group": "角色特征", "tag_type": "content",
     "definition": "群像式多主角、戏份并行展开"},
]

# 内容维度组 → 维度名（与 config.VALID_DIMENSIONS 一一对应）。
# 受众偏好不在此列：偏好型标签不由内容维度生成（user_tags_for 与
# MockGenClient 的词表均只取本表覆盖的组）。
DIMENSION_GROUPS = {
    "视觉风格": "style",
    "情绪氛围": "mood",
    "叙事主题": "theme",
    "叙事手法": "narrative",
    "角色特征": "character",
}


def _hash_unit(key: str) -> float:
    """哈希 → 确定性 [0,1) 均匀值。"""
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000


def base_score(title: str, tag: str) -> float:
    """电影×标签 的确定性基准分。ground truth 与 MockScoringClient 共用（相关性的来源）。"""
    return _hash_unit(f"base|{title}|{tag}")


def _truth(title: str, tag: str) -> float:
    """ground truth = 0.85*base + 0.15*独立噪声，clip 到 [0,1]。"""
    noise = _hash_unit(f"truth|{title}|{tag}")
    return min(1.0, max(0.0, 0.85 * base_score(title, tag) + 0.15 * noise))


def _drop_trailing_s(tag: str) -> str:
    """噪声变体之一：去掉末词尾 s（复数 → 单数式的手滑）。无尾 s 原样返回。"""
    words = tag.split()
    if words[-1].endswith("s"):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def _typo_swap(tag: str, title: str) -> str:
    """噪声变体之一：常见错拼。有 "ie" 则换 "y"，否则按哈希位置去一个元音。"""
    if "ie" in tag:
        return tag.replace("ie", "y", 1)
    vowels = [i for i, ch in enumerate(tag) if ch in "aeiou"]
    if not vowels:
        return tag
    pos = vowels[int(_hash_unit(f"vowel|{title}|{tag}") * len(vowels))]
    return tag[:pos] + tag[pos + 1:]


def _apply_noise(tag: str, title: str) -> str:
    """对规范标签施加一种确定性噪声变体（变体由哈希挑选）。

    变体池：全小写 / 去尾 s / 常见错拼。规范词表本身全小写，"全小写"对
    已小写标签是空转——只在真正改变标签的变体中挑选，保证被标记为噪声
    的标签确实非规范（测试据此断言每部电影至少一个非规范自由标签）。
    """
    variants = (
        lambda t: t.lower(),
        lambda t: _drop_trailing_s(t),
        lambda t: _typo_swap(t, title),
    )
    effective = [v for v in variants if v(tag) != tag]
    if not effective:
        return tag
    chosen = effective[int(_hash_unit(f"variant|{title}|{tag}") * len(effective))]
    return chosen(tag)


def user_tags_for(title: str) -> list[str]:
    """电影 → 4-8 个确定性自由标签（规范词表哈希采样 + 噪声变体）。

    从五个内容维度组的词表（DIMENSION_GROUPS 覆盖的 30 个规范标签）按
    哈希排序采样；噪声位置与变体均由哈希决定，且每部电影至少 1 个噪声
    标签。与 MockGenClient 共享规范词表，使相关性 / 覆盖度指标非退化。
    """
    pool = [t["tag"] for t in SYNTHETIC_TAGS if t["group"] in DIMENSION_GROUPS]
    n = 4 + int(_hash_unit(f"count|{title}") * 5)            # 4..8
    picked = sorted(pool, key=lambda tg: (_hash_unit(f"pick|{title}|{tg}"), tg))[:n]
    n_noisy = 1 + int(_hash_unit(f"nnoise|{title}") * 3)     # 1..3（至少 1 个）
    noisy = sorted(picked, key=lambda tg: (_hash_unit(f"noise|{title}|{tg}"), tg))[:n_noisy]
    return [_apply_noise(tg, title) if tg in noisy else tg for tg in picked]


def _make_frame(path: Path, color: tuple, index: int) -> None:
    """生成一张渐变关键帧（尺寸小，冒烟要快）。"""
    img = Image.new("RGB", (64, 36), color)
    draw = ImageDraw.Draw(img)
    shade = int(255 * index / 8)
    draw.rectangle([0, 0, 63, 18], fill=(min(255, color[0] + shade), color[1], color[2]))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=60)


_MOVIE_COLORS = [(120, 20, 20), (20, 20, 120), (20, 120, 60), (80, 80, 20), (60, 20, 80)]


def _script_text(title: str) -> str:
    return (
        f"INT. {title.upper()} - OPENING - NIGHT\n\n"
        "The city hums outside. Rain streaks the window.\n\n"
        f"Our protagonist studies a photograph. The story of {title} begins here.\n\n"
        "EXT. STREET - CONTINUOUS\n\nTraffic. Neon reflections in puddles.\n"
    )


def _srt_text(title: str) -> str:
    return (
        "1\n00:00:01,000 --> 00:00:03,500\n"
        f"They said {title} would change everything.\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\n"
        "Nothing changes. We just keep moving.\n\n"
        "3\n00:00:07,000 --> 00:00:09,500\n"
        "Then move faster.\n"
    )


def make_synthetic_dataset(out_root: Path) -> None:
    """生成完整合成数据集。out_root 即 data/synthetic/。"""
    out_root = Path(out_root)
    mn = out_root / "raw" / "movienet"
    ml = out_root / "raw" / "movielens"
    mn.mkdir(parents=True, exist_ok=True)
    ml.mkdir(parents=True, exist_ok=True)

    # MovieNet 侧（write_text 不会自动建目录，先建 scripts/ 与 subtitles/）
    (mn / "scripts").mkdir(parents=True, exist_ok=True)
    (mn / "subtitles").mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(SYNTHETIC_MOVIES):
        for f in range(8):
            _make_frame(mn / "keyframes" / m["movienet_id"] / f"frame_{f:03d}.jpg",
                        _MOVIE_COLORS[i], f)
        (mn / "scripts" / f'{m["movienet_id"]}.txt').write_text(
            _script_text(m["title"]), encoding="utf-8")
        (mn / "subtitles" / f'{m["movienet_id"]}.srt').write_text(
            _srt_text(m["title"]), encoding="utf-8")
    pd_rows = [{"movie_id": m["movienet_id"], "title": m["title"], "year": m["year"],
                "imdb_id": f'tt{m["imdb_id"]:07d}'} for m in SYNTHETIC_MOVIES]
    import pandas as pd
    pd.DataFrame(pd_rows).to_csv(mn / "movies.csv", index=False)

    # MovieLens 侧（movies.csv 镜像真实三列格式，年份由 load_movies 从标题解析）
    movies_rows = [{"movieId": m["movie_id"],
                    "title": f'{m["title"]} ({m["year"]})',
                    "genres": "|".join(m["genres"])}
                   for m in SYNTHETIC_MOVIES]
    pd.DataFrame(movies_rows).to_csv(ml / "movies.csv", index=False)
    pd.DataFrame([{"movieId": m["movie_id"], "imdbId": m["imdb_id"], "tmdbId": m["movie_id"]}
                  for m in SYNTHETIC_MOVIES]).to_csv(ml / "links.csv", index=False)
    pd.DataFrame([{"tagId": t["tag_id"], "tag": t["tag"]}
                  for t in SYNTHETIC_TAGS]).to_csv(ml / "genome-tags.csv", index=False)
    score_rows = [{"movieId": m["movie_id"], "tagId": t["tag_id"],
                   "relevance": round(_truth(m["title"], t["tag"]), 4)}
                  for m in SYNTHETIC_MOVIES for t in SYNTHETIC_TAGS]
    pd.DataFrame(score_rows).to_csv(ml / "genome-scores.csv", index=False)
    # 用户自由标签：每电影 4-8 条确定性噪声标签（userId 轮转 1-3，时间戳常量保确定性）
    user_rows = [{"userId": u % 3 + 1, "movieId": m["movie_id"], "tag": tag,
                  "timestamp": 1700000000}
                 for m in SYNTHETIC_MOVIES
                 for u, tag in enumerate(user_tags_for(m["title"]))]
    pd.DataFrame(user_rows).to_csv(ml / "tags.csv", index=False)

    # processed：tag_groups.json（合成分组已知，直接生成）
    processed = out_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "tag_groups.json").write_text(
        json.dumps(SYNTHETIC_TAGS, ensure_ascii=False, indent=1), encoding="utf-8")

    # processed：eval_test.txt（固定测试清单的合成等价物；k = max(1, len//5)，
    # 5 部 → 1 部。与真实清单同格式：每行一个 id，fixed_split 直接可读）
    ml_ids = sorted(m["movie_id"] for m in SYNTHETIC_MOVIES)
    eval_ids = random.Random(42).sample(ml_ids, max(1, len(ml_ids) // 5))
    (processed / "eval_test.txt").write_text(
        "".join(f"{mid}\n" for mid in eval_ids), encoding="utf-8")
