"""合成样本生成：结构、五维度分组、自由标签噪声、确定性。"""
import json

import pandas as pd

from movietaggen.config import VALID_DIMENSIONS
from movietaggen.data.sample import (
    DIMENSION_GROUPS,
    SYNTHETIC_MOVIES,
    SYNTHETIC_TAGS,
    base_score,
    make_synthetic_dataset,
    user_tags_for,
)

_NEW_GROUPS = ("叙事手法", "角色特征")


def test_constants_shape():
    assert len(SYNTHETIC_TAGS) == 36          # A 的 24 + 叙事手法 6 + 角色特征 6
    assert len({t["group"] for t in SYNTHETIC_TAGS}) == 6
    assert len(SYNTHETIC_MOVIES) == 5
    # tag_type 只有 content / preference 两种
    assert {t["tag_type"] for t in SYNTHETIC_TAGS} == {"content", "preference"}
    # 每个标签都带非空中文释义
    assert all(isinstance(t["definition"], str) and t["definition"] for t in SYNTHETIC_TAGS)
    # 新组 tag_id 连续接续 A 的编号（25-36）
    new_ids = [t["tag_id"] for t in SYNTHETIC_TAGS if t["group"] in _NEW_GROUPS]
    assert new_ids == list(range(25, 37))


def test_dimension_groups_mapping():
    """五个内容维度组 → 维度名；受众偏好是偏好型标签，不属内容维度。"""
    assert DIMENSION_GROUPS == {
        "视觉风格": "style",
        "情绪氛围": "mood",
        "叙事主题": "theme",
        "叙事手法": "narrative",
        "角色特征": "character",
    }
    assert "受众偏好" not in DIMENSION_GROUPS
    assert sorted(DIMENSION_GROUPS.values()) == sorted(VALID_DIMENSIONS)


def test_base_score_deterministic_and_bounded():
    assert base_score("Crimson Horizon", "neon visuals") == base_score("Crimson Horizon", "neon visuals")
    assert 0.0 <= base_score("A", "x") <= 1.0
    assert base_score("A", "x") != base_score("B", "x")  # 不同电影不同分


def test_user_tags_deterministic_and_noisy():
    """确定性（哈希采样、无随机模块）+ 每部电影至少一个非规范噪声标签。"""
    canonical = {t["tag"] for t in SYNTHETIC_TAGS}
    for movie in SYNTHETIC_MOVIES:
        title = movie["title"]
        a, b = user_tags_for(title), user_tags_for(title)
        assert a == b, title                      # 确定性
        assert 4 <= len(a) <= 8, title
        assert len(set(a)) == len(a), title       # 无重复
        # 噪声注入：至少存在一个非规范形式（大小写/错拼/别名变体）
        assert any(t not in canonical for t in a), title


def test_noise_variant_mechanics():
    """控制器决议的三种噪声变体：行为逐一可检验。"""
    from movietaggen.data.sample import _apply_noise, _drop_trailing_s, _typo_swap

    # 去尾 s：末词复数 → 单数；无尾 s 原样
    assert _drop_trailing_s("vivid colors") == "vivid color"
    assert _drop_trailing_s("tense") == "tense"
    # 错拼：含 "ie" → 换 "y"
    assert _typo_swap("movie night", "T") == "movy night"
    # 错拼：无 "ie" → 哈希位置去一个元音（确定性 + 确实少一个字符）
    assert _typo_swap("tense", "T") in {"tnse", "tens"}   # "tense" 去任一 e
    assert _typo_swap("tense", "T") == _typo_swap("tense", "T")
    # 应用层：规范词表全小写，"全小写"变体必空转；_apply_noise 只在真正
    # 改变标签的变体中按哈希挑选，故对任何规范标签总产出非规范形式
    for t in SYNTHETIC_TAGS:
        assert _apply_noise(t["tag"], "X") != t["tag"]


def test_synthetic_layout(tmp_path):
    make_synthetic_dataset(tmp_path)
    assert (tmp_path / "raw/movielens/tags.csv").exists()
    tags = pd.read_csv(tmp_path / "raw/movielens/tags.csv")
    assert len(tags) >= 20            # 每电影 4-8 个自由标签


def test_tags_csv_schema(tmp_path):
    """tags.csv 镜像真实 MovieLens 四列格式；userId 轮转 1-3；每电影 4-8 条。"""
    make_synthetic_dataset(tmp_path)
    utags = pd.read_csv(tmp_path / "raw/movielens/tags.csv")
    assert list(utags.columns) == ["userId", "movieId", "tag", "timestamp"]
    assert set(utags["userId"].unique()) <= {1, 2, 3}
    for mid in range(1, 6):
        n = int((utags["movieId"] == mid).sum())
        assert 4 <= n <= 8, mid


def test_new_groups_present(tmp_path):
    make_synthetic_dataset(tmp_path)
    groups = json.loads((tmp_path / "processed" / "tag_groups.json").read_text(encoding="utf-8"))
    names = {g["group"] for g in groups}
    assert {"叙事手法", "角色特征"} <= names       # narrative / character 两组
    for g in _NEW_GROUPS:
        sel = [t for t in groups if t["group"] == g]
        assert len(sel) == 6
        assert all(t["tag_type"] == "content" for t in sel)
        assert all(t["definition"] for t in sel)


def test_make_synthetic_dataset_structure(tmp_path):
    make_synthetic_dataset(tmp_path)
    # MovieLens 侧
    ml = tmp_path / "raw" / "movielens"
    scores = pd.read_csv(ml / "genome-scores.csv")
    tags = pd.read_csv(ml / "genome-tags.csv")
    movies = pd.read_csv(ml / "movies.csv")
    links = pd.read_csv(ml / "links.csv")
    assert len(scores) == 5 * 36
    assert len(tags) == 36
    assert len(movies) == 5
    assert len(links) == 5
    # 年份分割（2022）：4 部 < 2022，1 部 ≥ 2022（年份编码在标题内，同真实格式）
    titles = movies["title"].tolist()
    assert sum(1 for t in titles if "(2023)" in t) == 1
    assert sum(1 for t in titles if any(y in t for y in
               ("(2018)", "(2019)", "(2020)", "(2021)"))) == 4
    # MovieNet 侧
    mn = tmp_path / "raw" / "movienet"
    mn_movies = pd.read_csv(mn / "movies.csv")
    assert len(mn_movies) == 5
    frames = sorted((mn / "keyframes" / "mn01").glob("*.jpg"))
    assert len(frames) == 8  # 每部 8 帧
    assert (mn / "subtitles" / "mn01.srt").exists()
    assert (mn / "scripts" / "mn01.txt").exists()
    # processed：tag_groups.json 已随样本生成
    tg = json.loads((tmp_path / "processed" / "tag_groups.json").read_text(encoding="utf-8"))
    assert len(tg) == 36
    assert {"tag_id", "tag", "group", "tag_type", "definition"} <= set(tg[0].keys())
