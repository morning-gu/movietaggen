"""生成 prompt 组装与标签解析测试（无网络；prompt 文本即 Mock 解析契约）。"""
from movietaggen.config import VALID_DIMENSIONS, ModelConfig
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.generate.content import approx_tokens, truncate_to_budget
from movietaggen.generate.prompts import (
    DIMENSIONS_BLOCK,
    FewShotTags,
    SYSTEM_PROMPT,
    build_gen_messages,
    normalize_tag,
    parse_tags,
)


def _movie(tmp_path=None, *, n_frames=0, script="剧情内容", subtitles="台词内容", year=2018):
    frames = []
    if tmp_path is not None and n_frames:
        kf = tmp_path / "keyframes"
        kf.mkdir(exist_ok=True)
        for i in range(n_frames):
            p = kf / f"f{i}.jpg"
            p.write_bytes(b"jpeg-bytes-" + str(i).encode())
            frames.append(p)
    return MovieMultimodal(movienet_id="mn01", title="Crimson Horizon", year=year,
                           imdb_id="tt0000001", keyframes=frames,
                           script=script, subtitles=subtitles)


def _text(msgs):
    """user 消息的 text part。"""
    return msgs[1]["content"][0]["text"]


def _image_parts(msgs):
    return [p for p in msgs[1]["content"] if p["type"] == "image_url"]


def test_message_structure_mirrors_a():
    msgs = build_gen_messages(_movie(), ["style"], [], [], ModelConfig())
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    content = msgs[1]["content"]
    assert content[0]["type"] == "text"
    assert "电影：《Crimson Horizon》（2018）" in content[0]["text"]


def test_title_line_year_none():
    msgs = build_gen_messages(_movie(year=None), ["style"], [], [], ModelConfig())
    assert "电影：《Crimson Horizon》（未知年份）" in _text(msgs)


def test_instruction_present():
    text = _text(build_gen_messages(_movie(), list(VALID_DIMENSIONS), [], [], ModelConfig()))
    assert "基于提供的电影内容，围绕上述维度生成 15-20 个英文标签" in text
    assert '["tag one"]' in text


def test_dimension_block_full_and_order():
    text = _text(build_gen_messages(_movie(), list(VALID_DIMENSIONS), [], [], ModelConfig()))
    assert DIMENSIONS_BLOCK in text
    for frag in ("视觉风格", "情绪基调", "叙事主题", "叙事手法", "角色特征"):
        assert frag in text
    # VALID_DIMENSIONS 固定顺序渲染
    assert text.index("- style：") < text.index("- mood：") < text.index("- theme：") \
        < text.index("- narrative：") < text.index("- character：")


def test_dimension_subset_changes_instruction():
    text = _text(build_gen_messages(_movie(), ["style", "mood"], ["video"], [], ModelConfig()))
    assert "视觉风格" in text and "角色特征" not in text


def test_empty_modalities_is_title_only():
    msgs = build_gen_messages(_movie(), list(VALID_DIMENSIONS), [], [], ModelConfig())
    text = _text(msgs)
    assert "电影：《" in text
    assert "## 关键帧" not in text and "## 剧本节选" not in text and "## 台词/字幕" not in text
    # text-only：无图像 part
    assert all(p["type"] == "text" for p in msgs[1]["content"])


def test_content_sections_only_when_content_available(tmp_path):
    # script=None → 即使配置 script 模态也不渲染剧本节（沿 A 的 truthy 检查）
    m = _movie(tmp_path, n_frames=2, script=None)
    text = _text(build_gen_messages(m, ["style"], ["video", "script"], [], ModelConfig()))
    assert "## 关键帧" in text
    assert "## 剧本节选" not in text
    # video 未配置但有帧 → 无关键帧节、无图像 part
    m2 = _movie(tmp_path, n_frames=2)
    msgs2 = build_gen_messages(m2, ["style"], ["script"], [], ModelConfig())
    assert "## 关键帧" not in _text(msgs2) and _image_parts(msgs2) == []


def test_few_shot_block_rendered():
    fs = [FewShotTags(title="Old Film", year=2001, tags=["noir", "rain-soaked"])]
    text = _text(build_gen_messages(_movie(), ["style"], [], fs, ModelConfig()))
    assert "## 示例（已知标签的电影）" in text
    assert "《Old Film》（2001）：noir, rain-soaked" in text


def test_few_shot_absent_when_empty():
    text = _text(build_gen_messages(_movie(), ["style"], [], [], ModelConfig()))
    assert "## 示例" not in text


def test_budget_truncation():
    long_script = "scene " * 200          # 1200 字符 ≈ 300 tokens
    long_subs = "sub " * 100
    cfg = ModelConfig(script_token_budget=10, dialogue_token_budget=5)
    text = _text(build_gen_messages(
        _movie(script=long_script, subtitles=long_subs),
        ["style"], ["script", "dialogue"], [], cfg))
    assert "## 剧本节选" in text and "## 台词/字幕" in text
    assert long_script not in text        # 已截断
    assert long_script[: 10 * 4] in text  # 截断保留前 budget*4 字符
    assert long_subs not in text


def test_budget_within_limit_kept_whole():
    script = "short scene"
    text = _text(build_gen_messages(_movie(script=script), ["style"], ["script"], [],
                                    ModelConfig()))
    assert script in text


def test_images_attached_when_video(tmp_path):
    cfg = ModelConfig(max_frames=3)
    msgs = build_gen_messages(_movie(tmp_path, n_frames=8), ["style"], ["video"], [], cfg)
    imgs = _image_parts(msgs)
    assert len(imgs) == 3                 # 8 帧抽样至 max_frames
    for img in imgs:
        url = img["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")


def test_parse_tags_variants():
    assert parse_tags('["neon visuals", "slow burn"]') == ["neon visuals", "slow burn"]
    assert parse_tags("1. neon visuals\n2. slow burn") == ["neon visuals", "slow burn"]
    assert parse_tags("- Neon Visuals\n- neon visuals") == ["Neon Visuals"]   # 归一去重


def test_parse_tags_json_with_prose():
    text = '好的，标签如下：\n["neon visuals", "slow burn"]\n希望有帮助'
    assert parse_tags(text) == ["neon visuals", "slow burn"]


def test_parse_tags_fenced_json():
    text = '```json\n["tag a", "tag b"]\n```'
    assert parse_tags(text) == ["tag a", "tag b"]


def test_parse_tags_bullet_prefix():
    assert parse_tags("• neon visuals\n• slow burn") == ["neon visuals", "slow burn"]


def test_parse_tags_empty_and_garbage_never_raise():
    assert parse_tags("") == []
    assert parse_tags("没有可用内容") == []
    assert parse_tags('{"not": "a list"}') == []
    assert parse_tags("[broken json") == []
    assert parse_tags('["a", 1, null]') == []   # 非全字符串数组 → 降级逐行 → 空


def test_parse_tags_dedup_keeps_first_original():
    assert parse_tags('["Neon Visuals", "neon visuals", "NEON  VISUALS"]') == ["Neon Visuals"]


def test_normalize_tag():
    assert normalize_tag("  Neon   Visuals ") == "neon visuals"
    assert normalize_tag("Slow\tBurn") == "slow burn"


def test_content_helpers_extracted_from_a():
    """自 A 抽出的共享助手行为一致（A 的测试契约镜像）。"""
    assert approx_tokens("x" * 41) == 10
    assert truncate_to_budget("short", 100) == "short"
    assert truncate_to_budget("x" * 45, 10) == "x" * 40   # 45 字符 ≈ 11 tokens > 10 → 截断
