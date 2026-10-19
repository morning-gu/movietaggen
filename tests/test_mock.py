"""Mock 生成客户端：确定性 + 轮次噪声、维度子集可区分、模态门控、prompt 契约解析。

维度子集 / 模态门控的可区分性是整个消融框架的防伪绿保证（spec §10 mock 约束 b）：
若 mock 对维度/模态消融不敏感，消融实验测到的就是噪声而非结构。
"""
import json

from movietaggen.config import VALID_DIMENSIONS, ModelConfig
from movietaggen.data.movienet import MovieMultimodal
from movietaggen.data.sample import DIMENSION_GROUPS, SYNTHETIC_TAGS
from movietaggen.generate.mock import DIMENSION_MODALITIES, MockGenClient
from movietaggen.generate.prompts import build_gen_messages, normalize_tag

MODEL_CFG = ModelConfig()


def _movie(tmp_path=None, *, n_frames=0, title="Crimson Horizon"):
    frames = []
    if tmp_path is not None and n_frames:
        kf = tmp_path / "keyframes"
        kf.mkdir(exist_ok=True)
        for i in range(n_frames):
            p = kf / f"f{i}.jpg"
            p.write_bytes(b"jpeg-bytes-" + str(i).encode())
            frames.append(p)
    return MovieMultimodal(movienet_id="mn01", title=title, year=2018,
                           imdb_id="tt0000001", keyframes=frames,
                           script="剧情内容", subtitles="台词内容")


def _msgs(movie, dimensions, modalities):
    return build_gen_messages(movie, dimensions, modalities, [], MODEL_CFG)


def _complete(msgs, run_id="run-0", client=None):
    resp = (client or MockGenClient()).complete(msgs, run_id)
    assert resp.ok
    return json.loads(resp.content)


# 规范词表（五个内容维度组；受众偏好组不参与生成）
_VOCAB_ALL = {t["tag"] for t in SYNTHETIC_TAGS if t["group"] in DIMENSION_GROUPS}


def _vocab(*dims):
    return {t["tag"] for t in SYNTHETIC_TAGS if DIMENSION_GROUPS.get(t["group"]) in dims}


def _canonical(tags):
    """还原词序交换噪声：交换是前两词的对合，非词表标签交换后必须落回词表。"""
    out = []
    for t in tags:
        if t in _VOCAB_ALL:
            out.append(t)
        else:
            w = t.split()
            restored = " ".join([w[1], w[0]] + w[2:])
            assert restored in _VOCAB_ALL, f"未知标签 {t!r}（既非词表项也非可还原的交换变体）"
            out.append(restored)
    return out


def test_dimension_modalities_table():
    """门控表即控制器决议：五维度 × 依赖模态。"""
    assert DIMENSION_MODALITIES == {
        "style": {"video"},
        "mood": {"video", "dialogue"},
        "theme": {"script", "dialogue"},
        "narrative": {"script"},
        "character": {"script", "dialogue"},
    }


def test_deterministic_and_run_noise(tmp_path):
    m = _movie(tmp_path, n_frames=3)
    msgs = _msgs(m, list(VALID_DIMENSIONS), ["video", "script", "dialogue"])
    a = MockGenClient().complete(msgs, "run-0")
    b = MockGenClient().complete(msgs, "run-0")
    assert a.content == b.content           # 同 run 确定性
    c = MockGenClient().complete(msgs, "run-1")
    assert a.content != c.content           # 轮次噪声：逐轮内容确实变化
    # n_runs 并集非平凡：多轮并集大于单轮
    runs = [set(_complete(msgs, f"run-{i}")) for i in range(3)]
    assert len(runs[0] | runs[1] | runs[2]) > len(runs[0])


def test_run_noise_includes_word_swaps():
    """轮次噪声含词序交换变体：多轮多次后必出现非词表（可还原）标签。"""
    m = _movie()
    msgs = _msgs(m, list(VALID_DIMENSIONS), ["script", "dialogue"])
    seen = [t for i in range(8) for t in _complete(msgs, f"run-{i}")]
    assert any(t not in _VOCAB_ALL for t in seen)   # 交换机制确实在发火
    assert all(t in _VOCAB_ALL for t in _canonical(seen))  # 变体全部可还原


def test_dimension_subset_changes_output(tmp_path):
    """消融可区分（spec §10 mock 约束 b）：维度子集输出是全集输出的真子集。"""
    m = _movie(tmp_path, n_frames=3)
    full = set(_complete(_msgs(m, list(VALID_DIMENSIONS), ["video", "script", "dialogue"])))
    part = set(_complete(_msgs(m, ["style"], ["video", "script", "dialogue"])))
    assert part < full and len(part) < len(full)


def test_modality_gating(tmp_path):
    """同维度、模态不同 → 产标签的维度不同（DIMENSION_MODALITIES 门控）。"""
    m = _movie(tmp_path, n_frames=3)
    dims = list(VALID_DIMENSIONS)
    n = MockGenClient().n_tags_per_dimension

    video_only = set(_complete(_msgs(m, dims, ["video"])))
    assert len(video_only) == 2 * n                      # 仅 style+mood 过门控
    assert set(_canonical(video_only)) <= _vocab("style", "mood")
    assert not (set(_canonical(video_only)) & _vocab("narrative", "character"))

    script_only = set(_complete(_msgs(m, dims, ["script"])))
    assert len(script_only) == 3 * n                     # theme+narrative+character
    assert set(_canonical(script_only)) <= _vocab("theme", "narrative", "character")


def test_title_only_prompt_single_tag():
    """title-only 探针（无内容节）：只产 1 个标题派生标签。"""
    m = _movie()
    tags = _complete(_msgs(m, list(VALID_DIMENSIONS), []))
    assert tags == [f"{normalize_tag(m.title)} movie"]


def test_n_tags_per_dimension_param(tmp_path):
    m = _movie(tmp_path, n_frames=3)
    default = _complete(_msgs(m, ["style", "mood"], ["video"]))
    assert len(default) == 8                              # 默认 4 × 2 维度
    two = _complete(_msgs(m, ["style", "mood"], ["video"]),
                    client=MockGenClient(n_tags_per_dimension=2))
    assert len(two) == 4


def test_prompt_contract_parsed():
    """从 prompt 文本解析标题/维度/可用内容节（关键帧节带张数后缀 → 前缀匹配）。"""
    text = (
        "电影：《Test Film》（1999）\n\n"
        "## 生成维度\n"
        "- style：视觉风格（画面、色彩、镜头语言）\n"
        "- mood：情绪基调\n\n"
        "## 关键帧（99 张，见后附图像）\n帧内容占位"
    )
    tags = _complete([{"role": "user", "content": text}])
    assert len(tags) == 8                                 # style+mood 各 4
    assert set(_canonical(tags)) <= _vocab("style", "mood")
    # title-only 手工 prompt：标题行解析 → 标题派生标签
    title_text = "电影：《Test Film》（1999）\n\n说明文字"
    assert _complete([{"role": "user", "content": title_text}]) == ["test film movie"]


def test_response_envelope_and_usage():
    """信封形状同 A：缓存命中路径从 raw.choices[0].message.content 还原文本。"""
    resp = MockGenClient().complete(_msgs(_movie(), ["theme"], ["script"]), "run-0")
    assert resp.ok
    assert resp.usage == {"prompt_tokens": 100, "completion_tokens": 50}
    assert resp.raw["choices"][0]["message"]["content"] == resp.content
    assert resp.raw["mock"] is True
    assert isinstance(json.loads(resp.content), list)
