"""嵌入层：mock 确定性与规范化、缓存命中跳过 embedder、懒导入卫生。"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from movietaggen.align.embedding import (
    ApiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    embed_key,
    embed_texts,
    make_embedder,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def cos(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))


class CountingEmbedder:
    """测试桩：记录 embed 调用次数与送入文本，返回按文本可区分的确定性向量。"""

    name = "counting"
    dim = 64

    def __init__(self):
        self.calls = 0
        self.embedded = []

    def embed(self, texts):
        self.calls += 1
        self.embedded.extend(texts)
        return np.stack([_stub_vec(t) for t in texts])


def _stub_vec(text: str) -> np.ndarray:
    """按文本确定性取值（1..89），不同文本几乎必然不同行，可验证保序。"""
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 89 + 1
    return np.full(64, float(h), dtype=np.float32)


# ---------- MockEmbedder ----------

def test_mock_shape_and_unit_norm():
    a = MockEmbedder().embed(["aaa", "bbb", ""])
    assert a.shape == (2 + 1, 64)
    assert a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_mock_deterministic_and_canonical():
    a = MockEmbedder().embed(["Neon Visuals", "neon visuals"])
    assert np.allclose(a[0], a[1])                       # 规范化后同形
    assert abs(cos(MockEmbedder().embed(["aaa"])[0], MockEmbedder().embed(["bbb"])[0])) < 0.5


def test_mock_canonical_ignores_case_whitespace_punct():
    """大小写、连续空白、标点差异全部规范化抹平。"""
    a = MockEmbedder().embed(["Neon  Visuals!", "neon\tvisuals??", "NEON visuals"])
    assert np.allclose(a[0], a[1]) and np.allclose(a[1], a[2])


def test_mock_different_canonical_forms_differ():
    a = MockEmbedder().embed(["neon visuals", "neonvisuals"])
    assert not np.allclose(a[0], a[1])   # 规范化只留字母数字与空格：连写≠分写


def test_mock_deterministic_across_instances():
    v1 = MockEmbedder().embed(["identity crisis"])
    v2 = MockEmbedder().embed(["identity crisis"])
    assert np.array_equal(v1[0], v2[0])


def test_mock_empty_input():
    out = MockEmbedder().embed([])
    assert out.shape == (0, 64)


# ---------- embed_key ----------

def test_embed_key_stable_and_discriminating():
    k1 = embed_key("mock", "neon visuals")
    k2 = embed_key("mock", "neon visuals")
    assert k1 == k2 and len(k1) == 64
    assert embed_key("mock", "other") != k1
    assert embed_key("intfloat/multilingual-e5-base", "neon visuals") != k1  # 换模型即换键


# ---------- embed_texts 缓存 ----------

def test_cache_miss_then_hit_skips_embedder(tmp_path):
    e1 = CountingEmbedder()
    m1 = embed_texts(e1, ["t1", "t2"], tmp_path)
    assert e1.calls == 1 and e1.embedded == ["t1", "t2"]
    assert np.array_equal(m1[0], _stub_vec("t1")) and np.array_equal(m1[1], _stub_vec("t2"))

    # 同实例重放：全命中，0 次 embedder 调用
    m2 = embed_texts(e1, ["t1", "t2"], tmp_path)
    assert e1.calls == 1 and e1.embedded == ["t1", "t2"]
    assert np.array_equal(m2, m1)

    # 全新实例（calls=0）：命中来自磁盘而非内存
    e2 = CountingEmbedder()
    m3 = embed_texts(e2, ["t1", "t2"], tmp_path)
    assert e2.calls == 0
    assert np.array_equal(m3, m1)


def test_cache_partial_hit_only_sends_misses(tmp_path):
    embed_texts(CountingEmbedder(), ["t1"], tmp_path)   # 预热：只缓存 t1
    e = CountingEmbedder()
    m = embed_texts(e, ["t1", "t2"], tmp_path)
    assert e.calls == 1 and e.embedded == ["t2"]        # 命中项不进 embedder
    assert np.array_equal(m[0], _stub_vec("t1"))        # 缓存行
    assert np.array_equal(m[1], _stub_vec("t2"))        # 新算行
    # 乱序输入（miss 在前）：返回矩阵仍按输入顺序
    m2 = embed_texts(CountingEmbedder(), ["t2", "t1"], tmp_path)
    assert np.array_equal(m2[0], _stub_vec("t2")) and np.array_equal(m2[1], _stub_vec("t1"))


def test_cache_duplicate_texts_embedded_once(tmp_path):
    e = CountingEmbedder()
    m = embed_texts(e, ["dup", "dup", "other"], tmp_path)
    assert e.embedded.count("dup") == 1                 # 重复文本只送 embedder 一次
    assert np.array_equal(m[0], m[1])
    assert np.array_equal(m[0], _stub_vec("dup"))
    assert np.array_equal(m[2], _stub_vec("other"))


def test_cache_atomic_write_no_tmp_leftover(tmp_path):
    """落盘走 tmp + os.replace 原子替换，不留 .tmp 残留（同 ResponseCache）。"""
    embed_texts(CountingEmbedder(), ["t1"], tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []
    assert (tmp_path / f"{embed_key('counting', 't1')}.json").exists()


def test_cache_corrupt_file_treated_as_miss(tmp_path):
    """截断/损坏的缓存文件按未命中处理（重嵌自愈），不永久抛错。"""
    bad = tmp_path / f"{embed_key('counting', 't1')}.json"
    bad.write_text('[[1.0, 2.0', encoding="utf-8")
    e = CountingEmbedder()
    m = embed_texts(e, ["t1"], tmp_path)
    assert e.calls == 1
    assert np.array_equal(m[0], _stub_vec("t1"))


def test_cache_roundtrip_value_is_json_float_list(tmp_path):
    """缓存值格式契约：JSON 浮点列表（跨进程/跨语言可读）。"""
    embed_texts(CountingEmbedder(), ["t1"], tmp_path)
    raw = json.loads((tmp_path / f"{embed_key('counting', 't1')}.json").read_text())
    assert isinstance(raw, list) and all(isinstance(x, float) for x in raw) and len(raw) == 64


def test_embed_texts_empty_input(tmp_path):
    """空输入 → (0, d)，不触碰 embedder（不建目录、不调用）。"""
    e = CountingEmbedder()
    out = embed_texts(e, [], tmp_path)
    assert out.shape == (0, 64)
    assert e.calls == 0


def test_embed_texts_key_includes_embedder_name(tmp_path):
    """同文本不同 embedder 不互相命中（模型名进键）。"""
    e1 = CountingEmbedder()
    e1.name = "model-a"
    e2 = CountingEmbedder()
    e2.name = "model-b"
    embed_texts(e1, ["t1"], tmp_path)
    embed_texts(e2, ["t1"], tmp_path)
    assert e1.calls == 1 and e2.calls == 1


# ---------- make_embedder 分发 ----------

def test_make_embedder_dispatch():
    api = make_embedder("api:text-embedding-3", mock=True)   # api: 前缀无视 mock
    assert isinstance(api, ApiEmbedder)
    assert api.model == "text-embedding-3" and api.name == "api:text-embedding-3"

    assert isinstance(make_embedder("api:x", mock=False), ApiEmbedder)
    assert isinstance(make_embedder("intfloat/multilingual-e5-base", mock=True), MockEmbedder)

    local = make_embedder("intfloat/multilingual-e5-base", mock=False, device="cuda")
    assert isinstance(local, LocalEmbedder)
    assert local.name == "intfloat/multilingual-e5-base" and local.device == "cuda"


# ---------- LocalEmbedder 惰性 ----------

def test_local_embedder_construction_does_not_load_model():
    """构造与空输入不加载模型（不下载、不加载 sentence_transformers 权重）。"""
    emb = LocalEmbedder("some-model", device="cpu")
    assert emb.name == "some-model"
    assert emb._model is None
    out = emb.embed([])
    assert out.shape == (0, 0)   # dim 未知 → 0 列；绝不为此触发模型加载
    assert emb._model is None


def test_lazy_import_guard():
    """mock 路径全程不触发 sentence_transformers / openai 加载。

    子进程隔离：主 pytest 进程可能已被其他测试加载这些库，进程内断言不可靠。
    """
    code = "\n".join([
        "import sys",
        "import tempfile",
        "from pathlib import Path",
        "from movietaggen.align.embedding import embed_texts, make_embedder",
        "emb = make_embedder('mock', mock=True)",
        "assert emb.name == 'mock'",
        "assert emb.embed(['a', 'b']).shape == (2, 64)",
        "with tempfile.TemporaryDirectory() as d:",
        "    embed_texts(emb, ['x', 'y'], Path(d))",
        "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers 被加载'",
        "assert 'openai' not in sys.modules, 'openai 被加载'",
        "print('guard-ok')",
    ])
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=REPO_ROOT, timeout=120)
    assert r.returncode == 0, f"子进程失败:\nstdout={r.stdout}\nstderr={r.stderr}"
    assert "guard-ok" in r.stdout


# ---------- ApiEmbedder ----------

class _FakeEmbeddingsApi:
    """openai embeddings 接口最小替身：记录请求，返回与文本长度相关的确定性向量。"""

    def __init__(self):
        self.requests = []

    def create(self, model, input):
        batch = list(input)
        self.requests.append((model, batch))
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float(len(t)), 1.0])
            for i, t in enumerate(batch)
        ])


def _api_embedder_with_fake(model="mymodel"):
    fake = _FakeEmbeddingsApi()
    emb = ApiEmbedder(model, client=SimpleNamespace(embeddings=fake))
    return emb, fake


def test_api_embedder_batches_64_per_request():
    emb, fake = _api_embedder_with_fake()
    texts = [f"t{i}" for i in range(130)]
    mat = emb.embed(texts)
    assert emb.name == "api:mymodel"
    assert len(fake.requests) == 3                       # 64 + 64 + 2
    assert [len(b) for _, b in fake.requests] == [64, 64, 2]
    assert all(m == "mymodel" for m, _ in fake.requests)
    assert mat.shape == (130, 2) and emb.dim == 2


def test_api_embedder_l2_normalized_and_order_preserved():
    emb, _ = _api_embedder_with_fake()
    texts = ["t0", "t63", "t64", "t129"]                 # 首批末位/次批首尾各取样
    mat = emb.embed(texts)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0)
    for i, t in enumerate(texts):                        # [len, 1] 归一化后可由文本反推
        expected = np.array([float(len(t)), 1.0]) / np.hypot(len(t), 1.0)
        assert np.allclose(mat[i], expected)


def test_api_embedder_empty_input_never_calls_client():
    emb, fake = _api_embedder_with_fake()
    out = emb.embed([])
    assert out.shape == (0, 0)
    assert fake.requests == []


def test_api_embedder_missing_env_raises(monkeypatch):
    """env 未设置时构建即失败（不静默），且发生在 openai 导入之前。"""
    monkeypatch.delenv("NO_SUCH_EMB_BASE", raising=False)
    monkeypatch.delenv("NO_SUCH_EMB_KEY", raising=False)
    emb = ApiEmbedder("m", base_url_env="NO_SUCH_EMB_BASE", api_key_env="NO_SUCH_EMB_KEY")
    with pytest.raises(RuntimeError, match="NO_SUCH_EMB_BASE"):
        emb.embed(["x"])
