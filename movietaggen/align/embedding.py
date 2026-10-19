"""嵌入层：本地 E5 / OpenAI 兼容 API / 确定性 mock 三实现 + 逐文本磁盘缓存。

三实现统一契约（`.name` / `.dim` / `.embed(texts) -> np.ndarray (n, d)`）：
- MockEmbedder：哈希伪向量（零网络、零依赖），测试与冒烟路径。规范化
  （casefold + 仅留字母数字与空格 + 折叠空白）后同形文本同向量，不同文本
  近似正交——对齐/覆盖度指标在 mock 路径下可区分不同标签。
- LocalEmbedder：sentence-transformers 惰性导入（mock/测试路径不付首次
  import 开销，也不触发模型下载），encode 后 L2 归一化输出 float32。
- ApiEmbedder：openai SDK 惰性导入，env 同 ModelConfig（base_url/key），
  一次请求至多 64 条文本，结果 L2 归一化。

`embed_texts` 逐文本落盘缓存：键 = sha256(embedder.name + "\\0" + text)，
换模型即换键；值 = JSON 浮点列表；写入走 tmp + os.replace 原子替换（同
ResponseCache——崩溃不留截断 JSON 卡死续跑），损坏文件按未命中自愈。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

API_BATCH_SIZE = 64   # 单次 API 请求的文本数上限（端点限批）
_MOCK_DIM = 64


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """逐行 L2 归一化（零向量行按原样返回，不做除零）。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


# ---------- MockEmbedder ----------

def _canonical(text: str) -> str:
    """规范化：casefold → 仅保留字母数字与空格（其余丢弃）→ 折叠连续空白 → 去首尾。

    isalnum 是 Unicode 感知的（中文/重音字符保留），多语标签天然适用。
    """
    kept = "".join(ch if ch.isalnum() or ch.isspace() else "" for ch in text.casefold())
    return " ".join(kept.split())


def _mock_unit_vec(canonical: str) -> np.ndarray:
    """sha256 摘要按位展开成 64 维 ±1 符号向量，再归一化。

    无 random 模块、跨进程确定：同形必同向量；不同形相互独立，64 维符号
    向量的余弦近似 N(0, 1/8)，|cos| ≥ 0.5 概率 ~6e-5，满足近似正交。
    """
    bits = int.from_bytes(hashlib.sha256(canonical.encode("utf-8")).digest(), "big")
    vec = np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(_MOCK_DIM)],
                   dtype=np.float32)
    return vec / np.linalg.norm(vec)


class MockEmbedder:
    """确定性伪嵌入：哈希即向量，测试与 mock 配置专用。"""

    name = "mock"
    dim = _MOCK_DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, _MOCK_DIM), dtype=np.float32)
        return np.stack([_mock_unit_vec(_canonical(t)) for t in texts])


# ---------- LocalEmbedder ----------

class LocalEmbedder:
    """sentence-transformers 本地模型；模型加载惰性到首次 embed 非空输入。"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.name = model_name
        self.dim = None          # 首次真实编码后回填（模型维度随模型而定）
        self._model = None

    def _load(self):
        if self._model is None:
            # 懒导入：mock/测试路径不需要安装后首次 import 的开销，更不触发下载
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            # dim 未知时返回 (0, 0)：绝不为了空输入加载/下载模型
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        model = self._load()
        vecs = model.encode(list(texts), convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        mat = np.asarray(vecs, dtype=np.float32)
        # 双保险：normalize_embeddings 已归一化，再归一一次防后端差异
        mat = _l2_normalize(mat)
        self.dim = mat.shape[1]
        return mat


# ---------- ApiEmbedder ----------

class ApiEmbedder:
    """OpenAI 兼容 embeddings 端点；客户端惰性构建，测试可注入替身。"""

    def __init__(self, model: str, base_url_env: str = "OPENAI_BASE_URL",
                 api_key_env: str = "OPENAI_API_KEY", client=None):
        self.model = model
        self.name = f"api:{model}"   # 进缓存键：与 make_embedder 的 api: 前缀约定互逆
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.dim = None          # 首次真实请求后回填
        self._client = client    # None = 按 env 惰性构建（注入仅测试用）

    def _build_client(self):
        base_url = os.environ.get(self.base_url_env)
        api_key = os.environ.get(self.api_key_env)
        if not base_url or not api_key:
            raise RuntimeError(
                f"环境变量 {self.base_url_env} / {self.api_key_env} 未设置"
                "（复制 .env.example 为 .env 并填入）")
        from openai import OpenAI  # 懒导入：env 校验在前，缺失时不付 import 开销
        return OpenAI(base_url=base_url, api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32)
        if self._client is None:
            self._client = self._build_client()
        vecs: list[list[float]] = []
        for start in range(0, len(texts), API_BATCH_SIZE):
            batch = list(texts[start:start + API_BATCH_SIZE])
            resp = self._client.embeddings.create(model=self.model, input=batch)
            # 端点契约 data[i].index 对应输入位次；按 index 排序防御乱序返回
            ordered = sorted(resp.data, key=lambda d: d.index)
            vecs.extend(d.embedding for d in ordered)
        mat = _l2_normalize(np.asarray(vecs, dtype=np.float32))
        self.dim = mat.shape[1]
        return mat


# ---------- 工厂与缓存 ----------

def make_embedder(model_name: str, mock: bool, device: str = "cpu"):
    """模型名以 `api:` 前缀 → ApiEmbedder（无视 mock）；否则 mock ? Mock : Local。"""
    if model_name.startswith("api:"):
        return ApiEmbedder(model_name[len("api:"):])
    if mock:
        return MockEmbedder()
    return LocalEmbedder(model_name, device=device)


def embed_key(name: str, text: str) -> str:
    """缓存键 = sha256(embedder.name + "\\0" + text)：换模型即换键，同文本不串模型。"""
    return hashlib.sha256(f"{name}\x00{text}".encode("utf-8")).hexdigest()


def _vec_path(cache_root: Path, key: str) -> Path:
    return cache_root / f"{key}.json"


def _read_vec(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 中断残留的截断文件按未命中处理——重嵌重写，自愈（同 ResponseCache）
        return None
    if not isinstance(data, list):
        return None
    return np.asarray(data, dtype=np.float32)


def _write_vec(cache_root: Path, key: str, vec: np.ndarray) -> None:
    # 先写临时文件再原子替换：崩溃/断电不会留下截断的 JSON 卡死续跑
    tmp = cache_root / f"{key}.json.tmp"
    tmp.write_text(json.dumps(vec.tolist()), encoding="utf-8")
    os.replace(tmp, _vec_path(cache_root, key))


def embed_texts(embedder, texts: list[str], cache_root: Path) -> np.ndarray:
    """逐文本缓存嵌入：命中不调用 embedder，未命中批量送 embedder 后逐条落盘。

    重复文本同键去重（只嵌一次）；返回矩阵按输入顺序（float32）。
    空输入 → (0, dim)，不触碰 embedder（dim 未知的实现退化为 (0, 0)）。
    """
    if not texts:
        return np.zeros((0, embedder.dim or 0), dtype=np.float32)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    keys = [embed_key(embedder.name, t) for t in texts]
    results: dict[int, np.ndarray] = {}
    misses: list[int] = []
    for i, key in enumerate(keys):
        vec = _read_vec(_vec_path(cache_root, key))
        if vec is None:
            misses.append(i)
        else:
            results[i] = vec

    if misses:
        # 同键取首现位置去重：一批未命中只送 embedder 一次
        first_by_key: dict[str, int] = {}
        for i in misses:
            first_by_key.setdefault(keys[i], i)
        vecs = embedder.embed([texts[i] for i in first_by_key.values()])
        vec_by_key = {key: np.asarray(v, dtype=np.float32)
                      for key, v in zip(first_by_key, vecs)}
        for key, vec in vec_by_key.items():
            _write_vec(cache_root, key, vec)
        for i in misses:
            results[i] = vec_by_key[keys[i]]

    return np.stack([results[i] for i in range(len(texts))]).astype(np.float32)
