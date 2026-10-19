"""缓存：键稳定性、run_id/参数维度、落盘往返。"""
from movietaggen.generate.cache import ResponseCache, cache_key

MSGS = [{"role": "user", "content": "hi"}]


def test_key_stable():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 0.7}, "run-0")
    assert a == b and len(a) == 64


def test_key_differs_by_run_id():
    """Review Focus #1：同 prompt 不同轮次必须不同键，否则多轮取平均失效。"""
    a = cache_key(MSGS, "gpt-4o", {}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {}, "run-1")
    assert a != b


def test_key_changes_with_params():
    a = cache_key(MSGS, "gpt-4o", {"temperature": 0.0}, "run-0")
    b = cache_key(MSGS, "gpt-4o", {"temperature": 1.0}, "run-0")
    assert a != b


def test_cache_key_sensitive_to_run_id_and_params():
    m = [{"role": "user", "content": "x"}]
    assert cache_key(m, "mo", {"temperature": 0.7}, "run-0") != cache_key(m, "mo", {"temperature": 0.7}, "run-1")
    assert cache_key(m, "mo", {"temperature": 0.7}, "r") != cache_key(m, "mo", {"temperature": 0.3}, "r")


def test_put_get_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path)
    assert cache.get("nonexistent") is None
    resp = {"choices": [{"message": {"content": "{\"a\": 1}"}}], "usage": {"prompt_tokens": 3}}
    cache.put("k1", resp)
    assert cache.get("k1") == resp


def test_truncated_file_treated_as_miss(tmp_path):
    """终审 I3：崩溃残留的截断 JSON 按未命中处理（管道重调自愈），不永久抛错。"""
    cache = ResponseCache(tmp_path)
    (tmp_path / "broken.json").write_text('{"choices": [{"mess', encoding="utf-8")
    assert cache.get("broken") is None


def test_put_leaves_no_tmp_leftover(tmp_path):
    """终审 I3：put 走 tmp + os.replace 原子替换，不留 .json.tmp 残留。"""
    cache = ResponseCache(tmp_path)
    resp = {"a": 1}
    cache.put("k2", resp)
    assert cache.get("k2") == resp
    assert not (tmp_path / "k2.json.tmp").exists()
