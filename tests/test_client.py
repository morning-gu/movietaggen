"""客户端：transport 注入测试（无网络）。重试、response_format 降级、usage 提取。"""
import pytest

from movietaggen.generate.client import LLMResponse, OpenAIClient


def _ok_transport(content='{"a": 1}'):
    def transport(model, messages, temperature, response_format, extra_body=None):
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    return transport


def test_complete_success():
    client = OpenAIClient(model="m", base_url_env="X", transport=_ok_transport(),
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="run-0")
    assert isinstance(resp, LLMResponse)
    assert resp.ok and resp.content == '{"a": 1}'
    assert resp.usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_extra_body_forwarded_to_transport():
    """端点专有参数（如 DashScope enable_thinking）必须到达传输层。"""
    seen = {}

    def spy(model, messages, temperature, response_format, extra_body=None):
        seen.update(extra_body=extra_body, response_format=response_format)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=spy,
                          request_interval=0.0,
                          extra_body={"enable_thinking": False})
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok
    assert seen["extra_body"] == {"enable_thinking": False}
    assert seen["response_format"] == {"type": "json_object"}


def test_no_extra_body_sends_none():
    seen = {}

    def spy(model, messages, temperature, response_format, extra_body=None):
        seen["extra_body"] = extra_body
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=spy,
                          request_interval=0.0)
    client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert seen["extra_body"] is None


def test_sampling_params_fingerprint():
    """采样参数指纹：extra_body 变化必须改变缓存键，未设置时与旧键逐字节一致。"""
    from types import SimpleNamespace

    from movietaggen.generate.cache import cache_key
    from movietaggen.generate.client import sampling_params

    plain = SimpleNamespace(temperature=0.7, extra_body={})
    with_extra = SimpleNamespace(temperature=0.7,
                                 extra_body={"enable_thinking": False})
    msgs = [{"role": "user", "content": "hi"}]
    # 未设置 extra_body：与旧版手写 dict 的键完全一致（存量缓存不失效）
    assert cache_key(msgs, "m", sampling_params(plain), "r") == \
        cache_key(msgs, "m", {"temperature": 0.7}, "r")
    # extra_body 进键：开/关思考不得互相命中
    assert cache_key(msgs, "m", sampling_params(with_extra), "r") != \
        cache_key(msgs, "m", sampling_params(plain), "r")


def test_retry_on_transport_error_then_success():
    calls = {"n": 0}

    def flaky(model, messages, temperature, response_format, extra_body=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=flaky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and calls["n"] == 2


def test_gives_up_after_max_retries():
    def always_fail(model, messages, temperature, response_format, extra_body=None):
        raise ConnectionError("down")

    client = OpenAIClient(model="m", base_url_env="X", transport=always_fail,
                          request_interval=0.0, max_retries=2)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert not resp.ok and resp.content == ""


def test_response_format_fallback():
    """端点不支持 json_object → 去掉 response_format 重发一次。"""
    calls = {"n": 0}

    def picky(model, messages, temperature, response_format, extra_body=None):
        calls["n"] += 1
        if response_format is not None:
            raise ValueError("response_format is not supported")
        return {"choices": [{"message": {"content": "plain"}}], "usage": {}}

    client = OpenAIClient(model="m", base_url_env="X", transport=picky,
                          request_interval=0.0)
    resp = client.complete([{"role": "user", "content": "hi"}], run_id="r")
    assert resp.ok and resp.content == "plain" and calls["n"] == 2


def test_missing_env_raises():
    import os
    client = OpenAIClient(model="m", base_url_env="NO_SUCH_ENV_XYZ")
    with pytest.raises(RuntimeError, match="NO_SUCH_ENV_XYZ"):
        client._build_transport()  # 环境变量缺失时构建即失败（不静默）
