"""OpenAI 兼容 LLM 客户端：可注入 transport 便于测试，懒导入 openai SDK。

不做输出解析——parse_scores 在 prompts.py（Task 10），重试解析在 pipeline（Task 12）。
本层只负责：传输、重试、response_format 降级、限速间隔、usage 记录。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

Transport = Callable[[str, list[dict], float, dict | None, dict | None], dict]


def sampling_params(model_cfg) -> dict:
    """采样参数指纹：进缓存键，保证换参数（temperature/extra_body）即换键。
    模型无关：未设置 extra_body 的配置得到与旧版逐字节相同的键（存量缓存不失效）。
    """
    params: dict = {"temperature": model_cfg.temperature}
    if getattr(model_cfg, "extra_body", None):
        params["extra_body"] = dict(model_cfg.extra_body)
    return params


@dataclass
class LLMResponse:
    content: str
    usage: dict
    raw: dict
    ok: bool


class OpenAIClient:
    def __init__(self, model: str, base_url_env: str,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.7,
                 request_interval: float = 0.5, transport: Transport | None = None,
                 max_retries: int = 3, extra_body: dict | None = None):
        self.model = model
        self.base_url_env = base_url_env
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.request_interval = request_interval
        self.max_retries = max_retries
        self.extra_body = dict(extra_body) if extra_body else {}
        self._transport = transport

    def _build_transport(self) -> Transport:
        base_url = os.environ.get(self.base_url_env)
        api_key = os.environ.get(self.api_key_env)
        if not base_url or not api_key:
            raise RuntimeError(
                f"环境变量 {self.base_url_env} / {self.api_key_env} 未设置"
                "（复制 .env.example 为 .env 并填入）")
        from openai import OpenAI  # 懒导入：mock/测试路径不需要安装后首次 import 的开销

        sdk = OpenAI(base_url=base_url, api_key=api_key)

        def transport(model: str, messages: list[dict], temperature: float,
                      response_format: dict | None, extra_body: dict | None) -> dict:
            kwargs: dict = {"model": model, "messages": messages,
                            "temperature": temperature}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if extra_body:
                kwargs["extra_body"] = extra_body
            return sdk.chat.completions.create(**kwargs).model_dump()

        return transport

    def complete(self, messages: list[dict], run_id: str) -> LLMResponse:
        """run_id 仅用于日志语境；缓存键在 pipeline 层构造。

        重试语义：json_object 不被端点支持 → 降级为无 response_format 重试（不耗退避）；
        其他传输异常 → 指数退避重试至多 max_retries 次，仍失败返回 ok=False。
        """
        if self._transport is None:
            self._transport = self._build_transport()
        time.sleep(self.request_interval)

        last_err: Exception | None = None
        use_json = True
        for attempt in range(self.max_retries):
            try:
                return self._call(self._transport, messages, allow_json=use_json)
            except Exception as e:  # noqa: BLE001 —— 传输层任何异常统一处理
                if use_json and _detect_json_format_error(e):
                    use_json = False   # 端点不支持 json_object → 降级后重试
                    continue
                last_err = e
                time.sleep(0.5 * (2 ** attempt))
        return LLMResponse(content="", usage={}, raw={"error": str(last_err)}, ok=False)

    def _call(self, transport: Transport, messages: list[dict],
              allow_json: bool) -> LLMResponse:
        rf = {"type": "json_object"} if allow_json else None
        raw = transport(self.model, messages, self.temperature, rf,
                        self.extra_body or None)
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(content=content or "", usage=raw.get("usage", {}),
                           raw=raw, ok=True)


def _detect_json_format_error(e: Exception) -> bool:
    """端点不支持 response_format=json_object 的错误文本识别。"""
    text = str(e).lower()
    return "response_format" in text or "json_object" in text or "json mode" in text


def make_client(config):
    """工厂：mock 配置返回 MockGenClient（不联网、不碰缓存）。"""
    # README 的文档化步骤是 cp .env.example .env——此处负责把 .env 装入进程环境，
    # 否则该步骤是空操作。放在工厂内（而非模块顶层）保持 import 无副作用。
    from dotenv import load_dotenv
    load_dotenv()
    if config.mock:
        from movietaggen.generate.mock import MockGenClient
        return MockGenClient()
    return OpenAIClient(
        model=config.model.model, base_url_env=config.model.base_url_env,
        api_key_env=config.model.api_key_env, temperature=config.model.temperature,
        request_interval=config.model.request_interval,
        extra_body=config.model.extra_body)
