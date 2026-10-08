import json

import httpx
import pytest

from verdict.config import ModelCfg, ProviderCfg
from verdict.providers import ProviderError
from verdict.providers.openai_compat import OpenAICompatProvider

CFG = ProviderCfg(name="nvidia", type="openai_compatible", base_url="https://fake.test/v1", api_key_env="FAKE_KEY", max_retries=2)
MODEL = ModelCfg(id="kimi", provider="nvidia", model="moonshotai/kimi-k3", family="kimi", temperature=1.0,
                 max_tokens=99, max_tokens_param="max_completion_tokens", extra_body={"foo": 1})


def _provider(handler, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "nvapi-test")
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = OpenAICompatProvider(CFG)
    p._client = httpx.Client(base_url=CFG.base_url, transport=httpx.MockTransport(handler))
    return p


def ok(text="hello"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text, "reasoning_content": "thinking..."},
                                                  "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 11, "completion_tokens": 7}})


def test_request_shape_and_retry_on_429(monkeypatch):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer nvapi-test"
        return httpx.Response(429, headers={"retry-after": "0"}) if len(seen) == 1 else ok()

    r = _provider(handler, monkeypatch).complete(MODEL, [{"role": "user", "content": "hi"}])
    assert r.text == "hello" and (r.input_tokens, r.output_tokens) == (11, 7)
    body = seen[-1]
    assert body["model"] == "moonshotai/kimi-k3" and body["max_completion_tokens"] == 99 and "max_tokens" not in body
    assert body["foo"] == 1 and body["temperature"] == 1.0 and len(seen) == 2


def test_non_retryable_and_exhausted_retries(monkeypatch):
    calls = []

    def bad_key(req):
        calls.append(1)
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(ProviderError, match="401"):
        _provider(bad_key, monkeypatch).complete(MODEL, [{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    with pytest.raises(ProviderError, match="503"):
        _provider(lambda r: httpx.Response(503, text="busy"), monkeypatch).complete(MODEL, [{"role": "user", "content": "x"}])


def test_empty_content_from_reasoning_model(monkeypatch):
    with pytest.raises(ProviderError, match="raise max_tokens"):
        _provider(lambda r: ok(text=""), monkeypatch).complete(MODEL, [{"role": "user", "content": "x"}])


def test_missing_key(monkeypatch):
    monkeypatch.delenv("FAKE_KEY", raising=False)
    with pytest.raises(ProviderError, match="FAKE_KEY is not set"):
        OpenAICompatProvider(CFG).complete(MODEL, [{"role": "user", "content": "x"}])
