import json

import httpx
import pytest

from app.vllm_client import VLLMAPIError, VLLMClient, VLLMRequestError, VLLMTimeoutError


def _make_client(handler: httpx.MockTransport) -> VLLMClient:
    http_client = httpx.Client(transport=handler, base_url="http://127.0.0.1:8000/v1", timeout=1.0)
    return VLLMClient(client=http_client)


def test_list_models_returns_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "Qwen2.5-Coder-1.5B-Instruct-4bit"}]})

    client = _make_client(httpx.MockTransport(handler))

    try:
        result = client.list_models()
    finally:
        client.close()

    assert result["data"][0]["id"] == "Qwen2.5-Coder-1.5B-Instruct-4bit"


def test_chat_completions_posts_payload() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        seen_payload.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "chatcmpl-1", "choices": []})

    client = _make_client(httpx.MockTransport(handler))

    try:
        result = client.chat_completions(
            {
                "model": "Qwen2.5-Coder-1.5B-Instruct-4bit",
                "messages": [{"role": "user", "content": "こんにちは"}],
                "temperature": 0.2,
                "max_tokens": 128,
            }
        )
    finally:
        client.close()

    assert seen_payload["model"] == "Qwen2.5-Coder-1.5B-Instruct-4bit"
    assert result["id"] == "chatcmpl-1"


def test_non_success_response_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "vLLM is starting"}})

    client = _make_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(VLLMAPIError) as exc_info:
            client.list_models()
    finally:
        client.close()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "vLLM is starting"


def test_request_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _make_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(VLLMRequestError):
            client.list_models()
    finally:
        client.close()


def test_timeout_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = _make_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(VLLMTimeoutError):
            client.chat_completions({"model": "x", "messages": []})
    finally:
        client.close()
