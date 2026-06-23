import json

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.vllm_client import VLLMClient, get_vllm_client


def _override_client(handler: httpx.MockTransport) -> VLLMClient:
    http_client = httpx.Client(transport=handler, base_url="http://127.0.0.1:8000/v1", timeout=1.0)
    return VLLMClient(client=http_client)


def test_security_chat_proxies_to_vllm() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "こんにちは"}}],
            },
        )

    app.dependency_overrides[get_vllm_client] = lambda: _override_client(httpx.MockTransport(handler))
    client = TestClient(app)

    try:
        response = client.post(
            "/security/chat",
            json={
                "model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                "messages": [{"role": "user", "content": "こんにちは"}],
                "temperature": 0.2,
                "max_tokens": 128,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-test"
    assert seen_payload["model"] == "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"


def test_security_chat_returns_bad_gateway_when_vllm_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app.dependency_overrides[get_vllm_client] = lambda: _override_client(httpx.MockTransport(handler))
    client = TestClient(app)

    try:
        response = client.post(
            "/security/chat",
            json={
                "model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
                "messages": [{"role": "user", "content": "こんにちは"}],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
