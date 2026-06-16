"""HTTP client for the vLLM OpenAI-compatible API."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings


class VLLMClientError(RuntimeError):
    """Base error for vLLM client failures."""


class VLLMRequestError(VLLMClientError):
    """Raised when the request cannot be sent to vLLM."""


class VLLMTimeoutError(VLLMClientError):
    """Raised when vLLM does not respond before the timeout."""


class VLLMAPIError(VLLMClientError):
    """Raised when vLLM returns a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"vLLM returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class VLLMClient:
    """Small synchronous client for vLLM endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.vllm_base_url
        self.timeout = timeout if timeout is not None else settings.request_timeout_seconds
        self._client = client or httpx.Client(base_url=self.base_url, timeout=self.timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VLLMClient:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/models")

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/chat/completions", json=payload)

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise VLLMTimeoutError("vLLM request timed out") from exc
        except httpx.RequestError as exc:
            raise VLLMRequestError(f"Unable to reach vLLM: {exc}") from exc

        if response.is_error:
            raise VLLMAPIError(response.status_code, self._extract_error_detail(response))

        return response.json()

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text or "unknown error"

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message:
                    return message
            detail = body.get("detail")
            if isinstance(detail, str) and detail:
                return detail

        return response.text or "unknown error"


def get_vllm_client() -> VLLMClient:
    """Create a client using the current application settings."""

    return VLLMClient()
