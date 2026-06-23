"""FastAPI entrypoint for the vLLM Security Gateway."""

from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import ChatRequest
from app.vllm_client import (
    VLLMAPIError,
    VLLMClient,
    VLLMRequestError,
    VLLMTimeoutError,
    get_vllm_client,
)


SERVICE_NAME = "vllm-security-gateway"

app = FastAPI(title=SERVICE_NAME)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.post("/security/chat")
def security_chat(
    payload: ChatRequest,
    vllm_client: VLLMClient = Depends(get_vllm_client),
) -> dict[str, Any]:
    try:
        return vllm_client.chat_completions(payload.model_dump(exclude_none=True))
    except VLLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except VLLMRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except VLLMAPIError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    finally:
        vllm_client.close()
