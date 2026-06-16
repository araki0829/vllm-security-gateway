"""FastAPI entrypoint for the vLLM Security Gateway."""

from fastapi import FastAPI


SERVICE_NAME = "vllm-security-gateway"

app = FastAPI(title=SERVICE_NAME)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}
