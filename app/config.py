"""Application settings for the vLLM Security Gateway."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    vllm_base_url: str = Field(default="http://127.0.0.1:8000/v1", alias="VLLM_BASE_URL")
    gateway_api_key: str | None = Field(default=None, alias="GATEWAY_API_KEY")
    max_messages: int = Field(default=20, alias="MAX_MESSAGES")
    max_message_chars: int = Field(default=8000, alias="MAX_MESSAGE_CHARS")
    max_total_chars: int = Field(default=16000, alias="MAX_TOTAL_CHARS")
    max_tokens_limit: int = Field(default=1024, alias="MAX_TOKENS_LIMIT")
    max_n: int = Field(default=1, alias="MAX_N")
    temperature_min: float = Field(default=0.0, alias="TEMPERATURE_MIN")
    temperature_max: float = Field(default=2.0, alias="TEMPERATURE_MAX")
    request_timeout_seconds: int = Field(default=120, alias="REQUEST_TIMEOUT_SECONDS")
    log_full_prompts: bool = Field(default=False, alias="LOG_FULL_PROMPTS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
