#!/usr/bin/env bash
set -euo pipefail

exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8080
