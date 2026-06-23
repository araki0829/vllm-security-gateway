#!/usr/bin/env bash
set -euo pipefail

curl http://127.0.0.1:8080/security/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": "こんにちは。短く自己紹介して。"
      }
    ],
    "temperature": 0.2,
    "max_tokens": 128
  }'
