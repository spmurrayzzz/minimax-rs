#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"

printf '%s\n' '== health =='
curl --fail-with-body -sS "$BASE_URL/health"
printf '\n%s\n' '== models =='
curl --fail-with-body -sS "$BASE_URL/v1/models"
printf '\n%s\n' '== text completion =='
curl --fail-with-body -sS \
  -X POST "$BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M2.7",
    "prompt": "Write a one-sentence hello.",
    "max_tokens": 16,
    "temperature": 0
  }'
printf '\n%s\n' '== token-id completion =='
curl --fail-with-body -sS \
  -X POST "$BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M2.7",
    "prompt": [1],
    "max_tokens": 1,
    "temperature": 0
  }'
printf '\n%s\n' '== llama.cpp greedy golden =='
curl --fail-with-body -sS \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "test"}],
    "max_tokens": 24,
    "temperature": 0
  }' | python3 -c '
import json, sys
response = json.load(sys.stdin)
expected = "The user says \"test\". This is a simple test message. The assistant should respond appropriately. There\x27s no policy violation."
actual = response["choices"][0]["message"]["content"]
assert actual == expected, (actual, expected)
assert response["usage"]["prompt_tokens"] == 39, response["usage"]
print(json.dumps(response, ensure_ascii=False))
'
printf '\n'
