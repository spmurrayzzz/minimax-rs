#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

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
message = response["choices"][0]["message"]
actual = message["reasoning_content"]
assert actual == expected, (actual, expected)
assert message["content"] is None
assert response["choices"][0]["finish_reason"] == "length"
assert response["usage"]["prompt_tokens"] == 39, response["usage"]
print(json.dumps(response, ensure_ascii=False))
'
printf '\n%s\n' '== partial-prefix KV reuse =='
python3 - "$BASE_URL" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1]

def complete(prompt):
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps({"prompt": prompt, "max_tokens": 0, "temperature": 0}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

prefix = [777] * 39
assert complete(prefix)["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
assert complete(prefix + [778, 779])["usage"]["prompt_tokens_details"]["cached_tokens"] == 39
response = complete(prefix + [780])
assert response["usage"]["prompt_tokens_details"]["cached_tokens"] == 39, response
print(json.dumps(response))
PY
printf '\n%s\n' '== native tool call =='
curl --fail-with-body -sS \
  -X POST "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "Open README.md. You must use the read tool."}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string"}},
          "required": ["path"]
        }
      }
    }],
    "max_tokens": 128,
    "temperature": 0
  }' | python3 -c '
import json, sys
response = json.load(sys.stdin)
choice = response["choices"][0]
assert choice["finish_reason"] == "tool_calls", choice
call = choice["message"]["tool_calls"][0]
assert call["function"]["name"] == "read", call
assert json.loads(call["function"]["arguments"])["path"] == "README.md", call
print(json.dumps(response, ensure_ascii=False))
'
printf '\n'
