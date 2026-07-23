#!/usr/bin/env python3
"""Exercise the full server API, cache, streaming, stop, and stress matrix."""

import argparse
import json
import math
import time
import urllib.request

GOLDEN_REASONING = (
    'The user says "test". This is a simple test message. The assistant should '
    "respond appropriately. There's no policy violation."
)
EOG_USER_PROMPT = "Answer with exactly OK."


class Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base_url + path, timeout=30) as response:
            return json.load(response)

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.load(response)

    def stream(self, path: str, payload: dict) -> list[dict]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps({**payload, "stream": True}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        events = []
        saw_done = False
        with urllib.request.urlopen(request, timeout=900) as response:
            assert response.headers.get_content_type() == "text/event-stream"
            for raw_line in response:
                line = raw_line.decode().rstrip("\r\n")
                if not line:
                    continue
                assert line.startswith("data: "), line
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                else:
                    events.append(json.loads(data))
        assert saw_done, "stream ended without [DONE]"
        return events


def cached_tokens(response: dict) -> int:
    return response["usage"]["prompt_tokens_details"]["cached_tokens"]


def reset_with(client: Client, token: int) -> None:
    client.post(
        "/v1/completions",
        {"prompt": [token], "max_tokens": 0, "temperature": 0},
    )


def validate_fresh_prefill(client: Client) -> dict[int, float]:
    timings = {}
    for index, length in enumerate((1, 5, 39, 512, 513)):
        prompt = [1100 + index] + [777] * (length - 1)
        started = time.monotonic()
        response = client.post(
            "/v1/completions",
            {"prompt": prompt, "max_tokens": 1, "temperature": 0},
        )
        timings[length] = time.monotonic() - started
        usage = response["usage"]
        assert usage["prompt_tokens"] == length, response
        assert cached_tokens(response) == 0, response
        assert usage["completion_tokens"] == 1, response
    print("fresh 1/5/39/512/513-token prefills: ok")
    return timings


def validate_cache_rewind(client: Client) -> None:
    prefix = [2100] + [777] * 38
    base = client.post(
        "/v1/completions",
        {"prompt": prefix, "max_tokens": 0, "temperature": 0},
    )
    assert cached_tokens(base) == 0, base
    follow = client.post(
        "/v1/completions",
        {"prompt": prefix + [778, 779], "max_tokens": 0, "temperature": 0},
    )
    assert cached_tokens(follow) == 39, follow
    diverged = client.post(
        "/v1/completions",
        {"prompt": prefix + [780], "max_tokens": 0, "temperature": 0},
    )
    assert cached_tokens(diverged) == 39, diverged
    historical = client.post(
        "/v1/completions",
        {"prompt": prefix, "max_tokens": 0, "temperature": 0},
    )
    assert cached_tokens(historical) == 38, historical
    exact = client.post(
        "/v1/completions",
        {"prompt": prefix, "max_tokens": 0, "temperature": 0},
    )
    assert cached_tokens(exact) == 39, exact
    print("cache reset/reuse/divergence/historical-logit rewind: ok")


def validate_stop_and_eog(client: Client) -> None:
    stop_response = client.post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 8,
            "temperature": 0,
            "stop": "The",
        },
    )
    assert stop_response["choices"][0]["finish_reason"] == "stop", stop_response
    assert stop_response["usage"]["completion_tokens"] == 1, stop_response
    stop_follow = client.post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 0,
            "temperature": 0,
        },
    )
    assert stop_follow["usage"]["prompt_tokens"] == 39, stop_follow
    assert cached_tokens(stop_follow) == 39, stop_follow

    completed = client.post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": EOG_USER_PROMPT}],
            "max_tokens": 512,
            "temperature": 0,
        },
    )
    assert completed["choices"][0]["finish_reason"] == "stop", completed
    message = completed["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    assert reasoning and content, completed
    native_prefix = (
        "]~!b[]~b]system\n"
        "You are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax."
        f"[e~[\n]~b]user\n{EOG_USER_PROMPT}[e~[\n]~b]ai\n<think>\n"
    )
    through_last_token = native_prefix + reasoning + "\n</think>\n\n" + content
    eog_response = client.post(
        "/v1/completions",
        {"prompt": through_last_token, "max_tokens": 1, "temperature": 0},
    )
    assert eog_response["choices"][0]["finish_reason"] == "stop", eog_response
    assert eog_response["usage"]["completion_tokens"] == 0, eog_response
    eog_follow = client.post(
        "/v1/completions",
        {"prompt": through_last_token, "max_tokens": 0, "temperature": 0},
    )
    assert eog_follow["usage"]["prompt_tokens"] == eog_response["usage"]["prompt_tokens"]
    assert cached_tokens(eog_follow) == eog_response["usage"]["prompt_tokens"], eog_follow
    print("stop- and EOG-selected tokens remain outside KV: ok")


def validate_sampling_and_golden(client: Client) -> None:
    deterministic = client.post(
        "/v1/completions",
        {"prompt": "Hello", "max_tokens": 1, "temperature": 0},
    )
    assert deterministic["usage"]["completion_tokens"] == 1, deterministic
    stochastic = client.post(
        "/v1/completions",
        {
            "prompt": [3100, 777, 778, 779, 780],
            "max_tokens": 8,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
        },
    )
    assert stochastic["usage"]["prompt_tokens"] == 5, stochastic
    assert 0 <= stochastic["usage"]["completion_tokens"] <= 8, stochastic

    golden = client.post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 24,
            "temperature": 0,
        },
    )
    message = golden["choices"][0]["message"]
    assert message["reasoning_content"] == GOLDEN_REASONING, golden
    assert message["content"] is None, golden
    assert golden["usage"]["prompt_tokens"] == 39, golden
    assert golden["usage"]["completion_tokens"] == 24, golden
    print("deterministic, stochastic, and llama.cpp golden sampling: ok")


def validate_perplexity(client: Client) -> None:
    prompt = [6100, 777, 778, 779]
    response = client.post("/v1/perplexity", {"prompt": prompt})
    assert response["object"] == "perplexity", response
    assert response["execution_mode"] in {"pipeline", "tensor"}, response
    assert response["token_ids"] == prompt, response
    logprobs = response["token_logprobs"]
    assert len(logprobs) == len(prompt) - 1, response
    assert all(math.isfinite(value) and value <= 1e-5 for value in logprobs), response
    nll = math.fsum(-value for value in logprobs)
    mean_nll = nll / len(logprobs)
    assert math.isclose(response["negative_log_likelihood"], nll, rel_tol=1e-6), response
    assert math.isclose(
        response["mean_negative_log_likelihood"], mean_nll, rel_tol=1e-6
    ), response
    assert math.isclose(response["perplexity"], math.exp(mean_nll), rel_tol=1e-6), response
    assert response["scored_tokens"] == len(logprobs), response
    print("teacher-forced perplexity scoring: ok")


def validate_streaming(client: Client) -> None:
    completion_payload = {
        "prompt": [5100, 777, 778, 779, 780],
        "max_tokens": 24,
        "temperature": 0,
    }
    expected = client.post("/v1/completions", completion_payload)
    # Give both forms the same fresh-prefill execution shape. Rewinding a token
    # evaluates it through decode rather than the original batched prefill path.
    reset_with(client, 5199)
    events = client.stream("/v1/completions", completion_payload)
    text = "".join(
        event["choices"][0].get("text", "")
        for event in events
        if event.get("choices")
    )
    finishes = [
        event["choices"][0]["finish_reason"]
        for event in events
        if event.get("choices")
        and event["choices"][0].get("finish_reason") is not None
    ]
    assert text == expected["choices"][0]["text"], (text, expected)
    assert finishes == [expected["choices"][0]["finish_reason"]], events

    reset_with(client, 5299)
    chat_events = client.stream(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 24,
            "temperature": 0,
        },
    )
    reasoning = ""
    content = ""
    chat_finishes = []
    for event in chat_events:
        if not event.get("choices"):
            continue
        choice = event["choices"][0]
        delta = choice.get("delta", {})
        reasoning += delta.get("reasoning_content") or ""
        content += delta.get("content") or ""
        if choice.get("finish_reason") is not None:
            chat_finishes.append(choice["finish_reason"])
    assert reasoning == GOLDEN_REASONING, (reasoning, chat_events)
    assert content == "", (content, chat_events)
    assert chat_finishes == ["length"], chat_finishes

    tool_events = client.stream(
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Open README.md. You must use the read tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        },
    )
    calls = {}
    tool_finishes = []
    for event in tool_events:
        if not event.get("choices"):
            continue
        choice = event["choices"][0]
        for call in choice.get("delta", {}).get("tool_calls", []):
            state = calls.setdefault(
                call["index"],
                {"id": "", "type": "", "name": "", "arguments": ""},
            )
            state["id"] += call.get("id") or ""
            state["type"] += call.get("type") or ""
            function = call.get("function") or {}
            state["name"] += function.get("name") or ""
            state["arguments"] += function.get("arguments") or ""
        if choice.get("finish_reason") is not None:
            tool_finishes.append(choice["finish_reason"])
    assert list(calls) == [0], calls
    call = calls[0]
    assert call["id"].startswith("call_"), call
    assert call["type"] == "function", call
    assert call["name"] == "read", call
    assert json.loads(call["arguments"]) == {"path": "README.md"}, call
    assert tool_finishes == ["tool_calls"], tool_finishes
    print("completion, reasoning, and tool-call SSE reconstruction: ok")


def stress(client: Client, cycles: int) -> None:
    if cycles == 0:
        return
    lengths = (1, 5, 39, 512, 513)
    prompts = {
        length: [4200 + index] + [777] * (length - 1)
        for index, length in enumerate(lengths)
    }
    baselines = {}
    for length in lengths:
        response = client.post(
            "/v1/completions",
            {"prompt": prompts[length], "max_tokens": 16, "temperature": 0},
        )
        assert cached_tokens(response) == 0, response
        baselines[length] = (
            response["choices"][0]["text"],
            response["choices"][0]["finish_reason"],
            response["usage"]["completion_tokens"],
        )
    for cycle in range(cycles):
        for length in lengths:
            response = client.post(
                "/v1/completions",
                {"prompt": prompts[length], "max_tokens": 16, "temperature": 0},
            )
            actual = (
                response["choices"][0]["text"],
                response["choices"][0]["finish_reason"],
                response["usage"]["completion_tokens"],
            )
            assert actual == baselines[length], (cycle, length, actual, baselines[length])
            assert cached_tokens(response) == 0, (cycle, length, response)
        assert client.get("/health") == {"status": "ok"}
        print(f"stress cycle {cycle + 1}/{cycles}: ok", flush=True)
    print(f"stress: {cycles * len(lengths)} deterministic requests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--stress-cycles",
        type=int,
        default=0,
        help="repeat the five prompt lengths this many times",
    )
    args = parser.parse_args()
    if args.stress_cycles < 0:
        parser.error("--stress-cycles must be non-negative")

    client = Client(args.base_url)
    assert client.get("/health") == {"status": "ok"}
    models = client.get("/v1/models")
    assert models["data"][0]["ready"] is True, models
    timings = validate_fresh_prefill(client)
    validate_cache_rewind(client)
    validate_stop_and_eog(client)
    validate_sampling_and_golden(client)
    validate_perplexity(client)
    validate_streaming(client)
    stress(client, args.stress_cycles)
    print(
        json.dumps(
            {
                "status": "ok",
                "fresh_request_seconds": timings,
                "stress_cycles": args.stress_cycles,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
