#!/usr/bin/env python3
"""Measure uncached streaming TTFT and decode event throughput."""

import argparse
import json
import time
import urllib.request


def stream_request(base_url: str, prompt: list[int], max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_content = None
    content_events = []
    usage = None
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            now = time.perf_counter()
            if "usage" in event:
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                if first_content is None:
                    first_content = now
                content_events.append(now)
    end = time.perf_counter()
    span = content_events[-1] - content_events[0] if len(content_events) > 1 else 0.0
    return {
        "prompt_tokens": len(prompt),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "ttft_s": None if first_content is None else first_content - start,
        "total_s": end - start,
        "content_events": len(content_events),
        "event_span_s": span,
        "event_rate": (len(content_events) - 1) / span if span else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--length", type=int, default=39)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--token", type=int, default=1234)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.length < 1:
        parser.error("--length must be positive")

    for run in range(args.repeats):
        # Change the final id so a previous exact-prefix cache cannot apply.
        prompt = [args.token] * args.length
        prompt[-1] = args.token + run
        print(
            json.dumps(stream_request(args.base_url, prompt, args.max_tokens)),
            flush=True,
        )


if __name__ == "__main__":
    main()
