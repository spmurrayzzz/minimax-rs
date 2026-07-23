#!/usr/bin/env python3
"""Capture and compare deterministic teacher-forced perplexity results.

The server scores ln p(token[i] | token[:i]) for every token after the first.
Capture a pipeline baseline, restart in tensor mode, then use the baseline
artifact as the reference for a paired token-level comparison.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
import urllib.error
import urllib.request
from typing import Any

SCHEMA_VERSION = 1
ENDPOINT = "/v1/perplexity"


def load_prompt(path: pathlib.Path, token_ids: bool) -> tuple[str | list[int], dict[str, Any]]:
    raw = path.read_bytes()
    source = {
        "path": str(path.resolve()),
        "format": "token_ids" if token_ids else "text",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    if not token_ids:
        try:
            return raw.decode("utf-8"), source
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not valid UTF-8: {error}") from error

    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise ValueError("--token-ids input must be a JSON array of integers")
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("token IDs must be unsigned 32-bit integers")
    return values, source


def post_json(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"perplexity endpoint returned HTTP {error.code}: {body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"perplexity request failed: {error}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"perplexity endpoint returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("perplexity endpoint response is not a JSON object")
    return value


def metrics(logprobs: list[float]) -> dict[str, float | int]:
    if not logprobs:
        raise ValueError("response contains no scored token log probabilities")
    if any(not math.isfinite(value) for value in logprobs):
        raise ValueError("response contains a non-finite token log probability")
    # A tiny positive value can result from floating-point roundoff. Anything
    # materially above zero is not a valid natural log probability.
    if any(value > 1e-5 for value in logprobs):
        raise ValueError("response contains a positive token log probability")
    nll = math.fsum(-value for value in logprobs)
    mean_nll = nll / len(logprobs)
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as error:
        raise ValueError("perplexity overflowed") from error
    if not math.isfinite(perplexity):
        raise ValueError("perplexity is not finite")
    return {
        "scored_tokens": len(logprobs),
        "negative_log_likelihood": nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": perplexity,
    }


def validate_response(response: dict[str, Any]) -> dict[str, float | int]:
    token_ids = response.get("token_ids")
    logprobs = response.get("token_logprobs")
    if not isinstance(token_ids, list) or any(type(value) is not int for value in token_ids):
        raise ValueError("response token_ids must be an integer array")
    if not isinstance(logprobs, list) or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) for value in logprobs
    ):
        raise ValueError("response token_logprobs must be a numeric array")
    logprobs = [float(value) for value in logprobs]
    if len(token_ids) < 2 or len(logprobs) != len(token_ids) - 1:
        raise ValueError(
            f"response has {len(token_ids)} token IDs and {len(logprobs)} log probabilities"
        )
    result = metrics(logprobs)
    for name in (
        "negative_log_likelihood",
        "mean_negative_log_likelihood",
        "perplexity",
    ):
        reported = response.get(name)
        if not isinstance(reported, (int, float)) or not math.isclose(
            float(reported), float(result[name]), rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError(
                f"server {name}={reported!r} disagrees with recomputed {result[name]}"
            )
    if response.get("scored_tokens") != len(logprobs):
        raise ValueError("server scored_tokens disagrees with token_logprobs")
    return result


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compare(reference: dict[str, Any], candidate: dict[str, Any], worst_count: int) -> dict[str, Any]:
    reference_response = reference.get("response")
    candidate_response = candidate.get("response")
    if not isinstance(reference_response, dict) or not isinstance(candidate_response, dict):
        raise ValueError("comparison artifacts must contain response objects")
    reference_ids = reference_response.get("token_ids")
    candidate_ids = candidate_response.get("token_ids")
    if reference_ids != candidate_ids:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference_ids or [], candidate_ids or []))
                if left != right
            ),
            min(len(reference_ids or []), len(candidate_ids or [])),
        )
        raise ValueError(f"reference and candidate token IDs first differ at position {mismatch}")

    reference_logprobs = [float(value) for value in reference_response["token_logprobs"]]
    candidate_logprobs = [float(value) for value in candidate_response["token_logprobs"]]
    if len(reference_logprobs) != len(candidate_logprobs):
        raise ValueError("reference and candidate scored-token counts differ")

    # delta_nll > 0 means the candidate assigned less probability to the target.
    delta_nll = [
        reference_value - candidate_value
        for reference_value, candidate_value in zip(reference_logprobs, candidate_logprobs)
    ]
    absolute = [abs(value) for value in delta_nll]
    reference_ppl = float(reference["metrics"]["perplexity"])
    candidate_ppl = float(candidate["metrics"]["perplexity"])
    ranked = sorted(range(len(absolute)), key=absolute.__getitem__, reverse=True)[:worst_count]
    worst = [
        {
            "position": index + 1,
            "token_id": candidate_ids[index + 1],
            "reference_logprob": reference_logprobs[index],
            "candidate_logprob": candidate_logprobs[index],
            "delta_nll": delta_nll[index],
            "absolute_delta_nll": absolute[index],
        }
        for index in ranked
    ]
    return {
        "relative_perplexity_delta": candidate_ppl / reference_ppl - 1.0,
        "mean_token_nll_delta": math.fsum(delta_nll) / len(delta_nll),
        "mean_absolute_token_nll_delta": math.fsum(absolute) / len(absolute),
        "p50_absolute_token_nll_delta": percentile(absolute, 0.50),
        "p95_absolute_token_nll_delta": percentile(absolute, 0.95),
        "p99_absolute_token_nll_delta": percentile(absolute, 0.99),
        "max_absolute_token_nll_delta": max(absolute),
        "worst_tokens": worst,
    }


def load_artifact(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read reference artifact {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path} is not a perplexity artifact schema {SCHEMA_VERSION}")
    if not isinstance(value.get("metrics"), dict):
        raise ValueError(f"{path} has no metrics object")
    validate_response(value.get("response", {}))
    return value


def write_artifact(path: pathlib.Path, artifact: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(artifact, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    os.replace(temporary, path)


def print_capture(artifact: dict[str, Any]) -> None:
    response = artifact["response"]
    result = artifact["metrics"]
    elapsed_ms = float(response.get("elapsed_ms", 0.0))
    rate = len(response["token_ids"]) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    print(
        f"candidate mode={response.get('execution_mode', 'unknown')} "
        f"tokens={len(response['token_ids'])} scored={result['scored_tokens']} "
        f"mean_nll={result['mean_negative_log_likelihood']:.9f} "
        f"ppl={result['perplexity']:.9f} elapsed_ms={elapsed_ms:.2f} "
        f"evaluated_tokens_per_second={rate:.2f}"
    )


def nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("threshold must be a finite non-negative number")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and compare MiniMax teacher-forced perplexity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("corpus", type=pathlib.Path, help="UTF-8 text or JSON token-ID corpus")
    parser.add_argument(
        "--token-ids",
        action="store_true",
        help="interpret the corpus as a JSON array of token IDs instead of UTF-8 text",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="MiniMax-M2.7")
    parser.add_argument("--timeout", type=nonnegative, default=7200.0)
    parser.add_argument("--output", type=pathlib.Path, help="write the captured JSON artifact")
    parser.add_argument("--reference", type=pathlib.Path, help="compare with a prior artifact")
    parser.add_argument(
        "--max-relative-ppl-delta",
        type=nonnegative,
        help="fail when abs(candidate/reference - 1) exceeds this fraction",
    )
    parser.add_argument(
        "--max-mean-abs-token-nll-delta",
        type=nonnegative,
        help="fail when mean absolute paired token NLL delta exceeds this value",
    )
    parser.add_argument(
        "--max-p99-abs-token-nll-delta",
        type=nonnegative,
        help="fail when p99 absolute paired token NLL delta exceeds this value",
    )
    parser.add_argument("--worst-tokens", type=int, default=10)
    args = parser.parse_args()
    thresholds = (
        args.max_relative_ppl_delta,
        args.max_mean_abs_token_nll_delta,
        args.max_p99_abs_token_nll_delta,
    )
    if any(value is not None for value in thresholds) and args.reference is None:
        parser.error("comparison thresholds require --reference")
    if args.worst_tokens < 0:
        parser.error("--worst-tokens must be non-negative")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        prompt, source = load_prompt(args.corpus, args.token_ids)
        response = post_json(
            args.base_url,
            {"model": args.model, "prompt": prompt},
            args.timeout,
        )
        result = validate_response(response)
        artifact: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": source,
            "base_url": args.base_url.rstrip("/"),
            "response": response,
            "metrics": result,
        }
        if args.output is not None:
            write_artifact(args.output, artifact)
        print_capture(artifact)

        if args.reference is None:
            if args.output is not None:
                print(f"wrote {args.output}")
            return 0

        reference = load_artifact(args.reference)
        if reference.get("source", {}).get("sha256") != source["sha256"]:
            print("warning: source hashes differ; token IDs will determine comparability", file=sys.stderr)
        comparison = compare(reference, artifact, args.worst_tokens)
        artifact["reference"] = {
            "path": str(args.reference.resolve()),
            "source_sha256": reference.get("source", {}).get("sha256"),
            "execution_mode": reference["response"].get("execution_mode"),
            "perplexity": reference["metrics"].get("perplexity"),
        }
        artifact["comparison"] = comparison
        if args.output is not None:
            write_artifact(args.output, artifact)
            print(f"wrote comparison artifact {args.output}")
        reference_response = reference["response"]
        reference_metrics = reference["metrics"]
        print(
            f"reference mode={reference_response.get('execution_mode', 'unknown')} "
            f"mean_nll={reference_metrics['mean_negative_log_likelihood']:.9f} "
            f"ppl={reference_metrics['perplexity']:.9f}"
        )
        print(
            f"delta relative_ppl={comparison['relative_perplexity_delta']:+.6%} "
            f"mean_nll={comparison['mean_token_nll_delta']:+.9f} "
            f"mean_abs_token_nll={comparison['mean_absolute_token_nll_delta']:.9f} "
            f"p50_abs_token_nll={comparison['p50_absolute_token_nll_delta']:.9f} "
            f"p95_abs_token_nll={comparison['p95_absolute_token_nll_delta']:.9f} "
            f"p99_abs_token_nll={comparison['p99_absolute_token_nll_delta']:.9f} "
            f"max_abs_token_nll={comparison['max_absolute_token_nll_delta']:.9f}"
        )
        for worst in comparison["worst_tokens"]:
            print(
                f"worst position={worst['position']} token={worst['token_id']} "
                f"delta_nll={worst['delta_nll']:+.9f} "
                f"abs={worst['absolute_delta_nll']:.9f}"
            )

        failures = []
        if (
            args.max_relative_ppl_delta is not None
            and abs(comparison["relative_perplexity_delta"]) > args.max_relative_ppl_delta
        ):
            failures.append(
                f"absolute relative PPL delta {abs(comparison['relative_perplexity_delta']):.9g} "
                f"> {args.max_relative_ppl_delta:.9g}"
            )
        if (
            args.max_mean_abs_token_nll_delta is not None
            and comparison["mean_absolute_token_nll_delta"]
            > args.max_mean_abs_token_nll_delta
        ):
            failures.append(
                f"mean absolute token NLL delta "
                f"{comparison['mean_absolute_token_nll_delta']:.9g} "
                f"> {args.max_mean_abs_token_nll_delta:.9g}"
            )
        if (
            args.max_p99_abs_token_nll_delta is not None
            and comparison["p99_absolute_token_nll_delta"]
            > args.max_p99_abs_token_nll_delta
        ):
            failures.append(
                f"p99 absolute token NLL delta "
                f"{comparison['p99_absolute_token_nll_delta']:.9g} "
                f"> {args.max_p99_abs_token_nll_delta:.9g}"
            )
        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1
        if any(
            value is not None
            for value in (
                args.max_relative_ppl_delta,
                args.max_mean_abs_token_nll_delta,
                args.max_p99_abs_token_nll_delta,
            )
        ):
            print("PASS: all configured perplexity gates passed")
        else:
            print("comparison complete; no pass/fail thresholds were configured")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
