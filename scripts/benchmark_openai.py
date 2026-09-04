#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible /v1/completions endpoint with stdlib only."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def post_json(url: str, payload: dict[str, object], timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    elapsed = time.perf_counter() - started
    return json.loads(body), elapsed


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "openai"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--label", default="openai")
    parser.add_argument("--result-dir", type=pathlib.Path)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if args.tokens < 1 or args.repeat < 1:
        parser.error("--tokens and --repeat must be positive")

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    result_dir = args.result_dir or repo_root / "results" / "raw"
    result_dir.mkdir(parents=True, exist_ok=True)
    label = safe_label(args.label)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    endpoint = args.base_url.rstrip("/") + "/completions"

    if not args.no_warmup:
        post_json(
            endpoint,
            {"model": args.model, "prompt": "warmup", "max_tokens": 8, "temperature": 0},
            args.timeout,
        )

    rates: list[float] = []
    run_records: list[dict[str, object]] = []
    for index in range(1, args.repeat + 1):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.tokens,
            "temperature": 0,
        }
        try:
            response, elapsed = post_json(endpoint, payload, args.timeout)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SystemExit(f"request {index} failed: {exc}") from exc

        if "error" in response:
            raise SystemExit(f"request {index} returned an error: {response['error']}")

        usage = response.get("usage", {})
        generated = int(usage.get("completion_tokens", 0))
        rate = generated / elapsed if elapsed else 0.0
        rates.append(rate)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "run": index,
            "model": args.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": generated,
            "elapsed_seconds": elapsed,
            "generated_tokens_per_second_wall": rate,
        }
        run_records.append(record)
        (result_dir / f"{label}-{index}.response.json").write_text(
            json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"{label} run {index}: {rate:.2f} tok/s ({generated} tokens, {elapsed:.2f}s)")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "model": args.model,
        "repeat": args.repeat,
        "tokens_requested": args.tokens,
        "mean_tokens_per_second": statistics.mean(rates),
        "median_tokens_per_second": statistics.median(rates),
        "min_tokens_per_second": min(rates),
        "max_tokens_per_second": max(rates),
        "runs": run_records,
    }
    with (result_dir / "openai-runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(
        f"summary: mean {summary['mean_tokens_per_second']:.2f}, "
        f"median {summary['median_tokens_per_second']:.2f} tok/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
