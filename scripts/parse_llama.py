#!/usr/bin/env python3
"""Parse one llama-server completion response and optionally append JSONL."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime, timezone


ACCEPTANCE_RE = re.compile(
    r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), "
    r"mean len = ([\d.]+)"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("response", type=pathlib.Path)
    parser.add_argument("server_log", type=pathlib.Path)
    parser.add_argument("--jsonl", type=pathlib.Path)
    args = parser.parse_args()

    try:
        payload = json.loads(args.response.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{args.label}: response parse failed: {exc}", file=sys.stderr)
        return 1

    if "error" in payload:
        print(f"{args.label}: server error: {payload['error']}", file=sys.stderr)
        return 1

    timings = payload.get("timings", {})
    result: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "generated_tokens": timings.get("predicted_n"),
        "generated_tokens_per_second": timings.get("predicted_per_second"),
    }

    try:
        log_text = args.server_log.read_text(encoding="utf-8", errors="replace")
        matches = ACCEPTANCE_RE.findall(log_text)
    except OSError:
        matches = []

    if matches:
        acceptance, accepted, drafted, mean_length = matches[-1]
        result.update(
            {
                "draft_acceptance": float(acceptance),
                "draft_accepted": int(accepted),
                "draft_generated": int(drafted),
                "draft_mean_length": float(mean_length),
            }
        )

    pp = result["prompt_tokens_per_second"]
    tg = result["generated_tokens_per_second"]
    n = result["generated_tokens"]
    acceptance_note = ""
    if "draft_acceptance" in result:
        acceptance_note = (
            f" | acceptance {result['draft_acceptance']:.3f}"
            f" mean_len {result['draft_mean_length']:.2f}"
        )
    print(
        f"{args.label}: pp {float(pp or 0):.1f} tok/s | "
        f"tg {float(tg or 0):.2f} tok/s (n={n}){acceptance_note}"
    )

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
