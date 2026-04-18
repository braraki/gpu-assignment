#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPTS = [
    "In one sentence, explain what a GPU kernel launch does.",
    "In one sentence, explain why batching can improve throughput.",
    "In one sentence, explain what Nsight Systems is used for.",
    "In one sentence, explain what concurrency means for an inference server.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send exactly one concurrent round of chat requests."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def wait_for_server(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(1)
            continue
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for server at {health_url}")


def build_payload(model: str, prompt: str, max_tokens: int) -> bytes:
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    return json.dumps(body).encode("utf-8")


def send_request(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout_seconds: int
) -> dict:
    payload = build_payload(model, prompt, max_tokens)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.time()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw_body = response.read()
    finished_at = time.time()
    decoded_body = json.loads(raw_body.decode("utf-8"))
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_seconds": finished_at - started_at,
        "prompt": prompt,
        "response": decoded_body,
    }


def main() -> int:
    args = parse_args()

    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    wait_for_server(args.base_url, args.timeout_seconds)

    prompts: list[str] = []
    while len(prompts) < args.concurrency:
        prompts.extend(PROMPTS)
    prompts = prompts[:args.concurrency]

    run_started_at = time.time()
    results: list[dict] = [None] * args.concurrency  # type: ignore[list-item]

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_index = {
            executor.submit(
                send_request,
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout_seconds,
            ): index
            for index, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    run_finished_at = time.time()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "run_started_at": run_started_at,
        "run_finished_at": run_finished_at,
        "wall_clock_seconds": run_finished_at - run_started_at,
        "requests": results,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latencies = [item["latency_seconds"] for item in results]
    print(f"Server ready at {args.base_url}")
    print(f"Sent {args.concurrency} requests in one round")
    print(f"Wall clock seconds: {summary['wall_clock_seconds']:.3f}")
    print(f"Min latency seconds: {min(latencies):.3f}")
    print(f"Max latency seconds: {max(latencies):.3f}")
    print(f"Saved load artifact: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"single_round_chat_load.py failed: {exc}", file=sys.stderr)
        raise
