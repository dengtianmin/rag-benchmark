from __future__ import annotations

import argparse
import json
import sys

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a local OpenAI-compatible vLLM server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Llama-3.1-8B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--disable-auth", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    headers = {"Content-Type": "application/json"}
    if not args.disable_auth:
        headers["Authorization"] = f"Bearer {args.api_key}"

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "You are a concise test assistant."},
            {"role": "user", "content": "Reply with exactly: vllm_ok"},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }

    try:
        response = requests.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=args.timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"REQUEST_FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"INVALID_RESPONSE: {exc}", file=sys.stderr)
        print(response.text)
        return 2

    print(json.dumps({"content": content, "raw": data}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
