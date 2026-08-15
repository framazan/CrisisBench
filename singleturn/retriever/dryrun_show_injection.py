"""Dry-run demo for retrieval snippet injection.

This script is standalone and does not call APIs.
"""
from __future__ import annotations

import json
from typing import Any


def inject_retrieved_snippets_into_messages(request_json: dict[str, Any]) -> None:
    snippets = request_json.get("retrieved_snippets")
    messages = request_json.get("messages")

    if not snippets or not isinstance(messages, list):
        return

    lines = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = str(snippet.get("text", "")).strip()
        else:
            text = str(snippet).strip()

        if text:
            lines.append(f"- {text[:500]}")

    if not lines:
        return

    retrieval_message = {
        "role": "system",
        "content": "Relevant examples from prior high-quality conversations:\n" + "\n".join(lines),
    }

    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages.insert(1, retrieval_message)
    else:
        messages.insert(0, retrieval_message)


def main() -> None:
    request_json = {
        "messages": [
            {"role": "system", "content": "You are an expert evaluator."},
            {"role": "user", "content": "Assess this counselor response."},
        ],
        "retrieved_snippets": [
            {"text": "Helpful example: reflect feelings and ask an open question."},
            {"text": "High-quality example: validate client's emotion and offer support."},
        ],
    }

    print("Before injection:")
    print(json.dumps(request_json["messages"], indent=2))

    inject_retrieved_snippets_into_messages(request_json)

    print("\nAfter injection:")
    print(json.dumps(request_json["messages"], indent=2))


if __name__ == "__main__":
    main()
