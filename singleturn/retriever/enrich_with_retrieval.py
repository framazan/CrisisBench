"""Add retrieved_snippets to prompted JSONL records.

Example:
    PYTHONPATH=/Users/saketganti/vf_copilot python3 singleturn/retriever/enrich_with_retrieval.py \
      -i data/prompted/counselor_eval_20250604.jsonl \
      -o /tmp/counselor_eval_enriched.jsonl \
      -k 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multiturn.static_evals.retriever.mock_retriever import MockRetriever


def _record_to_query(record: dict[str, Any]) -> str:
    if isinstance(record.get("body"), str) and record["body"]:
        return record["body"]
    if isinstance(record.get("prompt"), str) and record["prompt"]:
        return record["prompt"]
    messages = record.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        if parts:
            return "\n".join(parts)
    return json.dumps(record.get("metadata", record), ensure_ascii=False)


def enrich_file(input_path: Path, output_path: Path, k: int = 3) -> int:
    retriever = MockRetriever()
    processed = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            record: dict[str, Any] = json.loads(line)
            query = _record_to_query(record)
            record["retrieved_snippets"] = retriever.get_top_k(query, k=k)

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich prompted JSONL with retrieved snippets.")
    parser.add_argument("-i", "--input", required=True, help="Path to input prompted JSONL.")
    parser.add_argument("-o", "--output", required=True, help="Path to output enriched JSONL.")
    parser.add_argument("-k", type=int, default=3, help="Number of snippets to attach.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = enrich_file(input_path, output_path, k=args.k)
    print(f"Wrote enriched JSONL to {output_path} ({processed} records)")


if __name__ == "__main__":
    main()
