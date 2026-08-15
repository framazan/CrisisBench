"""Mock retriever for prototyping retrieval-enrichment.

Returns deterministic placeholder snippets so the enrichment step can be tested
without building an embedding index, calling external services, or touching real data.
"""
from __future__ import annotations

from typing import Dict, List


class MockRetriever:
    """Simple mock retriever with the same shape as a future real retriever."""

    def __init__(self, source_label: str = "mock_snippet_db") -> None:
        self.source_label = source_label

    def get_top_k(self, query: str, k: int = 3) -> List[Dict]:
        snippets = []
        for i in range(k):
            snippets.append(
                {
                    "id": f"{self.source_label}_snippet_{i + 1}",
                    "text": (
                        "[MOCK SNIPPET] This is a placeholder high-quality snippet "
                        "used for prototyping retrieval-enrichment. Replace with a "
                        "real retriever before evaluation."
                    ),
                    "score": round(1.0 - (i * 0.05), 3),
                }
            )
        return snippets
