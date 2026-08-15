"""
rag.py – Retrieval-Augmented Generation index for historical counselor responses.

Builds an in-memory vector index from historical conversation CSVs, with
deterministic hash-based caching so only *new* (context, response) pairs are
embedded on subsequent runs.

Usage (from step1_generate_and_upload.py)
-----------------------------------------
    from rag import RAGIndex

    idx = RAGIndex(
        csv_path="path/to/200_perfect_scoring_historical.csv",
        embeddings_model="text-embedding-3-small",
        top_k=5,
    )
    pairs = idx.retrieve("counselor: Hi there ...\npatient: I feel ...")
    # pairs -> [(context_str, counselor_response_str), ...]
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# ── Import block-based conversation parser from vf_copilot ───────────────────
_VF_ROOT = Path("/home/stanford/CrisisBench")
if str(_VF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VF_ROOT))

from singleturn.convert_dialogue_to_static_eval_format import (  # noqa: E402
    split_conversation,
)

# ── Constants ────────────────────────────────────────────────────────────────
_CACHE_DIR_NAME = ".rag_cache"
_EMBEDDINGS_FILE = "embeddings.npy"
_METADATA_FILE = "metadata.json"
_MAX_TOKENS_PER_BATCH = 250_000  # stay under OpenAI's 300K token/request limit
_MAX_ITEMS_PER_BATCH = 2048      # OpenAI supports up to 2048 inputs per call


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_hash(context: str, response: str) -> str:
    """SHA-256 hash of a (context, response) pair – deterministic across runs."""
    payload = (context + "\x00" + response).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _batch_embed(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """Embed a list of texts in token-aware batches, returning (N, dim) array.

    Batches are split so each request stays under the OpenAI per-request token
    limit (300 K tokens).  We use ``cl100k_base`` encoding which is used by
    OpenAI embedding models.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    all_embeddings: list[list[float]] = []

    batch: list[str] = []
    batch_tokens = 0
    batches_sent = 0

    def _flush(b: list[str]) -> None:
        nonlocal batches_sent
        if not b:
            return
        resp = client.embeddings.create(model=model, input=b)
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        all_embeddings.extend([d.embedding for d in sorted_data])
        batches_sent += 1
        print(f"[rag]   embedded batch {batches_sent} ({len(b)} texts)")

    for text in texts:
        if not text.strip():
            text = " "
            
        tokens = enc.encode(text)
        if len(tokens) > 8190:
            # Over the 8192 token limit for text-embedding-3 models.
            # Truncate to safely fit. (Keep the end of the context where the latest turns are)
            tokens = tokens[-8190:]
            text = enc.decode(tokens)
            text_tokens = len(enc.encode(text))  # Recalculate accurately after decoding
        else:
            text_tokens = len(tokens)
            
        # If adding this text would exceed limits, flush first
        if batch and (
            batch_tokens + text_tokens > _MAX_TOKENS_PER_BATCH
            or len(batch) >= _MAX_ITEMS_PER_BATCH
        ):
            _flush(batch)
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += text_tokens

    _flush(batch)  # remaining
    return np.array(all_embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RAGIndex
# ─────────────────────────────────────────────────────────────────────────────

class RAGIndex:
    """In-memory vector index over historical (context → counselor response) pairs.

    Parameters
    ----------
    csv_path : str
        Path to the CSV containing historical conversations.  Must have a
        ``convo_str`` column.
    embeddings_model : str
        OpenAI embeddings model name (e.g. ``text-embedding-3-small``).
    top_k : int
        Number of similar pairs to return from :meth:`retrieve`.
    cache_dir : str | None
        Directory for the ``.npy`` / ``.json`` cache.  Defaults to
        ``<dynamic_evals dir>/.rag_cache/``.
    conversation_column : str
        Name of the column in the CSV that contains conversation text.
    """

    def __init__(
        self,
        csv_path: str,
        embeddings_model: str,
        top_k: int = 5,
        cache_dir: str | None = None,
        conversation_column: str = "convo_str",
    ):
        self._model = embeddings_model
        self._top_k = top_k
        self._client = OpenAI()  # reads OPENAI_API_KEY from env

        cache_path = Path(cache_dir) if cache_dir else Path(__file__).parent / _CACHE_DIR_NAME
        cache_path.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_path

        # 1. Parse CSV → (context, response) pairs
        pairs = self._parse_csv(csv_path, conversation_column)
        print(f"[rag] Parsed {len(pairs)} (context, response) pairs from CSV")

        # 2. Hash each pair
        hashes = [_deterministic_hash(ctx, resp) for ctx, resp in pairs]

        # 3. Load existing cache
        cached_hashes, cached_embeddings = self._load_cache()

        # 4. Identify new pairs
        cached_set = set(cached_hashes)
        new_indices = [i for i, h in enumerate(hashes) if h not in cached_set]
        print(f"[rag] {len(new_indices)} new pairs to embed, "
              f"{len(hashes) - len(new_indices)} already cached")

        # 5. Embed new contexts (with cost estimate + confirmation)
        if new_indices:
            new_contexts = [pairs[i][0] for i in new_indices]

            # Token count & cost estimate
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            total_tokens = sum(len(enc.encode(t)) for t in new_contexts)

            cost_per_million = {
                "text-embedding-3-small": 0.02,
                "text-embedding-3-large": 0.13,
            }
            rate = cost_per_million.get(self._model)
            if rate is not None:
                est_cost = (total_tokens / 1_000_000) * rate
                print(
                    f"[rag] Embedding cost estimate: "
                    f"{total_tokens:,} tokens × ${rate}/1M = ${est_cost:.4f} "
                    f"(model: {self._model})"
                )
            else:
                print(
                    f"[rag] Embedding {total_tokens:,} tokens with model "
                    f"'{self._model}' (unknown pricing — check OpenAI docs)"
                )

            confirm = input("[rag] Proceed with embedding? [y/n]: ").strip().lower()
            if confirm != "y":
                sys.exit("[rag] Aborted by user.")

            new_embeds = _batch_embed(self._client, new_contexts, self._model)

            # Merge into cache
            new_hashes = [hashes[i] for i in new_indices]
            new_responses = [pairs[i][1] for i in new_indices]

            if cached_embeddings is not None and len(cached_embeddings) > 0:
                merged_embeddings = np.vstack([cached_embeddings, new_embeds])
                merged_hashes = cached_hashes + new_hashes
            else:
                merged_embeddings = new_embeds
                merged_hashes = new_hashes

            # Build full metadata (hash → response) for the cached entries too
            merged_metadata = self._load_metadata_list()
            merged_metadata.extend(
                {"hash": h, "response": r} for h, r in zip(new_hashes, new_responses)
            )

            # Save updated cache
            self._save_cache(merged_embeddings, merged_metadata)
            all_cache_hashes = merged_hashes
            all_cache_embeddings = merged_embeddings
            all_cache_metadata = merged_metadata
        else:
            all_cache_hashes = cached_hashes
            all_cache_embeddings = cached_embeddings
            all_cache_metadata = self._load_metadata_list()

        # 6. Build the in-memory index for only the pairs in *this* CSV
        #    (cache may contain entries from other CSVs)
        hash_to_row: dict[str, int] = {h: i for i, h in enumerate(all_cache_hashes)}
        active_rows = []
        self._contexts: list[str] = []
        self._responses: list[str] = []

        for i, h in enumerate(hashes):
            row_idx = hash_to_row[h]
            active_rows.append(row_idx)
            self._contexts.append(pairs[i][0])
            self._responses.append(all_cache_metadata[row_idx]["response"])

        self._embeddings = all_cache_embeddings[active_rows]
        print(f"[rag] Index ready: {self.num_documents} documents, "
              f"embedding dim={self._embeddings.shape[1]}")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def num_documents(self) -> int:
        return len(self._responses)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, current_context: str) -> list[tuple[str, str]]:
        """Return top-k (context, counselor_response) pairs most similar to
        *current_context*."""
        if not current_context.strip():
            return []
            
        query_embed = _batch_embed(self._client, [current_context], self._model)
        sims = cosine_similarity(query_embed, self._embeddings)[0]
        top_indices = np.argsort(sims)[::-1][: self._top_k]
        return [(self._contexts[i], self._responses[i]) for i in top_indices]

    # ── CSV Parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_csv(csv_path: str, conversation_column: str) -> list[tuple[str, str]]:
        """Parse the CSV and return all (context, counselor_response) pairs.

        Pairs whose context is empty (e.g. the very first counselor message in
        a conversation) are dropped because there is nothing to embed.
        """
        df = pd.read_csv(csv_path)
        if conversation_column not in df.columns:
            raise ValueError(
                f"Column '{conversation_column}' not found in CSV. "
                f"Available: {', '.join(df.columns)}"
            )

        all_pairs: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            convo = row[conversation_column]
            if not isinstance(convo, str) or not convo.strip():
                continue
            pairs = split_conversation(convo)
            for ctx, resp in pairs:
                # Skip pairs with empty/whitespace-only context
                if ctx.strip():
                    all_pairs.append((ctx, resp))

        return all_pairs

    # ── Cache I/O ─────────────────────────────────────────────────────────

    def _load_cache(self) -> tuple[list[str], np.ndarray | None]:
        """Load cached hashes and embeddings. Returns (hashes, embeddings)."""
        emb_path = self._cache_path / _EMBEDDINGS_FILE
        meta_path = self._cache_path / _METADATA_FILE

        if not emb_path.exists() or not meta_path.exists():
            return [], None

        embeddings = np.load(emb_path)
        with open(meta_path) as f:
            metadata = json.load(f)

        hashes = [entry["hash"] for entry in metadata]
        return hashes, embeddings

    def _load_metadata_list(self) -> list[dict]:
        """Load the full metadata list from cache (or empty list)."""
        meta_path = self._cache_path / _METADATA_FILE
        if not meta_path.exists():
            return []
        with open(meta_path) as f:
            return json.load(f)

    def _save_cache(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Persist embeddings and metadata to the cache directory."""
        np.save(self._cache_path / _EMBEDDINGS_FILE, embeddings)
        with open(self._cache_path / _METADATA_FILE, "w") as f:
            json.dump(metadata, f)
        print(f"[rag] Cache saved: {len(metadata)} entries → {self._cache_path}")
