"""
Shared cost-estimation utilities for the static-evals pipeline.

Uses tiktoken for accurate input token counting and heuristic output token
estimates based on the expected JSON structure of each eval prompt.
"""

import tiktoken

# ── Pricing per 1M tokens (USD) ──────────────────────────────────────────────
# Source: https://platform.openai.com/docs/pricing (as of 2025-Q1)
# Add new models here as needed.

MODEL_PRICING: dict[str, dict[str, float]] = {
    # {model: {input: $/1M, output: $/1M, batch_input: $/1M, batch_output: $/1M}}
    "gpt-5-nano": {
        "input": 0.05, "output": 0.40,
        "batch_input": 0.025, "batch_output": 0.20,
    },
    "gpt-5-mini": {
        "input": 0.25, "output": 2.00,
        "batch_input": 0.125, "batch_output": 1.00,
    },
    "gpt-5.2": {
        "input": 1.75, "output": 14.00,
        "batch_input": 0.875, "batch_output": 7.00,
    },
    "gpt-4.1": {
        "input": 2.00, "output": 8.00,
        "batch_input": 1.00, "batch_output": 4.00,
    },
    "gpt-4.1-mini": {
        "input": 0.40, "output": 1.60,
        "batch_input": 0.20, "batch_output": 0.80,
    },
    "gpt-4.1-nano": {
        "input": 0.10, "output": 0.40,
        "batch_input": 0.05, "batch_output": 0.20,
    },
    "gpt-4o": {
        "input": 2.50, "output": 10.00,
        "batch_input": 1.25, "batch_output": 5.00,
    },
    "gpt-4o-mini": {
        "input": 0.15, "output": 0.60,
        "batch_input": 0.075, "batch_output": 0.30,
    },
}

# Fallback for unknown models
_DEFAULT_PRICING = {
    "input": 2.50, "output": 10.00,
    "batch_input": 1.25, "batch_output": 5.00,
}

# ── Estimated output tokens per eval prompt type ─────────────────────────────
# Based on the expected JSON structure of each eval prompt template.
# rubric_eval: 4 metrics × {score, justification} ≈ 200 tokens
# unit tests:  1 metric  × {score, justification} ≈ 60 tokens
# forbidden:   1 metric  × {score, justification, breakdown(5)} ≈ 100 tokens

ESTIMATED_OUTPUT_TOKENS: dict[str, int] = {
    "rubric_eval":       200,
    "empathy_chaining":   60,
    "forbidden":         70,
    "multiple_tasks":     60,
    "pacing":             60,
    "questions":          60,
    "risk_assessment":    60,
}
_DEFAULT_OUTPUT_TOKENS = 100


def get_pricing(model: str) -> dict[str, float]:
    """Return pricing dict for a model, falling back to defaults."""
    return MODEL_PRICING.get(model, _DEFAULT_PRICING)


def count_tokens(text: str, model: str = "gpt-4.1") -> int:
    """Count tokens in a string using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def count_message_tokens(messages: list[dict], model: str = "gpt-4.1") -> int:
    """Estimate token count for a chat-completions messages array.

    Adds per-message overhead (~4 tokens each) following OpenAI's counting guide.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    total = 3  # every reply is primed with <|start|>assistant<|message|>
    for msg in messages:
        total += 4  # <|start|>{role/name}\n
        for key, val in msg.items():
            total += len(enc.encode(str(val)))
    return total


def estimate_output_tokens(eval_name: str) -> int:
    """Return estimated output tokens for a given eval prompt key."""
    return ESTIMATED_OUTPUT_TOKENS.get(eval_name, _DEFAULT_OUTPUT_TOKENS)


def estimate_cost(
    *,
    requests: list[dict],
    model: str,
    use_batch: bool,
) -> dict:
    """
    Estimate cost for a list of requests.

    Each request dict must have:
      - "messages": list[dict]  (the chat messages)
      - "eval_name": str        (eval prompt key, for output estimation)

    Returns dict with total_input_tokens, total_output_tokens,
    input_cost, output_cost, total_cost.
    """
    pricing = get_pricing(model)
    input_key  = "batch_input"  if use_batch else "input"
    output_key = "batch_output" if use_batch else "output"

    total_input_tokens  = 0
    total_output_tokens = 0

    for req in requests:
        total_input_tokens  += count_message_tokens(req["messages"], model)
        total_output_tokens += estimate_output_tokens(req.get("eval_name", ""))

    input_cost  = total_input_tokens  / 1_000_000 * pricing[input_key]
    output_cost = total_output_tokens / 1_000_000 * pricing[output_key]

    return {
        "total_input_tokens":  total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost":  input_cost,
        "output_cost": output_cost,
        "total_cost":  input_cost + output_cost,
        "pricing_tier": "batch" if use_batch else "standard",
        "model": model,
    }


def format_cost_summary(est: dict, *, show_comparison: bool = False, alt_est: dict | None = None) -> str:
    """Format a cost estimate dict into a human-readable summary."""
    lines = [
        f"  Model:          {est['model']}",
        f"  Pricing tier:   {est['pricing_tier']}",
        f"  Input tokens:   {est['total_input_tokens']:,}",
        f"  Output tokens:  ~{est['total_output_tokens']:,} (estimated)",
        f"  Input cost:     ${est['input_cost']:.4f}",
        f"  Output cost:    ${est['output_cost']:.4f}",
        f"  Total cost:     ${est['total_cost']:.4f}",
    ]
    if show_comparison and alt_est is not None:
        savings = alt_est["total_cost"] - est["total_cost"]
        if savings > 0:
            lines.append(f"  Savings vs {alt_est['pricing_tier']}: ${savings:.4f} ({savings/alt_est['total_cost']*100:.0f}%)")
        else:
            extra = est["total_cost"] - alt_est["total_cost"]
            lines.append(f"  Extra vs {alt_est['pricing_tier']}: +${extra:.4f}")
    return "\n".join(lines)
