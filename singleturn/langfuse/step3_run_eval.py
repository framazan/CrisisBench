"""
Step 3 – Run LLM-judge evals on an existing generation run.

Supports two modes:
  • Standard (default): synchronous calls via langfuse.openai wrapper (auto-traced)
  • Batch API (--use-batch or interactive prompt): 50 % cheaper, async

Before running, the script estimates total cost and asks for confirmation.

Modes
-----
  Default      : standard API (synchronous, fully traced via langfuse.openai)
  --use-batch  : use the OpenAI Batch API (cheaper, async)
  --submit-only: submit the batch and exit (resume later with --batch-id)
  --batch-id ID: skip prepare/submit, resume polling an existing batch

Usage
-----
  python step3_run_eval.py --dataset-name my-ds --run-name copilot-v1
  python step3_run_eval.py --dataset-name my-ds --run-name copilot-v1 --use-batch
  python step3_run_eval.py --dataset-name my-ds --run-name copilot-v1 --batch-id batch_abc123
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import openai
import yaml
from langfuse import Langfuse
from langfuse.openai import OpenAI

from cost_utils import (
    estimate_cost,
    estimate_output_tokens,
    format_cost_summary,
)

# ── Constants ────────────────────────────────────────────────────────────────

BATCH_ARTIFACTS_DIR = Path(__file__).parent / "_batch_artifacts"


def _check_env() -> None:
    """Exit early if required environment variables are missing."""
    missing = [v for v in ("OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"[step3] ❌ Missing required environment variable(s): {', '.join(missing)}\n"
            "  Set them in ~/.bashrc (or export them) before running this script."
        )


def _is_fatal_api_error(exc: Exception) -> bool:
    """Return True for errors that will affect every future API call."""
    return isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError))


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_static_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _print_resume_instructions(args, batch_id):
    """Print resume commands after Ctrl+C or --submit-only."""
    standalone = (
        f"  python step3_run_eval.py \\\n"
        f"    --dataset-name {args.dataset_name} \\\n"
        f"    --run-name {args.run_name} \\\n"
        f"    --batch-id {batch_id}"
    )
    pipeline = (
        f"  python run_pipeline.py \\\n"
        f"    --dataset-name {args.dataset_name} \\\n"
        f"    --run-name {args.run_name} \\\n"
        f"    --model {args.model} \\\n"
        f"    --start-from 3 \\\n"
        f"    --use-batch \\\n"
        f"    --batch-id-step3 {batch_id}"
    )
    print(
        f"\n[step3] ⏸️  Batch {batch_id} is still running on OpenAI's servers.\n"
        f"         It will continue processing even though you stopped polling.\n\n"
        f"         Resume step3 standalone:\n{standalone}\n\n"
        f"         Or resume the full pipeline from step 3:\n{pipeline}\n"
    )


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run LLM-judge evals on a Langfuse generation run")
    p.add_argument("--dataset-name",  required=True,
                   help="Langfuse dataset name (resolved from step 1/2)")
    p.add_argument("--run-name",      required=True,
                   help="Langfuse experiment run name to evaluate")
    p.add_argument("--model",         default=cfg["eval"]["model"])
    p.add_argument("--max-completion-tokens", type=int, default=cfg["eval"]["max_completion_tokens"])
    p.add_argument("--temperature",   type=float, default=cfg["eval"]["temperature"])
    p.add_argument("--max-workers",   type=int,   default=cfg["max_workers"])
    p.add_argument("--prompts",       nargs="+",  default=None,
                   help="Subset of eval prompt keys to run (default: all)")
    p.add_argument("--use-batch",     action="store_true",
                   help="Use the OpenAI Batch API (50%% cheaper, async)")
    p.add_argument("--batch-id",      default=None,
                   help="Resume polling an already-submitted batch")
    p.add_argument("--submit-only",   action="store_true",
                   help="Submit the batch and exit (resume later with --batch-id)")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="Seconds between batch status polls (default: 60)")
    p.add_argument("--yes", "-y",     action="store_true",
                   help="Skip interactive confirmation prompts")
    return p.parse_args(), cfg


def extract_json_block(text: str) -> Optional[dict]:
    """Extract the first JSON object from an LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    for candidate in [text, re.search(r"\{.*\}", text, re.DOTALL)]:
        if candidate is None:
            continue
        s = candidate if isinstance(candidate, str) else candidate.group(0)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    return None


# ── Build eval payloads (shared by both modes) ───────────────────────────────

def build_eval_payloads(
    *,
    lf: Langfuse,
    run_items,
    dataset_item_map: dict,
    eval_prompts: dict,
) -> list[dict]:
    """
    Build a list of eval payload dicts, each containing:
      trace_id, dataset_item_id, eval_name, prompt_obj, messages
    """
    payloads: list[dict] = []

    for ri in run_items:
        trace_id = ri.trace_id
        dataset_item = dataset_item_map.get(ri.dataset_item_id)
        if dataset_item is None:
            print(f"  [warn] no dataset item for run_item {ri.id}")
            continue

        try:
            resolved_trace_id = trace_id
            try:
                trace = lf.api.trace.get(resolved_trace_id)
            except Exception:
                # Backward compatibility: some runs may store an observation ID
                # instead of a trace ID in dataset run items.
                observation = lf.api.observations.get(trace_id)
                obs_trace_id = getattr(observation, "trace_id", None)
                if not obs_trace_id:
                    raise
                resolved_trace_id = obs_trace_id
                trace = lf.api.trace.get(resolved_trace_id)

            raw_output = trace.output
            copilot_output = (
                raw_output if isinstance(raw_output, str)
                else (json.dumps(raw_output) if raw_output else "")
            )
        except Exception as exc:
            print(f"  [warn] could not fetch trace {trace_id}: {exc}")
            continue

        convo_history = (dataset_item.input or {}).get("convo_history", "")
        human_output  = dataset_item.expected_output or ""

        for eval_name, prompt_obj in eval_prompts.items():
            comparison = (
                f"Conversation History:\n{convo_history}\n\n"
                f"Response A (Human Counselor):\n{human_output}\n\n"
                f"Response B (AI Copilot):\n{copilot_output}"
            )
            try:
                compiled = prompt_obj.compile(note=comparison)
            except Exception:
                compiled = prompt_obj.prompt.replace("{{note}}", comparison)

            messages = [{"role": "user", "content": compiled}]
            payloads.append({
                "trace_id": resolved_trace_id,
                "dataset_item_id": ri.dataset_item_id,
                "eval_name": eval_name,
                "prompt_obj": prompt_obj,
                "messages": messages,
            })

    return payloads


# ── Score parsing (shared) ───────────────────────────────────────────────────

def parse_and_attach_scores(lf: Langfuse, trace_id: str, eval_name: str, raw: str) -> int:
    """Parse JSON scores from LLM output and attach to trace. Returns count of scores added."""
    parsed = extract_json_block(raw)
    if not parsed:
        print(f"  [warn] could not parse JSON from eval '{eval_name}': {raw[:120]}")
        return 0

    added = 0
    for metric_key, metric_val in parsed.items():
        if not isinstance(metric_val, dict):
            score_value = float(metric_val) if metric_val is not None else None
            justification = ""
        else:
            score_value = metric_val.get("score")
            justification = metric_val.get("justification", "")

        if score_value is None:
            continue

        lf.create_score(
            trace_id=trace_id,
            name=f"{eval_name}.{metric_key}" if eval_name not in metric_key else metric_key,
            value=float(score_value),
            comment=justification[:1000] if justification else None,
            data_type="NUMERIC",
        )
        added += 1

    return added


# ══════════════════════════════════════════════════════════════════════════════
# STANDARD (non-batch) PATH
# ══════════════════════════════════════════════════════════════════════════════

def score_one_item_standard(
    *,
    lf: Langfuse,
    payload: dict,
    model: str,
    max_completion_tokens: int,
    temperature: float,
) -> int:
    """Run a single eval call via the langfuse.openai wrapper (auto-traced)."""
    eval_name  = payload["eval_name"]
    trace_id   = payload["trace_id"]
    prompt_obj = payload["prompt_obj"]
    messages   = payload["messages"]

    client = OpenAI()  # langfuse.openai wrapper — auto-traces each eval call
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            langfuse_prompt=prompt_obj,
            name=f"eval-{eval_name}",
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as exc:
        if _is_fatal_api_error(exc):
            raise
        print(f"  [warn] LLM call failed for eval '{eval_name}' on trace {trace_id}: {exc}")
        return 0

    return parse_and_attach_scores(lf, trace_id, eval_name, raw)


def run_standard_evals(lf, payloads, args):
    """Run all evals synchronously (with thread pool) via langfuse.openai wrapper."""
    total_scores = 0
    errors = 0
    fatal_error: Optional[Exception] = None
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                score_one_item_standard,
                lf=lf,
                payload=p,
                model=args.model,
                max_completion_tokens=args.max_completion_tokens,
                temperature=args.temperature,
            ): p
            for p in payloads
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                n = fut.result()
                total_scores += n
                if n > 0:
                    short_id = p["dataset_item_id"][:8]
                    print(f"  [ok] {short_id}/{p['eval_name']}: {n} scores")
            except Exception as exc:
                if _is_fatal_api_error(exc):
                    fatal_error = exc
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                errors += 1
                print(f"  [err] {p['eval_name']}/{p['trace_id'][:8]}: {exc}")

    if fatal_error is not None:
        sys.exit(
            f"\n[step3] ❌ Fatal API error — no scores flushed to Langfuse.\n"
            f"  {fatal_error}"
        )

    lf.flush()
    elapsed = time.time() - t0
    print(
        f"\n[step3] ✅ Done in {elapsed:.1f}s — "
        f"{total_scores} scores added, {errors} errors.\n"
        f"         View run: {lf._base_url}/datasets/{args.dataset_name}/runs/{args.run_name}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# BATCH API PATH
# ══════════════════════════════════════════════════════════════════════════════

def payloads_to_batch(payloads, model, max_completion_tokens, temperature):
    """Convert payloads to JSONL lines + manifest for the OpenAI Batch API."""
    jsonl_lines: list[str] = []
    manifest: dict[str, dict] = {}

    for idx, p in enumerate(payloads):
        custom_id = f"req-{idx}"
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": p["messages"],
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
            },
        }
        jsonl_lines.append(json.dumps(request))
        manifest[custom_id] = {
            "trace_id": p["trace_id"],
            "dataset_item_id": p["dataset_item_id"],
            "eval_name": p["eval_name"],
            "prompt_name": p["prompt_obj"].name,
            "prompt_version": p["prompt_obj"].version,
            "messages": p["messages"],
        }

    return jsonl_lines, manifest


def submit_batch(client: openai.OpenAI, jsonl_path: Path) -> str:
    """Upload the JSONL file and create a batch. Returns the batch ID."""
    print(f"[step3] Uploading {jsonl_path.name} …")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"[step3] File uploaded: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"step3 eval batch from {jsonl_path.stem}"},
    )
    print(f"[step3] Batch created: {batch.id}")
    return batch.id


def poll_batch(client: openai.OpenAI, batch_id: str, poll_interval: int):
    """Poll until the batch reaches a terminal state. Returns the batch object."""
    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        if counts:
            print(
                f"  [{time.strftime('%H:%M:%S')}] status={batch.status}  "
                f"completed={counts.completed}/{counts.total}  "
                f"failed={counts.failed}"
            )
        else:
            print(f"  [{time.strftime('%H:%M:%S')}] status={batch.status}")
        if batch.status in terminal:
            return batch
        time.sleep(poll_interval)


def process_batch_results(
    *,
    lf: Langfuse,
    client: openai.OpenAI,
    batch,
    manifest: dict,
    eval_prompts: dict,
    model: str,
) -> tuple[int, int]:
    """Download batch results, log Langfuse traces, attach scores."""
    if batch.output_file_id is None:
        print("[step3] No output file — nothing to process.")
        return 0, 0

    content = client.files.content(batch.output_file_id)
    result_lines = content.text.strip().split("\n")

    total_scores = 0
    errors = 0

    for line in result_lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        meta = manifest.get(custom_id)
        if meta is None:
            print(f"  [warn] unknown custom_id: {custom_id}")
            errors += 1
            continue

        trace_id   = meta["trace_id"]
        eval_name  = meta["eval_name"]
        messages   = meta["messages"]

        if result.get("error"):
            print(f"  [err] {custom_id} ({eval_name}): {result['error']}")
            errors += 1
            continue

        response_body = result["response"]["body"]
        raw   = response_body["choices"][0]["message"]["content"].strip()
        usage = response_body.get("usage", {})

        # Log eval call to Langfuse as a trace + generation
        prompt_obj = eval_prompts.get(eval_name)
        gen = lf.start_generation(
            name=f"eval-{eval_name}",
            model=model,
            input=messages,
            output=raw,
            usage_details={
                "input":  usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
                "total":  usage.get("total_tokens", 0),
            },
            prompt=prompt_obj,
        )
        gen.update_trace(
            name=f"eval-{eval_name}",
            metadata={
                "batch_id": batch.id,
                "source_trace_id": trace_id,
                "dataset_item_id": meta["dataset_item_id"],
            },
            tags=["batch-eval"],
        )
        gen.end()

        n = parse_and_attach_scores(lf, trace_id, eval_name, raw)
        total_scores += n
        if n > 0:
            print(f"  [ok] {meta['dataset_item_id'][:8]}/{eval_name}: {n} scores")
        else:
            errors += 1

    return total_scores, errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args, cfg = parse_args()
    _check_env()

    lf = Langfuse()
    client = openai.OpenAI()

    # ── Flatten prompt config ────────────────────────────────────────────
    def _flatten_prompts(cfg_prompts: dict) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in cfg_prompts.items():
            if k == "generation":
                continue
            if isinstance(v, str):
                out[k] = v
            elif isinstance(v, dict):
                out.update(v)
        return out

    all_eval_prompt_keys = _flatten_prompts(cfg["prompts"])
    requested_keys = args.prompts or list(all_eval_prompt_keys.keys())

    print(f"[step3] Loading {len(requested_keys)} eval prompt(s) from Langfuse …")
    eval_prompts: dict = {}
    for key in requested_keys:
        prompt_name = all_eval_prompt_keys.get(key, key)
        try:
            eval_prompts[key] = lf.get_prompt(prompt_name)
            print(f"  ✓ '{key}' → '{prompt_name}' (v{eval_prompts[key].version})")
        except Exception as exc:
            print(f"  ✗ '{key}' → '{prompt_name}': {exc}")

    if not eval_prompts:
        print("[step3] No eval prompts loaded — aborting.")
        return

    # ── Resume from existing batch? ──────────────────────────────────────
    if args.batch_id:
        print(f"\n[step3] Resuming batch {args.batch_id} …")
        manifest_path = BATCH_ARTIFACTS_DIR / f"{args.batch_id}_manifest.json"
        if not manifest_path.exists():
            alt = BATCH_ARTIFACTS_DIR / f"{args.run_name}_manifest.json"
            if alt.exists():
                manifest_path = alt
            else:
                sys.exit(
                    f"[step3] ❌ Manifest not found at {manifest_path}.\n"
                    f"  The manifest is needed to map batch results back to traces."
                )
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[step3] Loaded manifest with {len(manifest)} requests.")

        print(f"\n[step3] Polling batch {args.batch_id} (every {args.poll_interval}s) …")
        try:
            batch = poll_batch(client, args.batch_id, args.poll_interval)
        except KeyboardInterrupt:
            _print_resume_instructions(args, args.batch_id)
            return
        if batch.status != "completed":
            sys.exit(f"\n[step3] ❌ Batch ended with status '{batch.status}'.\n  Errors: {batch.errors}")

        t0 = time.time()
        total_scores, errors = process_batch_results(
            lf=lf, client=client, batch=batch, manifest=manifest,
            eval_prompts=eval_prompts, model=args.model,
        )
        lf.flush()
        elapsed = time.time() - t0
        print(
            f"\n[step3] ✅ Done in {elapsed:.1f}s — "
            f"{total_scores} scores added, {errors} errors.\n"
            f"         Batch ID: {args.batch_id}\n"
            f"         View run: {lf._base_url}/datasets/{args.dataset_name}/runs/{args.run_name}"
        )
        return

    # ── Build eval payloads ──────────────────────────────────────────────
    print(f"\n[step3] Fetching dataset '{args.dataset_name}' …")
    dataset = lf.get_dataset(args.dataset_name)
    dataset_item_map = {item.id: item for item in dataset.items}
    print(f"[step3] {len(dataset_item_map)} dataset items loaded.")

    print(f"[step3] Fetching run '{args.run_name}' …")
    try:
        run = lf.get_dataset_run(dataset_name=args.dataset_name, run_name=args.run_name)
    except Exception as exc:
        sys.exit(f"[step3] ❌ Could not fetch run '{args.run_name}': {exc}")

    run_items = run.dataset_run_items
    print(f"[step3] {len(run_items)} run items.\n")

    print("[step3] Building eval payloads …")
    payloads = build_eval_payloads(
        lf=lf, run_items=run_items,
        dataset_item_map=dataset_item_map,
        eval_prompts=eval_prompts,
    )
    if not payloads:
        print("[step3] No eval payloads — aborting.")
        return
    print(f"[step3] {len(payloads)} eval requests prepared.")

    # ── Cost estimation ──────────────────────────────────────────────────
    cost_requests = [{"messages": p["messages"], "eval_name": p["eval_name"]} for p in payloads]
    batch_est = estimate_cost(requests=cost_requests, model=args.model, use_batch=True)
    std_est   = estimate_cost(requests=cost_requests, model=args.model, use_batch=False)

    # ── Mode selection ───────────────────────────────────────────────────
    use_batch = args.use_batch

    if not args.yes and not use_batch:
        print(f"\n[step3] Cost estimate for {len(payloads)} eval calls:\n")
        print("  ── Standard API ──")
        print(format_cost_summary(std_est))
        print("\n  ── Batch API (50% cheaper) ──")
        print(format_cost_summary(batch_est, show_comparison=True, alt_est=std_est))
        print()
        answer = input("[step3] Do you want to use the Batch API? (y/n): ").strip().lower()
        use_batch = answer in ("y", "yes")

    chosen_est = batch_est if use_batch else std_est

    if not args.yes:
        tier_label = "batch" if use_batch else "standard"
        print(f"\n[step3] Estimated {tier_label} API cost: ${chosen_est['total_cost']:.4f}")
        confirm = input(
            f"[step3] Given the ${chosen_est['total_cost']:.4f} estimated cost, "
            f"do you want to proceed? (y/n): "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("[step3] Aborted.")
            return

    # ── Execute ──────────────────────────────────────────────────────────
    if use_batch:
        jsonl_lines, manifest = payloads_to_batch(
            payloads, args.model, args.max_completion_tokens, args.temperature,
        )

        BATCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path    = BATCH_ARTIFACTS_DIR / f"{args.run_name}_requests.jsonl"
        manifest_path = BATCH_ARTIFACTS_DIR / f"{args.run_name}_manifest.json"

        with open(jsonl_path, "w") as f:
            f.write("\n".join(jsonl_lines) + "\n")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[step3] Saved requests  → {jsonl_path}")
        print(f"[step3] Saved manifest  → {manifest_path}")

        batch_id = submit_batch(client, jsonl_path)

        with open(BATCH_ARTIFACTS_DIR / f"{batch_id}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if args.submit_only:
            _print_resume_instructions(args, batch_id)
            return

        print(f"\n[step3] Polling batch {batch_id} (every {args.poll_interval}s) …")
        try:
            batch = poll_batch(client, batch_id, args.poll_interval)
        except KeyboardInterrupt:
            _print_resume_instructions(args, batch_id)
            return
        if batch.status != "completed":
            sys.exit(f"\n[step3] ❌ Batch ended with status '{batch.status}'.\n  Errors: {batch.errors}")

        completed_count = batch.request_counts.completed if batch.request_counts else "??"
        print(f"\n[step3] Batch completed. Processing {completed_count} results …\n")

        t0 = time.time()
        total_scores, errors = process_batch_results(
            lf=lf, client=client, batch=batch, manifest=manifest,
            eval_prompts=eval_prompts, model=args.model,
        )
        lf.flush()
        elapsed = time.time() - t0
        print(
            f"\n[step3] ✅ Done in {elapsed:.1f}s — "
            f"{total_scores} scores added, {errors} errors.\n"
            f"         Batch ID: {batch_id}\n"
            f"         View run: {lf._base_url}/datasets/{args.dataset_name}/runs/{args.run_name}"
        )
    else:
        run_standard_evals(lf, payloads, args)


if __name__ == "__main__":
    main()
