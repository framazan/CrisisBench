"""
Step 2 – Run the copilot generation experiment in Langfuse.

Supports two modes:
  • Standard (default): uses lf.run_experiment() + langfuse.openai wrapper
  • Batch API (--use-batch or interactive prompt): 50 % cheaper, but async

In both modes every LLM call is traced in Langfuse and linked to the prompt
version in the Langfuse library.

Usage
-----
  python step2_run_generation.py --dataset-name static-evals --run-name copilot-v1 --model gpt-4.1
  python step2_run_generation.py --dataset-name ds --run-name v1 --model gpt-4.1 --use-batch
  python step2_run_generation.py --dataset-name ds --run-name v1 --model gpt-4.1 --batch-id batch_abc
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import openai
import yaml
from langfuse import Langfuse
from langfuse.api import CreateDatasetRunItemRequest
from langfuse.openai import OpenAI

from cost_utils import (
    count_message_tokens,
    estimate_cost,
    format_cost_summary,
    get_pricing,
)

BATCH_ARTIFACTS_DIR = Path(__file__).parent / "_batch_artifacts"


def _check_env() -> None:
    """Exit early if required environment variables are missing."""
    missing = [v for v in ("OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"[step2] ❌ Missing required environment variable(s): {', '.join(missing)}\n"
            "  Set them in ~/.bashrc (or export them) before running this script."
        )


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_static_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _print_resume_instructions(args, batch_id):
    """Print resume commands after Ctrl+C or --submit-only."""
    standalone = (
        f"  python step2_run_generation.py \\\n"
        f"    --dataset-name {args.dataset_name} \\\n"
        f"    --run-name {args.run_name} \\\n"
        f"    --model {args.model} \\\n"
        f"    --batch-id {batch_id}"
    )
    pipeline = (
        f"  python run_pipeline.py \\\n"
        f"    --dataset-name {args.dataset_name} \\\n"
        f"    --run-name {args.run_name} \\\n"
        f"    --model {args.model} \\\n"
        f"    --start-from 2 \\\n"
        f"    --use-batch \\\n"
        f"    --batch-id-step2 {batch_id}"
    )
    print(
        f"\n[step2] ⏸️  Batch {batch_id} is still running on OpenAI's servers.\n"
        f"         It will continue processing even though you stopped polling.\n\n"
        f"         Resume step2 standalone:\n{standalone}\n\n"
        f"         Or resume the full pipeline from step 2:\n{pipeline}\n"
    )


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run copilot generation experiment in Langfuse")
    p.add_argument("--dataset-name",  required=True,
                   help="Langfuse dataset name (resolved from step 1)")
    p.add_argument("--run-name",      required=True,
                   help="Langfuse experiment run name (shown in UI)")
    p.add_argument("--model",         required=True,
                   help="OpenAI model to use for generation")
    p.add_argument("--prompt-name",   default=cfg["prompts"]["generation"])
    p.add_argument("--max-completion-tokens", type=int, default=cfg["generation"]["max_completion_tokens"])
    p.add_argument("--temperature",   type=float, default=cfg["generation"]["temperature"])
    p.add_argument("--max-workers",   type=int,   default=cfg["max_workers"])
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
    return p.parse_args()


def _estimate_generation_cost(items, compiled_counselor, model, max_completion_tokens, use_batch, has_editor=False):
    """Estimate cost for all generation requests."""
    requests = []
    for item in items:
        convo_history = (item.input or {}).get("convo_history", "")
        messages = [
            {"role": "system", "content": compiled_counselor},
            {"role": "user",   "content": convo_history},
        ]
        requests.append({"messages": messages, "eval_name": "__generation"})
        if has_editor:
            # Add a second request for editor
            editor_messages = [
                {"role": "user", "content": convo_history + " placeholder draft text " * 50},
            ]
            requests.append({"messages": editor_messages, "eval_name": "__generation"})

    # For generation, output estimate = max_completion_tokens (worst case)
    # Use a more realistic heuristic: ~60% of max
    from cost_utils import ESTIMATED_OUTPUT_TOKENS
    ESTIMATED_OUTPUT_TOKENS["__generation"] = int(max_completion_tokens * 0.6)

    return estimate_cost(requests=requests, model=model, use_batch=use_batch)


# ── Batch API helpers ────────────────────────────────────────────────────────

def prepare_generation_batch(items, compiled_counselor, model, max_completion_tokens, temperature, rag_index=None, rag_template=None):
    """Build JSONL lines and manifest for the batch API."""
    jsonl_lines = []
    manifest = {}
    for idx, item in enumerate(items):
        convo_history = (item.input or {}).get("convo_history", "")
        
        system_prompt = compiled_counselor
        
        # Apply RAG if enabled
        if rag_index is not None and rag_template is not None:
            retrieved_pairs = rag_index.retrieve(convo_history)
            example_block = "\n\n".join(
                f"Example {i+1}:\nContext: {ctx}\nCounselor response: {resp}"
                for i, (ctx, resp) in enumerate(retrieved_pairs)
            )
            system_prompt = rag_template.replace("{example_responses}", example_block)
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": convo_history},
        ]
        custom_id = f"gen-{idx}"
        body = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        if "gpt-5" in model.lower() and not 'gpt-5.2' in model.lower():
            body["reasoning_effort"] = "minimal"
        else:
            body["reasoning_effort"] = 'low'
            
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        jsonl_lines.append(json.dumps(request))
        manifest[custom_id] = {
            "dataset_item_id": item.id,
            "uid": (item.input or {}).get("uid", item.id),
            "messages": messages,
            "convo_history": convo_history,
        }
    return jsonl_lines, manifest


def submit_batch(client, jsonl_path):
    """Upload a JSONL file and create a batch. Returns batch ID."""
    print(f"[step2] Uploading {jsonl_path.name} …")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"[step2] File uploaded: {file_obj.id}")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"step2 generation batch from {jsonl_path.stem}"},
    )
    print(f"[step2] Batch created: {batch.id}")
    return batch.id

def prepare_editor_batch(results, manifest, editor_prompt_text, model, max_completion_tokens, temperature):
    jsonl_lines = []
    editor_manifest = {}
    
    for custom_id, result in results.items():
        if result.get("error"): continue
        
        meta = manifest[custom_id]
        dataset_item_id = meta["dataset_item_id"]
        convo_history = meta.get("convo_history", "")
        output = result["response"]["body"]["choices"][0]["message"]["content"].strip()
        
        editor_input = (
            f"{editor_prompt_text.strip()}\n\n"
            f"Conversation history:\n{convo_history}\n\n"
            f"New message from counselor:\n{output}\n"
            f"Output: "
        )
        
        messages = [
            {"role": "user", "content": editor_input},
        ]
        
        edit_custom_id = f"edit-{custom_id}"
        body = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        if "gpt-5" in model.lower() and not 'gpt-5.2' in model.lower():
            body["reasoning_effort"] = "minimal"
        else:
            body["reasoning_effort"] = "low"
            
        request = {
            "custom_id": edit_custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        jsonl_lines.append(json.dumps(request))
        editor_manifest[edit_custom_id] = {
            "dataset_item_id": dataset_item_id,
            "uid": meta.get("uid", dataset_item_id),
            "messages": messages,
            "draft_output": output, # store the draft 
            "source_custom_id": custom_id,
        }
        
    return jsonl_lines, editor_manifest
    
def download_batch_results(client, batch):
    if batch.output_file_id is None: return {}
    content = client.files.content(batch.output_file_id)
    results = {}
    for line in content.text.strip().split("\n"):
        res = json.loads(line)
        results[res["custom_id"]] = res
    return results


def poll_batch(client, batch_id, poll_interval):
    """Poll until the batch reaches a terminal state."""
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


def process_generation_results(*, lf, client, batch, manifest, compiled_counselor, args, is_editor=False):
    """Download batch results, log traces in Langfuse, link to dataset run."""
    if batch.output_file_id is None:
        print("[step2] No output file — nothing to process.")
        return

    content = client.files.content(batch.output_file_id)
    result_lines = content.text.strip().split("\n")

    dataset = lf.get_dataset(args.dataset_name)
    dataset_item_map = {item.id: item for item in dataset.items}

    ok = 0
    errors = 0

    for line in result_lines:
        result = json.loads(line)
        custom_id = result["custom_id"]
        meta = manifest.get(custom_id)
        if meta is None:
            print(f"  [warn] unknown custom_id: {custom_id}")
            errors += 1
            continue

        if result.get("error"):
            print(f"  [err] {custom_id}: {result['error']}")
            errors += 1
            continue

        response_body = result["response"]["body"]
        output = response_body["choices"][0]["message"]["content"].strip()
        usage  = response_body.get("usage", {})

        tag = f"{args.dataset_name}/{args.run_name}"
        shared_metadata = {
            "batch_id": batch.id,
            "dataset_name": args.dataset_name,
            "run_name": args.run_name,
            "model": args.model,
            "dataset_item_id": meta["dataset_item_id"],
            "uid": meta.get("uid", meta["dataset_item_id"]),
            "custom_id": custom_id,
        }

        parent_name = "copilot-editor-response" if is_editor else "copilot-response"
        parent = lf.start_observation(name=parent_name, as_type="span")
        parent.update_trace(
            name=parent_name,
            input=meta["messages"],
            output=output,
            metadata={
                **shared_metadata,
                "trace_stage": "editor" if is_editor else "generation",
            },
            tags=[tag],
        )

        child_trace_context = {
            "trace_id": parent.trace_id,
            "parent_span_id": parent.id,
        }

        if is_editor:
            draft_gen = lf.start_generation(
                trace_context=child_trace_context,
                name="copilot-draft-response",
                model=args.model,
                input="[Draft request from prior generation batch]",
                output=meta.get("draft_output", "N/A"),
            )
            draft_gen.end()

            editor_gen = lf.start_generation(
                trace_context=child_trace_context,
                name="copilot-editor-response",
                model=args.model,
                input=meta["messages"],
                output=output,
                usage_details={
                    "input":  usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total":  usage.get("total_tokens", 0),
                },
            )
            editor_gen.end()
            parent.update(
                metadata={
                    "draft_observation_id": draft_gen.id,
                    "editor_observation_id": editor_gen.id,
                    "editor_source_custom_id": meta.get("source_custom_id"),
                }
            )
        else:
            gen = lf.start_generation(
                trace_context=child_trace_context,
                name="copilot-response",
                model=args.model,
                input=meta["messages"],
                output=output,
                usage_details={
                    "input":  usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total":  usage.get("total_tokens", 0),
                },
            )
            gen.end()
            parent.update(metadata={"generation_observation_id": gen.id})

        parent.end()
        run_trace_id = parent.trace_id

        # Link this trace to the dataset item as a run item
        dataset_item = dataset_item_map.get(meta["dataset_item_id"])
        if dataset_item:
            lf.api.dataset_run_items.create(
                request=CreateDatasetRunItemRequest(
                    runName=args.run_name,
                    datasetItemId=dataset_item.id,
                    traceId=run_trace_id,
                ),
            )

        uid_short = meta.get("uid", meta["dataset_item_id"])[:8]
        print(f"  [ok] {uid_short} → '{output[:60]}…'")
        ok += 1

    print(f"\n[step2] Processed {ok} results, {errors} errors.")


# ── Standard (non-batch) path ────────────────────────────────────────────────

def run_standard(lf, args, dataset, compiled_counselor, editor_prompt_text, rag_index, rag_template):
    """Run generation via lf.run_experiment() + langfuse.openai wrapper."""
    def task(*, item, **kwargs) -> str:
        convo_history = (item.input or {}).get("convo_history", "")
        
        system_prompt = compiled_counselor
        
        # Apply RAG if enabled
        if rag_index is not None and rag_template is not None:
            retrieved_pairs = rag_index.retrieve(convo_history)
            example_block = "\n\n".join(
                f"Example {i+1}:\nContext: {ctx}\nCounselor response: {resp}"
                for i, (ctx, resp) in enumerate(retrieved_pairs)
            )
            system_prompt = rag_template.replace("{example_responses}", example_block)

        tag = f"{args.dataset_name}/{args.run_name}"
        lf.update_current_trace(
            tags=[tag],
            metadata={
                "dataset_name": args.dataset_name,
                "run_name": args.run_name,
                "model": args.model,
            },
        )

        client = OpenAI()
        
        req_kwargs = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": convo_history},
            ],
            "max_completion_tokens": args.max_completion_tokens,
            "temperature": args.temperature,
            "name": "copilot-response",
        }
        if "gpt-5" in args.model.lower() and "gpt-5.2" not in args.model.lower():
            req_kwargs["reasoning_effort"] = "minimal"
        else:
            req_kwargs["reasoning_effort"] = "low"
            
        response = client.chat.completions.create(**req_kwargs)
        output = response.choices[0].message.content.strip()
        print(f"  [draft] {(item.input or {}).get('uid', item.id)[:8]} → '{output[:60]}…'")
        
        # Apply Editor if enabled
        if editor_prompt_text:
            editor_input = (
                f"{editor_prompt_text.strip()}\n\n"
                f"Conversation history:\n{convo_history}\n\n"
                f"New message from counselor:\n{output}\n"
                f"Output: "
            )
            
            edit_req_kwargs = {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": editor_input},
                ],
                "max_completion_tokens": args.max_completion_tokens,
                "temperature": args.temperature,
                "name": "copilot-editor-response",
            }
            if "gpt-5" in args.model.lower() and "gpt-5.2" not in args.model.lower():
                edit_req_kwargs["reasoning_effort"] = "minimal"
            else:
                edit_req_kwargs["reasoning_effort"] = "low"
            
            edited_response = client.chat.completions.create(**edit_req_kwargs)
            output = edited_response.choices[0].message.content.strip()
            print(f"  [edited] {(item.input or {}).get('uid', item.id)[:8]} → '{output[:60]}…'")
            
        return output

    try:
        result = lf.run_experiment(
            name=args.dataset_name,
            run_name=args.run_name,
            data=dataset.items,
            task=task,
            max_concurrency=args.max_workers,
        )
    except Exception as exc:
        sys.exit(f"\n[step2] ❌ Experiment failed: {exc}\n  No data flushed to Langfuse.")

    lf.flush()
    print(
        f"\n[step2] ✅ Done. Run '{result.run_name}' complete.\n"
        f"         View at: {result.dataset_run_url}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _check_env()
    lf = Langfuse()

    # Fetch the system prompt from the Langfuse prompt library
    print(f"[step2] Fetching prompt '{args.prompt_name}' from Langfuse …")
    prompt_obj = lf.get_prompt(args.prompt_name)
    print(f"[step2] Using prompt version {prompt_obj.version}")

    # Fetch dataset items
    print(f"[step2] Fetching dataset '{args.dataset_name}' …")
    dataset = lf.get_dataset(args.dataset_name)
    print(f"[step2] {len(dataset.items)} items — run: '{args.run_name}'")

    cfg = load_config()
    gen_cfg = cfg.get("generation", {})
    
    # Optional counselor prompt file
    counselor_config_path = gen_cfg.get("counselor_config_path")
    counselor_prompt_key = gen_cfg.get("counselor_prompt")
    prompt_dictionary = {}
    if counselor_config_path and counselor_prompt_key:
        with open(counselor_config_path) as f:
            prompt_dictionary = yaml.safe_load(f)
        compiled_counselor = prompt_dictionary[counselor_prompt_key]
        print(f"[step2] Using counselor prompt '{counselor_prompt_key}' from {counselor_config_path}")
    else:
        compiled_counselor = prompt_obj.compile()

    # Optional editor
    editor_prompt_file = gen_cfg.get("editor_prompt_file")
    editor_prompt_text = None
    if editor_prompt_file:
        with open(editor_prompt_file) as f:
            editor_prompt_text = f.read()
        print(f"[step2] using editor prompt from {editor_prompt_file}")

    # Optional RAG
    rag_index = None
    rag_template = None
    if gen_cfg.get("use_rag"):
        rag_prompt_file = gen_cfg.get("rag_prompt_file")
        if not rag_prompt_file:
            sys.exit("[step2] ❌ rag_prompt_file required when use_rag is true.")
        with open(rag_prompt_file) as f:
            rag_template = f.read()

        sys.path.insert(0, str(Path(__file__).parent.parent / "dynamic_evals"))
        from rag import RAGIndex
        print("[step2] 🔍 Building RAG index from historical conversations …")
        rag_index = RAGIndex(
            csv_path=gen_cfg["historical_conversations"],
            embeddings_model=gen_cfg["rag_embeddings_model"],
            top_k=gen_cfg.get("rag_top_k", 5),
        )
        print(f"[step2] ✅ RAG index ready ({rag_index.num_documents} context–response pairs)")


    # ── Resume from existing batch? ──────────────────────────────────────
    if args.batch_id:
        print(f"\n[step2] Resuming batch {args.batch_id} …")
        manifest_path = BATCH_ARTIFACTS_DIR / f"{args.batch_id}_manifest.json"
        if not manifest_path.exists():
            editor_alt = BATCH_ARTIFACTS_DIR / f"{args.run_name}_editor_manifest.json"
            gen_alt = BATCH_ARTIFACTS_DIR / f"{args.run_name}_gen_manifest.json"
            if editor_alt.exists():
                manifest_path = editor_alt
            elif gen_alt.exists():
                manifest_path = gen_alt
            else:
                sys.exit(f"[step2] ❌ Manifest not found at {manifest_path}.")
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"[step2] Loaded manifest with {len(manifest)} requests.")
        is_editor = "edit" in list(manifest.keys())[0] if manifest else False
        client = openai.OpenAI()

        print(f"\n[step2] Polling batch {args.batch_id} (every {args.poll_interval}s) …")
        try:
            batch = poll_batch(client, args.batch_id, args.poll_interval)
        except KeyboardInterrupt:
            _print_resume_instructions(args, args.batch_id)
            return
        if batch.status != "completed":
            sys.exit(f"\n[step2] ❌ Batch ended with status '{batch.status}'.\n  Errors: {batch.errors}")

        results = download_batch_results(client, batch)

        if editor_prompt_text is not None and not is_editor:
            print(f"\n[step2] Preparing EDITOR batch requests (from resume) …")
            editor_jsonl_lines, editor_manifest = prepare_editor_batch(
                results, manifest, editor_prompt_text, args.model, 
                args.max_completion_tokens, args.temperature
            )
            print(f"[step2] {len(editor_jsonl_lines)} editor requests prepared.")
            
            e_jsonl_path   = BATCH_ARTIFACTS_DIR / f"{args.run_name}_editor_requests.jsonl"
            e_manifest_path = BATCH_ARTIFACTS_DIR / f"{args.run_name}_editor_manifest.json"

            with open(e_jsonl_path, "w") as f:
                f.write("\n".join(editor_jsonl_lines) + "\n")
            with open(e_manifest_path, "w") as f:
                json.dump(editor_manifest, f, indent=2)

            editor_batch_id = submit_batch(client, e_jsonl_path)

            with open(BATCH_ARTIFACTS_DIR / f"{editor_batch_id}_manifest.json", "w") as f:
                json.dump(editor_manifest, f, indent=2)

            print(f"\n[step2] Polling EDITOR batch {editor_batch_id} (every {args.poll_interval}s) …")
            try:
                editor_batch = poll_batch(client, editor_batch_id, args.poll_interval)
            except KeyboardInterrupt:
                _print_resume_instructions(args, editor_batch_id)
                return
            if editor_batch.status != "completed":
                sys.exit(f"\n[step2] ❌ Editor Batch ended with status '{editor_batch.status}'.\n  Errors: {editor_batch.errors}")
            
            process_generation_results(
                lf=lf, client=client, batch=editor_batch, manifest=editor_manifest,
                compiled_counselor=compiled_counselor, args=args, is_editor=True
            )
        else:
            process_generation_results(
                lf=lf, client=client, batch=batch, manifest=manifest,
                compiled_counselor=compiled_counselor, args=args, is_editor=is_editor
            )

        lf.flush()
        print(f"\n[step2] ✅ Done.\n         View run in Langfuse datasets.")
        return

    # ── Cost estimation & mode selection ─────────────────────────────────
    use_batch = args.use_batch

    if not args.yes and not use_batch:
        # Show cost comparison and ask
        batch_est = _estimate_generation_cost(
            dataset.items, compiled_counselor, args.model,
            args.max_completion_tokens, use_batch=True,
            has_editor=(editor_prompt_text is not None)
        )
        std_est = _estimate_generation_cost(
            dataset.items, compiled_counselor, args.model,
            args.max_completion_tokens, use_batch=False,
            has_editor=(editor_prompt_text is not None)
        )
        print(f"\n[step2] Cost estimate for {len(dataset.items)} items:\n")
        print("  ── Standard API ──")
        print(format_cost_summary(std_est))
        print("\n  ── Batch API (50% cheaper) ──")
        print(format_cost_summary(batch_est, show_comparison=True, alt_est=std_est))
        print()
        answer = input("[step2] Do you want to use the Batch API? (y/n): ").strip().lower()
        use_batch = answer in ("y", "yes")

    if use_batch:
        # Show batch cost and confirm
        batch_est = _estimate_generation_cost(
            dataset.items, compiled_counselor, args.model,
            args.max_completion_tokens, use_batch=True,
            has_editor=(editor_prompt_text is not None)
        )
        if not args.yes:
            print(f"\n[step2] Estimated batch cost: ${batch_est['total_cost']:.4f}")
            confirm = input(f"[step2] Given the ${batch_est['total_cost']:.4f} estimated cost, "
                            f"do you want to proceed? (y/n): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("[step2] Aborted.")
                return

        print("\n[step2] Preparing batch requests …")
        jsonl_lines, manifest = prepare_generation_batch(
            dataset.items, compiled_counselor, args.model,
            args.max_completion_tokens, args.temperature,
            rag_index=rag_index, rag_template=rag_template
        )
        print(f"[step2] {len(jsonl_lines)} generation requests prepared.")

        BATCH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_path   = BATCH_ARTIFACTS_DIR / f"{args.run_name}_gen_requests.jsonl"
        manifest_path = BATCH_ARTIFACTS_DIR / f"{args.run_name}_gen_manifest.json"

        with open(jsonl_path, "w") as f:
            f.write("\n".join(jsonl_lines) + "\n")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        client = openai.OpenAI()
        batch_id = submit_batch(client, jsonl_path)

        with open(BATCH_ARTIFACTS_DIR / f"{batch_id}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if args.submit_only:
            _print_resume_instructions(args, batch_id)
            return

        print(f"\n[step2] Polling batch {batch_id} (every {args.poll_interval}s) …")
        try:
            batch = poll_batch(client, batch_id, args.poll_interval)
        except KeyboardInterrupt:
            _print_resume_instructions(args, batch_id)
            return
        if batch.status != "completed":
            sys.exit(f"\n[step2] ❌ Batch ended with status '{batch.status}'.\n  Errors: {batch.errors}")

        results = download_batch_results(client, batch)
        
        if editor_prompt_text is not None:
            print(f"\n[step2] Preparing EDITOR batch requests …")
            editor_jsonl_lines, editor_manifest = prepare_editor_batch(
                results, manifest, editor_prompt_text, args.model, 
                args.max_completion_tokens, args.temperature
            )
            print(f"[step2] {len(editor_jsonl_lines)} editor requests prepared.")
            
            e_jsonl_path   = BATCH_ARTIFACTS_DIR / f"{args.run_name}_editor_requests.jsonl"
            e_manifest_path = BATCH_ARTIFACTS_DIR / f"{args.run_name}_editor_manifest.json"

            with open(e_jsonl_path, "w") as f:
                f.write("\n".join(editor_jsonl_lines) + "\n")
            with open(e_manifest_path, "w") as f:
                json.dump(editor_manifest, f, indent=2)

            editor_batch_id = submit_batch(client, e_jsonl_path)

            with open(BATCH_ARTIFACTS_DIR / f"{editor_batch_id}_manifest.json", "w") as f:
                json.dump(editor_manifest, f, indent=2)

            print(f"\n[step2] Polling EDITOR batch {editor_batch_id} (every {args.poll_interval}s) …")
            try:
                editor_batch = poll_batch(client, editor_batch_id, args.poll_interval)
            except KeyboardInterrupt:
                _print_resume_instructions(args, editor_batch_id)
                return
            if editor_batch.status != "completed":
                sys.exit(f"\n[step2] ❌ Editor Batch ended with status '{editor_batch.status}'.\n  Errors: {editor_batch.errors}")
            
            process_generation_results(
                lf=lf, client=client, batch=editor_batch, manifest=editor_manifest,
                compiled_counselor=compiled_counselor, args=args, is_editor=True
            )
        else:
            process_generation_results(
                lf=lf, client=client, batch=batch, manifest=manifest,
                compiled_counselor=compiled_counselor, args=args, is_editor=False
            )
        
        lf.flush()
        print(f"\n[step2] ✅ Done.\n         View run in Langfuse datasets.")
    else:
        # Standard path
        if not args.yes:
            std_est = _estimate_generation_cost(
                dataset.items, prompt_obj, args.model,
                args.max_completion_tokens, use_batch=False,
            )
            print(f"\n[step2] Estimated standard API cost: ${std_est['total_cost']:.4f}")
            confirm = input(f"[step2] Given the ${std_est['total_cost']:.4f} estimated cost, "
                            f"do you want to proceed? (y/n): ").strip().lower()
            if confirm not in ("y", "yes"):
                print("[step2] Aborted.")
                return

        run_standard(lf, args, dataset, compiled_counselor, editor_prompt_text, rag_index, rag_template)


if __name__ == "__main__":
    main()
