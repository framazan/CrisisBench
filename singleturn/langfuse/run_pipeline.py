"""
run_pipeline.py – Run the full static-evals Langfuse pipeline end to end.

Steps executed:
  0. Upload prompts to the Langfuse prompt library
  1. Upload the dataset (CSV → Langfuse dataset items)
  2. Run generation (copilot responses logged as Langfuse experiment run)
  3. Run eval & scoring (LLM judge, scores attached to each trace)
  4. Create analysis dashboard  [optional, requires --email / --password]

Both steps 2 and 3 support standard (synchronous) and Batch API modes.
When run interactively, each step will ask whether to use the Batch API
and confirm before spending money. Pass --yes to skip all prompts.

Usage
-----
  # Full pipeline from a raw dialogue CSV:
  python run_pipeline.py \\
      --input /path/to/raw_convos.csv \\
      --convert \\
      --dataset-name static-evals \\
      --run-name copilot-v1 \\
      --model gpt-4.1 \\
      --email admin@example.com --password yourpassword

  # Skip prompt upload (already done) and dashboard creation:
  python run_pipeline.py \\
      --input static_eval_input.csv \\
      --dataset-name static-evals \\
      --run-name copilot-v2 \\
      --model gpt-4.1 \\
      --skip-prompts --skip-dashboard

  # Force batch API for both steps and skip confirmations:
  python run_pipeline.py \\
      --input data.csv \\
      --dataset-name static-evals \\
      --run-name copilot-v1 \\
      --model gpt-4.1 \\
      --use-batch --yes

  # Resume from step 2 (dataset already uploaded):
  python run_pipeline.py \\
      --input ignored \\
      --dataset-name static-evals \\
      --run-name copilot-v1 \\
      --model gpt-4.1 \\
      --start-from 2 --skip-dashboard

  # Resume a batch that was submitted for step 2:
  python run_pipeline.py \\
      --dataset-name static-evals \\
      --run-name copilot-v1 \\
      --model gpt-4.1 \\
      --start-from 2 --use-batch \\
      --batch-id-step2 batch_abc123

  # Resume a batch that was submitted for step 3:
  python run_pipeline.py \\
      --dataset-name static-evals \\
      --run-name copilot-v1 \\
      --model gpt-4.1 \\
      --start-from 3 --use-batch \\
      --batch-id-step3 batch_def456

Environment variables (set in ~/.bashrc)
-----------------------------------------
  LANGFUSE_SECRET_KEY   – Langfuse secret key
  LANGFUSE_PUBLIC_KEY   – Langfuse public key
  LANGFUSE_BASE_URL     – e.g. http://localhost:3000
  OPENAI_API_KEY        – for the generation and eval LLM calls
  LANGFUSE_ADMIN_EMAIL  – for dashboard creation (step 4)
  LANGFUSE_ADMIN_PASSWORD
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def run_step(script: str, extra_args: list[str], step_num: int, step_name: str) -> None:
    cmd = [sys.executable, str(HERE / script)] + extra_args
    print(f"\n{'='*60}")
    print(f"  STEP {step_num}: {step_name}")
    print(f"{'='*60}")
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[pipeline] ❌ Step {step_num} failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"\n[pipeline] ✅ Step {step_num} complete.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full static-evals Langfuse pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    p.add_argument("--input", default="",
                   help="Path to input CSV (required for steps 0-1)")
    p.add_argument("--convert", action="store_true",
                   help="Convert raw dialogue CSV before uploading")
    p.add_argument("--conversation-column", default="conversation")
    p.add_argument("--filter-messages", type=int, default=5)
    p.add_argument("--dataset-name", required=True,
                   help="Langfuse dataset name")

    # ── Experiment ─────────────────────────────────────────────────────────
    p.add_argument("--run-name", required=True,
                   help="Langfuse experiment run name")
    p.add_argument("--model", required=True,
                   help="Generation model (e.g. gpt-4.1, gpt-4o-mini)")

    # ── Batch API / confirmation ───────────────────────────────────────────
    p.add_argument("--use-batch", action="store_true",
                   help="Force Batch API for steps 2 and 3 (otherwise interactive)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip all interactive confirmation prompts")
    p.add_argument("--batch-id-step2", default=None,
                   help="Resume step 2 with an already-submitted batch ID")
    p.add_argument("--batch-id-step3", default=None,
                   help="Resume step 3 with an already-submitted batch ID")

    # ── Flow control ───────────────────────────────────────────────────────
    p.add_argument("--start-from", type=int, default=0,
                   help="Start from step N (0=prompts, 1=dataset, 2=gen, 3=eval, 4=dashboard)")
    p.add_argument("--skip-prompts", action="store_true",
                   help="Skip step 0 (prompts already uploaded)")
    p.add_argument("--skip-dashboard", action="store_true",
                   help="Skip step 4 (dashboard creation)")
    p.add_argument("--force-prompts", action="store_true",
                   help="Re-upload prompts even if text is unchanged")

    # ── Dashboard auth ─────────────────────────────────────────────────────
    p.add_argument("--email", default="",
                   help="Langfuse admin email for dashboard creation")
    p.add_argument("--password", default="",
                   help="Langfuse admin password for dashboard creation")
    p.add_argument("--dashboard-name", default="Static Evals Analysis")

    return p.parse_args()


def main():
    args = parse_args()
    start = args.start_from

    print("\n[pipeline] 🚀 Starting static-evals Langfuse pipeline")
    print(f"[pipeline]    Dataset  : {args.dataset_name}")
    print(f"[pipeline]    Run name : {args.run_name}")
    print(f"[pipeline]    Start at : step {start}")

    # ── Step 0: Upload prompts ──────────────────────────────────────────────
    if start <= 0 and not args.skip_prompts:
        flags = ["--force"] if args.force_prompts else []
        run_step("step0_upload_prompts.py", flags, 0, "Upload prompts to Langfuse")
    else:
        print("\n[pipeline] Skipping step 0 (prompts)")

    # ── Step 1: Upload dataset ──────────────────────────────────────────────
    if start <= 1:
        if not args.input:
            sys.exit("[pipeline] --input is required for step 1.")
        flags = [
            "--input", args.input,
            "--dataset-name", args.dataset_name,
        ]
        if args.convert:
            flags += [
                "--convert",
                "--conversation-column", args.conversation_column,
                "--filter-messages", str(args.filter_messages),
            ]
        run_step("step1_upload_dataset.py", flags, 1, "Upload dataset to Langfuse")
        # Pick up the name step1 settled on (user may have renamed it)
        _state = HERE / "__pipeline_state.txt"
        if _state.exists():
            resolved = _state.read_text().strip()
            if resolved and resolved != args.dataset_name:
                print(f"[pipeline]    Dataset name resolved to: '{resolved}'")
                args.dataset_name = resolved
            _state.unlink(missing_ok=True)
    else:
        print("\n[pipeline] Skipping step 1 (dataset)")

    # ── Step 2: Run generation ──────────────────────────────────────────────
    if start <= 2:
        flags = [
            "--dataset-name", args.dataset_name,
            "--run-name", args.run_name,
            "--model", args.model,
        ]
        if args.use_batch:
            flags.append("--use-batch")
        if args.yes:
            flags.append("--yes")
        if args.batch_id_step2:
            flags += ["--batch-id", args.batch_id_step2]
        run_step("step2_run_generation.py", flags, 2, "Run copilot generation experiment")
    else:
        print("\n[pipeline] Skipping step 2 (generation)")

    # ── Step 3: Run eval ──────────────────────────────────────────────────────────
    if start <= 3:
        if start <= 2:
            print("\n[pipeline] Waiting 15 seconds before starting Step 3 for Langfuse processing...")
            time.sleep(10)
            
        flags = [
            "--dataset-name", args.dataset_name,
            "--run-name", args.run_name,
        ]
        if args.use_batch:
            flags.append("--use-batch")
        if args.yes:
            flags.append("--yes")
        if args.batch_id_step3:
            flags += ["--batch-id", args.batch_id_step3]
        run_step("step3_run_eval.py", flags, 3, "Run LLM eval and attach scores")
    else:
        print("\n[pipeline] Skipping step 3 (eval)")

    # ── Step 4: Create dashboard ────────────────────────────────────────────
    if start <= 4 and not args.skip_dashboard:
        flags = ["--dashboard-name", args.dashboard_name]
        if args.email:
            flags += ["--email", args.email]
        if args.password:
            flags += ["--password", args.password]
        run_step("step4_create_dashboard.py", flags, 4, "Create analysis dashboard")
    else:
        print("\n[pipeline] Skipping step 4 (dashboard)")

    print("\n[pipeline] 🎉 All steps complete!")
    print(f"[pipeline]    View results at: $LANGFUSE_BASE_URL/datasets/{args.dataset_name}")


if __name__ == "__main__":
    main()
