"""
run_pipeline.py – Run the full dynamic-evals Langfuse pipeline end to end.

Steps executed:
  1. Generate synthetic patient–counselor dialogues and upload them as a
     Langfuse dataset (one item per patient profile).
  2. Run LLM-judge evaluation on every dialogue; post scores onto each trace.

The dataset name is used as the Langfuse tag on generation traces (step 1).
Evaluation traces (step 2) are tagged as "{dataset_name}/{run_name}".

Usage
-----
  # Full pipeline – run all profiles matching "financial_loss":
  python run_pipeline.py \\
      --dataset-name dynamic-evals-v1 \\
      --run-name eval-run-1

  # Override which profiles to generate (category or exact key):
  python run_pipeline.py \\
      --dataset-name dynamic-evals-v1 \\
      --run-name eval-run-1 \\
      --patient-prompts anxiety panic_attack_14

  # Skip dialogue generation (dataset already uploaded), only run eval:
  python run_pipeline.py \\
      --dataset-name dynamic-evals-v1 \\
      --run-name eval-run-2 \\
      --start-from 2

Required environment variables (OPENAI_API_KEY + Langfuse keys):
  OPENAI_API_KEY         – for dialogue generation and LLM-judge eval calls
  LANGFUSE_SECRET_KEY    – Langfuse secret key
  LANGFUSE_PUBLIC_KEY    – Langfuse public key
  LANGFUSE_BASE_URL      – e.g. http://localhost:3000
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run_step(script: str, extra_args: list, step_num: int, step_name: str) -> None:
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
        description="Run the full dynamic-evals Langfuse pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core ───────────────────────────────────────────────────────────────
    p.add_argument("--dataset-name", required=True,
                   help="Langfuse dataset name (created in step 1)")
    p.add_argument("--run-name", required=True,
                   help="Langfuse experiment run name for the eval step (step 2)")

    # ── Patient profile selection ──────────────────────────────────────────
    p.add_argument("--patient-prompts", nargs="+", default=None,
                   help="Override config patient_prompts – accepts exact keys "
                        "(e.g. 'financial_loss_0') or category prefixes "
                        "(e.g. 'financial_loss') or 'all'")

    # ── Eval overrides ─────────────────────────────────────────────────────
    p.add_argument("--eval-prompts", nargs="+", default=None,
                   help="Subset of eval prompt keys to run in step 2 "
                        "(default: all from config)")
    p.add_argument("--eval-model", default=None,
                   help="Override eval model (default: from config)")

    # ── Flow control ───────────────────────────────────────────────────────
    p.add_argument("--start-from", type=int, default=1,
                   help="Start from step N (1=generate+upload, 2=eval)")
    p.add_argument(
        "--on-existing-dataset",
        choices=["prompt", "generate", "reuse"],
        default="generate",
        help=(
            "Step 1 behavior when dataset already exists: 'generate' regenerates "
            "and upserts selected profiles, 'reuse' skips step 1 work, 'prompt' asks"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()
    start = args.start_from

    print("\n[pipeline] 🚀 Starting dynamic-evals Langfuse pipeline")
    print(f"[pipeline]    Dataset  : {args.dataset_name}")
    print(f"[pipeline]    Run name : {args.run_name}")
    print(f"[pipeline]    Start at : step {start}")

    # ── Step 1: Generate dialogues + upload dataset ────────────────────────
    if start <= 1:
        flags = [
            "--dataset-name", args.dataset_name,
            "--run-name", args.run_name,
            "--on-existing-dataset", args.on_existing_dataset,
        ]
        if args.patient_prompts:
            flags += ["--patient-prompts"] + args.patient_prompts
        run_step("step1_generate_and_upload.py", flags, 1,
                 "Generate dialogues & upload to Langfuse dataset")

        # Pick up the resolved dataset name if the user renamed it interactively
        _state = HERE / "__pipeline_state.txt"
        if _state.exists():
            resolved = _state.read_text().strip()
            if resolved and resolved != args.dataset_name:
                print(f"[pipeline]    Dataset name resolved to: '{resolved}'")
                args.dataset_name = resolved
            _state.unlink(missing_ok=True)
    else:
        print("\n[pipeline] Skipping step 1 (generate + upload)")

    # ── Step 2: Run LLM eval ───────────────────────────────────────────────
    if start <= 2:
        flags = [
            "--dataset-name", args.dataset_name,
            "--run-name",     args.run_name,
        ]
        if args.eval_prompts:
            flags += ["--prompts"] + args.eval_prompts
        if args.eval_model:
            flags += ["--model", args.eval_model]
        run_step("step2_run_eval.py", flags, 2,
                 "Run LLM-judge eval & attach scores to traces")
    else:
        print("\n[pipeline] Skipping step 2 (eval)")

    print("\n[pipeline] 🎉 All steps complete!")
    import os
    base_url = os.environ.get("LANGFUSE_BASE_URL", "")
    print(f"[pipeline]    View dataset : {base_url}/datasets/{args.dataset_name}")
    print(f"[pipeline]    View run     : {base_url}/datasets/{args.dataset_name}/runs/{args.run_name}")


if __name__ == "__main__":
    main()
