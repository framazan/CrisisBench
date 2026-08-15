"""
Step 2 – Run LLM-judge evaluations on the generated dialogues.

Uses lf.run_experiment() to create a named experiment run on the dataset.
For each dataset item (patient profile + linked generation trace):
  1. Tags the run trace as "{dataset_name}/{run_name}".
  2. Loads the full conversation from the generation trace output.
  3. Calls an OpenAI LLM judge (via the Langfuse-traced OpenAI wrapper) with
      each configured eval prompt.
  4. Posts numeric scores to BOTH:
         - the current run trace (so scores are visible on dataset run items), and
         - the source generation trace (so conversation traces carry eval scores).

The task output returned to the dataset run is the full conversation text.

Score naming convention:  "{eval_name}.{metric_key}"
  Examples:
    language_patterns.grammar_and_punctuation
    presenting_concern.score
    emotional_realism.cognitive_distortions

Usage
-----
  python step2_run_eval.py --dataset-name dynamic-evals-v1 --run-name eval-run-1
  python step2_run_eval.py --dataset-name dynamic-evals-v1 --run-name eval-run-1 \\
      --prompts language_patterns presenting_concern
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import yaml
from langfuse import Langfuse
from langfuse.openai import OpenAI  # type: ignore[attr-defined]

try:
    from json_repair import repair_json  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    repair_json = None


def _check_env() -> None:
    missing = [v for v in ("OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"[step2] Missing environment variable(s): {', '.join(missing)}\n"
            "  Export them before running."
        )


# helpers

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_dynamic_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


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


def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Run LLM-judge evals on generated dialogues")
    p.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    p.add_argument("--run-name",     required=True, help="Experiment run name")
    p.add_argument("--model",        default=cfg["eval"]["model"])
    p.add_argument("--max-completion-tokens", type=int,
                   default=cfg["eval"]["max_completion_tokens"])
    p.add_argument("--temperature",  type=float, default=cfg["eval"]["temperature"])
    p.add_argument("--max-workers",  type=int,   default=cfg["max_workers"])
    p.add_argument("--prompts",      nargs="+",  default=None,
                   help="Subset of eval prompt keys to run (default: all from config)")
    p.add_argument(
        "--all-items",
        action="store_true",
        help=(
            "Evaluate all dataset items. By default, only items generated in the "
            "current run-name are evaluated (metadata.generation_run_name == run-name)."
        ),
    )
    return p.parse_args(), cfg


def parse_json_with_repair(raw: str) -> Optional[dict]:
    """Parse JSON, then fall back to json_repair if needed."""
    parsed = extract_json_block(raw)
    if parsed is not None:
        return parsed

    if repair_json is None:
        return None

    try:
        repaired = repair_json(raw)
    except Exception:
        return None

    if isinstance(repaired, dict):
        return repaired

    if isinstance(repaired, str):
        parsed = extract_json_block(repaired)
        if parsed is not None:
            return parsed
        try:
            obj = json.loads(repaired)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


# main

def main():
    args, cfg = parse_args()
    _check_env()
    lf = Langfuse()

    # Cache source-trace lookups to avoid repeated API calls for legacy items.
    trace_run_name_cache: dict[str, Optional[str]] = {}

    def _trace_run_name(source_trace_id: Optional[str]) -> Optional[str]:
        if not source_trace_id:
            return None
        if source_trace_id in trace_run_name_cache:
            return trace_run_name_cache[source_trace_id]
        try:
            src = lf.api.trace.get(source_trace_id)
            md = getattr(src, "metadata", None) or {}
            run_name_val = md.get("run_name") if isinstance(md, dict) else None
            trace_run_name_cache[source_trace_id] = run_name_val
            return run_name_val
        except Exception:
            trace_run_name_cache[source_trace_id] = None
            return None

    # Load eval prompts from the YAML file on disk
    eval_prompts_path = cfg["patient_eval_prompts_file"]
    with open(eval_prompts_path) as f:
        all_eval_prompts_raw: dict = yaml.safe_load(f)

    # Determine which eval prompts to run
    requested_prompts = args.prompts or cfg["eval"].get("prompts") or list(all_eval_prompts_raw.keys())
    eval_prompts: dict = {}
    for key in requested_prompts:
        if key in all_eval_prompts_raw:
            entry = all_eval_prompts_raw[key]
            eval_prompts[key] = entry["prompt"] if isinstance(entry, dict) else str(entry)
            print(f"[step2] Loaded eval prompt '{key}'")
        else:
            print(f"[step2] Eval prompt '{key}' not found in {eval_prompts_path}")

    if cfg.get("eval", {}).get("realism_eval", False):
        realism_prompts_path = cfg.get("realism_eval_prompts_file")
        if realism_prompts_path and os.path.exists(realism_prompts_path):
            with open(realism_prompts_path) as f:
                realism_prompts_raw = yaml.safe_load(f)
            for key in ["language_patterns", "presenting_concern", "emotional_realism"]:
                if key in realism_prompts_raw:
                    entry = realism_prompts_raw[key]
                    eval_prompts[key] = entry["prompt"] if isinstance(entry, dict) else str(entry)
                    print(f"[step2] Loaded realism eval prompt '{key}'")

    if not eval_prompts:
        sys.exit("[step2] No eval prompts loaded -- aborting.")

    # Fetch dataset items
    print(f"\n[step2] Fetching dataset '{args.dataset_name}' ...")
    dataset = lf.get_dataset(args.dataset_name)

    dataset_items = list(dataset.items)
    total_items = len(dataset_items)
    if args.all_items:
        eval_items = dataset_items
        print(
            f"[step2] {total_items} items -- run: '{args.run_name}' "
            f"(all-items mode)\n"
        )
    else:
        eval_items = []
        matched_by_item_metadata = 0
        matched_by_source_trace = 0

        for item in dataset_items:
            metadata = item.metadata or {}
            item_run_name = metadata.get("generation_run_name")

            if item_run_name is not None:
                if item_run_name == args.run_name:
                    eval_items.append(item)
                    matched_by_item_metadata += 1
                continue

            # Legacy compatibility: infer run from source trace metadata.
            source_trace_id = (
                metadata.get("generation_trace_id")
                or getattr(item, "source_trace_id", None)
            )
            if _trace_run_name(source_trace_id) == args.run_name:
                eval_items.append(item)
                matched_by_source_trace += 1

        if not eval_items:
            sys.exit(
                f"[step2] No dataset items matched run '{args.run_name}'. "
                "Refusing to evaluate all items by default. "
                "If intentional, pass --all-items."
            )

        print(
            f"[step2] {total_items} items in dataset; evaluating {len(eval_items)} "
            f"item(s) for run '{args.run_name}' "
            f"({matched_by_item_metadata} via item metadata, "
            f"{matched_by_source_trace} via source trace).\n"
        )

    # Capture args for use inside the task closure
    dataset_name          = args.dataset_name
    run_name              = args.run_name
    model                 = args.model
    max_completion_tokens = args.max_completion_tokens
    temperature           = args.temperature

    # Task function: called once per dataset item by run_experiment()
    def task(*, item, **kwargs) -> str:
        profile_key    = (item.input or {}).get("patient_profile_key", item.id)
        profile_prompt = (item.input or {}).get("patient_profile_prompt", "")
        category       = (item.input or {}).get("category", "")
        source_trace_id = (
            (item.metadata or {}).get("generation_trace_id")
            or getattr(item, "source_trace_id", None)
        )
        tag            = f"{dataset_name}/{run_name}"

        # Resolve full conversation from the source generation trace.
        conversation = ""
        if source_trace_id:
            import time
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    source_trace = lf.api.trace.get(source_trace_id)
                    source_output = source_trace.output
                    if isinstance(source_output, str):
                        conversation = source_output
                    elif source_output is not None:
                        conversation = json.dumps(source_output)
                    break # Success!
                except Exception as exc:
                    if attempt < max_retries - 1:
                        time.sleep(15) # Wait for Langfuse ingestion
                    else:
                        print(f"  [warn] Could not fetch source trace for '{profile_key}' after retries: {exc}")

        if not conversation:
            # Backward compatibility for previously uploaded datasets.
            conversation = (item.input or {}).get("conversation", "")

        # Tag this eval trace and attach metadata
        lf.update_current_trace(
            name="counselor-eval-rubric",
            tags=[tag],
            metadata={
                "dataset_name": dataset_name,
                "run_name":     run_name,
                "profile_key":  profile_key,
                "category":     category,
                "source_generation_trace_id": source_trace_id,
            },
        )

        client     = OpenAI()   # Langfuse-traced OpenAI wrapper
        all_scores: dict = {}

        for eval_name, base_prompt in eval_prompts.items():
            rubric_prompt = base_prompt

            if eval_name == "presenting_concern":
                match = re.search(r"(Presenting Concern:.*?)\n\*\*\* Communication patterns \*\*\*", profile_prompt, re.DOTALL)
                if match:
                    rubric_prompt += f"\n\nPatient Profile Details:\n{match.group(1).strip()}"

            elif eval_name == "language_patterns":
                match = re.search(r"(\*\*\* Language examples \*\*\*.*?)(?:MUST FOLLOW THESE GUIDELINES:|\Z)", profile_prompt, re.DOTALL)
                if match:
                    rubric_prompt += f"\n\n{match.group(1).strip()}"

            lf.update_current_trace(name=eval_name, input=rubric_prompt)
            try:
                req_kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": rubric_prompt},
                        {"role": "user", "content": conversation},
                    ],
                    "response_format": {"type": "json_object"},
                    "max_completion_tokens": max_completion_tokens,
                    "temperature": temperature,
                    "name": f"eval-{eval_name}",
                }
                # GPT-5 models can spend completion budget on reasoning and
                # return empty visible text; constrain reasoning effort.
                if "gpt-5" in model.lower():
                    req_kwargs["reasoning_effort"] = "low"

                resp = client.chat.completions.create(**req_kwargs) # type: ignore
                raw = resp.choices[0].message.content.strip()
            except Exception as exc:
                print(f"  [warn] LLM call failed for eval '{eval_name}' on '{profile_key}': {exc}")
                continue

            parsed = parse_json_with_repair(raw)
            if not parsed:
                print(f"  [warn] Could not parse JSON for '{eval_name}' on '{profile_key}': {raw[:120]}")
                continue

            lf.update_current_trace(
                output=json.dumps(parsed, ensure_ascii=True, indent=2),
            )

            if "score" in parsed and not isinstance(parsed["score"], dict):
                parsed_metrics = {"score": parsed}
            else:
                parsed_metrics = parsed

            for metric_key, metric_val in parsed_metrics.items():
                if isinstance(metric_val, dict):
                    score_value   = metric_val.get("score")
                    justification = str(metric_val.get("rationale", metric_val.get("justification", "")))
                else:
                    try:
                        score_value = float(metric_val)
                    except (TypeError, ValueError):
                        continue
                    
                    group = metric_key[0] if metric_key else ""
                    just_key = f"{group}_justification"
                    justification = str(parsed_metrics.get(just_key, ""))

                if score_value is None:
                    continue

                score_name = (
                    f"{eval_name}.{metric_key}"
                    if not metric_key.startswith(eval_name)
                    else metric_key
                )

                lf.score_current_trace(
                    name=score_name,
                    value=float(score_value),
                    comment=justification[:1000] if justification else None,
                    data_type="NUMERIC",
                )

                # Also attach to the source generation trace so conversation
                # traces hold their own eval outcomes.
                if source_trace_id:
                    lf.create_score(
                        trace_id=source_trace_id,
                        name=score_name,
                        value=float(score_value),
                        comment=justification[:1000] if justification else None,
                        data_type="NUMERIC",
                        metadata={
                            "dataset_name": dataset_name,
                            "run_name": run_name,
                            "profile_key": profile_key,
                        },
                    )

                all_scores[score_name] = float(score_value)

        short = {k.split(".")[-1]: v for k, v in list(all_scores.items())[:4]}
        print(f"  [ok] {profile_key}: {short}")
        return conversation

    # Run the experiment
    t0 = time.time()
    try:
        result = lf.run_experiment(
            name=dataset_name,
            run_name=run_name,
            data=eval_items,
            task=task,
            max_concurrency=args.max_workers,
        )
    except Exception as exc:
        sys.exit(f"\n[step2] Experiment failed: {exc}\n  No data flushed to Langfuse.")

    lf.flush()
    elapsed = time.time() - t0
    base_url = os.environ.get("LANGFUSE_BASE_URL", "")
    run_url  = getattr(result, "dataset_run_url",
                       f"{base_url}/datasets/{dataset_name}/runs/{run_name}")
    print(
        f"\n[step2] Done in {elapsed:.1f}s.\n"
        f"         View run: {run_url}"
    )


if __name__ == "__main__":
    main()
