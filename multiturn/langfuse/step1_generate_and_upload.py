"""
Step 1 – Generate synthetic dialogues and upload them as a Langfuse dataset.

For each resolved patient profile:
  1. Generates a counselor ↔ patient dialogue with **every LLM call** traced
     individually as a Langfuse generation (patient turns, counselor turns,
     optional editor / phase-selection calls).
  2. Creates a Langfuse trace per dialogue, tagged with the dataset name.
  3. Uploads a dataset item:
       input  = { patient_profile_key, patient_profile_prompt, conversation }
       expected_output = "" (no fixed reference; the conversation IS the artefact)

Dialogues are generated in parallel via ThreadPoolExecutor (``max_workers``
from config.yaml).

Patient-prompt resolution:
  - Exact key  (e.g. "financial_loss_3")  → runs only that profile
  - Category   (e.g. "financial_loss")    → runs all financial_loss_* profiles
  - "all"                                 → runs every profile in the YAML

Usage
-----
  python step1_generate_and_upload.py --dataset-name dynamic-evals-v1
  python step1_generate_and_upload.py --dataset-name dynamic-evals-v1 \\
      --patient-prompts anxiety panic_attack_14
"""

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from langfuse import Langfuse, get_client, observe

# ── vf_copilot path setup ─────────────────────────────────────────────────────
_VF_ROOT = Path("/home/stanford/CrisisBench")
if str(_VF_ROOT) not in sys.path:
    sys.path.insert(0, str(_VF_ROOT))

from singleturn.clients.llm_client import chat_complete  # noqa: E402
from multiturn.agents import load_prompts     # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────

COUNSELOR_PROMPT_SUFFIX = (
    "You can (optionally) send multiple messages in a row, separated by the "
    "delimiter '//'. Maintain variety in message lengths and number of messages "
    "per counselor reply\n"
    'If you want to end the conversation, type "[end conversation]".'
)


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_dynamic_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def expand_patient_prompts(requested: list[str], all_keys: list[str]) -> list[str]:
    """Resolve each entry (exact key, category prefix, or 'all') into concrete keys."""
    result: list[str] = []
    for req in requested:
        if req == "all":
            result.extend(all_keys)
        elif req in all_keys:
            result.append(req)
        else:
            # Treat as category prefix: match anything that starts with req + "_"
            matches = [k for k in all_keys if k.startswith(req + "_")]
            if matches:
                result.extend(sorted(matches))
            else:
                print(f"[step1] ⚠️  No profiles found for '{req}' – skipping.")
    # Preserve order, remove duplicates
    seen: set[str] = set()
    deduped: list[str] = []
    for k in result:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped


def _print_dialogue_line(profile_key: str, role: str, message: str) -> None:
    """Print one dialogue line with a stable, profile-prefixed format."""
    clean = (message or "").strip()
    if not clean:
        return
    print(f"[ok] [{profile_key}] {role}: {clean}")


def check_dataset_name(
    lf: Langfuse,
    requested_name: str,
    on_existing_dataset: str = "prompt",
) -> tuple[str, bool]:
    """Resolve dataset name conflict.

    Returns:
      (final_dataset_name, should_add_items)

        Behavior when dataset already exists:
            - generate -> continue and generate/upsert selected profiles
            - reuse    -> reuse existing dataset without generation
            - prompt   -> ask interactively (Enter defaults to generate/upsert)
    """
    try:
        existing = lf.get_dataset(requested_name)
        item_count = len(existing.items)
    except Exception:
        existing = None
        item_count = 0

    if existing is None:
        return requested_name, True

    if on_existing_dataset == "reuse":
        print(f"[step1] Reusing dataset '{requested_name}' without generating new dialogues.")
        return requested_name, False

    if on_existing_dataset == "generate":
        print(
            f"[step1] Dataset '{requested_name}' exists ({item_count} items). "
            "Will generate and upsert selected profiles."
        )
        return requested_name, True

    print(
        f"\n[step1] ⚠️  Dataset '{requested_name}' already exists "
        f"({item_count} items).\n"
        f"  • Press Enter to GENERATE/UPSERT selected profiles\n"
        f"  • Type 'reuse' to SKIP generation and reuse as-is\n"
        f"  • Or type a new dataset name to create a fresh one: ",
        end="", flush=True,
    )
    user_input = input().strip()
    if not user_input or user_input.lower() in {"generate", "regen", "regenerate"}:
        print(f"[step1] Continuing with existing dataset '{requested_name}'.")
        return requested_name, True
    if user_input.lower() == "reuse":
        print(f"[step1] Reusing dataset '{requested_name}' without adding new items.")
        return requested_name, False
    return check_dataset_name(lf, user_input, on_existing_dataset)


def _stable_dataset_item_id(dataset_name: str, profile_key: str) -> str:
    raw = f"dynamic-evals::{dataset_name}::{profile_key}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ─── Traced LLM call ─────────────────────────────────────────────────────────

@observe(as_type="generation")
def _traced_llm_call(
    *,
    call_name: str,
    role: str,
    model: str,
    api_url: str,
    api_key_env: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    reasoning_effort: str = None,
) -> str:
    """Single LLM call, posted to Langfuse as a generation."""
    lf = get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    lf.update_current_generation(
        name=call_name,
        model=model,
        input=messages,
        model_parameters={"temperature": temperature},
        metadata={"role": role},
    )

    chat_kwargs = {
        "model": model,
        "api_url": api_url,
        "api_key_env": api_key_env,
        "temperature": temperature,
        "max_tokens": max_completion_tokens,
        "messages": messages,
    }
    if reasoning_effort:
        chat_kwargs["reasoning_effort"] = reasoning_effort

    result = chat_complete(**chat_kwargs)

    lf.update_current_generation(output=result)
    return result


# ─── Dialogue generation with full tracing ────────────────────────────────────

@observe(capture_input=False)
def _generate_and_upload_one(
    *,
    lf_client: Langfuse,
    cfg: dict,
    profile_key: str,
    profile_prompt: str,
    dataset_name: str,
    run_name: str,
    category: str,
    counselor_prompts: dict,
    editor_prompts: dict,
    rag_index=None,
) -> dict:
    """Generate one dialogue with every LLM call traced, then upload."""
    d = cfg["dialogue"]
    lf = get_client()

    tag = f"{dataset_name}/{run_name}"
    lf.update_current_trace(
        name="generated-conversation",
        tags=[tag],
        metadata={
            "dataset_name": dataset_name,
            "run_name": run_name,
            "profile_key":  profile_key,
            "category":     category,
        },
    )

    # ── Resolve prompt templates (once per dialogue) ──────────────────────
    counselor_prompt_template = (
        counselor_prompts[d["counselor_prompt"]] + COUNSELOR_PROMPT_SUFFIX
    )
    patient_prompt_template = profile_prompt  # profile text IS the system prompt

    editor_key = d.get("editor_prompt")
    editor_prompt_template = None
    if editor_key:
        raw = counselor_prompts.get(editor_key)
        if raw:
            editor_prompt_template = raw + COUNSELOR_PROMPT_SUFFIX

    patient_editor_key = d.get("patient_editor_prompt")
    patient_editor_prompt_template = (
        editor_prompts.get(patient_editor_key) if patient_editor_key else None
    )

    phase_selection_key = d.get("phase_selection_prompt")
    phase_selection_template = (
        counselor_prompts.get(phase_selection_key) if phase_selection_key else None
    )

    # ── Dialogue loop ─────────────────────────────────────────────────────
    conversation_history: list[str] = []
    initial_msg = d["initial_counselor_message"]
    conversation_history.append(f"counselor: {initial_msg}")
    _print_dialogue_line(profile_key, "counselor", initial_msg)

    max_messages = d["max_messages"]
    turn = 0

    try:
        while (
            not any("[end conversation]" in line.lower() for line in conversation_history)
            and len(conversation_history) < max_messages * 2
        ):
            turn += 1
            conv_text = "\n".join(conversation_history)

            # ── Patient turn ──────────────────────────────────────────────
            patient_draft = _traced_llm_call(
                call_name=f"patient-turn-{turn}",
                role="patient",
                model=d["patient_model"],
                api_url=d["patient_api_url"],
                api_key_env=d["patient_api_key_env"],
                temperature=d["patient_temperature"],
                system_prompt=patient_prompt_template,
                user_prompt=conv_text,
                max_completion_tokens=d['max_patient_tokens'],
                reasoning_effort=d.get("patient_reasoning_effort")
            )

            if patient_editor_prompt_template:
                editor_input = (
                    patient_editor_prompt_template
                    .replace("{conversation_history}", conv_text)
                    .replace("{message_draft}", patient_draft)
                )
                patient_draft = _traced_llm_call(
                    call_name=f"patient-editor-turn-{turn}",
                    role="patient-editor",
                    model=d["patient_model"],
                    api_url=d["patient_api_url"],
                    api_key_env=d["patient_api_key_env"],
                    temperature=d["patient_temperature"],
                    system_prompt="",
                    user_prompt=editor_input,
                    max_completion_tokens=d['max_patient_tokens'],
                    reasoning_effort=d.get("patient_reasoning_effort")
                )

            for msg in patient_draft.split("//"):
                conversation_history.append(f"patient: {msg}")
                conversation_history.append("\n")
                _print_dialogue_line(profile_key, "patient", msg)

            if "[end conversation]" in patient_draft.lower():
                break

            conv_text = "\n".join(conversation_history)

            # ── Counselor turn ────────────────────────────────────────────
            effective_counselor_prompt = counselor_prompt_template
            phase_selected = None

            # ── RAG augmentation (every turn, draft prompt only) ──────
            if rag_index is not None:
                retrieved_pairs = rag_index.retrieve(conv_text)
                example_block = "\n\n".join(
                    f"Example {i+1}:\nContext: {ctx}\nCounselor response: {resp}"
                    for i, (ctx, resp) in enumerate(retrieved_pairs)
                )
                rag_prompt_key = d.get(
                    "counselor_rag_prompt", "rag_dynamic_instructions"
                )
                rag_template = counselor_prompts.get(rag_prompt_key, "")
                if rag_template:
                    effective_counselor_prompt = (
                        rag_template.replace(
                            "{example_responses}", example_block
                        )
                        + COUNSELOR_PROMPT_SUFFIX
                    )

            if phase_selection_template:
                phase_selected = _traced_llm_call(
                    call_name=f"counselor-phase-select-turn-{turn}",
                    role="counselor-phase-select",
                    model=d["counselor_model"],
                    api_url=d["counselor_api_url"],
                    api_key_env=d["counselor_api_key_env"],
                    temperature=d["counselor_temperature"],
                    system_prompt=phase_selection_template,
                    user_prompt=conv_text,
                    max_completion_tokens=1000,
                    reasoning_effort=d.get("counselor_reasoning_effort")
                )
                phased = counselor_prompts.get("phased_prompt_dictionary", {})
                effective_counselor_prompt += phased.get(phase_selected, "")

            counselor_human = (
                f"{phase_selected}\n{conv_text}" if phase_selected else conv_text
            )
            counselor_draft = _traced_llm_call(
                call_name=f"counselor-turn-{turn}",
                role="counselor",
                model=d["counselor_model"],
                api_url=d["counselor_api_url"],
                api_key_env=d["counselor_api_key_env"],
                temperature=d["counselor_temperature"],
                system_prompt=effective_counselor_prompt,
                user_prompt=counselor_human,
                max_completion_tokens=d['max_counselor_tokens'],
                reasoning_effort=d.get("counselor_reasoning_effort")
            )

            if editor_prompt_template:
                editor_input = (
                    f"{editor_prompt_template.strip()}\n\n"
                    f"Conversation history:\n{conv_text}\n\n"
                    f"New message from counselor:\n{counselor_draft}\n"
                    f"Output: "
                )
                counselor_draft = _traced_llm_call(
                    call_name=f"counselor-editor-turn-{turn}",
                    role="counselor-editor",
                    model=d["counselor_model"],
                    api_url=d["counselor_api_url"],
                    api_key_env=d["counselor_api_key_env"],
                    temperature=d["counselor_temperature"],
                    system_prompt="",
                    user_prompt=editor_input,
                    max_completion_tokens=d['max_counselor_tokens'],
                    reasoning_effort=d.get("counselor_reasoning_effort")
                )

            for msg in counselor_draft.split("//"):
                conversation_history.append(f"counselor: {msg}")
                conversation_history.append("\n")
                _print_dialogue_line(profile_key, "counselor", msg)

            if "[end conversation]" in counselor_draft.lower():
                break

        conversation = "\n".join(conversation_history)

        # Keep the full conversation on the generation trace so this trace remains
        # the single source of truth for LM-call-level debugging + scored outcomes.
        lf.update_current_trace(
            input=profile_prompt,
            output=conversation,
            metadata={
                "dataset_name": dataset_name,
                "run_name": run_name,
                "profile_key": profile_key,
                "category": category,
                "turn_count": turn,
            },
        )

        trace_id = lf.get_current_trace_id()

        # Upload dataset item with profile-centric input; conversation is linked
        # via source_trace_id and later surfaced as dataset run output.
        item_id = _stable_dataset_item_id(dataset_name, profile_key)
        lf_client.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "patient_profile_key":    profile_key,
                "patient_profile_prompt": profile_prompt,
                "category":               category,
            },
            expected_output="",
            metadata={
                "profile_key":          profile_key,
                "category":             category,
                "generation_trace_id":  trace_id,
                "generation_run_name":  run_name,
            },
            source_trace_id=trace_id,
            id=item_id,
        )

        print(
            f"[step1]   ✓ '{profile_key}' uploaded "
            f"({len(conversation)} chars, {turn} turns)"
        )
        return {
            "success": True,
            "profile_key": profile_key,
            "conversation_length": len(conversation),
            "turn_count": turn,
        }

    except Exception as exc:
        print(f"[step1]   ✗ Generation failed for '{profile_key}': {exc}")
        return {
            "success": False,
            "profile_key": profile_key,
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────

def run(
    dataset_name: str,
    run_name: str = "generation",
    patient_prompts_override: list[str] | None = None,
    on_existing_dataset: str = "prompt",
) -> str:
    """Generate dialogues and upload to Langfuse. Returns the resolved dataset name."""
    _check_env()
    cfg = load_config()

    # Load patient profile prompts
    profiles_path = cfg["patient_profile_prompts_file"]
    with open(profiles_path) as f:
        all_profiles: dict = yaml.safe_load(f)
    all_keys = list(all_profiles.keys())

    # Resolve which profiles to run
    requested = patient_prompts_override or cfg["patient_prompts"]
    profile_keys = expand_patient_prompts(requested, all_keys)
    if not profile_keys:
        sys.exit("[step1] ERROR: No matching patient profiles found. Check your patient_prompts config.")

    print(f"[step1] Resolved {len(profile_keys)} profile(s) to generate: {profile_keys[:5]}"
          + (f"… (+{len(profile_keys)-5} more)" if len(profile_keys) > 5 else ""))

    lf = Langfuse()
    dataset_name, should_add_items = check_dataset_name(
        lf,
        dataset_name,
        on_existing_dataset=on_existing_dataset,
    )

    if not should_add_items:
        print(f"[step1] Reuse selected. Skipping generation/upload for dataset '{dataset_name}'.")
        return dataset_name

    lf.create_dataset(
        name=dataset_name,
        description=cfg["dataset"]["description"],
    )
    print(f"[step1] Dataset '{dataset_name}' ready in Langfuse.\n")

    # Pre-load prompt templates once (shared across threads, read-only)
    d = cfg["dialogue"]
    counselor_prompts = load_prompts(d["counselor_config_path"])
    
    editor_prompts = counselor_prompts
    if d.get("patient_editor_prompt") and d.get("patient_editor_config_path"):
        editor_prompts = load_prompts(d["patient_editor_config_path"])

    # ── RAG validation & index construction ───────────────────────────────
    rag_index = None
    if d.get("use_rag"):
        if not d.get("rag_embeddings_model"):
            sys.exit(
                "[step1] ❌ use_rag is true but rag_embeddings_model is null. "
                "Set it to an OpenAI embeddings model (e.g. 'text-embedding-3-small')."
            )
        csv_path = d.get("historical_conversations") or ""
        if not csv_path.strip():
            sys.exit(
                "[step1] ❌ use_rag is true but historical_conversations is null/blank. "
                "Provide the path to a historical conversations CSV."
            )
        if not Path(csv_path).exists():
            sys.exit(
                f"[step1] ❌ historical_conversations file not found: {csv_path}"
            )
        rag_prompt_key = d.get("counselor_rag_prompt", "rag_dynamic_instructions")
        if rag_prompt_key not in counselor_prompts:
            sys.exit(
                f"[step1] ❌ counselor_rag_prompt '{rag_prompt_key}' not found "
                f"in counselor prompts file. Available keys: "
                f"{list(counselor_prompts.keys())}"
            )

        from rag import RAGIndex  # noqa: E402  (lazy import)

        print("[step1] 🔍 Building RAG index from historical conversations …")
        rag_index = RAGIndex(
            csv_path=csv_path,
            embeddings_model=d["rag_embeddings_model"],
            top_k=d.get("rag_top_k", 5),
        )
        print(
            f"[step1] ✅ RAG index ready "
            f"({rag_index.num_documents} context–response pairs)"
        )

    total = len(profile_keys)
    max_workers = cfg.get("max_workers", 1)
    uploaded = 0
    failed = 0

    def _task(profile_key: str) -> dict:
        profile_prompt = all_profiles[profile_key]
        category = "_".join(profile_key.rsplit("_", 1)[:-1]) or profile_key
        return _generate_and_upload_one(
            lf_client=lf,
            cfg=cfg,
            profile_key=profile_key,
            profile_prompt=profile_prompt,
            dataset_name=dataset_name,
            run_name=run_name,
            category=category,
            counselor_prompts=counselor_prompts,
            editor_prompts=editor_prompts,
            rag_index=rag_index,
        )

    print(f"[step1] Generating {total} dialogue(s) with up to {max_workers} worker(s) …\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_task, pk): pk for pk in profile_keys}
        for i, future in enumerate(as_completed(futures), 1):
            pk = futures[future]
            try:
                result = future.result()
                if result.get("success"):
                    uploaded += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"[step1]   ✗ '{pk}' raised: {exc}")
                failed += 1
            print(f"[step1] Progress: {i}/{total}")

    lf.flush()
    print(
        f"\n[step1] ✅ Done — {uploaded} uploaded, {failed} failed.\n"
        f"         View dataset at: {os.environ.get('LANGFUSE_BASE_URL', '')}/datasets/{dataset_name}"
    )
    return dataset_name


# ─────────────────────────────────────────────────────────────────────────────

def _check_env() -> None:
    missing = [v for v in ("OPENAI_API_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit(
            f"[step1] ❌ Missing environment variable(s): {', '.join(missing)}\n"
            "  Export them before running."
        )


def parse_args():
    p = argparse.ArgumentParser(description="Generate dialogues and upload as Langfuse dataset")
    p.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    p.add_argument("--run-name", default="generation", help="Run name used for Langfuse trace tags")
    p.add_argument("--patient-prompts", nargs="+", default=None,
                   help="Override config patient_prompts list (keys or categories)")
    p.add_argument(
        "--on-existing-dataset",
        choices=["prompt", "generate", "reuse"],
        default="prompt",
        help=(
            "Behavior when dataset exists: 'generate' regenerates and upserts selected "
            "profiles, 'reuse' skips generation, 'prompt' asks interactively"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    final_name = run(
        args.dataset_name,
        args.run_name,
        args.patient_prompts,
        args.on_existing_dataset,
    )

    # Write resolved name for the pipeline orchestrator
    state_file = Path(__file__).parent / "__pipeline_state.txt"
    state_file.write_text(final_name)


if __name__ == "__main__":
    main()
