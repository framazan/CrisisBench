"""
Step 1 – Upload a static-eval dataset to Langfuse.

Each row becomes one Langfuse dataset item:
  input           = { "convo_history": "...", "uid": "...", "conversation_id": "..." }
  expected_output = the real counselor message (ground-truth / human response)

Input flavours
--------------
  A) Already-converted CSV (columns: uid, conversation_id, convo_history,
     counselor_message).
       python step1_upload_dataset.py --input static_eval_input.csv

  B) Raw dialogue CSV (one full conversation per row). Pass --convert and the
     script will run the vf_copilot conversion internally first.
       python step1_upload_dataset.py --input raw_convos.csv --convert
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import yaml
from langfuse import Langfuse

# Optionally import the conversion helper from vf_copilot
_REPO_ROOT = Path("/home/stanford/CrisisBench")
sys.path.insert(0, str(_REPO_ROOT))
try:
    from singleturn.convert_dialogue_to_static_eval_format import process_dialogues as _convert
except ImportError:
    _convert = None


# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_static_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def convert_raw_csv(input_csv: str, conv_col: str, filter_n: int, max_messages_per_convo : int) -> pd.DataFrame:
    if _convert is None:
        sys.exit("[step1] ERROR: Could not import vf_copilot conversion helper.")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _convert(input_csv, tmp_path, conv_col, filter_n, max_messages_per_convo)
        return pd.read_csv(tmp_path)
    finally:
        os.unlink(tmp_path)


def check_dataset_name(lf: Langfuse, requested_name: str) -> tuple[str, bool]:
    """Return the dataset name to use and whether to upload new items, prompting if it already exists."""
    try:
        existing = lf.get_dataset(requested_name)
        item_count = len(existing.items)
    except Exception:
        existing = None
        item_count = 0

    if existing is None:
        return requested_name, True

    print(
        f"\n[step1] ⚠️  Dataset '{requested_name}' already exists "
        f"({item_count} items).\n"
        f"  • Press Enter to ADD new items into it\n"
        f"  • Type 'reuse' to use it without adding items\n"
        f"  • Or type a new dataset name to create a fresh one: ",
        end="", flush=True
    )
    user_input = input().strip()
    if not user_input:
        print(f"[step1] Continuing with existing dataset '{requested_name}'.")
        return requested_name, True
    if user_input.lower() == "reuse":
        print(f"[step1] Reusing dataset '{requested_name}' without adding new items.")
        return requested_name, False
    # Recursively resolve in case the new name also exists
    return check_dataset_name(lf, user_input)


def upload_dataset(df: pd.DataFrame, dataset_name: str, description: str) -> str:
    """Upload items and return the final dataset name actually used."""
    lf = Langfuse()

    dataset_name, should_upload = check_dataset_name(lf, dataset_name)
    
    if not should_upload:
        return dataset_name

    # create_dataset is idempotent in Langfuse
    lf.create_dataset(name=dataset_name, description=description)
    print(f"[step1] Dataset '{dataset_name}' ready in Langfuse.")

    total = len(df)
    print(f"[step1] Uploading {total} items …")

    for i, row in df.iterrows():
        uid = str(row.get("uid", uuid.uuid4()))
        conv_id = str(row.get("conversation_id", ""))

        lf.create_dataset_item(
            dataset_name=dataset_name,
            input={
                "convo_history": str(row["convo_history"]),
                "conversation_id": conv_id,
                "uid": uid,
            },
            expected_output=str(row["counselor_message"]),
            metadata={"uid": uid, "conversation_id": conv_id},
        )

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"[step1]   {i + 1}/{total} uploaded")

    lf.flush()
    print(f"[step1] ✅ Done. Dataset '{dataset_name}' is live in Langfuse.")
    return dataset_name


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    cfg = load_config()
    p = argparse.ArgumentParser(description="Upload static-eval dataset to Langfuse")
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--dataset-name", required=True, help="Langfuse dataset name")
    p.add_argument("--description", default=cfg["dataset"]["description"])
    p.add_argument("--convert", action="store_true",
                   help="Convert raw dialogue CSV before uploading")
    p.add_argument("--conversation-column",
                   default=cfg["conversion"]["conversation_column"])
    p.add_argument("--filter-messages", type=int,
                   default=cfg["conversion"]["filter_messages"])
    return p.parse_args()


def main():
    args = parse_args()

    if args.convert:
        print(f"[step1] Converting raw dialogue CSV: {args.input}")
        df = convert_raw_csv(args.input, args.conversation_column, args.filter_messages, 7)
        print(f"[step1] Conversion produced {len(df)} rows.")
    else:
        df = pd.read_csv(args.input)
        required = {"convo_history", "counselor_message"}
        missing = required - set(df.columns)
        if missing:
            sys.exit(f"[step1] ERROR: CSV missing columns {missing}. "
                     "Use --convert if this is a raw dialogue CSV.")

    final_name = upload_dataset(df, args.dataset_name, args.description)

    # Write resolved name so run_pipeline.py can pass it to subsequent steps.
    state_file = Path(__file__).parent / "__pipeline_state.txt"
    state_file.write_text(final_name)
    if final_name != args.dataset_name:
        print(f"[step1] ℹ️  Dataset name changed to '{final_name}' — saved for downstream steps.")


if __name__ == "__main__":
    main()
