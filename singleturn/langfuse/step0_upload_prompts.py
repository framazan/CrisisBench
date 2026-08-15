"""
Step 0 – Upload all prompts to the Langfuse prompt library.

Run this once (or whenever you update a prompt template). Each template file
is stored as a versioned text prompt in Langfuse. Downstream steps fetch them
by name, so the actual text is always managed in Langfuse.

Template variables use {note} in the source files; we normalise to the
Langfuse double-brace convention {{note}} on upload.

Usage
-----
  python step0_upload_prompts.py
  python step0_upload_prompts.py --force   # bump version even if text unchanged
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from langfuse import Langfuse


# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config" / "langfuse_static_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def to_langfuse_template(text: str) -> str:
    """Convert {note} (Python-style) → {{note}} (Langfuse-style)."""
    # Only replace single-brace {note}; leave existing double-braces alone.
    return re.sub(r"(?<!\{)\{note\}(?!\})", "{{note}}", text)


def read_prompt_file(path: str) -> str:
    text = Path(path).read_text().strip()
    return to_langfuse_template(text)


def upload_prompt(lf: Langfuse, name: str, text: str, force: bool) -> None:
    """Create or update a Langfuse text prompt."""
    try:
        existing = lf.get_prompt(name, cache_ttl_seconds=0)
        if existing.prompt == text and not force:
            print(f"  [skip] '{name}' – text unchanged (use --force to re-upload)")
            return
    except Exception:
        pass  # prompt doesn't exist yet

    lf.create_prompt(
        name=name,
        prompt=text,
        labels=["production"],
        type="text",
    )
    print(f"  [ok]   '{name}' uploaded")


def main():
    parser = argparse.ArgumentParser(description="Upload prompts to Langfuse")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload even if prompt text has not changed")
    args = parser.parse_args()

    cfg = load_config()
    lf = Langfuse()

    pf = cfg["prompt_files"]
    pn = cfg["prompts"]

    uploads = [
        (pn["generation"], pf["generation"]),
        (pn["rubric_eval"], pf["rubric_eval"]),
    ]
    for key, name in pn["unit_tests"].items():
        uploads.append((name, pf["unit_tests"][key]))

    print(f"Uploading {len(uploads)} prompts to Langfuse …")
    for name, file_path in uploads:
        try:
            text = read_prompt_file(file_path)
            upload_prompt(lf, name, text, args.force)
        except FileNotFoundError:
            print(f"  [WARN] Source file not found: {file_path}", file=sys.stderr)

    lf.flush()
    print("\nDone. View prompts at: $LANGFUSE_BASE_URL/prompts")


if __name__ == "__main__":
    main()
