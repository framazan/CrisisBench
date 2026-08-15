import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import yaml


###############################################################################
# Helper functions
###############################################################################


def extract_json_from_text(input_string: str) -> Dict[str, Any]:
    """Return first JSON object found inside *input_string* or an empty dict."""
    try:
        match = re.search(r"\{.*\}", input_string, re.DOTALL)
        if not match:
            raise ValueError("No JSON-like content found.")
        return json.loads(match.group(0))
    except Exception:
        return {}


def extract_client_messages(convo_text: str) -> List[str]:
    """Return list of client messages extracted from the raw conversation text."""
    lines = convo_text.strip().split("\n")
    client_msgs, buffer = [], []
    for line in lines:
        line = line.strip()
        if line.lower().startswith("client:"):
            msg = re.sub(r"^client:\s*", "", line, flags=re.IGNORECASE)
            buffer.append(msg)
        else:
            if buffer:
                client_msgs.append(" // ".join(buffer))
                buffer = []
    if buffer:
        client_msgs.append(" // ".join(buffer))
    return client_msgs


###############################################################################
# Sampling logic
###############################################################################

_STANDARD_CONCERNS = {
    "Financial loss",
    "Loneliness",
    "Exam or study related",
    "Depression",
    "Anxiety",
    "Suicidality",
    "Relationship",
    "Panic attack",
    "Family relationship",
    "Non-suicidal self-injury",
    "Workplace related",
}


def _prepare_traits_df(merged: pd.DataFrame) -> pd.DataFrame:
    traits_df = pd.DataFrame(
        {
            "uid": merged["uid"],
            "traits": merged["client_traits_obscured_json_raw"].apply(
                extract_json_from_text
            ),
            "presenting_concern": merged["presenting_concern"],
        }
    )
    # Vectorize fields for quick filtering
    traits_df["age"] = traits_df["traits"].apply(
        lambda x: pd.to_numeric(x.get("age"), errors="coerce")
    )
    traits_df["sex"] = traits_df["traits"].apply(lambda x: x.get("sex_or_gender"))
    return traits_df


def find_matching_profiles(target_uid: str, merged: pd.DataFrame) -> List[str]:
    """Return list of UIDs with demographics & concern similar to *target_uid*."""
    traits_df = _prepare_traits_df(merged)
    target_profile = traits_df[traits_df["uid"] == target_uid].iloc[0]
    others = traits_df[traits_df["uid"] != target_uid].copy()

    age_match = (others["age"] - target_profile["age"]).abs() <= 3
    sex_match = others["sex"] == target_profile["sex"]
    concern_match = (
        others["presenting_concern"] == target_profile["presenting_concern"]
    )

    # Apply tiered matching logic mirroring the notebook
    if target_profile["presenting_concern"] not in _STANDARD_CONCERNS:
        if pd.isna(target_profile["age"]):
            final = sex_match
        elif target_profile["sex"] in ["unspecified", "unknown", None]:
            final = age_match
        else:
            final = age_match & sex_match
    else:
        if pd.isna(target_profile["age"]) and target_profile["sex"] in [
            "unspecified",
            "unknown",
            None,
        ]:
            final = concern_match
        elif target_profile["sex"] in ["unspecified", "unknown", None]:
            final = concern_match & age_match
        elif pd.isna(target_profile["age"]):
            final = concern_match & sex_match
        else:
            final = age_match & sex_match & concern_match

    matches = others[final]

    # Fallback relaxations
    if len(matches) < 5:
        matches = others[age_match & sex_match & concern_match]
    if len(matches) < 5:
        matches = others[concern_match]

    return matches["uid"].tolist()


def sample_messages(
    target_uid: str,
    merged: pd.DataFrame,
    matched_uids: List[str],
    num_messages: int = 45,
    messages_per_convo: int = 9,
) -> List[str]:
    sampled_match_uids = (
        random.sample(matched_uids, min(5, len(matched_uids))) if matched_uids else []
    )
    sampled_messages: List[str] = []

    for idx, uid in enumerate(sampled_match_uids):
        convo_row = merged[merged["uid"] == uid]
        if convo_row.empty:
            continue
        client_msgs_field = convo_row.iloc[0]["conversation"]
        client_msg_chunks = extract_client_messages(client_msgs_field)

        # Ensure list
        if isinstance(client_msg_chunks, str):
            try:
                client_msg_chunks = ast.literal_eval(client_msg_chunks)
            except Exception:
                client_msg_chunks = client_msg_chunks.split(" // ")
        if not isinstance(client_msg_chunks, list):
            client_msg_chunks = [client_msg_chunks]

        total = len(client_msg_chunks)
        if total == 0:
            continue
        start = int((idx / 5) * total)
        end = int(((idx + 1) / 5) * total)
        segment = client_msg_chunks[start:end] if end > start else client_msg_chunks
        if len(segment) <= messages_per_convo:
            sampled_messages.extend(segment)
        else:
            sampled_messages.extend(random.sample(segment, messages_per_convo))

    # Cap total messages
    if len(sampled_messages) > num_messages:
        sampled_messages = random.sample(sampled_messages, num_messages)

    return sampled_messages


###############################################################################
# Prompt builders
###############################################################################


_PROMPT_TEMPLATE = """Role Description: You will simulate a patient in a mental health crisis so that counselors may practice handling conversations with these patients.
 In this scenario, you will portray the following patient, who is reaching out to a Whatsapp crisis chat line based in India.
 Respond as if you were this patient:
*** Personal details ***
Name: {name}
Age: {age}
Sex or gender: {sex_or_gender}
Occupation: {job}
Residence: {residence}
Socioeconomic Status: {socioeconomic_status}
Presenting Concern: {presenting_concern}
Key Details: {key_details}
*** Communication patterns ***
Language patterns: {language_patterns}
Emotional Expression & Vulnerability: {emotional_expression_vulnerability}
Depth & Pacing of Disclosure: {depth_pacing_of_disclosure}
Engagement & Response to Support: {engagement_response_to_support}
Focus & Structure of Discussion: {focus_structure_of_discussion}
*** Language examples ***
Use the following language patterns as model responses of what the patient actually sounds like.
Copy the language in the examples closely, modifying details (names, events, etc.) as needed.
{client_messages}
MUST FOLLOW THESE GUIDELINES:
1. It is essential that you respond in a way that is consistent with the profile above.
2. Do not repeat the same phrases or similar variations over and over.
3. Emulate the language examples closely, including punctuation, grammar, spelling, etc.
Remember that many patients are Indians with limited English proficiency. Incorrect punctuation, spelling, and grammar (e.g., omission of articles) may be common.
4. Do not add a prefix like \"patient:\" or \"[name]:\" to the message.
5. Vary the length of your responses. Some may be short, others longer. Beware of using perfect grammar and punctuation too frequently.
Just output the message to send and nothing else. If you want to end the conversation, type \"[end conversation]\".
6. If it's the first message, do not reveal everything immediately. Keep some information vague and allow the counselor to extract key details over time.
7. You can (optionally) send multiple messages in a row, separated by the delimiter \"//\". Maintain variety in message lengths and number of messages per counselor reply."""


def _tidy_newlines(text: str) -> str:
    """Collapse runs of 3+ newlines into a double newline and strip ends."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


###############################################################################
# Main driver
###############################################################################


def generate_patient_profiles(
    traits_file: Path,
    sampled_convos_file: Path,
    deid_convo_file: Path,
    output_file: Path,
    output_yaml: Path | None = None,
    num_messages: int = 45,
    messages_per_convo: int = 9,
):
    print("[INFO] Loading input files…")
    traits_df = pd.read_csv(traits_file)
    print(f"  • traits_df loaded: {traits_df.shape} (rows, cols)")

    # ------------------------------------------------------------------
    # Expand JSON fields from `client_traits_obscured_json_raw` so that
    # columns like `name`, `age`, etc. exist for prompt filling.
    # ------------------------------------------------------------------
    print("[INFO] Parsing client trait JSON into separate columns…")
    trait_cols = [
        "name",
        "age",
        "sex_or_gender",
        "socioeconomic_status",
        "residence",
        "job",
        "key_details",
        "language_patterns",
        "emotional_expression_vulnerability",
        "depth_pacing_of_disclosure",
        "engagement_response_to_support",
        "focus_structure_of_discussion",
    ]

    def _parse_and_extract(row_json: str):
        parsed = extract_json_from_text(row_json)
        return {k: parsed.get(k, "") for k in trait_cols}

    parsed_trait_df = traits_df["client_traits_obscured_json_raw"].apply(_parse_and_extract).apply(pd.Series)
    # Keep the original JSON column for downstream functions that may rely on it
    traits_df = pd.concat([traits_df, parsed_trait_df], axis=1)
    print(f"  • traits_df now expanded to: {traits_df.shape}")

    sampled_convos_df = pd.read_csv(sampled_convos_file)
    print(f"  • sampled_convos_df loaded: {sampled_convos_df.shape}")

    deid_convo_df = pd.read_csv(deid_convo_file)
    print(f"  • deid_convo_df loaded: {deid_convo_df.shape}")

    merged = traits_df.merge(sampled_convos_df, on="uid").merge(deid_convo_df, on="uid")
    print(f"[INFO] Merged dataset shape: {merged.shape}")

    prompts: List[Dict[str, Any]] = []

    total_rows = len(merged)
    print(f"[INFO] Generating prompts for {total_rows} patient profiles…")

    for idx, (_, row) in enumerate(merged.iterrows(), start=1):
        print(f"  [{idx}/{total_rows}] Processing UID: {row['uid']}")

        uid = row["uid"]
        matches = find_matching_profiles(uid, merged)
        print(f"    • Found {len(matches)} matching profiles for sampling")

        message_examples = sample_messages(
            uid, merged, matches, num_messages=num_messages, messages_per_convo=messages_per_convo
        )
        print(f"    • Sampled {len(message_examples)} message examples")

        # Build prompt
        prompt_text = _PROMPT_TEMPLATE.format(
            name=row.get("name", "[Name not provided]"),
            age=row.get("age", "[Age unknown]"),
            sex_or_gender=row.get("sex_or_gender", "unspecified"),
            job=row.get("job", ""),
            residence=row.get("residence", ""),
            socioeconomic_status=row.get("socioeconomic_status", ""),
            presenting_concern=row.get("presenting_concern", ""),
            key_details=row.get("key_details", "[Details not provided]"),
            language_patterns=row.get("language_patterns", ""),
            emotional_expression_vulnerability=row.get("emotional_expression_vulnerability", ""),
            depth_pacing_of_disclosure=row.get("depth_pacing_of_disclosure", ""),
            engagement_response_to_support=row.get("engagement_response_to_support", ""),
            focus_structure_of_discussion=row.get("focus_structure_of_discussion", ""),
            client_messages=" // ".join(message_examples),
        )
        prompt_text = _tidy_newlines(prompt_text)

        prompts.append({
            "uid": uid,
            "prompt": prompt_text,
            "presenting_concern": row.get("presenting_concern", "unknown"),
            "presenting_concern_cat": row.get("presenting_concern_cat", row.get("presenting_concern", "unknown")),
        })

    prompts_df = pd.DataFrame(prompts)

    # Save CSV
    prompts_df.to_csv(output_file, index=False)
    print(f"[SUCCESS] Saved {len(prompts)} patient profiles → {output_file}")

    # Optionally save YAML
    if output_yaml is not None:
        print(f"[INFO] Writing YAML to {output_yaml}…")

        # Prefer the categorical concern if present, and turn it into lowercase
        concern_source = (
            prompts_df["presenting_concern_cat"].fillna(prompts_df["presenting_concern"])
        )
        prompts_df["concern_key"] = concern_source.str.replace(" ", "_", regex=False).str.lower()

        prompts_df["idx_within_concern"] = prompts_df.groupby("concern_key").cumcount()
        prompts_df["yaml_key"] = (
            prompts_df["concern_key"] + "_" + prompts_df["idx_within_concern"].astype(str)
        )

        yaml_dict = dict(zip(prompts_df["yaml_key"], prompts_df["prompt"]))

        with open(output_yaml, "w") as yf:
            yaml.safe_dump(yaml_dict, yf, allow_unicode=True, width=10000, sort_keys=False)
        print(f"[SUCCESS] YAML saved → {output_yaml}")


###############################################################################
# CLI
###############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patient profile prompts with language examples.")
    parser.add_argument("--traits_file", required=True, help="Path to patient traits CSV (de-identified).")
    parser.add_argument("--sampled_convos_file", required=True, help="Path to sampled conversation summaries CSV.")
    parser.add_argument("--deid_convo_file", required=True, help="Path to de-identified conversations CSV.")
    parser.add_argument("--output_file", required=True, help="Path for output CSV of generated prompts.")
    parser.add_argument("--output_yaml", help="Optional path to also write patient profiles YAML.")
    parser.add_argument("--num_messages", type=int, default=45, help="Max total language examples to include.")
    parser.add_argument("--messages_per_convo", type=int, default=9, help="Number of messages to sample from each matched convo.")

    args = parser.parse_args()

    generate_patient_profiles(
        Path(args.traits_file),
        Path(args.sampled_convos_file),
        Path(args.deid_convo_file),
        Path(args.output_file),
        output_yaml=Path(args.output_yaml) if args.output_yaml else None,
        num_messages=args.num_messages,
        messages_per_convo=args.messages_per_convo,
    ) 