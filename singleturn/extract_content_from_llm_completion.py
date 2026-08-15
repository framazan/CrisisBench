import sys
sys.path.append("/home/stanford/vf_copilot/evals")
import argparse
import json
import pandas as pd
import os

def parse_single_jsonl(input_path, content_name="generated_differential"):
    """
    Parse a .jsonl file that can contain either:
    1) the legacy 3‑element list format; or
    2) a single flat object with keys `prompt` + `completion`.
    Returns a list of dicts ready for a DataFrame.
    """
    import json, os, uuid

    def remove_suffix(input_string, suffix):
        if suffix and input_string.endswith(suffix):
            return input_string[:-len(suffix)]
        return input_string

    extracted = []
    file_name = remove_suffix(os.path.basename(input_path), ".jsonl")

    with open(input_path) as jf:
        for line in jf:
            data = json.loads(line)

            # ────────── Case 1: new flat Claude format ──────────
            if isinstance(data, dict) and "completion" in data:
                extracted.append(
                    {
                        "uid": data.get("uid", str(uuid.uuid4())),          # fabricate if missing
                        "prompt_template_name": data.get("prompt_template_name", ""),
                        content_name: data["completion"],
                        "file_name": file_name,
                        "input_file_name": data.get("input_file_name", ""),
                    }
                )
                continue

            # ────────── Case 2: legacy 3‑element list ──────────
            if isinstance(data, list) and len(data) == 3:
                meta = data[2].get("metadata", {})
                choices = data[1].get("choices", []) if isinstance(data[1], dict) else data[1]

                if choices and "message" in choices[0]:
                    extracted.append(
                        {
                            "uid": meta.get("uid", str(uuid.uuid4())),
                            "prompt_template_name": meta.get("prompt_template_name", ""),
                            content_name: choices[0]["message"]["content"],
                            "file_name": file_name,
                            "input_file_name": meta.get("input_file_name", ""),
                        }
                    )
    return extracted


def parse_jsonl(input_path, output_path, content_name = "generated_differential", write_to_file = True):
    """
    If input_path is a file, parse it.
    If input_path is a directory, parse all .jsonl files in that directory.
    Write the results to output_path as a CSV.
    """
    all_data = []

    if os.path.isdir(input_path):
        # Process all .jsonl files in the directory
        for filename in os.listdir(input_path):
            if filename.endswith(".jsonl"):
                file_path = os.path.join(input_path, filename)
                all_data.extend(parse_single_jsonl(file_path, content_name))
    else:
        # Process single file
        all_data.extend(parse_single_jsonl(input_path, content_name))

    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    if write_to_file:
        df.to_csv(output_path, index=False)
        print(f"Data successfully extracted to {output_path}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Extract relevant data from JSONL file(s) and save to CSV.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input .jsonl file or directory of .jsonl files.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output CSV file.")
    parser.add_argument("-c", "--content_name", type=str, default="generated_differential", help="Name of the content field to extract.")
    args = parser.parse_args()

    parse_jsonl(args.input, args.output, args.content_name)

if __name__ == "__main__":
    main()
