import os
import json
import argparse
import pandas as pd

def parse_filename(file_name):
    """Parse the filename into its components."""
    parts = file_name.split('|')
    dialogue_profile = parts[1]
    dialogue_profile_split = dialogue_profile.split("_")
    dialogue_name = dialogue_profile_split[0]
    patient_type = dialogue_profile_split[2] + "_" + dialogue_profile_split[3]

    counselor_prompt = parts[2]
    editor_prompt = parts[3]
    model_name = parts[4]
    timestamp = parts[5].split('.')[0]
    return {
        "dialogue_name": dialogue_name,
        "patient_type": patient_type,
        "counselor_prompt": counselor_prompt,
        "editor_prompt": editor_prompt,
        "model_name": model_name,
        "timestamp": timestamp
    }

def process_jsonl_file(file_path):
    """Process a single JSONL file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) != 3:
        raise ValueError(f"File {file_path} does not have exactly 3 lines.")

    # Parse each evaluation type
    patient_eval = json.loads(lines[0])
    counselor_unit_test = json.loads(lines[1])
    counselor_eval = json.loads(lines[2])
    
    return patient_eval, counselor_unit_test, counselor_eval

def analyze_directory(jsonl_dir):
    """Analyze all JSONL files in a directory."""
    results = []
    
    for file_name in os.listdir(jsonl_dir):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(jsonl_dir, file_name)
            try:
                # Extract filename metadata
                file_metadata = parse_filename(file_name)
                
                # Process the JSONL file
                patient_eval, counselor_unit_test, counselor_eval = process_jsonl_file(file_path)
                
                # Aggregate results
                results.append({
                    **file_metadata,
                    "patient_eval": patient_eval,
                    "counselor_unit_test": counselor_unit_test,
                    "counselor_eval": counselor_eval
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return results

def flatten_results(results):
    """Flatten the JSON results into a table format."""
    data = []

    for result in results:
        # Unpack patient evaluation
        patient_eval = result["patient_eval"]
        for key, value in patient_eval.items():
            data.append({
                "dialogue_name": result["dialogue_name"],
                "patient_type": result["patient_type"],
                "counselor_prompt": result["counselor_prompt"],
                "editor_prompt": result["editor_prompt"],
                "model_name": result["model_name"],
                "timestamp": result["timestamp"],
                "evaluation_type": "patient_eval",
                "category": key,
                "score": value.get("score"),
                "rationale": value.get("rationale")
            })

        # Unpack counselor unit tests
        counselor_unit_test = result["counselor_unit_test"]
        for key, value in counselor_unit_test.items():
            if isinstance(value, dict):
                # Extract score dynamically based on available keys in the dictionary
                score_keys = ["n_multi_task_messages", "n_instances_not_followed", "build_rapport", "n_skipped_stages",
                              "n_forbidden_statements", "n_question_messages", "n_empathy_chained_messages",
                              "explore_situation", "assess_risks", "explore_next_steps", "offer_resources", 
                              "end_conversation", "n_emotional_language"]
                score = next((value.get(k) for k in score_keys if k in value), None)
                
                data.append({
                    "dialogue_name": result["dialogue_name"],
                    "patient_type": result["patient_type"],
                    "counselor_prompt": result["counselor_prompt"],
                    "editor_prompt": result["editor_prompt"],
                    "model_name": result["model_name"],
                    "timestamp": result["timestamp"],
                    "evaluation_type": "counselor_unit_test",
                    "category": key,
                    "score": score,
                    "rationale": value.get("rationale") or value.get("explanation")
                })

        # Unpack counselor evaluation
        counselor_eval = result["counselor_eval"]
        for key, value in counselor_eval.items():
            data.append({
                "dialogue_name": result["dialogue_name"],
                "patient_type": result["patient_type"],
                "counselor_prompt": result["counselor_prompt"],
                "editor_prompt": result["editor_prompt"],
                "model_name": result["model_name"],
                "timestamp": result["timestamp"],
                "evaluation_type": "counselor_eval",
                "category": key,
                "score": value.get("score"),
                "rationale": value.get("rationale")
            })

    return pd.DataFrame(data)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze JSONL dialogue evaluation files.")
    parser.add_argument("--jsonl_dir", required=True, help="Path to the directory containing JSONL files.")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    # Analyze the directory
    results = analyze_directory(args.jsonl_dir)

    # Flatten results into a DataFrame
    df = flatten_results(results)

    # Save results to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Analysis saved to {args.output_csv}")

if __name__ == "__main__":
    main()

# example usage:
# python crisisbench/analyze_dialogue_evals.py --jsonl_dir data/evaluator_outputs --output_csv evals/dialogue_evals.csv