import pandas as pd
import argparse
import os

def load_csv(file_path):
    """Load a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)

def prepare_second_call_prompt(base_prompt_file, copilot_data_file, output_file):
    """Prepare second call prompt input by joining two CSVs and formatting the column."""
    # Load CSVs
    base_prompt_df = load_csv(base_prompt_file)
    copilot_df = load_csv(copilot_data_file)

    # Merge on 'uid'
    merged_df = pd.merge(base_prompt_df, copilot_df, on="uid", how="inner")

    # Create second_call_prompt_input column
    merged_df["second_call_prompt_input"] = merged_df.apply(
        lambda row: f"Conversation history:\n\n{row['conversation']}\n\nNew message from counselor:{row['base_prompt_response']}",
        axis=1
    )

    # Select required columns
    output_df = merged_df[["uid", "second_call_prompt_input"]]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the processed CSV
    output_df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare second call prompt inputs by joining two CSVs.")
    
    parser.add_argument(
        "--base_prompt_file",
        default="evals/data/llm_inference/extracted_completions/base_prompt_completions.csv",
        help="Path to the base prompt completions CSV file"
    )
    
    parser.add_argument(
        "--copilot_data_file",
        default="evals/data/for_prompting/copilot_data.csv",
        help="Path to the copilot data CSV file"
    )
    
    parser.add_argument(
        "--output_file",
        default="evals/data/for_prompting/second_call_prompt_input.csv",
        help="Path to save the output CSV file"
    )
    
    args = parser.parse_args()
    
    prepare_second_call_prompt(args.base_prompt_file, args.copilot_data_file, args.output_file)
