import pandas as pd
import argparse
import json
from typing import List

def extract_llm_response(completion_json: str) -> str:
    """
    Extract the LLM response from the completion JSON.
    
    Args:
        completion_json: JSON string containing the LLM completion
        
    Returns:
        Extracted response text
    """
    try:
        completion = json.loads(completion_json)
        # Extract from OpenAI chat completion format
        if isinstance(completion, dict) and 'choices' in completion:
            message = completion['choices'][0]['message']
            if isinstance(message, dict) and 'content' in message:
                return message['content'].strip()
        return completion_json
    except:
        return completion_json

def prepare_comparison_data(
    input_csv: str,
    completions_csvs: List[str],
    output_csv: str
) -> None:
    """
    Prepare data for comparison by combining original input with LLM completions.
    
    Args:
        input_csv: Path to the original input CSV with conversation history
        completions_csvs: List of paths to CSVs containing LLM completions
        output_csv: Path to save the combined comparison data
    """
    # Read input data
    input_df = pd.read_csv(input_csv)
    print("Input CSV columns:", input_df.columns.tolist())
    
    # ---- concatenate multiple completion files --------------------------------------
    completion_frames = []
    for path in completions_csvs:
        print(f"Loading completions: {path}")
        try:
            df = pd.read_csv(path)
            completion_frames.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found {path}")
            continue

    if not completion_frames:
        raise ValueError("No completion CSVs provided")

    completions_df = pd.concat(completion_frames, ignore_index=True)
    print("Concatenated completions shape:", completions_df.shape)
    
    # Extract the completion text from the JSON
    completions_df['extracted_completion'] = completions_df['llm_output_json_raw'].apply(extract_llm_response)
    
    # Optional sanity check
    missing_count = input_df['uid'].nunique() - completions_df['uid'].nunique()
    if missing_count != 0:
        print(f"Warning: UID mismatch – input has {input_df['uid'].nunique()} unique rows, completions have {completions_df['uid'].nunique()}")
    
    # Merge the dataframes
    merged_df = pd.merge(
        input_df,
        completions_df,
        on='uid',  # Use uid to match rows
        suffixes=('', '_completion')
    )
    
    # Print merged columns to debug
    print("Merged columns:", merged_df.columns.tolist())
    
    # Create comparison input format
    merged_df['comparison_input'] = merged_df.apply(
        lambda row: f"Conversation History:\n{row['convo_history']}\n\nResponse A:\n{row['counselor_message']}\n\nResponse B:\n{row['extracted_completion']}",
        axis=1
    )
    
    # Save the result
    merged_df.to_csv(output_csv, index=False)
    print(f"Comparison data saved to {output_csv}")

def split_paths(paths):
    """Split paths that might contain newlines into separate path strings."""
    result = []
    for path in paths:
        # Split if the path contains newlines
        if '\n' in path:
            result.extend([p for p in path.split('\n') if p.strip()])
        else:
            result.append(path)
    return result

def main():
    parser = argparse.ArgumentParser(description='Prepare data for response comparison')
    parser.add_argument('input_csv', help='Path to input CSV file')
    parser.add_argument('completions_csv', nargs='+', help='Path(s) to completions CSV file(s)')
    parser.add_argument('output_csv', help='Path to save the comparison data')
    
    args = parser.parse_args()
    
    # Process paths that might contain newlines
    completions_paths = split_paths(args.completions_csv)
    print(f"Processing {len(completions_paths)} completion files:")
    for path in completions_paths:
        print(f"  - {path}")
    
    prepare_comparison_data(
        args.input_csv,
        completions_paths,
        args.output_csv
    )

if __name__ == '__main__':
    main() 