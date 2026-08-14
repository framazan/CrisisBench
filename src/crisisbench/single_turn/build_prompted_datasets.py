import os
import json
import timeit
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from langchain_core.prompts import PromptTemplate

########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Generate prompt dataset")

input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument(
    "-i", "--input_dir", type=str, help="Path to the input directory containing CSV files to process"
)
input_group.add_argument(
    "-f", "--input_file", type=str, help="Path to a single CSV file to process"
)

parser.add_argument(
    "-p", "--path_to_prompt_dir", type=str, help="Path to the directory containing prompt templates", required=True
)

parser.add_argument(
    "-o", "--path_to_output_dir", type=str, help="Path to the output JSONL file", required=True
)

parser.add_argument(
    "-n", "--file_name_prefix", type=str, help="Prefix name of prompted dataset file", default="factehr"
)

parser.add_argument(
    "-fmt", "--completion_format", type=str,
    help="Completion format: 'messages' for chat, 'prompt' for single-prompt completions",
    choices=["messages", "prompt"], default="messages"
)

parser.add_argument("-m", "--max_tokens", type=int, help="Maximum tokens per completion", default=4096)

parser.add_argument(
    "-c", "--input_column", type=str, help="Name of the column containing the input text", default="note"
)

########## Misc Data Loaders ##########

def load_prompt_templates(path_to_prompt_dir: str):
    """Load prompt templates from a directory. Name of prompt is the same as the file name without extension."""
    prompt_templates = []

    for file in os.listdir(path_to_prompt_dir):
        if file.endswith(".tmpl"):
            text = open(os.path.join(path_to_prompt_dir, file), "r").read()
            prompt_templates.append(
                PromptTemplate.from_template(template=text, name=file.split(".")[0])
            )
    logging.info(f"Loaded {len(prompt_templates)} prompt templates")
    return prompt_templates

def create_prompt_record(uid, template_name, prompt, completion_format, max_tokens, optional_metadata=None):
    """
    Create a standardized prompt record structure.
    
    Args:
        uid (str): Unique identifier for the prompt
        template_name (str): Name of the prompt template used
        prompt (str): The formatted prompt text
        completion_format (str): Format type - 'messages' or 'prompt'
        max_tokens (int): Maximum tokens for completion
        optional_metadata (dict, optional): Additional metadata fields to include
        
    Returns:
        dict: The formatted prompt record
    """
    metadata = {
        "uid": uid,
        "prompt_template_name": template_name,
    }
    
    # Add any optional metadata fields
    if optional_metadata:
        metadata.update(optional_metadata)
    
    record = {
        "metadata": metadata
    }
    
    # Format based on completion type
    if completion_format == "messages":
        record["messages"] = [{"role": "user", "content": prompt}]
    elif completion_format == "prompt":
        record["prompt"] = prompt
    else:
        raise ValueError(f"Invalid completion format: {completion_format}")
    
    record["max_tokens"] = max_tokens
    
    return record

def process_csv_file(csv_path, templates, input_column, completion_format, max_tokens):
    """Process a single CSV file and generate prompt records."""
    dataset = pd.read_csv(csv_path)

    if input_column not in dataset.columns:
        logger.error(f"Column '{input_column}' not found in {csv_path}. Skipping...")
        return []

    prompt_dataset = []
    # Flag to check presence of optional file_name column once
    has_file_name_col = "file_name" in dataset.columns
    has_id_col = "id" in dataset.columns

    for tmpl in templates:
        for index, doc in dataset.iterrows():
            if pd.isnull(doc[input_column]):
                continue

            uid = doc.get("uid", f"{os.path.basename(csv_path)}_{index}")
            id_value = str(doc["id"]) if has_id_col else uid 
            prompt = tmpl.format(**{
                'note': doc[input_column],
                'id': id_value,
                'uid': uid
                })

            
            # Build metadata and include optional input file_name if available
            metadata = {
                "uid": uid,
                "prompt_template_name": tmpl.name,
                "id": id_value
            }
            if has_file_name_col:
                metadata["input_file_name"] = doc["file_name"]
            if has_id_col: 
                metadata["id"] = doc["id"]  

            record = create_prompt_record(uid, tmpl.name, prompt, completion_format, max_tokens, metadata)
            prompt_dataset.append(record)

    return prompt_dataset

def main():
    args = parser.parse_args()

    # Load prompt templates
    templates = load_prompt_templates(args.path_to_prompt_dir)
    prompt_dataset = []

    if args.input_file:
        # Process a single file
        prompt_dataset.extend(
            process_csv_file(
                args.input_file, templates, args.input_column, args.completion_format, args.max_tokens
            )
        )
    else:
        # Process all files in the directory
        all_csv_files = [os.path.join(args.input_dir, file) for file in os.listdir(args.input_dir) if file.endswith('.csv')]
        
        for csv_file in all_csv_files:
            prompt_dataset.extend(
                process_csv_file(
                    csv_file, templates, args.input_column, args.completion_format, args.max_tokens
                )
            )

    # Ensure output directory exists
    os.makedirs(args.path_to_output_dir, exist_ok=True)

    # Generate output file path
    version_str = datetime.now().strftime("%Y%m%d")
    output_file_path = os.path.join(args.path_to_output_dir, f"{args.file_name_prefix}_{version_str}.jsonl")

    # Write to output file
    with open(output_file_path, "w") as f:
        for item in prompt_dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(prompt_dataset)} prompts to {output_file_path}")

if __name__ == "__main__":
    # Change this for getting run statistics
    num_runs = 1
    wall_times = np.array(
        [
            timeit.timeit("main()", setup="from __main__ import main", number=1)
            for _ in range(num_runs)
        ]
    )
    logger.info(
        f"Execution time: Mean (SD) = {np.mean(wall_times):.1f} ({np.std(wall_times):.1f}) seconds"
    )
