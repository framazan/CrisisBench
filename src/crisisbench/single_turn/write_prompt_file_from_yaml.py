import yaml
import argparse
import os

DEFAULT_OUTPUT_DIR = "evals/data/prompts"
DEFAULT_APPEND_TEXT = "{note}"

def load_yaml(yaml_file):
    """Load YAML file and return as a dictionary."""
    with open(yaml_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def write_prompt_file(output_dir, prompt_key, prompt_text, append_text):
    """Write prompt text to a .tmpl file with the specified append text."""
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    file_path = os.path.join(output_dir, f"{prompt_key}.tmpl")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{prompt_text}\n\n{append_text}\n")

    print(f"Created: {file_path}")

def main(yaml_file, prompt_keys, append_text, output_dir=DEFAULT_OUTPUT_DIR):
    """Process YAML file and write the specified prompts to .tmpl files."""
    prompts = load_yaml(yaml_file)

    for key in prompt_keys:
        if key in prompts:
            if "prompt" in prompts[key]:
                prompt = prompts[key]["prompt"]
            else:
                prompt = prompts[key]
            write_prompt_file(output_dir, key, prompt, append_text)
        else:
            print(f"Warning: Key '{key}' not found in {yaml_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract prompts from YAML and write to .tmpl files.")
    parser.add_argument("yaml_file", help="Path to YAML file")
    parser.add_argument("prompt_keys", nargs="+", help="Keys of the prompts to extract")
    parser.add_argument("--append_text", default=DEFAULT_APPEND_TEXT, help="Text to append at the end of each prompt (default: {note})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save .tmpl files (default: evals/data/prompts)")
    
    args = parser.parse_args()
    main(args.yaml_file, args.prompt_keys, args.append_text, args.output_dir)
