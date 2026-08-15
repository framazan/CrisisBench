import os
import yaml
import argparse
from datetime import datetime

def extract_prompts_from_file(input_file):
    print(f"Extracting prompts from input file: {input_file}")
    with open(input_file, 'r') as file:
        lines = file.readlines()

    full_profile = []
    editor_prompt = []
    is_capturing_full_profile = False
    is_capturing_editor_prompt = False

    for line in lines:
        if 'full_profile = f"""' in line:
            is_capturing_full_profile = True
            continue
        elif 'editor_prompt = f"""' in line:
            is_capturing_editor_prompt = True
            continue
        elif '"""' in line:
            if is_capturing_full_profile:
                is_capturing_full_profile = False
            elif is_capturing_editor_prompt:
                is_capturing_editor_prompt = False
            continue

        if is_capturing_full_profile:
            full_profile.append(line)  
        elif is_capturing_editor_prompt:
            editor_prompt.append(line)  

    # Join lines with explicit newlines for YAML
    full_profile = ''.join(full_profile)
    editor_prompt = ''.join(editor_prompt)

    return full_profile, editor_prompt

def save_historical_prompts(input_file, output_dir):
    # Extract full profile and editor prompt from the input file
    full_profile, editor_prompt = extract_prompts_from_file(input_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get current date in month-date-year format
    current_date = datetime.now().strftime("%m-%d-%Y")

    # Save full profile to YAML
    full_profile_path = os.path.join(output_dir, f'full_profile_prompt_{current_date}.yaml')
    with open(full_profile_path, 'w') as f:
        f.write(full_profile)
    print(f"Saved full profile to {full_profile_path}.")

    # Save editor prompt to YAML
    editor_prompt_path = os.path.join(output_dir, f'editor_prompt_{current_date}.yaml')
    with open(editor_prompt_path, 'w') as f:
        f.write(editor_prompt)
    print(f"Saved editor prompt to {editor_prompt_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save historical prompts.")
    parser.add_argument("--input_file", required=True, help="Path to the generate_patient_profiles.py file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save historical prompts.")
    args = parser.parse_args()

    save_historical_prompts(args.input_file, args.output_dir)