import yaml
import os
import argparse

def loop_patient_prompts(config_path, patient_prompts_path, output_dir):
    # Load the main config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the patient prompts file
    with open(patient_prompts_path, 'r') as f:
        patient_prompts = yaml.safe_load(f)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each patient prompt
    for prompt_key in patient_prompts.keys():
        # Update the patient_prompt field in the config
        config['prompts']['patient_prompt'] = prompt_key
        config['prompts']['patient_editor_prompt'] = (
            prompt_key if config['prompts'].get('patient_editor_prompt') is not None else None
        )

        # Generate a new config file for this patient, replacing whitespace in the prompt key with underscores
        output_config_path = os.path.join(output_dir, f"{prompt_key.replace(' ', '_')}.yaml")
        with open(output_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"Generated config for {prompt_key}: {output_config_path}")

# Example usage with command line arguments
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Loop through patient prompts and generate config files")
    parser.add_argument('--config', required=True, help="Path to the main config file")
    parser.add_argument('--patient_prompts', required=True, help="Path to the patient prompts YAML file")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory for the new config files")

    args = parser.parse_args()

    loop_patient_prompts(args.config, args.patient_prompts, args.output_dir)

    # Example usage:
    # python make_config_files.py --config config.yaml --patient_prompts patient_prompts.yaml --output_dir output_configs/