import os
import random
import yaml
from datetime import datetime
import argparse

# Paths
patient_profiles_path = 'evals/llm_patient/data/patient_profile_prompts.yaml'
config_template_path = 'evals/dialogue_config.yaml'

# Define command line arguments
parser = argparse.ArgumentParser(description='Generate config files and dialogues for various counselor models.')
parser.add_argument('--num_samples', type=int, default=5, help='Number of patient profiles to sample')
parser.add_argument('--output_config_dir', type=str, default='evals/llm_patient/dialogue_configs', help='Directory to save generated config files')
parser.add_argument('--output_dialogue_dir', type=str, default='evals/llm_patient/generated_dialogues', help='Directory to save generated dialogues')
args = parser.parse_args()

# Counselor models to iterate over
counselor_models = [
    {
        'model': 'claude-3-7-sonnet-20250219-v1',
        'api_url': 'https://apim.stanfordhealthcare.org/awssig4claude37/aswsig4claude37',
        'api_key_env': 'CLAUDE_API_KEY'
    },
    {
        'model': 'gemini-2.0-flash',
        'api_url': 'https://apim.stanfordhealthcare.org/gcp-gem20flash-fa/apim-gcp-gem20flash-fa',
        'api_key_env': 'GEMINI_API_KEY'
    },
    {
        'model': 'gemini-1.5-pro',
        'api_url': 'https://apim.stanfordhealthcare.org/gcpgemini/apim-gcp-oauth-fa',
        'api_key_env': 'GEMINI_API_KEY'
    },
    {
        'model': 'Llama-3.3-70B-Instruct',
        'api_url': 'https://apim.stanfordhealthcare.org/llama3370b/v1/chat/completions',
        'api_key_env': 'LLAMA_API_KEY'
    },
    {
        'model': 'gpt-4o',
        'api_url': 'https://apim.stanfordhealthcare.org/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21',
        'api_key_env': 'OPENAI_API_KEY_VICKY'
    }
]

# Number of patient profiles to sample
num_samples = args.num_samples

print("Starting the generation of config files and dialogues...")

# Load patient profiles
print(f"Loading patient profiles from {patient_profiles_path}...")
with open(patient_profiles_path, 'r') as file:
    patient_profiles = yaml.safe_load(file)

# Convert patient profiles to a list if it's a dictionary
if isinstance(patient_profiles, dict):
    patient_profiles = list(patient_profiles.keys())

# Sample patient profiles
print(f"Sampling {num_samples} patient profiles...")
sampled_profiles = random.sample(patient_profiles, num_samples)

print(f"Sampled profiles: {sampled_profiles}")

# Create output directories
print(f"Creating output directories: {args.output_config_dir} and {args.output_dialogue_dir}...")
os.makedirs(args.output_config_dir, exist_ok=True)
os.makedirs(args.output_dialogue_dir, exist_ok=True)

# Generate config files and dialogues for each sampled patient profile
for profile_index, profile in enumerate(sampled_profiles):
    print(f"Processing profile {profile_index + 1}/{num_samples}...")
    
    # Generate config files for each counselor model
    for model in counselor_models:
        print(f"Generating config file for model: {model['model']} with profile {profile}...")
        # Load config template
        with open(config_template_path, 'r') as file:
            config = yaml.safe_load(file)

        # Update config with model details and specific patient profile
        config['counselor_model'] = model['model']
        config['counselor_api_url'] = model['api_url']
        config['counselor_api_key_env'] = model['api_key_env']
        config['prompts']['patient_prompt'] = profile

        # Save the new config file
        date_str = datetime.now().strftime('%m-%d-%Y')
        config_filename = f"config_{model['model']}_profile_{profile}_{date_str}.yaml"
        config_path = os.path.join(args.output_config_dir, config_filename)
        print(f"Saving config file to {config_path}...")
        with open(config_path, 'w') as file:
            yaml.dump(config, file)

        # Generate dialogues using the new config file
        dialogue_output_dir = os.path.join(args.output_dialogue_dir, date_str)
        print(f"Generating dialogues for model: {model['model']} with profile {profile}...")
        os.makedirs(dialogue_output_dir, exist_ok=True)
        os.system(f"python evals/generate_dialogue.py --config {config_path} --output_dir {dialogue_output_dir}")

print("Config files and dialogues generated successfully.") 