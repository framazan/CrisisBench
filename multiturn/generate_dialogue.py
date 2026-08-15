import os
import argparse
import yaml
from datetime import datetime
import timeit

from multiturn.agents import process_patient_message, process_counselor_message, load_prompts

def generate_response(role, message, prompt, config_path, prompt_suffix=None, editor_prompt=None, 
                      model=None, phase_selection_prompt=None,
                      editor_prompt_config=None, temperature=0.7, api_url=None, api_key_env=None):
    # Load prompts from the configuration file
    prompts = load_prompts(config_path)
    editor_prompts = load_prompts(editor_prompt_config) if editor_prompt_config else prompts

    if prompt_suffix:
        prompt = prompts[prompt] + prompt_suffix
        editor_prompt = editor_prompts[editor_prompt] + prompt_suffix if editor_prompt else None
    else:
        prompt = prompts[prompt]
        editor_prompt = editor_prompts[editor_prompt] if editor_prompt else None

    if role == "counselor":
        return process_patient_message(
            patient_message=message,
            counselor_prompt_template=prompt,
            phase_selection_template=prompts.get(phase_selection_prompt), 
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            editor_prompt_template=editor_prompt,
            temperature=temperature,
            prompt_dictionary=prompts
        )
    elif role == "patient":
        return process_counselor_message(
            counselor_message=message,
            patient_prompt_template=prompt,
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            temperature=temperature,
            editor_prompt_template=editor_prompt
        )

# Load the configuration file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Main function to generate the dialogue
def generate_dialogue(config, output_dir):
    conversation_history = []

    # Load initial counselor message from config
    counselor_message = config['initial_counselor_message']
    conversation_history.append(f"counselor: {counselor_message}")
    print("counselor: " + counselor_message)
    
    # Get paths, prompts, and settings from config
    counselor_config_path = config['counselor_config_path']
    patient_config_path = config['patient_config_path']
    
    patient_editor_prompt_config_path = config.get('patient_editor_config_path')
    patient_editor_prompt = config['prompts'].get('patient_editor_prompt')
    
    counselor_prompt = config['prompts']['counselor_prompt'] 
    editor_prompt = config['prompts'].get('editor_prompt')
    
    patient_prompt = config['prompts']['patient_prompt']
    
    phase_selection_prompt = config['prompts'].get('phase_selection_prompt')
    counselor_model = config['counselor_model']
    patient_model = config['patient_model']
    
    counselor_temperature = config.get('counselor_temperature', 0.0)
    patient_temperature = config.get('patient_temperature', 0.0)
    
    counselor_api_url = config.get('counselor_api_url', os.getenv('COUNSELOR_API_URL', 'https://api.openai.com/v1/chat/completions'))
    counselor_api_key_env = config.get('counselor_api_key_env', os.getenv('COUNSELOR_API_KEY_ENV', 'OPENAI_API_KEY'))

    patient_api_url = config.get('patient_api_url', os.getenv('PATIENT_API_URL', 'https://api.openai.com/v1/chat/completions'))
    patient_api_key_env = config.get('patient_api_key_env', os.getenv('PATIENT_API_KEY_ENV', 'OPENAI_API_KEY'))
    
    # Simulate a back-and-forth conversation until the string "[end conversation]" is detected
    while "[end conversation]" not in conversation_history[-1] and len(conversation_history) < config.get('max_messages', 20) * 2:
        # Step 1: Patient response
        patient_response = generate_response(
            role="patient",
            message='\n'.join(conversation_history),
            prompt=patient_prompt,
            editor_prompt=patient_editor_prompt,
            editor_prompt_config=patient_editor_prompt_config_path,
            config_path=patient_config_path,
            model=patient_model,
            api_url=patient_api_url,
            api_key_env=patient_api_key_env,
            temperature=patient_temperature
        )
        
        patient_messages = patient_response.split("//")
        for message in patient_messages:
            print(f"patient: {message}")
            conversation_history.append(f"patient: {message}")
            conversation_history.append("\n")
        if "[end conversation]" in patient_response:
            break
        
        # Step 2: Counselor response
        counselor_response = generate_response(
            role="counselor",
            message='\n'.join(conversation_history),
            prompt=counselor_prompt,
            prompt_suffix="You can (optionally) send multiple messages in a row, separated by the delimiter '//'. Maintain variety in message lengths and number of messages per counselor reply\nIf you want to end the conversation, type \"[end conversation]\".",
            config_path=counselor_config_path,
            editor_prompt=editor_prompt,
            model=counselor_model,
            api_url=counselor_api_url,
            api_key_env=counselor_api_key_env,
            phase_selection_prompt=phase_selection_prompt,
            temperature=counselor_temperature
        )
        
        counselor_messages = counselor_response.split("//")
        for message in counselor_messages:
            print(f"counselor: {message}")
            conversation_history.append(f"counselor: {message}")
            conversation_history.append("\n")
        
        if "[end conversation]" in counselor_response:
            break

    conversation_text = "\n".join(conversation_history)

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"dialogue|{patient_prompt}|{counselor_prompt}|{editor_prompt if editor_prompt else 'none'}|{counselor_model}|{timestamp}.txt")

    with open(output_file_path, 'w') as f:
        f.write(conversation_text)

    print(f"\nConversation saved to: {output_file_path}")
    
    out = {}
    out[os.path.basename(output_file_path)] = conversation_text
    
    return out

def process_all_dialogue_configs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".yaml"):
            input_file_path = os.path.join(input_dir, filename)
            print(f"Generating dialogue for {input_file_path}")
            config = load_config(input_file_path)
            generate_dialogue(config, output_dir)

def single_turn_response(args):
    prompts = load_prompts(args.config_path)
    
    if args.role == "counselor":
        if not args.counselor_prompt or not args.patient_message:
            print("Error: --counselor_prompt and --patient_message are required when role is 'counselor'.")
            return
        response = process_patient_message(
            patient_message=args.patient_message,
            counselor_prompt_template=prompts[args.counselor_prompt],
            model=args.model,
            api_url=os.getenv("API_URL", "https://api.openai.com/v1/chat/completions"),
            api_key_env=os.getenv("API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=args.temperature,
            phase_selection_template=prompts.get(args.phase_selection_prompt),
            editor_prompt_template=prompts.get(args.editor_prompt),
            prompt_dictionary=prompts
        )
    elif args.role == "patient":
        if not args.patient_prompt or not args.counselor_message:
            print("Error: --patient_prompt and --counselor_message are required when role is 'patient'.")
            return
        response = process_counselor_message(
            counselor_message=args.counselor_message,
            patient_prompt_template=prompts[args.patient_prompt],
            model=args.model,
            api_url=os.getenv("API_URL", "https://api.openai.com/v1/chat/completions"),
            api_key_env=os.getenv("API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=args.temperature,
        )
    print("\n--- Generated Response ---")
    print(response)

def main():
    parser = argparse.ArgumentParser(description="Unified Dialogue Generation Script")
    parser.add_argument('--mode', required=True, choices=['single_turn', 'dialogue', 'batch_dialogues'], help="Mode of execution")
    
    # Arguments for 'dialogue' and 'batch_dialogues' mode
    parser.add_argument('--config', help="Path to the YAML configuration file (used in 'dialogue' mode)")
    parser.add_argument('--config_dir', help="Directory containing YAML configuration files (used in 'batch_dialogues' mode)")
    parser.add_argument('--output_dir', help="Path to the output directory where dialogues will be saved")

    # Arguments for 'single_turn' mode
    parser.add_argument('--config_path', help="Path to the YAML configuration file (used in 'single_turn' mode)")
    parser.add_argument('--role', choices=["counselor", "patient"], help="Agent role for single turn")
    parser.add_argument('--model', default="gpt-4", help="Model to use for single turn")
    parser.add_argument('--temperature', type=float, default=0.0, help="Temperature for single turn")
    parser.add_argument('--counselor_prompt', help="Counselor prompt name")
    parser.add_argument('--patient_prompt', help="Patient prompt name")
    parser.add_argument('--editor_prompt', help="Editor prompt name")
    parser.add_argument('--phase_selection_prompt', help="Phase selection prompt name")
    parser.add_argument('--patient_message', help="Message from patient")
    parser.add_argument('--counselor_message', help="Message from counselor")

    args = parser.parse_args()

    if args.mode == "single_turn":
        single_turn_response(args)
    elif args.mode == "dialogue":
        if not args.config or not args.output_dir:
            parser.error("--config and --output_dir are required for 'dialogue' mode.")
            return
        config = load_config(args.config)
        generate_dialogue(config, args.output_dir)
    elif args.mode == "batch_dialogues":
        if not args.config_dir or not args.output_dir:
            parser.error("--config_dir and --output_dir are required for 'batch_dialogues' mode.")
            return
        process_all_dialogue_configs(args.config_dir, args.output_dir)

if __name__ == "__main__":
    main()
