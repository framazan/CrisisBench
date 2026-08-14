import yaml
import argparse
import os
from datetime import datetime
from crisisbench.multi_turn.generate_response import process_patient_message, process_counselor_message, load_prompts
import timeit

# Replace generate_response with direct function calls, adding support for phase selection
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
        # Process using the counselor agent and optional editor
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
        # Process using the patient agent
        return process_counselor_message(
            counselor_message=message,
            patient_prompt_template=prompt,
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            temperature=temperature,
            editor_prompt_template=editor_prompt
        )


# Helper function to handle phase selection logic
def process_phase_selection(patient_message, phase_selection_template, llm):
    phase_selection_chain = create_chain(phase_selection_template, llm, role="phase_selection")
    phase_selected = phase_selection_chain.invoke({"patient_message": patient_message}).content
    print(f"Phase Selected: {phase_selected}")
    return phase_selected


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
    
    patient_editor_prompt_config_path = config['patient_editor_config_path']
    patient_editor_prompt = config['prompts']['patient_editor_prompt']
    
    counselor_prompt = config['prompts']['counselor_prompt'] 
    editor_prompt = config['prompts']['editor_prompt']
    
    patient_prompt = config['prompts']['patient_prompt']
    
    phase_selection_prompt = config['prompts'].get('phase_selection_prompt')  # Optional phase selection prompt
    # conversation_turns = config['conversation_turns']
    counselor_model = config['counselor_model']
    patient_model = config['patient_model']
    
    counselor_temperature = config['counselor_temperature']
    patient_temperature = config['patient_temperature']
    
    counselor_api_url = config.get('counselor_api_url', os.getenv('COUNSELOR_API_URL', 'https://api.openai.com/v1/chat/completions'))
    counselor_api_key_env = config.get('counselor_api_key_env', os.getenv('COUNSELOR_API_KEY_ENV', 'OPENAI_API_KEY'))

    patient_api_url = config.get('patient_api_url', os.getenv('PATIENT_API_URL', 'https://api.openai.com/v1/chat/completions'))
    patient_api_key_env = config.get('patient_api_key_env', os.getenv('PATIENT_API_KEY_ENV', 'OPENAI_API_KEY'))
    
    # Simulate a back-and-forth conversation until the string "[end conversation]" is detected
    while "[end conversation]" not in conversation_history[-1] and len(conversation_history) < config['max_messages'] * 2:
        # Step 1: Patient response to the counselor's message
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
        
        # Split the patient response into multiple messages if it contains the delimiter "//"
        patient_messages = patient_response.split("//")
        for message in patient_messages:
            print(f"patient: {message}")
            conversation_history.append(f"patient: {message}")
            conversation_history.append("\n")
        if "[end conversation]" in patient_response:
            break
        
        # Step 2: Counselor response to the patient message (with optional phase selection)
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
            phase_selection_prompt=phase_selection_prompt,  # Add phase selection prompt to the call
            temperature=counselor_temperature
        )
        # conversation_history.append(f"counselor: {counselor_response}")
        # conversation_history.append("\n")
        
         # Split the patient response into multiple messages if it contains the delimiter "//"
        counselor_messages = counselor_response.split("//")
        for message in counselor_messages:
            print(f"counselor: {message}")
            conversation_history.append(f"counselor: {message}")
            conversation_history.append("\n")
        # print(f"counselor: {counselor_response}")
        
        if "[end conversation]" in counselor_response:
            break

    # Output the full conversation
    conversation_text = "\n".join(conversation_history)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"dialogue|{patient_prompt}|{counselor_prompt}|{editor_prompt if editor_prompt else 'none'}|{counselor_model}|{timestamp}.txt")

    # Save the conversation to the output file
    with open(output_file_path, 'w') as f:
        f.write(conversation_text)

    print(f"\nConversation saved to: {output_file_path}")
    
    out = {}
    out[os.path.basename(output_file_path)] = conversation_text
    
    return out

def main():
    parser = argparse.ArgumentParser(description="Dialogue generation script with configurable YAML file")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file")
    parser.add_argument('--output_dir', required=True, help="Path to the output directory where the dialogue will be saved")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Generate the dialogue using the loaded configuration and save it to the output directory
    print(f"generate_dialogue.py with config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    generate_dialogue(config, args.output_dir)

# Entry point
if __name__ == "__main__":
    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    print(f"Execution time: {elapsed_time:.2f} seconds")
