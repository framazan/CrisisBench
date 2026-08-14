import yaml
import argparse
import os
from dotenv import load_dotenv

# internal lightweight client (OpenAI-compatible endpoints)
from crisisbench.multi_turn.clients.llm_client import chat_complete

# Load .env file if it exists
if os.path.exists('.env'):
    load_dotenv()

# Fetch the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Load prompts from a YAML file
def load_prompts(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

###############################################################################
# Wrapper to call the LLM via chat_complete
###############################################################################

def _call_llm(
    *,
    model: str,
    api_url: str,
    api_key_env: str,
    temperature: float,
    system_prompt: str,
    human_prompt: str,
) -> str:
    """Single call helper returning assistant content."""
    # if the system prompt is empty, don't include it
    if system_prompt and system_prompt != "":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": human_prompt},
        ]
    return chat_complete(
        model=model,
        api_url=api_url,
        api_key_env=api_key_env,
        temperature=temperature,
        messages=messages
    )

# Function to process a patient message (Crisis Counselor) with optional phase selection
def process_patient_message(
    *,
    patient_message: str,
    counselor_prompt_template: str,
    model: str,
    api_url: str,
    api_key_env: str,
    temperature: float,
    phase_selection_template: str | None = None,
    editor_prompt_template: str | None = None,
    prompt_dictionary: dict | None = None,
):
    phase_selected = None
    
    # Step 1: Optional Phase Selection
    if phase_selection_template:
        phase_selected = _call_llm(
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            temperature=temperature,
            system_prompt=phase_selection_template,
            human_prompt=patient_message,
        )
        print(f"Phase Selected: {phase_selected}")
        counselor_prompt_template += prompt_dictionary['phased_prompt_dictionary'][phase_selected]

    # Step 2: Create the counselor chain with optional phase selection
    draft_response = _call_llm(
        model=model,
        api_url=api_url,
        api_key_env=api_key_env,
        temperature=temperature,
        system_prompt=counselor_prompt_template,
        human_prompt=f"{phase_selected}\n{patient_message}" if phase_selected else patient_message,
    )

    # Step 3: Optionally use the editor agent to refine the draft
    if editor_prompt_template:
        # Fill placeholders
        human_prompt = (
            f"{editor_prompt_template.strip()}\n\n"
            f"Conversation history:\n{patient_message}\n\n"
            f"New message from counselor:\n{draft_response}\n"
            f"Output: "
        )

        
        final_response = _call_llm(
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            temperature=temperature,
            system_prompt="",  # treat full template as user prompt
            human_prompt=human_prompt,
        )
    else:
        final_response = draft_response

    return final_response

# Function to process a counselor message (Patient Agent)
def process_counselor_message(
    *,
    counselor_message: str,
    patient_prompt_template: str,
    model: str,
    api_url: str,
    api_key_env: str,
    temperature: float,
    editor_prompt_template: str | None = None,
):
    draft_response = _call_llm(
        model=model,
        api_url=api_url,
        api_key_env=api_key_env,
        temperature=temperature,
        system_prompt=patient_prompt_template,
        human_prompt=counselor_message,
    )
    
    # Optionally use the editor agent to refine the draft
    if editor_prompt_template:
        # Fill placeholders
        human_prompt = (
            editor_prompt_template.replace("{conversation_history}", counselor_message)
            .replace("{message_draft}", draft_response)
        )
        patient_response = _call_llm(
            model=model,
            api_url=api_url,
            api_key_env=api_key_env,
            temperature=temperature,
            system_prompt="",
            human_prompt=human_prompt,
        )
    else:
        patient_response = draft_response
    
    return patient_response

# Main function to handle command-line arguments and call the appropriate workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Agent Script for Crisis Counselor and Patient")

    # Common arguments
    parser.add_argument('--config_path', required=True, help="Path to the YAML configuration file")
    parser.add_argument('--role', required=True, choices=["counselor", "patient"], help="Specify which agent to use: counselor or patient")
    parser.add_argument('--model', help="Specify the model to use (e.g., gpt-3.5-turbo, gpt-4). Default is gpt-4.")
    parser.add_argument('--temperature', type=float, default=0.0, help="Set the model's temperature (creativity). Default is 0 (deterministic).")

    # Crisis Counselor-specific arguments
    parser.add_argument('--counselor_prompt', required=False, help="Name of the counselor prompt in the YAML file")
    parser.add_argument('--editor_prompt', required=False, help="Name of the editor prompt (optional) in the YAML file")
    parser.add_argument('--phase_selection_prompt', required=False, help="Name of the phase selection prompt (optional) in the YAML file")
    parser.add_argument('--patient_message', required=False, help="Message from the patient")

    # Patient-specific arguments
    parser.add_argument('--patient_prompt', required=False, help="Name of the patient prompt in the YAML file")
    parser.add_argument('--counselor_message', required=False, help="Message from the counselor")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the prompts from the YAML file
    prompts = load_prompts(args.config_path)
    
    # Handle logic based on role
    if args.role == "counselor":
        if not args.counselor_prompt or not args.patient_message:
            parser.error("--counselor_prompt and --patient_message are required when role is 'counselor'.")
        
        counselor_prompt_template = prompts[args.counselor_prompt]
        editor_prompt_template = prompts.get(args.editor_prompt)
        phase_selection_template = prompts.get(args.phase_selection_prompt)

        # Process patient message via the counselor agent with optional phase selection
        response = process_patient_message(
            patient_message=args.patient_message,
            counselor_prompt_template=counselor_prompt_template,
            model=args.model,
            api_url=os.getenv("API_URL", "https://api.openai.com/v1/chat/completions"),
            api_key_env=os.getenv("API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=args.temperature,
            phase_selection_template=phase_selection_template,
            editor_prompt_template=editor_prompt_template,
            prompt_dictionary=prompts
        )

    elif args.role == "patient":
        if not args.patient_prompt or not args.counselor_message:
            parser.error("--patient_prompt and --counselor_message are required when role is 'patient'.")
        
        patient_prompt_template = prompts[args.patient_prompt]

        # Process counselor message via the patient agent
        response = process_counselor_message(
            counselor_message=args.counselor_message,
            patient_prompt_template=patient_prompt_template,
            model=args.model,
            api_url=os.getenv("API_URL", "https://api.openai.com/v1/chat/completions"),
            api_key_env=os.getenv("API_KEY_ENV", "OPENAI_API_KEY"),
            temperature=args.temperature,
        )

# Entry point for the script
if __name__ == "__main__":
    main()
