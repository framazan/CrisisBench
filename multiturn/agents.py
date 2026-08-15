import yaml
import os
from dotenv import load_dotenv

# internal lightweight client (OpenAI-compatible endpoints)
from multiturn.clients.llm_client import chat_complete

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
