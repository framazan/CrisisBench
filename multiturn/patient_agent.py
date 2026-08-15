"""
Patient Agent Script

This script allows the creation of a "Patient Agent" that processes a message from a counselor
and generates a patient response using a predefined template.

The script uses LangChain to create a chain:
    1. A patient agent that generates a response based on the counselor's message.

The prompt for the patient agent is loaded from a YAML configuration file,
allowing flexibility to modify the behavior of the agent by editing the YAML file.

Command-Line Arguments:
------------------------
--config_path : str
    Required. Path to the YAML configuration file containing the patient prompt.

--patient_prompt : str
    Required. The name of the patient prompt in the YAML file.

--counselor_message : str
    Required. The message from the counselor that needs to be processed by the patient agent.

Usage Examples:
---------------
$ python patient_agent.py --config_path config.yaml --counselor_message "Please take deep breaths and tell me how you're feeling." --patient_prompt patient_response
"""

import yaml
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Load prompts from a YAML file
def load_prompts(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize the Language Models (LLMs)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Function to create a patient response chain
def create_patient_chain(patient_prompt_template):
    # Create prompt template for the patient
    patient_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", patient_prompt_template),
            ("human", "{counselor_message}")
        ]
    )

    # Create chain for the patient
    patient_chain = patient_prompt | llm

    return patient_chain

# Function to process a counselor message and generate a patient response
def process_counselor_message(counselor_message, patient_prompt_template):
    # Create the patient chain
    patient_chain = create_patient_chain(patient_prompt_template)

    # Generate the patient's response
    patient_response = patient_chain.invoke({"counselor_message": counselor_message}).content

    return patient_response

# Main function to handle command-line arguments and call the workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Patient Agent Script")

    # Add argument for the path to the YAML configuration file
    parser.add_argument('--config_path', required=True, help="Path to the YAML configuration file")
    parser.add_argument('--patient_prompt', required=True, help="Name of the patient prompt in the YAML file")
    parser.add_argument('--counselor_message', required=True, help="The message from the counselor")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the prompts from the YAML file
    prompts = load_prompts(args.config_path)

    # Get the patient prompt template
    patient_prompt_template = prompts[args.patient_prompt]

    # Call the process function with the provided arguments
    response = process_counselor_message(
        counselor_message=args.counselor_message,
        patient_prompt_template=patient_prompt_template
    )

    # Print the final response
    print("Patient Response:\n", response)

# Entry point for the script
if __name__ == "__main__":
    main()
