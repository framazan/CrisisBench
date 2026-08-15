"""
Crisis Counselor Agent Script

This script allows the creation of a "Crisis Counselor" agent that processes a patient message
and drafts a response using predefined templates. Optionally, the response can be passed
through an editor agent to refine and improve the response.

The script uses LangChain to create two chains:
    1. A crisis counselor agent that drafts a response.
    2. An optional editor agent that refines the drafted response.

The prompts for both the counselor and editor agents are loaded from a YAML configuration file,
allowing flexibility to modify the behavior of the agents by editing the YAML file.

Command-Line Arguments:
------------------------
--config_path : str
    Required. Path to the YAML configuration file containing the counselor and editor prompts.
    
--counselor_prompt : str
    Required. The name of the counselor prompt in the YAML file.

--editor_prompt : str, optional
    The name of the editor prompt in the YAML file. If not provided, the editor agent is skipped.

--patient_message : str
    Required. The message from the patient that needs to be processed by the counselor agent.

Usage Examples:
---------------
1. With editor agent:
    $ python crisis_counselor.py --config_path config.yaml --patient_message "I feel really overwhelmed and don't know what to do." --counselor_prompt counselor --editor_prompt editor

2. Without editor agent:
    $ python crisis_counselor.py --config_path config.yaml --patient_message "I feel really overwhelmed and don't know what to do." --counselor_prompt counselor
"""

import yaml
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import RunnableSequence

load_dotenv()

# Load prompts from a YAML file
def load_prompts(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

# Initialize the Language Models (LLMs)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Function to create chains for counselor and editor
def create_chains(counselor_prompt_template, editor_prompt_template=None):
    # Create prompt templates for the counselor
    counselor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", counselor_prompt_template),
            ("human", "{patient_message}")
        ]
    )
    
    # Create chain for the counselor
    counselor_chain = counselor_prompt | llm

    # Create chain for the editor if provided
    if editor_prompt_template:
        editor_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", editor_prompt_template),
                ("human", "{draft_response}")
            ]
        )
        editor_chain = editor_prompt | llm
    else:
        editor_chain = None

    return counselor_chain, editor_chain

# Function to process a patient message through the workflow
def process_patient_message(patient_message, counselor_prompt_template, editor_prompt_template=None):
    # Create chains for both agents using the loaded prompts
    counselor_chain, editor_chain = create_chains(counselor_prompt_template, editor_prompt_template)

    # Step 1: Counselor drafts a response
    draft_response = counselor_chain.invoke({"patient_message": patient_message}).content

    # Step 2: Optionally use the editor agent to refine the draft
    if editor_chain:
        final_response = editor_chain.invoke({"draft_response": draft_response}).content
    else:
        final_response = draft_response

    # Return the final response (either edited or draft)
    return final_response

# Main function to handle command-line arguments and call the workflow
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Crisis Counselor Agent Script")

    # Add argument for the path to the YAML configuration file
    parser.add_argument('--config_path', required=True, help="Path to the YAML configuration file")
    parser.add_argument('--counselor_prompt', required=True, help="Name of the counselor prompt in the YAML file")
    parser.add_argument('--editor_prompt', required=False, default=None, help="Name of the editor prompt (second call) in the YAML file")
    parser.add_argument('--patient_message', required=True, help="The message from the patient")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the prompts from the YAML file
    prompts = load_prompts(args.config_path)

    # Get the counselor prompt
    counselor_prompt_template = prompts[args.counselor_prompt]

    # Get the editor prompt (if provided)
    if args.editor_prompt:
        editor_prompt_template = prompts.get(args.editor_prompt)
    else:
        editor_prompt_template = None

    # Call the process function with the provided arguments
    response = process_patient_message(
        patient_message=args.patient_message, 
        counselor_prompt_template=counselor_prompt_template, 
        editor_prompt_template=editor_prompt_template
    )

    # Print the final response
    print("Final Response:\n", response)

# Entry point for the script
if __name__ == "__main__":
    main()
