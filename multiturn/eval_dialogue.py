import os
import yaml
import argparse
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils.data_processing import extract_json_from_text
from multiturn.generate_response import format_prompt_for_model

# Load environment variables from .env file if needed
load_dotenv()

# Load the YAML configuration file (e.g., agents_config.yaml)
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Main function to create and evaluate the conversation
def evaluate_conversation(conversation_file, prompt_template, llm, patient_config=None):
    # Read the conversation from the text file
    with open(conversation_file, 'r') as f:
        conversation_text = f.read()
    
    # If the patient config is specified, load the patient profile
    if patient_config:
        with open(patient_config, 'r') as f:
            patient_prompts = yaml.safe_load(f)
        
        patient_name = conversation_file.split('|')[1]
        patient_profile = patient_prompts[patient_name]
        
        human_prompt = f"""
        Patient profile: {patient_profile}

        --------------------------------

        Dialogue: {conversation_text}
        """
    else:
        human_prompt = f"Dialogue: {conversation_text}"
    
    # Format the prompt using the helper function
    prompt = format_prompt_for_model(llm.model_name, prompt_template, human_prompt)

    # Create a RunnableSequence, equivalent to the previous LLMChain, that combines the prompt and the language model
    chain = prompt | llm

    # Run the chain to evaluate the conversation and get the result
    evaluation_result = chain.invoke({}).content

    return evaluation_result


# Main function to evaluate conversations and save the outputs as JSON
def evaluate_conversations(config, input_path, output_dir, evaluator_prompts, patient_config=None,
                           model="gpt-4o", temperature=0.2):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the LLM (using OpenAI's ChatGPT model, customizable)
    if "o1" in model:
        temperature = 1.0
        response_format_dict = None
    else:
        response_format_dict = {"type": "json_object"}
    
    llm = ChatOpenAI(model=model, temperature=temperature)  # Customize model and temperature as needed

    all_results = {}
    
    # Check if the input path is a directory or a file
    if os.path.isdir(input_path):
        # List all .txt files in the directory
        files_to_process = [os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.endswith(".txt")]
    elif os.path.isfile(input_path):
        # Process only the specified file
        files_to_process = [input_path]
    else:
        raise ValueError(f"The input path '{input_path}' is neither a valid file nor a directory.")
    
    # Loop through the files to process
    for conversation_file in files_to_process:
        combined_results = {}
        filename = os.path.basename(conversation_file)

        # Loop through each evaluator prompt
        for prompt_name in evaluator_prompts:
            print("Evaluating conversation with prompt:", prompt_name)
            evaluator_prompt_template = config[prompt_name]['prompt']

            # Evaluate the conversation using the LangChain evaluator agent
            evaluation_result = evaluate_conversation(conversation_file, evaluator_prompt_template, llm, patient_config)

            # Extract and store the result in the combined dictionary
            combined_results[prompt_name] = extract_json_from_text(evaluation_result)

        # Create output JSONL file name
        jsonl_filename = filename.replace(".txt", ".jsonl")
        output_file_path = os.path.join(output_dir, jsonl_filename)

        # Save the combined evaluator output as a JSONL file, appending to the file if it exists
        with open(output_file_path, 'a') as jsonl_file:
            # Add the combined result as a new JSON object on a new line
            jsonl_file.write(json.dumps(combined_results) + '\n')
        
        all_results[filename] = combined_results

        print(f"Evaluator outputs saved to: {output_file_path}")
    
    return all_results

# Entry point
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluator agent script to process conversation text files")
    parser.add_argument('--config', required=True, help="Path to the YAML configuration file (e.g., agents_config.yaml)")
    parser.add_argument('--input_dir', required=True, help="Path to the directory with conversation text files")
    parser.add_argument('--output_dir', required=True, help="Path to the directory where JSON outputs will be saved")
    parser.add_argument('--evaluator_prompts', nargs='+', required=True, help="Names of the evaluator prompts from the config file (space-separated for multiple)")
    parser.add_argument('--patient_config', required=False, help="Path to the patient profile configuration file")
    parser.add_argument('--model', default="gpt-4o", help="Specify the model to use (e.g., gpt-3.5-turbo, gpt-4). Default is gpt-4.")
    parser.add_argument('--temperature', type=float, default=0.2, help="Set the model's temperature (creativity). Default is 0.2.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load configuration file
    config = load_config(args.config)

    # Evaluate the conversations and save the outputs
    print("Calling eval_dialogue.py")
    print(f"config: {config}")
    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"evaluator_prompts: {args.evaluator_prompts}")
    print(f"patient_config: {args.patient_config}")
    print(f"model: {args.model}")
    print(f"temperature: {args.temperature}")
    evaluate_conversations(config, args.input_dir, args.output_dir, args.evaluator_prompts, patient_config=args.patient_config,
                           model=args.model, temperature=args.temperature)

    # Example usage:
    # python eval_dialogue.py --config agents_config.yaml --input_dir conversations/ --output_dir evaluator_outputs/ --evaluator_prompts language_patterns presenting_concern realism --patient_config patient_profiles.yaml
