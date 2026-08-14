import os
import argparse
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from openai import OpenAI

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def deidentify_text(text, entity_to_placeholder, model, tokenizer, max_length, threshold):
    # Initialize NER pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Tokenize text and handle long texts
    max_length = max_length - 2  # Adjust for special tokens
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

    # Process each chunk and replace entities
    deidentified_chunks = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        entities = nlp(chunk_text)
        
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            if entity['score'] >= threshold:
                label = entity['entity_group']
                placeholder = entity_to_placeholder.get(label, "<PHI>")
                chunk_text = chunk_text[:entity['start']] + placeholder + chunk_text[entity['end']:]
        
        deidentified_chunks.append(chunk_text)

    # Combine the processed chunks back into a single string
    deidentified_text = ' '.join(deidentified_chunks)
    return deidentified_text

def stanford_deidentify_text(text, entity_to_placeholder, model, tokenizer, max_length, threshold):
    # Initialize NER pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Tokenize text and handle long texts
    max_length = max_length - 2  # Adjust for special tokens
    tokens = tokenizer.tokenize(text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

    # Process each chunk and replace entities
    deidentified_chunks = []
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        entities = nlp(chunk_text)
        
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            if entity['score'] >= threshold:
                label = entity['entity_group']
                placeholder = entity_to_placeholder.get(label, "<PHI>")
                chunk_text = chunk_text[:entity['start']] + placeholder + chunk_text[entity['end']:]
        
        deidentified_chunks.append(chunk_text)

    # Combine the processed chunks back into a single string
    deidentified_text = ' '.join(deidentified_chunks)
    return deidentified_text

def extend_redaction(extended_data_df, entity_to_placeholder):
    # Define patterns that might indicate incomplete redaction, including trailing punctuation
    punctuation = r'[.,;:!?"]'  # You can expand this set as needed
    phi_patterns = {
        "PERSON": re.compile(rf"<PERSON>[a-zA-Z]+{punctuation}?|[a-zA-Z]+<PERSON>{punctuation}?"),  # Name partially redacted
        "ORGANIZATION": re.compile(rf"<ORGANIZATION>[a-zA-Z0-9\s]+{punctuation}?|[a-zA-Z0-9\s]+<ORGANIZATION>{punctuation}?"),  # Organization partially redacted
        "DATE_TIME": re.compile(rf"<DATE_TIME>\d+[\/\-]\d+[\/\-]\d+{punctuation}?|\d+[\/\-]\d+[\/\-]\d+<DATE_TIME>{punctuation}?"),  # Date partially redacted
        "PHI": re.compile(rf"<PHI>[a-zA-Z0-9]+{punctuation}?|[a-zA-Z0-9]+<PHI>{punctuation}?")  # Generic PHI partially redacted
    }

    # Function to replace found patterns with full placeholders
    def replace_patterns(text, patterns):
        for label, pattern in patterns.items():
            placeholder = entity_to_placeholder.get(label, "<PHI>")  # Default to generic PHI if no specific placeholder is defined
            matches = pattern.findall(text)
            for match in matches:
                # Replace only the part that matches the pattern, including any trailing punctuation
                text = re.sub(re.escape(match), placeholder, text)
        return text

    # Apply the function to each column containing de-identified messages
    for col in extended_data_df.columns:
        if col.endswith("_de_id_message"):
            extended_data_df[col] = extended_data_df[col].apply(lambda x: replace_patterns(x, phi_patterns))

    return extended_data_df

def load_names(file_paths):
    """Load names from a file into a set."""
    all_names = set()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            names = {line.strip().lower() for line in file}
            all_names.update(names)
    print(len(all_names))
    return all_names


def redact_names(text, names):
    """Redact names in the given text."""
    # Regular expression pattern to match words and punctuation
    pattern = re.compile(r'\b\w+\b')

    def replace_name(match):
        word = match.group(0)
        stripped_word = re.sub(r'^\W+|\W+$', '', word).lower()
        return "[NAME]" if stripped_word in names else word

    redacted_text = re.sub(pattern, replace_name, text)
    return redacted_text

def redact_phi(message, client):
    system_prompt = f"""
    Redact all protected health information (PHI) from the message I share with you. If there is no PHI in the message, then return the original message. Here are some examples:
    
    Message: Hello my name is Bob, I was born in California on January 1st 1990 and my phone number is 3226258081
    Your output: Hello my name is <PHI>, I was born in <PHI> on <PHI> and my phone number is <PHI>

    Message: i spoke with devin yesterday and he told me now to worry about contacting Vandrevala Foundation
    Your output: i spoke with <PHI> yesterday and he told me now to worry about contacting <PHI>

    Message: My number is +91 33 12345678 and please tell Divya to hurry i dont know what else to do
    Your output: My number is <PHI> and please tell <PHI> to hurry i dont know what else to do

    Message: do you think you can help me with a task that will require at least 30 minutes of your time?
    Your output: do you think you can help me with a task that will require at least 30 minutes of your time?

    Message: You can contact me at my email devvrat.dgu@gmail.com or visit me in San Francisco or Los Angeles. whatever is most convenient for you i don’t care
    Your output: You can contact me at my email <PHI> or visit me in <PHI> or <PHI>. whatever is most convenient for you i don’t care
    """
    user_prompt = f"""
    Message: {message}
    Your output: """
    output = client.chat.completions.create(
      model='gpt-4o',
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ],
      temperature=0.2
    )

    formatted_output = output.choices[0].message.content.replace('Your output: ', '')
    formatted_output = ' '.join(formatted_output.split())
    return formatted_output

def de_id_pipeline(
        test,
        data_path,
        first_n_rows,
        model_list,
        max_length,
        threshold,
        entity_to_placeholder,
        save_output_path,
        save_extended_output_path
):
    print(f"Loading data")
    data_df = pd.read_csv(data_path)
    data_df.columns = [col.lower() for col in data_df.columns]
    data_df.rename(columns={'sent_date_and_time': 'sent_date', 'thread_content': 'message'}, inplace=True)
    data_df['sent_date'] = pd.to_datetime(data_df['sent_date'])
    data_df.sort_values(by=['ticket_id', 'sent_date'], ascending=True, inplace=True)
    data_df = data_df.groupby('ticket_id').apply(lambda x: x.head(first_n_rows)).reset_index(drop=True)
    data_df['user'] = np.where(data_df['agent_id'].isnull() | (data_df['agent_id'] == 'None'), 'client', 'counselor')
    data_df = data_df[['id', 'ticket_id', 'sent_date', 'message']]
    data_df = data_df.dropna(subset=['message'])
    data_df.reset_index(inplace=True)
    print(f"Data loaded:\n{data_df}")

    if test:
        print(f"\nTESTING ENV")
        data_df = data_df.head(10)
    
    for model_name in model_list:
        print(f"\nLoading the tokenizer and model for: {model_name}")
        model_suffix = model_name.split('/')[-1]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        print(f"\nRunning de-id pipeline")
        de_id_texts = []
        for index, row in data_df.iterrows():
            text = row['message']
            if index % 100 == 0:
                print(f"Processing row {index}")
            if 'stanford' in model_name:
                de_id_text = deidentify_text(text, entity_to_placeholder, model, tokenizer, max_length, threshold)
            else:
                de_id_text = deidentify_text(text, entity_to_placeholder, model, tokenizer, max_length, threshold)
            de_id_texts.append(de_id_text)
        data_df[f'{model_suffix}_de_id_message'] = de_id_texts # Assign the de-identified texts back to the DataFrame
    
    print(f"\nExtending PHI redaction")
    # Define patterns that might indicate incomplete redaction for the specified categories
    phi_patterns = {
        "PERSON": re.compile(r"<PERSON>[a-zA-Z]+|[a-zA-Z]+<PERSON>"),  # Name partially redacted
        "ORGANIZATION": re.compile(r"<ORGANIZATION>[a-zA-Z0-9\s]+|[a-zA-Z0-9\s]+<ORGANIZATION>"),  # Organization partially redacted
        "DATE_TIME": re.compile(r"<DATE_TIME>\d+[\/\-]\d+[\/\-]\d+|\d+[\/\-]\d+[\/\-]\d+<DATE_TIME>"),  # Date partially redacted
        "PHI": re.compile(r"<PHI>[a-zA-Z0-9]+|[a-zA-Z0-9]+<PHI>")  # Generic PHI partially redacted
    }
    extended_data_df = data_df.copy()
    extended_data_df = extend_redaction(extended_data_df, phi_patterns, entity_to_placeholder)

    print(f"\nSaving eval outputs:\n{data_df}")
    if test:
        path_parts = save_output_path.rsplit('/', 1)
        save_output_path = f"{path_parts[0]}/TEST_{path_parts[1]}"
        path_parts = save_extended_output_path.rsplit('/', 1)
        save_extended_output_path = f"{path_parts[0]}/TEST_{path_parts[1]}"
    data_df.to_excel(save_output_path, index=False)
    extended_data_df.to_excel(save_extended_output_path, index=False)

def run_additional_de_id_pipeline(
    test,
    openai_api_key,
    data_path,
    additional_save_path,
):
    client = OpenAI(
        api_key=openai_api_key,
    )
    
    print(f"\nLoading data")
    data_df = pd.read_csv(data_path)
    data_df.columns = [col.lower() for col in data_df.columns]
    data_df.rename(columns={'sent_date_and_time': 'sent_date', 'thread_content': 'message'}, inplace=True)
    data_df['sent_date'] = pd.to_datetime(data_df['sent_date'])
    data_df.sort_values(by=['ticket_id', 'sent_date'], ascending=True, inplace=True)
    data_df = data_df.groupby('ticket_id').apply(lambda x: x.head(first_n_rows)).reset_index(drop=True)
    data_df['user'] = np.where(data_df['agent_id'].isnull() | (data_df['agent_id'] == 'None'), 'client', 'counselor')
    data_df = data_df[['id', 'ticket_id', 'sent_date', 'message']]
    data_df = data_df.dropna(subset=['message'])
    data_df.reset_index(inplace=True)
    print(f"Data loaded:\n{data_df}")

    print(f"\nLoad annotated data")
    annotations1 = pd.read_excel("/home/stanford/de_id_data/annotation/annotated_ivan_de_id_extended_model_eval_outputs.xlsx")
    annotations2 = pd.read_csv("/home/stanford/de_id_data/annotation/sharang_de_id_extended_model_eval_outputs.csv")
    annotations2 = annotations2[1000:]
    all_annotations = pd.concat([annotations1, annotations2])
    all_annotations = all_annotations[all_annotations['label_deid_roberta_i2b2_de_id_message'].notnull()]

    print(f"\nSelecting test dataset rows")
    data_df = data_df[data_df['index'].isin(all_annotations['index'])]
    print(f"Test dataset:\n{data_df}")

    if test:
        print(f"\nTESTING DATASET")
        data_df = data_df.head(10)
        data_df

    print(f"\nRunning de-id pipeline")
    print(f"OpenAI")
    data_df['deid_gpt4o_message'] = data_df['message'].apply(lambda x: redact_phi(x, client))
    print(f"Regex")
    names_list = load_names(['/home/stanford/vf_copilot/de_id_pipeline/custom_dictionary/indian_names.txt', '/home/stanford/vf_copilot/de_id_pipeline/custom_dictionary/random_names.txt'])
    data_df['deid_regex_message'] = data_df['message'].apply(lambda x: redact_names(x, names_list))

    print(f"\nSaving eval outputs:\n{data_df}")
    if test:
        path_parts = additional_save_path.rsplit('/', 1)
        additional_save_path = f"{path_parts[0]}/TEST_{path_parts[1]}"
    data_df.to_excel(additional_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run de-identification using different models.")
    parser.add_argument("--function", choices=['main', 'additional'], help="Specify which function to run")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key for authentication")
    parser.add_argument("--test", action='store_true', help="Run the script in test mode.")
    args = parser.parse_args()

    data_path = '/home/stanford/de_id_data/phi_data.csv'
    first_n_rows = 10
    model_list = [
        'obi/deid_roberta_i2b2',
        'obi/deid_bert_i2b2',
        'StanfordAIMI/stanford-deidentifier-base',
    ]
    max_length = 512
    threshold = 0.45 # lower values = more strict scrubbing
    # Define a mapping from entity labels to placeholders
    entity_to_placeholder = {
        "PER": "<PERSON>",
        "ORG": "<ORGANIZATION>",
        "DATE": "<DATE_TIME>",
    }
    save_output_path = '/home/stanford/de_id_data/de_id_model_eval_outputs.xlsx'
    save_extended_output_path = '/home/stanford/de_id_data/de_id_extended_model_eval_outputs.xlsx'

    additional_save_path = "/home/stanford/de_id_data/additional_de_id_model_eval_outputs.xlsx"

    if args.function == 'main':
        de_id_pipeline(
            test=args.test,
            data_path=data_path,
            first_n_rows=first_n_rows,
            model_list=model_list,
            max_length=max_length,
            threshold=threshold,
            entity_to_placeholder=entity_to_placeholder,
            save_output_path=save_output_path,
            save_extended_output_path=save_extended_output_path,
            )
    elif args.function == 'additional':
        run_additional_de_id_pipeline(
            test=args.test,
            openai_api_key=args.openai_api_key,
            data_path=data_path,
            additional_save_path=additional_save_path,
        )