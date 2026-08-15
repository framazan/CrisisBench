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

pattern = re.compile(r'(?<!^)(?<!\n)(Client:|Counselor:)')
def clean_convo(text):
    """Insert a newline before every Client: or Counselor: label."""
    if not isinstance(text, str):
        return text
    return pattern.sub(r'\n\1', text.strip())


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
      model='gpt-4.1',
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
      ],
      temperature=0.2
    )

    formatted_output = output.choices[0].message.content.replace('Your output: ', '')
    formatted_output = ' '.join(formatted_output.split())
    return formatted_output

def openai_de_id_pipeline(
    test,
    openai_api_key,
    data_path,
    columns_to_deid_list,
    save_output_path,
):
    client = OpenAI(
        api_key=openai_api_key,
    )
    
    print(f"\nLoading data")
    data_df = pd.read_csv(data_path)
    data_df.columns = [col.lower() for col in data_df.columns]
    print(f"Data loaded:\n{data_df}")

    cross_walk_df = data_df[['uid']].copy()
    cross_walk_df['ai_convo'] = False if 'non_ai' in crosswalk_path else True
    print(f"\nSaving crosswalk to {crosswalk_path}")
    cross_walk_df.to_csv(crosswalk_path, index=False)

    if test:
        print(f"\nTESTING DATASET")
        data_df = data_df.head(10)
        data_df

    for column_to_deid in columns_to_deid_list:
        print(f"\nRunning OpenAI de-id pipeline for column '{column_to_deid}'")
        deid_column_name = f'deid_{column_to_deid}'
        if deid_column_name not in data_df.columns:
            data_df[deid_column_name] = None
        for count, row in enumerate(data_df.itertuples(index=True), start=1):
            if pd.notna(data_df.at[row.Index, deid_column_name]):
                continue

            print(f"Processing row {count} of {len(data_df)} (column '{column_to_deid}')")
            val = getattr(row, column_to_deid)
            if pd.notna(val):
                deid_val = redact_phi(val, client)
            else:
                deid_val = val
            data_df.at[row.Index, deid_column_name] = deid_val
        data_df.drop(columns=[column_to_deid], inplace=True)

    data_df['deid_conversation'] = data_df['deid_conversation'].apply(clean_convo)


    print(f"\nSaving eval outputs:\n{data_df}")
    if test:
        path_parts = save_output_path.rsplit('/', 1)
        save_output_path = f"{path_parts[0]}/TEST_{path_parts[1]}"
    data_df.to_excel(save_output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run de-identification using different models.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key for authentication")
    parser.add_argument("--test", action='store_true', help="Run the script in test mode.")
    args = parser.parse_args()

    data_path = '/home/stanford/vf_copilot/log_analysis/sampled_non_ai_conversations.csv'
    crosswalk_path = '/home/stanford/vf_copilot/evals/human_data/july_2025_uid_non_ai_crosswalk.csv'
    columns_to_deid_list = ['conversation']
    save_output_path = '/home/stanford/vf_copilot/log_analysis/sampled_non_ai_conversations_de_id.xlsx'

    openai_de_id_pipeline(
        test=args.test,
        openai_api_key=args.openai_api_key,
        data_path=data_path,
        columns_to_deid_list=columns_to_deid_list,
        save_output_path=save_output_path,
    )