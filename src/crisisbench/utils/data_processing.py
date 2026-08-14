import re
import json
import pandas as pd
import time
import yaml

from crisisbench.utils.gpt_query import query_gpt

with open("evals/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
    
retry_time = config['retry_delay_seconds']

def extract_json_from_text(input_string):
    try:
        match = re.search(r'\{.*\}', input_string, re.DOTALL)
        if not match:
            raise ValueError("No JSON-like content found.")
        json_string = match.group(0)
        json_data = json.loads(json_string)
        return json_data
    except:
        return {"result": "Unable to parse JSON",
                "original_response": input_string}

def convert_series_to_list(evals_dict):
    for key, value in evals_dict.items():
        if isinstance(value, pd.Series):
            evals_dict[key] = value.tolist()
        elif isinstance(value, dict):
            evals_dict[key] = convert_series_to_list(value)
    return evals_dict

def process_row(row, conversation_col, model, sys_prompt, full_history=False):
    full_query = row[conversation_col] if not full_history else f"Conversation history:\n{row['conversation_history']}\n Latest response from counselor: {row[conversation_col]}"
    
    try:
        result = query_gpt(conversation_history=full_query, system_prompt=sys_prompt, model=model)
    except:
        time.sleep(retry_time)
        result = query_gpt(conversation_history=full_query, system_prompt=sys_prompt, model=model)
        
    parsed = extract_json_from_text(result)
    parsed['exp_id'] = row['exp_id']
    
    return parsed
