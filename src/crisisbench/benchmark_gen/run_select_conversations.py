import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch

def load_historical_data():
    """
    this code is already written, refer to the rag_dynamic_prompting() function in app.py and also outside of the prompt you should find additional code
    for this task:
    This function loads the historical data. First time using the data I already embeded:
    1. load the client embeddings
    2. load the client messages
    3. load the full conversation data
    4. return 4 objects
    """
    pass

def get_embedding(text, tokenizer, model):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state[:, 0, :].numpy()

def select_historical_conversations(
        tokenized_historical_data_path,
        model_name,
        rag_top_k,
        save_data_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    with open(os.path.expanduser(json_definitions_path), 'r') as file:
        data = json.load(file)
        categories = data['categories']

    df = pd.read_csv(tokenized_historical_data_path)
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: eval(x))

    results = pd.DataFrame()
    for category in categories:
        print("Processing category:", category['name'])
        cat_embedding = get_embedding(category['definition'], tokenizer, model)
        df['cosine_sim'] = df['tokenized_text'].apply(lambda x: cosine_similarity(cat_embedding, get_embedding(x)).flatten()[0])
        sorted_df = df.sort_values(by='cosine_sim', ascending=False)
        top_conversations = sorted_df.groupby('conversation_id').head(rag_top_k) # THIS PROBABLY WONT WORK REFER TO THE CODE IN make_historical_embeddings
        results = pd.concat([results, top_conversations], ignore_index=True)
        save_data_path = base_save_data_path.format(category['name'].replace(' ', '_'))
        top_conversations.to_csv(save_data_path, index=False)

def expand_benchmark():
    """
    load json_definitions_path, pull the categories, put them in a list, loop through the list:
    take the CSV files created in select_historical_conversations() [stored in base_save_data_path]
    and append the conversations to our existing benchmark dataset
    and saved the final dataset in base_expanded_data_path
    """
    pass

if __name__ == "__main__":
    tokenized_historical_data_path = 'path/to/tokenized_historical_data.csv'
    json_definitions_path = '~/vf_copilot/expand_benchmark/category_definitions.json'
    model_name = 'example'
    rag_top_k = 100
    base_save_data_path = '/datadrive/expanded_benchmark/selected_category_data/{category}_results.csv'

    base_expanded_data_path = '/datadrive/expanded_benchmark/expanded_eval_dataset_{task_timestamp}'

    select_historical_conversations(
        tokenized_historical_data_path=tokenized_historical_data_path,
        json_definitions_path=json_definitions_path,
        model_name=model_name,
        rag_top_k=rag_top_k,
        base_save_data_path=base_save_data_path
    )
    expand_benchmark(
        json_definitions_path=json_definitions_path,
        base_save_data_path=base_save_data_path,
        base_expanded_data_path=base_expanded_data_path
    )