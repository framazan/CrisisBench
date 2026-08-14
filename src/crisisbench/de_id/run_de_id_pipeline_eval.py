import argparse
import pandas as pd
import numpy as np
from openai import OpenAI

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def evaluate_deid_binary(df, column):
    # Mock-up function to return correct, incorrect, and accuracy
    correct = df[column].sum()  # Assuming binary 0/1 for incorrect/correct
    total = len(df)
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0
    return correct, incorrect, accuracy

def error_typology(input_text, client):
    system_prompt = """Review the following message which has been processed to remove personally identifiable information (PHI). Identify any errors in the scrubbing process and label them using the following categories: date_time, location, age, name, phone, email, other_id. If no error is found, return 'none'.

    Here are some examples:

    message: Hello, my name is <PERSON>. I was born on January 1st, 1990 in <PHI>. I moved in with my uncle Bobby after
    error_types: date_time (January 1st, 1990), name (Bobby)

    message: You can contact me at <PHONE> or visit me in San Francisco or Los Angeles whatever is most convenient for you i don’t care
    error_types: location (San Francisco), location (Los Angeles)

    message: i spoke with <PHI> yesterday and he told me now to worry about contacting <PHI>
    error_types: none

    message: Hi Jacob, my email address jane.doe@example.com i dont regularly use. It is best if you call me at my phone instead
    error_types: name (Jacob), email (jane.doe@example.com)

    message: I was treated at <ORGANIZATION> on <DATE_TIME>. I still feel depression and anxiety. I don’t know what else to do. can i speak with jamal or <PERSON> because they understand what franco and eric did last night
    error_types: name (jamal), name (franco), name (eric)

    message: My number is +91 33 12345678 and please tell <PERSON> to hurry i dont know what else to do
    error_types: phone (+91 33 12345678)
    """
    user_prompt = f"""
    Review this message for error types: 
    message: '{input_text}'
    error_types: """
    
    output = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
        )

    formatted_output = output.choices[0].message.content.replace('error_types: ', '')
    formatted_output = ' '.join(formatted_output.split())
    return formatted_output

def de_id_pipeline_eval(
        test,
        openai_api_key,
        performance_outputs_save_path
):
    
    client = OpenAI(
        api_key=openai_api_key,
    )

    print(f"\nLoad annotated data")
    annotations1 = pd.read_excel("/home/stanford/de_id_data/annotation/annotated_ivan_de_id_extended_model_eval_outputs.xlsx")
    annotations2 = pd.read_csv("/home/stanford/de_id_data/annotation/sharang_de_id_extended_model_eval_outputs.csv")
    annotations2 = annotations2[1000:]
    first_set = pd.concat([annotations1, annotations2])
    first_set = first_set[first_set['label_deid_roberta_i2b2_de_id_message'].notnull()]
    first_set = first_set.drop("Notes", axis='columns')
    annotations3 = pd.read_excel("/home/stanford/de_id_data/annotation/annotated_ivan_additional_de_id_model_eval_outputs.xlsx")
    annotations4 = pd.read_csv("/home/stanford/de_id_data/annotation/sharang_additional_de_id_model_eval_outputs.csv")
    second_set = pd.concat([annotations3, annotations4])
    second_set = second_set[second_set['label_deid_gpt4o_message'].notnull()]
    data_df = first_set.merge(
        second_set[['index', 'deid_gpt4o_message', 'label_deid_gpt4o_message']], 
        how='left', 
        on='index'
        )
    print(f"Data loaded:\n{data_df}")

    if test:
        print(f"TESTING DATASET")
        data_df = data_df.head(10)
        data_df

    print(f"\nRunning performance evaluation")
    error_types = ['date_time', 'location', 'age', 'name', 'phone', 'email', 'other_id']
    eval_results = pd.DataFrame(columns=['eval'] + error_types + ['n_messages_correct', 'n_messages_incorrect', 'overall_accuracy'])
    message_columns = [col for col in data_df.columns if '_message' in col and not col.startswith('label_')]
    label_columns = {col: 'label_' + col for col in message_columns if 'label_' + col in data_df.columns}
    for de_id_message_column in message_columns:
        label_column = label_columns[de_id_message_column]
        data_df[label_column] = pd.to_numeric(data_df[label_column], errors='coerce')
        print(f"\nEval for: {de_id_message_column, label_column}")
        n_correct, n_incorrect, overall_accuracy = evaluate_deid_binary(data_df, label_column)
        # Initialize a DataFrame to hold error typology results for the current column
        error_typology_results = pd.DataFrame()
        # Analyze errors and populate error_typology_results
        mistake_df = data_df[data_df[label_column] == 0]
        error_typology_results[de_id_message_column] = mistake_df[de_id_message_column].apply(lambda x: error_typology(x, client))
        # Initialize counters for error types
        error_counts = {error: 0 for error in error_types}
        # Sum up counts from error_typology_results
        for index, message_errors in error_typology_results[de_id_message_column].items():
            errors = message_errors.split(',')
            for error in errors:
                error = error.strip()
                for error_type in error_types:
                    if error_type in error.lower():
                        error_counts[error_type] += 1
        # Create a new row for eval_results with accumulated counts and accuracy
        new_row = {'eval': de_id_message_column, 'n_messages_correct': n_correct, 'n_messages_incorrect': n_incorrect, 'overall_accuracy': overall_accuracy}
        new_row.update(error_counts)
        eval_results = pd.concat([eval_results, pd.DataFrame([new_row])], ignore_index=True)
    print(f"Performance eval complete:\n{eval_results}")
    
    print(f"\nSaving outputs")
    if test:
        path_parts = performance_outputs_save_path.rsplit('/', 1)
        performance_outputs_save_path = f"{path_parts[0]}/TEST_{path_parts[1]}"
    eval_results.to_excel(performance_outputs_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run de-identification using different models.")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key for authentication")
    parser.add_argument("--test", action='store_true', help="Run the script in test mode.")
    args = parser.parse_args()

    performance_outputs_save_path = '/home/stanford/vf_copilot/de_id_pipeline/eval_outputs/de_id_pipeline_performance.xlsx'

    de_id_pipeline_eval(
        test=args.test,
        openai_api_key=args.openai_api_key,
        performance_outputs_save_path=performance_outputs_save_path,
    )