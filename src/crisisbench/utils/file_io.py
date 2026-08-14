import os
import json
import pandas as pd
from crisisbench.utils.data_processing import convert_series_to_list

def save_evals(evals, timestamp, output_dir):
    evals_list = convert_series_to_list(evals)
    file_path = f"{output_dir}evals_{timestamp}.json"

    with open(file_path, 'w') as file:
        json.dump(evals_list, file, indent=4)

def save_flattened_scores(evals, timestamp, output_dir):
    rows = []

    for model_output, output_data in evals.items():
        for category, entries in output_data.items():
            for entry in entries:
                if isinstance(entry, dict):
                    row = {"model_output": model_output, "category": category}
                    row.update(entry)
                    rows.append(row)

    final_scores_df = pd.DataFrame(rows)
    final_scores_df.to_csv(f"{output_dir}benchmark_scores_{timestamp}.csv")
