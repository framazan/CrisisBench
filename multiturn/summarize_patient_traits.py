import json
import pandas as pd
import argparse
from collections import Counter

def load_jsonl(jsonl_file):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(jsonl_file, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def summarize_patients(jsonl_file, output_file):
    """Summarizes patient characteristics from a JSONL file and saves the result as CSV."""
    data = load_jsonl(jsonl_file)

    # Extract relevant attributes
    ages, genders, concerns = [], [], []

    for entry in data:
        for _, details in entry.items():  # Each entry contains a dictionary with a single patient
            ages.append(int(details.get("age", 0)))
            genders.append(details.get("gender", "Unknown"))
            concerns.append(details.get("presenting_concern", "Unknown"))

    total_patients = len(ages)

    # Categorizing age groups
    age_bins = [18, 25, 35, 50, 100]
    age_labels = ["18-25", "26-35", "36-50", "51+"]
    age_categories = pd.cut(ages, bins=age_bins, labels=age_labels, right=False)

    # Counting occurrences
    age_counts = Counter(age_categories)
    gender_counts = Counter(genders)
    concern_counts = Counter(concerns)

    # Creating the summary table
    summary_data = {
        "Characteristic": [],
        "Category": [],
        "n (%)": []
    }

    # Adding age group distribution
    for category, count in age_counts.items():
        summary_data["Characteristic"].append("Demographic Factors - Age Group")
        summary_data["Category"].append(category)
        summary_data["n (%)"].append(f"{count} ({(count / total_patients) * 100:.1f}%)")

    # Adding gender distribution
    for category, count in gender_counts.items():
        summary_data["Characteristic"].append("Demographic Factors - Gender")
        summary_data["Category"].append(category.capitalize())
        summary_data["n (%)"].append(f"{count} ({(count / total_patients) * 100:.1f}%)")

    # Adding presenting concerns distribution
    for category, count in concern_counts.items():
        summary_data["Characteristic"].append("Presenting Concerns - Primary Concern")
        summary_data["Category"].append(category)
        summary_data["n (%)"].append(f"{count} ({(count / total_patients) * 100:.1f}%)")

    # Convert summary to dataframe
    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Summary table saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize LLM patient characteristics from a JSONL file.")
    parser.add_argument("jsonl_file", help="Path to the JSONL file containing LLM patient data.")
    parser.add_argument("output_file", help="Path to save the output summary CSV file.")

    args = parser.parse_args()
    summarize_patients(args.jsonl_file, args.output_file)
