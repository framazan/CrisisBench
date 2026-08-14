import pandas as pd
import hashlib
import argparse

def generate_uid(row):
    """Generate a unique identifier based on a hash of the Phone, Date, Counsellor, conversation."""
    # Create a base string from the relevant columns
    # Ensure the columns are strings to avoid type errors
    row['Phone'] = str(row['Phone'])
    row['Date'] = str(row['Date'])
    row['Counsellor'] = str(row['Counsellor'])
    row['conversation'] = str(row['conversation'])
    # Concatenate the relevant columns to form a base string
    # Use the first 8 characters of the hash for brevity
    # Note: The hash is not shortened here, but you can use a slice if needed
    base_string = f"{row['Phone']}_{row['Date']}_{row['Counsellor']}_{row['conversation']}"
    # Generate a hash of the base string
    hash_part = hashlib.md5(base_string.encode()).hexdigest()  # Longer hash

    return f"{hash_part}"

def process_csv(input_file, output_file):
    """Read input CSV, process columns, and save to output CSV."""
    # Load CSV
    df = pd.read_csv(input_file)
    df = df.rename(columns={
        'patient_phone': 'Phone',
        'DateInserted': 'Date',
        'counsellor': 'Counsellor',
        'convo_history': 'conversation',
        'true_counselor_response': 'response_sent',
        'ai_suggestion': 'suggestion',
    })

    # Filter out advanced prompted columns
    df = df[df[['advanced_prompt', 'advanced_text']].isna().all(axis=1)]

    # Select required columns
    df["uid"] = df.apply(generate_uid, axis=1)
    df_output = df[["uid", "suggestion", "response_sent", "conversation"]].copy()
    
    # Filter out rows where "Response Sent", "conversation", or "Suggestion" is NaN
    df_output = df_output.dropna(subset=["response_sent", "conversation", "suggestion"])
    
    # Save processed CSV
    df_output.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file and generate a formatted output.")
    parser.add_argument("input_file", help="Path to input CSV file")
    parser.add_argument("output_file", help="Path to output CSV file")
    
    args = parser.parse_args()
    process_csv(args.input_file, args.output_file)
