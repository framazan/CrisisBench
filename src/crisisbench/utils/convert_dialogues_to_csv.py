import argparse
import os
import hashlib
import csv
import yaml

def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_patient_profiles(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = yaml.safe_load(f)
    return profiles

def extract_patient_key(filename):
    # Example filename: dialogue|relationship_78|base_instructions3|second_call_with_history3|gpt-4o|20250326_163053.txt
    # An alternative format is: dialogue_{number}_{profile_name}|{model_name}.txt
    
    # Try the first format (pipe-separated)
    parts = filename.split("|")
    
    if len(parts) > 1:
        # For the first format, return the second part (relationship_78)
        if len(parts) > 2:
            return parts[1]
        # For the alternative format, extract profile_name from dialogue_{number}_{profile_name}
        elif filename.startswith("dialogue_"):
            try:
                # Split by underscore and get the profile name part
                # dialogue_1_family_relationship_69 -> ['dialogue', '1', 'family', 'relationship', '69']
                name_parts = parts[0].split("_")
                if len(name_parts) >= 3:  # Ensure we have at least dialogue_{number}_{profile_name}
                    # Join all parts after the number to get the full profile name
                    
                    return "_".join(name_parts[2:])
            except:
                pass
    
    return None

def process_text_files(input_dir, output_csv, profile_path):
    profiles = load_patient_profiles(profile_path)
    rows = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                uid = hash_text(content)
                patient_key = extract_patient_key(filename)
                profile = profiles.get(patient_key, "PROFILE NOT FOUND")
                full_combined = f"***CONVERSATION***\n\n{content}\n\n***PATIENT PROFILE***\n\n{profile}"
                rows.append({
                    "uid": uid,
                    "file_name": filename,
                    "conversation": content,
                    "patient_profile": profile,
                    "conversation_plus_profile": full_combined
                })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "file_name", "conversation", "patient_profile", "conversation_plus_profile"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV file written to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text dialogues into CSV format and attach patient profiles.")
    parser.add_argument("input_dir", help="Directory containing .txt dialogue files")
    parser.add_argument("output_csv", help="Path to output CSV file")
    parser.add_argument("profile_path", help="Path to patient profile YAML file", default="evals/llm_patient/data/patient_profile_prompts.yaml")
    args = parser.parse_args()

    process_text_files(args.input_dir, args.output_csv, args.profile_path)
