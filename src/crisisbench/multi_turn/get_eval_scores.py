import json
import csv
import argparse
import os

def parse_eval_scores(json_file, file_name):
    with open(json_file, 'r') as f:
        eval_data = json.load(f)

    rows = []

    # Parse main categories
    for category, content in eval_data.items():
        if category == "big_five_traits":
            # Parse subcategories for Big Five Traits
            big_five_scores = []
            for trait, trait_data in content.items():
                rows.append({
                    "category": category,
                    "subcategory": trait,
                    "score": trait_data["score"],
                    "rationale": trait_data["rationale"],
                    "file_name": file_name
                })
                big_five_scores.append(trait_data["score"])
            
            # Calculate Big Five average
            big_five_avg = sum(big_five_scores) / len(big_five_scores)
            rows.append({
                "category": "big_five_traits_average",
                "subcategory": "",
                "score": big_five_avg,
                "rationale": "Average score across all Big Five traits.",
                "file_name": file_name
            })
            
        elif category == "language_patterns":
            language_patterns_scores = []
            for pattern, pattern_data in content.items():
                rows.append({
                    "category": category,
                    "subcategory": pattern,
                    "score": pattern_data["score"],
                    "rationale": pattern_data["rationale"],
                    "file_name": file_name
                })
                language_patterns_scores.append(pattern_data["score"])
            
            # Calculate language patterns average
            language_patterns_avg = sum(language_patterns_scores) / len(language_patterns_scores)
            rows.append({
                "category": "language_patterns_average",
                "subcategory": "",
                "score": language_patterns_avg,
                "rationale": "Average score across all language patterns.",
                "file_name": file_name
            })
        
        elif category == "emotional_realism":
            emotional_realism_scores = []
            for trait, trait_data in content.items():
                rows.append({
                    "category": category,
                    "subcategory": trait,
                    "score": trait_data["score"],
                    "rationale": trait_data["rationale"],
                    "file_name": file_name
                })
                emotional_realism_scores.append(trait_data["score"])
            
            # Calculate emotional realism average
            emotional_realism_avg = sum(emotional_realism_scores) / len(emotional_realism_scores)
            rows.append({
                "category": "emotional_realism_average",
                "subcategory": "",
                "score": emotional_realism_avg, 
                "rationale": "Average score across all emotional realism traits.",
                "file_name": file_name
            })
        
        else:
            # Parse regular categories
            print("content", content)
            print("category", category)
            rows.append({
                "category": category,
                "subcategory": category,
                "score": content["score"],
                "rationale": content["rationale"],
                "file_name": file_name
            })
            
    # Calculate the overall score
    if "big_five_traits_average" in eval_data:
        big_five_score = eval_data["big_five_traits_average"]["score"]
    else: 
        big_five_score = eval_data["big_five_traits_simple"]["score"]
    presenting_concern_score = eval_data["presenting_concern"]["score"]
    overall_score = (big_five_score + language_patterns_avg + emotional_realism_avg + presenting_concern_score) / 5

    rows.append({
        "category": "overall_score",
        "subcategory": "",
        "score": overall_score,
        "rationale": "Overall average score across Language Patterns, Presenting Concern, Realism, and Big Five average.",
        "file_name": file_name
    })

    return rows

def process_directory(json_dir, output_csv):
    all_rows = []
    print(f"Processing JSON files in directory: {json_dir}")
    # Process each JSON file in the directory
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".jsonl"):
            json_file = os.path.join(json_dir, file_name)
            file_rows = parse_eval_scores(json_file, file_name)
            all_rows.extend(file_rows)

    # Write all rows to a single CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["category", "subcategory", "score", "rationale", "file_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert evaluation scores JSONs in a directory to a single CSV.")
    parser.add_argument('--json_dir', required=True, help="Path to the directory containing evaluation scores JSON files.")
    parser.add_argument('--csv_file', required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    process_directory(args.json_dir, args.csv_file)

    print(f"CSV file generated at: {args.csv_file}")
