import pandas as pd
import json
import argparse
from typing import Dict, Any
import numpy as np
import re

SCORE_CATEGORIES = ['resolution', 'micro_skills', 'context', 'language', 'empathy_chaining', 'forbidden_statements', 'multiple_tasks', 'pacing', 'questions']

def extract_scores(comparison_json: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract scores from the comparison JSON string.
    
    Args:
        comparison_json: JSON string containing the comparison scores
        
    Returns:
        Dictionary of scores and justifications
    """
    try:
        # Remove markdown code block if present
        cleaned_json = re.sub(r'^```json\n', '', comparison_json)
        cleaned_json = re.sub(r'\n```$', '', cleaned_json)
        
        # First try to parse the outer JSON
        outer_json = json.loads(cleaned_json)
        
        # If it's an OpenAI completion, extract the content
        if isinstance(outer_json, dict) and 'choices' in outer_json:
            content = outer_json['choices'][0]['message']['content']
            # Try to parse the content as JSON
            return json.loads(content)
        return outer_json
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"JSON content: {comparison_json[:200]}...")  # Print first 200 chars for debugging
        return {}

def calculate_statistics(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics for each evaluation category.
    
    Args:
        scores_df: DataFrame containing the scores
        
    Returns:
        DataFrame with summary statistics
    """
    
    stats = []
    
    for category in SCORE_CATEGORIES:
        category_scores = scores_df[category + '_score'].dropna()
        if len(category_scores) > 0:
            stats.append({
                'Category': category.replace('_', ' ').title(),
                'Mean': np.mean(category_scores),
                'Median': np.median(category_scores),
                'Std Dev': np.std(category_scores),
                'Min': np.min(category_scores),
                'Max': np.max(category_scores),
                'Count': len(category_scores),
                'Score Distribution': {
                    '1': len(category_scores[category_scores == 1]),
                    '2': len(category_scores[category_scores == 2]),
                    '3': len(category_scores[category_scores == 3]),
                    '4': len(category_scores[category_scores == 4]),
                    '5': len(category_scores[category_scores == 5])
                }
            })
    
    return pd.DataFrame(stats)

def analyze_comparison_scores(
    input_csv: str,
    output_csv: str
) -> None:
    """
    Analyze comparison scores and generate summary statistics.
    
    Args:
        input_csv: Path to the CSV containing comparison results
        output_csv: Path to save the analysis results
    """
    # Read the comparison results
    df = pd.read_csv(input_csv)
    
    # Informative log of available columns (minimal)
    print("Available columns detected:", df.columns.tolist())
    
    # Find the column containing the comparison JSON
    json_columns = [col for col in df.columns if 'json' in col.lower()]
    if not json_columns:
        raise ValueError("No JSON column found in the input CSV. Available columns: " + ", ".join(df.columns))
    
    comparison_column = json_columns[0]
    print(f"Using column '{comparison_column}' for comparison data")
    
    # Extract scores from the comparison JSON
    scores = df[comparison_column].apply(extract_scores)
    
    # Create columns for each score category
    for category in SCORE_CATEGORIES:
        df[category + '_score'] = scores.apply(lambda x: x.get(category, {}).get('score', np.nan))
        df[category + '_justification'] = scores.apply(lambda x: x.get(category, {}).get('justification', ''))
    
    # ──────────────────────────────────────────────────────────────────────
    #  STRATIFY BY SOURCE FILE (if available)
    # ──────────────────────────────────────────────────────────────────────

    if "input_file_name" in df.columns:
        df['model_name'] = df['input_file_name'].str.split('|').str[-1]
        grouped_stats = []
        for file_name, group in df.groupby("model_name"):
            stats = calculate_statistics(group)
            stats["model"] = file_name
            grouped_stats.append(stats)
        stats_df = pd.concat(grouped_stats, ignore_index=True)
    else:
        stats_df = calculate_statistics(df)
    
    # Save the results (stratified if applicable)
    stats_df.to_csv(output_csv, index=False)

    # Print only the aggregated summary statistics
    header_msg = "Summary Statistics by model:" if "input_file_name" in df.columns else "Summary Statistics:"
    print("\n" + header_msg)
    print(stats_df.to_string(index=False))
    
    # Still provide score distribution details per category but without per-row spam
    print("\nScore Distributions (by category):")
    for _, row in stats_df.iterrows():
        print(f"\n{row.get('model', 'All Models')} → {row['Category']}")
        for score, count in row['Score Distribution'].items():
            print(f"  Score {score}: {count} responses ({count/row['Count']*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Analyze comparison scores and generate summary statistics')
    parser.add_argument('input_csv', help='Path to CSV file containing comparison results')
    parser.add_argument('output_csv', help='Path to save the analysis results')
    
    args = parser.parse_args()
    analyze_comparison_scores(
        args.input_csv,
        args.output_csv
    )

if __name__ == '__main__':
    main() 