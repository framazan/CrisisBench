import argparse
import os
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from evaluate import load

# Ensure NLTK tokenizer is available
nltk.download('punkt')

# Load BERTScore module
bertscore = load("bertscore")

def compute_bleu_score(reference, hypothesis):
    """Compute BLEU score between a reference and a hypothesis."""
    if pd.isna(reference) or pd.isna(hypothesis) or reference.strip() == "" or hypothesis.strip() == "":
        return None  # Handle missing or empty values gracefully

    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)
    return sentence_bleu([reference_tokens], hypothesis_tokens)

def compute_bertscore_batch(predictions, references, model_type='distilbert-base-uncased'):
    """
    Computes BERTScore for a batch of predictions and references.
    
    Args:
        predictions (list of str): List of candidate responses.
        references (list of str): List of ground truth responses.
        model_type (str): Pretrained model to use for BERTScore computation.

    Returns:
        list: F1 scores for each pair in the batch.
    """
    # Handle missing values
    valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) if not pd.isna(pred) and not pd.isna(ref)]
    valid_predictions, valid_references = zip(*valid_pairs) if valid_pairs else ([], [])

    if not valid_predictions:
        return [None] * len(predictions)  # Return None scores if everything is missing

    results = bertscore.compute(predictions=valid_predictions, references=valid_references, model_type=model_type)
    f1_scores = results['f1']

    # Map back to original order (handle missing values)
    f1_dict = {idx: score for idx, score in enumerate(f1_scores)}
    return [f1_dict.get(i, None) for i in range(len(predictions))]

def main(original_messages_path, new_messages_path, output_path):
    """Compute BLEU and BERT scores for message comparisons."""
    
    # Load datasets
    original_messages_df = pd.read_csv(original_messages_path)
    new_messages_df = pd.read_csv(new_messages_path)

    # Merge data on uid
    merged_df = pd.merge(original_messages_df, new_messages_df, on="uid", how="inner")

    # Compute BLEU scores
    merged_df['bleu_score_original'] = merged_df.apply(lambda row: compute_bleu_score(row['suggestion'], row['response_sent']), axis=1)
    merged_df['bleu_score_new'] = merged_df.apply(lambda row: compute_bleu_score(row['second_call_response'], row['response_sent']), axis=1)

    # Compute BERT scores in batch
    merged_df['bert_score_original'] = compute_bertscore_batch(merged_df['suggestion'].tolist(), merged_df['response_sent'].tolist())
    merged_df['bert_score_new'] = compute_bertscore_batch(merged_df['second_call_response'].tolist(), merged_df['response_sent'].tolist())

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BLEU and BERT scores for messages.")
    parser.add_argument("-o", "--original", type=str, required=True, help="Path to original messages CSV file.")
    parser.add_argument("-n", "--new", type=str, required=True, help="Path to new messages CSV file.")
    parser.add_argument("-out", "--output", type=str, required=True, help="Path to output CSV file.")
    
    args = parser.parse_args()
    main(args.original, args.new, args.output)
