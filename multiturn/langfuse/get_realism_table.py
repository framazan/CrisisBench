import os
import math
import numpy as np
import scipy.stats as st
from langfuse import Langfuse
from collections import defaultdict

# CONFIGURATION
DATASET_RUN_PAIRS_ENV = os.environ.get("DATASET_RUN_PAIRS")
if DATASET_RUN_PAIRS_ENV:
    # Expected format: "dataset1:run1,dataset2:run2"
    PAIRS = [pair.split(":") for pair in DATASET_RUN_PAIRS_ENV.split(",")]
else:
    DATASET_NAME = os.environ.get("DATASET_NAME") or "dynamic-evals-96-final"
    RUN_NAMES = os.environ.get("RUN_NAMES", "4.1-realism-200tk-0.5tmp").split(",")
    PAIRS = [(DATASET_NAME, r) for r in RUN_NAMES]

TRACE_FETCH_LIMIT = 100
MAX_PAGES = 200

TARGET_CATEGORIES = ["suicidality", "depression", "anxiety", "relationship"]

lf = Langfuse()

def _normalize_batch(resp):
    data = resp.data if hasattr(resp, "data") else resp
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    try:
        return list(data)
    except TypeError:
        return []

def get_all_generation_traces(dataset_name, run_name):
    tag = f"{dataset_name}/{run_name}"
    all_traces = []
    for page in range(1, MAX_PAGES + 1):
        resp = lf.api.trace.list(limit=TRACE_FETCH_LIMIT, page=page, tags=tag, name="generated-conversation")
        batch = _normalize_batch(resp)
        if not batch:
            break
        all_traces.extend(batch)
        if len(batch) < TRACE_FETCH_LIMIT:
            break
    return all_traces

def calculate_mean_ci(data):
    if not data:
        return 0.0, 0.0, 0.0
    n = len(data)
    mean = np.mean(data)
    if n < 2:
        return mean, mean, mean
    
    # 95% Confidence Interval using standard error
    se = st.sem(data)
    ci = se * st.t.ppf((1 + 0.95) / 2., n - 1)
    
    return mean, mean - ci, mean + ci

def main():
    print(f"Fetching traces for {PAIRS}...")
    
    # Structure: category -> dict of score_type -> list of values
    # score_types: "linguistic", "emotional", "trait"
    category_scores = defaultdict(lambda: {"linguistic": [], "emotional": [], "trait": []})
    
    for dataset_name, run_name in PAIRS:
        traces = get_all_generation_traces(dataset_name, run_name)
        
        for trace_meta in traces:
            # We need to fetch the full trace to get the scores
            try:
                full_trace = lf.api.trace.get(trace_meta.id)
            except Exception as e:
                print(f"Warning: Failed to fetch full trace {trace_meta.id}: {e}")
                continue
            
            category_raw = (full_trace.metadata or {}).get("category", "other").lower()
            if category_raw not in TARGET_CATEGORIES:
                category_bucket = "Other"
            else:
                category_bucket = category_raw.capitalize()
                
            scores = getattr(full_trace, "scores", [])
            if not scores:
                continue
                
            score_dict = {s.name: s.value for s in scores if getattr(s, "value", None) is not None}
            
            # 1. Linguistic realism
            # language_patterns.grammar_and_punctuation, .message_length_and_complexity, .authenticity_and_natural_flow
            ling_keys = [
                "language_patterns.grammar_and_punctuation",
                "language_patterns.message_length_and_complexity",
                "language_patterns.authenticity_and_natural_flow"
            ]
            ling_vals = [score_dict[k] for k in ling_keys if k in score_dict]
            if ling_vals:
                category_scores[category_bucket]["linguistic"].append(np.mean(ling_vals))
                
            # 2. Emotional realism
            # emotional_realism.cognitive_distortions, .physical_or_behavioral_symptoms_of_distress, .depth_of_emotion_presenting_issue, .authentic_progression_of_emotional_states
            emo_keys = [
                "emotional_realism.cognitive_distortions",
                "emotional_realism.physical_or_behavioral_symptoms_of_distress",
                "emotional_realism.depth_of_emotion_presenting_issue",
                "emotional_realism.authentic_progression_of_emotional_states"
            ]
            emo_vals = [score_dict[k] for k in emo_keys if k in score_dict]
            if emo_vals:
                category_scores[category_bucket]["emotional"].append(np.mean(emo_vals))
                
            # 3. Trait consistency
            # Mapping "Trait consistency" to the presenting_concern score
            if "presenting_concern.score" in score_dict:
                category_scores[category_bucket]["trait"].append(score_dict["presenting_concern.score"])

    # Categories to display in order
    display_order = ["Suicidality", "Depression", "Anxiety", "Relationship", "Other"]
    
    print("\nSupplementary Table 1. Patient realism metrics by presenting concern.\n")
    print("| Presenting concern | Linguistic realism (mean [CI]) | Emotional realism (mean [CI]) | Trait consistency (mean [CI]) |")
    print("|-------------------|-------------------------------|-----------------------------|-----------------------------|")
    
    for cat in display_order:
        data = category_scores[cat]
        
        # Format strings
        def fmt(vals):
            if not vals:
                return "N/A"
            mean, lower, upper = calculate_mean_ci(vals)
            # Ensure lower/upper bounds aren't illogical (e.g. outside 1-3 if that's the scale)
            # However, for CI it's standard to report mathematical bounds.
            return f"{mean:.2f} [{lower:.2f}, {upper:.2f}]"
        
        ling_str = fmt(data["linguistic"])
        emo_str = fmt(data["emotional"])
        trait_str = fmt(data["trait"])
        
        print(f"| {cat} | {ling_str} | {emo_str} | {trait_str} |")

if __name__ == "__main__":
    main()
