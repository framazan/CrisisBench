import os
import numpy as np
from langfuse import Langfuse

# Hardcoded dictionary as confirmed by user
SYSTEMS = [
    ("GPT-4.1", "dynamic-evals-96-final", "4.1-realism-200tk-0.5tmp"),
    ("GPT-4.1-mini", "eval-gpt-4.1-mini", "eval-run"),
    ("GPT-5 mini", "eval-gpt-5-mini", "eval-run"),
    ("GPT-5 nano", "eval-gpt-5-nano", "eval-run"),
    ("GPT-5.2", "eval-gpt-5.2", "eval-run"),
    ("Double prompt", "eval-gpt-5.2-double-prompt", "eval-run"),
    ("RAG", "eval-gpt-5.2-rag", "5.2-rag-eval-run")
]

RUBRIC_WEIGHTS = {
    'P1': 2, 'P2': 1, 'P3': 1, 'P4': 1, 'P5': 1, 'P6': 1, 'P7': 1, 'P8': 2,
    'M1': 2, 'M2': 1, 'M3': 1, 'M4': 1, 'M5': 1,
    'S1': 1, 'S2': 1, 'S3': 2, 'S4': 1, 'S5': 1, 'S6': 1, 'S7': 1,
    'X0': 0, 'X1': 4, 'X2': 4, 'X3': 2, 'X4': 2
}

PREFIX = "counselor-eval-rubric."
MAX_PAGES = 100
TRACE_FETCH_LIMIT = 100

lf = Langfuse()

def _normalize_batch(resp):
    data = resp.data if hasattr(resp, "data") else resp
    if data is None: return []
    if isinstance(data, list): return data
    if isinstance(data, tuple): return list(data)
    try: return list(data)
    except TypeError: return []

def get_all_traces(dataset_name, run_name):
    tag = f"{dataset_name}/{run_name}"
    all_traces = []
    for page in range(1, MAX_PAGES + 1):
        resp = lf.api.trace.list(limit=TRACE_FETCH_LIMIT, page=page, tags=tag, name="generated-conversation")
        batch = _normalize_batch(resp)
        if not batch: break
        all_traces.extend(batch)
        if len(batch) < TRACE_FETCH_LIMIT: break
    return all_traces

def compute_trace_fractional_score(trace_meta):
    try:
        full_trace = lf.api.trace.get(trace_meta.id)
    except Exception as e:
        print(f"  Warning: Failed to fetch full trace {trace_meta.id}: {e}")
        return None
    
    scores = getattr(full_trace, "scores", [])
    if not scores: return None
    
    score_dict = {s.name: s.value for s in scores if getattr(s, "value", None) is not None}
    
    # Extract only rubric values
    rubric_scores = {}
    for q in RUBRIC_WEIGHTS.keys():
        key = f"{PREFIX}{q}"
        if key in score_dict:
            # Normalize to 0/1 int
            val = score_dict[key]
            if str(val).lower() in ["1", "1.0", "true", "yes"]: rubric_scores[q] = 1
            elif str(val).lower() in ["0", "0.0", "false", "no"]: rubric_scores[q] = 0
            else: rubric_scores[q] = None
        else:
            rubric_scores[q] = None
            
    # Calculate Possible Score
    excluded = []
    if rubric_scores.get('X0') == 0:
        excluded += ['X1', 'X2', 'X3', 'X4']
    if rubric_scores.get('P5') is None:
        excluded += ['P5']
        
    possible_score = sum(RUBRIC_WEIGHTS[q] for q in RUBRIC_WEIGHTS if q not in excluded)
    
    # Calculate Total Score
    total_score = 0
    for q in RUBRIC_WEIGHTS:
        if q in excluded: continue
        # X0 is weight 0, so it doesn't add to total_score anyway.
        # If the LLM scored 1, add the weight.
        if rubric_scores.get(q) == 1:
            total_score += RUBRIC_WEIGHTS[q]
            
    if possible_score == 0:
        return None
        
    return total_score / possible_score

def main():
    print("System | Overall Rubric Score (Mean Frac)")
    print("--- | ---")
    
    for sys_name, dataset, run in SYSTEMS:
        traces = get_all_traces(dataset, run)
        if not traces:
            print(f"{sys_name} | N/A (no traces)")
            continue
            
        frac_scores = []
        for t in traces:
            score = compute_trace_fractional_score(t)
            if score is not None:
                frac_scores.append(score)
                
        if frac_scores:
            mean_score = np.mean(frac_scores)
            print(f"{sys_name} | {mean_score:.4f}")
        else:
            print(f"{sys_name} | N/A (no valid scores)")

if __name__ == "__main__":
    main()
