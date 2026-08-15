# Langfuse Evaluation Pipelines

The CrisisBench repository integrates directly with [Langfuse](https://langfuse.com) for tracking model execution, evaluating outputs via LLM judges, and building dashboards to monitor overall performance.

We provide two distinct Langfuse pipelines depending on your evaluation setup:
1. **Dynamic Evaluations (Multi-turn)**: Located in `multiturn/langfuse/`
2. **Static Evaluations (Single-turn)**: Located in `singleturn/langfuse/`

## Prerequisites

Before running any pipeline, ensure that you edit the relevant `config.yaml` located centrally in `/config`.

**Environment Variables**

You must have your Langfuse API keys exported in your environment:
```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com" # or your local host
```

Additionally, ensure your LLM API keys are set up (e.g., `OPENAI_API_KEY`).

---

## Dynamic Evaluations (Multi-turn)

Located in `multiturn/langfuse/`. This pipeline automatically simulates multi-turn conversations between a designated LLM counselor and an LLM patient profile, logs them, and scores the conversation.

**Steps Executed:**
1. Generate synthetic patient–counselor dialogues and upload them as a Langfuse dataset.
2. Run LLM-judge evaluation on every dialogue, posting scores to each trace.
3. Create an analysis dashboard in Langfuse.

### Full Pipeline
Run all profiles matching the config file:
```bash
python multiturn/langfuse/run_pipeline.py \
    --dataset-name dynamic-evals-v1 \
    --run-name eval-run-1 \
    --email admin@example.com --password yourpassword
```

### Resume Execution
If you've already generated the dataset and just want to run evaluations and create the dashboard:
```bash
python multiturn/langfuse/run_pipeline.py \
    --dataset-name dynamic-evals-v1 \
    --run-name eval-run-2 \
    --start-from 2 \
    --email admin@example.com --password yourpassword
```

### Skip Dashboard
```bash
python multiturn/langfuse/run_pipeline.py \
    --dataset-name dynamic-evals-v1 \
    --run-name eval-run-1 \
    --skip-dashboard
```

---

## Static Evaluations (Single-turn)

Located in `singleturn/langfuse/`. This pipeline uploads fixed prompts and a dataset of static message exchanges, runs the LLM counselor, and then evaluates their single response.

**Steps Executed:**
0. Upload prompts to the Langfuse prompt library.
1. Upload the dataset (CSV → Langfuse dataset items).
2. Run generation (copilot responses logged as Langfuse experiment run).
3. Run eval & scoring (LLM judge, scores attached to each trace).
4. Create an analysis dashboard.

### Full Pipeline
Provide your raw conversation CSV:
```bash
python singleturn/langfuse/run_pipeline.py \
    --input /path/to/raw_convos.csv \
    --convert \
    --dataset-name static-evals \
    --run-name copilot-v1 \
    --model gpt-4o \
    --email admin@example.com --password yourpassword
```

### Skip Dashboard Creation
```bash
python singleturn/langfuse/run_pipeline.py \
    --input static_eval_input.csv \
    --dataset-name static-evals \
    --run-name copilot-v2 \
    --model gpt-4o \
    --skip-prompts --skip-dashboard
```

### Resume from Step 2 (Dataset already uploaded)
```bash
python singleturn/langfuse/run_pipeline.py \
    --input ignored \
    --dataset-name static-evals \
    --run-name copilot-v1 \
    --model gpt-4o \
    --start-from 2
```
