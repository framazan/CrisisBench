# CrisisBench: An Evaluation Suite for Crisis Counseling AI

CrisisBench is a comprehensive evaluation suite for large language models (LLMs) used in text-based crisis counseling. This repository contains tools for simulating crisis conversations, evaluating model performance across specific metrics, and de-identifying crisis transcript data.

## Repository Structure

The `CrisisBench` repository is organized into the following key components:

- `src/crisisbench/de_id/`: Scripts and utilities for de-identifying real-world crisis text transcripts while preserving context.
- `src/crisisbench/single_turn/`: Code to evaluate LLMs in single-turn conversational environments (using static message-level rubrics).
- `src/crisisbench/multi_turn/`: Code to run and evaluate multi-turn dialogues between counselor systems and patient LLMs.
- `src/crisisbench/benchmark_gen/`: Tools for expanding the benchmark to new datasets.
- `src/crisisbench/utils/`: Shared utilities for model prompting, file IO, and data processing.
- `prompts/rubrics/`: Configuration files and yaml templates containing the rubrics used by the LLM Judges.
- `data/patient_profiles/`: Templates and configurations for patient LLMs (dummy profiles are provided for open-source).

## Installation

To set up the environment, clone the repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/yourusername/CrisisBench.git
cd CrisisBench
pip install -r requirements.txt
```

Set up your `.env` file with the required API keys (e.g., `OPENAI_API_KEY`) to run the LLM judges.

## Usage Guide

### 1. Multi-Turn Benchmark Evaluation
The multi-turn evaluation framework generates conversations between an LLM counselor and patient LLMs, then scores the conversation based on the 23-item conversation-level rubric.

**Generate and evaluate dialogues:**
```bash
python -m src.crisisbench.multi_turn.generate_and_evaluate_dialogue
```
This script handles the full pipeline: initializing a patient agent, initializing a counselor agent, running the multi-turn dialogue until completion, and evaluating the final transcript against the rubric.

### 2. Single-Turn Benchmark Evaluation
The single-turn evaluation tests counselor systems on specific, fixed crisis situations, analyzing the responses using message-level rubrics.

**Run the static evaluation:**
```bash
# Example command using the relevant script in the single_turn module
python -m src.crisisbench.single_turn.evals_main
```

### 3. De-identification Pipeline
If you have access to real crisis logs, you must de-identify them to preserve patient privacy before using them to generate benchmarks. 

**Run the de-identification pipeline:**
```bash
python -m src.crisisbench.de_id.run_de_id_pipeline
```

### 4. Expanding Benchmarks
To generate new benchmarking data from other valid sources (like local databases), you can use the benchmark generation scripts.

**Run benchmark expansion:**
```bash
python -m src.crisisbench.benchmark_gen.run_select_conversations
```

## Security and Privacy
For privacy and security reasons, real patient profiles derived from the hotline metadata are **not included** in this open-source release. We provide dummy patient profiles located in `data/patient_profiles/dummy_patient_profiles.yaml` that can be used to run and test the pipeline safely.

## License
MIT License
