# CrisisBench: An Evaluation Suite for Crisis Counseling AI

[comment]: <> (When using the img tag, which allows us to specify size, src has to be a URL.)
<img src="logo.png" alt="CrisisBench logo" width="480"/>

<!-- Badges -->
<a href="https://github.com/framazan/CrisisBench"><img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/framazan/CrisisBench"></a>&nbsp;
<a><img alt="GitHub contributors" src="https://img.shields.io/badge/contributors-18-brightgreen"></a>&nbsp;
<a href="https://github.com/framazan/CrisisBench/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/framazan/CrisisBench?color=blue"></a>

**CrisisBench** is a comprehensive evaluation suite for large language models (LLMs) used in text-based crisis counseling. This repository contains tools for simulating crisis conversations, evaluating model performance across specific metrics, and de-identifying crisis transcript data.

The repository is organized into the following key components:

- `de_id/`: Scripts and utilities for de-identifying real-world crisis text transcripts while preserving context.
- `singleturn/`: Code to evaluate LLMs in single-turn conversational environments (using static message-level rubrics).
- `multiturn/`: Code to run and evaluate multi-turn dialogues between counselor systems and patient LLMs.
- `utils/`: Shared utilities for model prompting, file IO, and data processing.
- `prompts/rubrics/`: Configuration files and yaml templates containing the rubrics used by the LLM Judges.
- `data/patient_profiles/`: Templates and configurations for patient LLMs (dummy profiles are provided for open-source).

## Documentation

Please refer to the documentation files in the `docs/` directory for full instructions:
- [Multi-Turn Evaluation](docs/multiturn_evals.md)
- [Single-Turn Evaluation](docs/singleturn_evals.md)
- [Langfuse UI Execution](docs/langfuse_evals.md)
- [De-identification Pipeline](docs/deid_pipeline.md)

## Quick Start

<!--quick-start-begin-->

To set up the environment, clone the repository and install the required dependencies using `pip`:

```sh
git clone https://github.com/yourusername/CrisisBench.git
cd CrisisBench
pip install -r requirements.txt
```

Set up your `.env` file with the required API keys (e.g., `OPENAI_API_KEY`) to run the LLM judges.

### Usage Guide

CrisisBench offers two distinct paradigms for running evaluations depending on your workflow needs:

#### Mode A: Manual Scripts Execution
This mode uses core python scripts to generate text-file outputs and CSVs locally. It is lightweight and easy to run without any external dashboarding tools.

- **Multi-Turn Evaluation**: Generates full dialogues and outputs raw text files containing the patient/counselor transcripts and scores.
  ```sh
  python -m multiturn.generate_and_evaluate_dialogue ...
  ```

- **Single-Turn Evaluation**: Tests systems on fixed message exchanges using a standard bash pipeline.
  ```sh
  bash singleturn/run_singleturn_pipeline.sh ...
  ```

#### Mode B: Langfuse UI Execution
This mode uses our custom CLI to hook directly into [Langfuse](https://langfuse.com), providing a rich web UI to track dialogue traces, view step-by-step LLM-judge scores, and visualize metrics on a dashboard. 

- **Dynamic (Multi-turn) UI Pipeline**:
  ```sh
  python multiturn/langfuse/run_pipeline.py \
      --dataset-name dynamic-evals-v1 \
      --run-name eval-run-1 \
      --email admin@example.com --password yourpassword
  ```

- **Static (Single-turn) UI Pipeline**:
  ```sh
  python singleturn/langfuse/run_pipeline.py \
      --input /path/to/raw_convos.csv \
      --convert \
      --dataset-name static-evals \
      --run-name copilot-v1 \
      --model gpt-4o \
      --email admin@example.com --password yourpassword
  ```

#### De-identification Pipeline
If you have access to real crisis logs, you must de-identify them to preserve patient privacy before using them to generate benchmarks. 

```sh
python -m de_id.run_de_id_pipeline
```
<!--quick-start-end-->

## Security and Privacy

For privacy and security reasons, real patient profiles derived from the hotline metadata are **not included** in this open-source release. We provide dummy patient profiles located in `data/patient_profiles/dummy_patient_profiles.yaml` that can be used to run and test the pipeline safely.

## Citation

If you use this software in your research, please cite our paper as below.

```bibtex
@article{
crisisbench2026,
title={CrisisBench: An Evaluation Suite for Crisis Counseling AI},
author={Filip Ramazan},
journal={arXiv preprint arXiv:[Placeholder]},
year={2026},
url={[Placeholder for URL]}
}
```

## License

MIT License
