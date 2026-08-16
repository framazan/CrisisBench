# Multi-Turn & Dynamic Evaluation Pipelines

This guide covers evaluating counselors interactively, creating LLM patient profiles, generating synthetic dialogues, and testing the stability of these patient agents.

## 1. Creating LLM Patient Profiles

To evaluate a counselor interactively, you must first create LLM Patient Profiles.

1. **Extract Traits from Real Conversations**:
   First, build a prompted dataset to extract clinical traits from a de-identified sample of real conversations.
   ```bash
   python utils/build_prompted_datasets.py \
     -f data/singleturn/for_prompting/sampled_for_patient_profiles_deid.csv \
     -p prompts/rubrics/convo_traits_for_profile \
     -o data/prompted/ \
     -n convo_traits_for_profile \
     -fmt messages \
     -c deid_conversation
   ```
   Next, run an LLM to generate the trait responses:
   ```bash
   bash singleturn/run_llm_client.sh data/prompted/convo_traits_for_profile.jsonl "gpt-4o" "openai-batch" 5 1000 "https://api.openai.com/v1/chat/completions"
   ```
   Finally, extract the completions into a CSV:
   ```bash
   python singleturn/extract_content_from_llm_completion.py \
     -i data/llm_inference/completions/final_merged_gpt-4o_max1000.jsonl \
     -o data/llm_inference/extracted_completions/patient_profiles_deidentified.csv \
     -c "client_traits_obscured_json_raw"
   ```

2. **Generate Patient Profiles**:
   Once you have the extracted traits and de-identified conversations, you can generate the patient profiles.

   ```bash
   python multiturn/create_patient_profiles.py \
     --traits_file data/llm_inference/extracted_completions/patient_profiles_deidentified.csv \
     --sampled_convos_file data/llm_inference/extracted_completions/sample_conversations_for_patient_profiles.csv \
     --deid_convo_file data/llm_inference/extracted_completions/sampled_for_patient_profiles_deid.csv \
     --output_file data/multiturn/patient_profiles/generated_patient_profiles.csv \
     --output_yaml data/multiturn/patient_profiles/patient_profile_prompts.yaml
   ```

3. **(Optional) Save Prompts to History**:
   To save the generated profiles to your historical prompts directory for version control:
   ```bash
   python multiturn/save_historical_prompts.py \
       --input_file data/multiturn/patient_profiles/patient_profile_prompts.yaml \
       --output_dir multiturn/historical_prompts/$(date +%m-%d-%Y)
   ```

## 2. Generating Synthetic Dialogues & Evaluating

The easiest way to run the full pipeline (Generate dialogue -> Evaluate) is to use the `generate_and_evaluate_dialogue.py` or the `inference_config.yaml` methods.

### Batch Generation & Evaluation

You can configure exactly which pipelines to run using `crisisbench/inference_config.yaml`. For example, this file can specify parallel configurations to evaluate both the Patient and Counselor on various rubrics.

1. Create a config file via `multiturn/make_config_files.py`.
2. Run the inference pipeline:
   ```bash
   utils/run_inference_from_config.sh crisisbench/inference_config.yaml dialogue_evals
   ```

### Analyzing Dialogue Scores

After running the evaluations, you can summarize the results:

```bash
LATEST_COUNSELOR_RUBRIC_FILE=$(ls -t data/llm_inference/extracted_completions/dynamic_counselor_rubric_*_extracted_*.csv | head -n 1)

python singleturn/analyze_comparison_scores.py \
    $LATEST_COUNSELOR_RUBRIC_FILE \
    data/analysis/dynamic_evals/dynamic_eval_rubric_stats.csv
```

## 3. Stability Analysis

To test how stable an LLM patient profile is (i.e. does the patient answer basic demographic questions consistently?), we can run stability tests.

```bash
export OPENAI_API_KEY='your-api-key-here'

# Run the stability analysis across a directory of patient dialogue configs
python multiturn/run_stability_tests.py \
    --config_dir multiturn/dialogue_configs/ \
    --n_samples 5
```

You can optionally override the default stability questions by providing `--prompt_1`, `--prompt_2`, etc., directly to the script.
