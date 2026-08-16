# De-identification Pipeline

If you are working with raw, sensitive conversation data, it must be completely de-identified to remove Protected Health Information (PHI) before it can be used for modeling, evaluation, or benchmarking.

> **Note**: Always use an API key from an endpoint that has a strict **Zero Data Retention** policy when transmitting any raw PHI to a Large Language Model.

## Running the De-id Pipeline

The pipeline uses the standard Single-Turn batching flow under the hood.

### 1. Prepare your input data
Save a CSV with the raw data you want to run through the de-id pipeline.
Ensure this CSV has a `uid` column corresponding to the primary key.

```bash
mkdir -p data/singleturn/for_prompting
mkdir -p data/llm_inference/extracted_completions
```

### 2. Build the Prompted Dataset
Merge your raw conversations with the De-identification prompt template to create the JSONL file ready for batching:

```bash
python singleturn/build_prompted_datasets.py \
    -f data/singleturn/for_prompting/sample_conversations.csv \
    -p data/prompts/deid \
    -o data/prompted/ \
    -n deid_job \
    -fmt messages \
    -c conversation
```

### 3. Run Batch Inference
Transmit the JSONL file to the LLM. 

```bash
singleturn/run_llm_client.sh \
    data/prompted/deid_job.jsonl \
    "gpt-4o" "openai-batch" 5 5000 "https://api.openai.com/v1/chat/completions"
```

### 4. Extract the De-identified Content
Once the LLM batch has finished generating, parse the nested output to extract the clean, de-identified texts:

```bash
python singleturn/extract_content_from_llm_completion.py \
    -i data/llm_inference/completions/final_merged_gpt-4o_max5000_<timestamp>.jsonl \
    -o data/llm_inference/extracted_completions/sample_conversations_deidentified.csv \
    -c "deid_conversation" 
```

The resulting `sample_conversations_deidentified.csv` file can now be safely used in downstream benchmark generation and modeling tasks.
