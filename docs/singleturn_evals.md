# Single-Turn & Static Evaluation Pipelines

"Static" evals refers to passing a pre-defined conversation history to an LLM counselor, having it generate the next message, then evaluating that generation. This setup allows for highly efficient inference via batching and parallel calls.

We use this identical pipeline to run several different message-level evaluations (Counselor Eval, Demographics, Message Level, Risk Assessment, Presenting Concern, etc.). 

## The Generic 3-Step Pipeline

For any of the static evaluations, the core workflow is exactly the same:

### 1. Build the Prompted Dataset
First, you take your raw conversation CSV and join it with the specific prompt template for the task you want to run. This creates a `.jsonl` file ready for batch processing.

*(Note: If you are just testing the pipeline, you can use the provided `data/singleturn/for_prompting/dummy_raw_conversations.csv` as your input file).*

```bash
python3 singleturn/build_prompted_datasets.py \
    -f data/singleturn/for_prompting/your_raw_conversations.csv \
    -p data/prompts/<prompt_folder> \
    -o data/prompted/ \
    -n <experiment_name> \
    -fmt messages \
    -c <column_name_containing_conversation>
```

### 2. Run the LLM to Generate Responses
Using the generated `.jsonl` file, you launch the LLM client to evaluate or respond to the prompts.

```bash
singleturn/run_llm_client.sh data/prompted/<experiment_name>.jsonl "gpt-4o" "openai-batch" 4 500 "https://api.openai.com/v1/chat/completions"
```

### 3. Extract the Completions
The batch API returns heavily nested JSON. This script extracts just the response content you need and saves it as a clean CSV.

```bash
python3 singleturn/extract_content_from_llm_completion.py \
    -i data/llm_inference/completions/<batch_output_file>.jsonl \
    -o data/llm_inference/extracted_completions/<experiment_name>_results.csv \
    -c "llm_eval_json" 
```

---

## Configuration Reference by Task

Depending on what you are evaluating, you simply swap out the parameters in Step 1. Here is the reference table for our standard evaluations:

| Evaluation Task | Prompt Folder (`-p`) | Suggested Output Name (`-n`) | Target Column (`-c`) |
| :--- | :--- | :--- | :--- |
| **Counselor Eval** | `data/prompts/counselor_eval` | `counselor_eval` | `deid_conversation` |
| **Counselor Checklist** | `data/prompts/counselor_eval_checklist` | `counselor_eval_checklist` | `Conversation` |
| **Demographics** | `data/prompts/demographics_all_convos` | `client_demographics_extra` | `conversation` |
| **Client Traits** | `data/prompts/convo_traits_for_profile` | `convo_traits_for_profile` | `deid_conversation` |
| **Message Level Eval** | `data/prompts/message_level_eval` | `message_level_eval` | `prompt` |
| **Presenting Concern** | `data/prompts/presenting_concern` | `presenting_concern_eval` | `deid_convo` |
| **Risk Assessment** | `data/prompts/risk_assessment` | `risk_assessment_eval` | `deid_convo` |

## Advanced: Comparing Models to Humans

If you want to compare the LLM's static generation directly against the human counselor's historical response:

1. Convert raw dialogues to static eval format:
```bash
python singleturn/convert_dialogue_to_static_eval_format.py \
    data/singleturn/for_prompting/dialogues.csv \
    data/singleturn/for_prompting/static_eval_input.csv \
    --conversation-column conversation
```
2. Generate LLM responses using the Generic Pipeline above (Step 1-3).
3. Combine human responses with LLM generated responses:
```bash
python singleturn/prepare_comparison_data.py \
    data/singleturn/for_prompting/static_eval_input.csv \
    data/llm_inference/extracted_completions/<generation_file>.csv \
    data/singleturn/for_prompting/static_eval_comparison.csv
```
4. Run the Generic Pipeline (Steps 1-3) again, but this time using the `static_eval_comparison.csv` and a comparison rubric prompt.
5. Analyze the scores:
```bash
python singleturn/analyze_comparison_scores.py \
    data/llm_inference/extracted_completions/<comparison_results>.csv \
    data/analysis/static_evals/static_eval_rubric_stats.csv
```
