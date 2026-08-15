#!/bin/bash

# Usage: ./run_inference_client.sh <request_file> <model> <client> [splits] [max_tokens]

# Parse command-line arguments
REQUEST_FILE="$1"  # Input request file as first argument
models=("$2")      # Model name as second argument (ignored for kb_bioannotator)
client="$3"        # Client type as third argument
SPLITS="${4:-25}"  # Optional splits argument with default of 25
MAX_TOKENS="${5:-4000}" # max tokens for generation
GPT_ENDPOINT="${6:-https://shcopenaisandbox.openai.azure.com/openai/deployments/shc-gpt-4o/chat/completions?api-version=2023-03-15-preview}" # Default endpoint
API_KEY_ENV="${7:-OPENAI_API_KEY}" # API key environment variable name

# Generate timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Directory paths
save_directory="${DATA_ROOT}/completions"
splits_directory_base="${save_directory}/splits_${timestamp}"
base_directory="evals/static_evals"
generation_config="${base_directory}/clients/generation_params.json"
expected_cost=300

# Create save directory
mkdir -p "$save_directory"

# Function to split the input file for parallel processing
split_input_file() {
    local input_file=$1
    local num_splits=$2
    local output_dir=$3

    echo "Splitting $input_file into $num_splits parts..."
    lines_per_split=$(( $(wc -l < "$input_file") / num_splits ))
    [ "$lines_per_split" -lt 1 ] && lines_per_split=1
    split -l "$lines_per_split" -d -a 2 "$input_file" "$output_dir/split_"
}

echo "Processing request file: $REQUEST_FILE with max tokens: $MAX_TOKENS"

# Split and process files if client is vertex or openai-batch
if [[ "$client" == "vertex" || "$client" == "openai-batch" ]]; then
    # Loop through models and run inference in parallel
    for model in "${models[@]}"; do
        echo "Running inference for model: $model on request file"

        model_safe_name=$(echo "$model" | sed 's/\//_/g')
        splits_directory="${splits_directory_base}/${model_safe_name}/splits_${timestamp}"

        echo "Clearing old split files in $splits_directory..."
        [ -d "$splits_directory" ] && rm -r "$splits_directory"
        mkdir -p "$splits_directory"

        split_input_file "$REQUEST_FILE" "$SPLITS" "$splits_directory"
        
        a=0
        for file in "$splits_directory"/split_*; do
            mv "$file" "${file}.jsonl"
            a=$((a + 1))
        done

        for split_file in "$splits_directory"/split_*.jsonl; do
            i=$(basename "$split_file" | sed 's/split_//' | sed 's/.jsonl//')
            save_filepath="${splits_directory}/merged_${model_safe_name}_split_${i}_max${MAX_TOKENS}_${timestamp}.jsonl"
            log_file="${splits_directory}/log_${model_safe_name}_split_${i}_${timestamp}.log"

            echo "Processing split $i for model: $model"

            {
                if [ "$client" == "vertex" ]; then
                    python "${base_directory}/clients/vertex_api.py" \
                        --input_jsonl "$split_file" \
                        --output_jsonl "$save_filepath" \
                        --model_name "$model" \
                        --generation_config "$generation_config" \
                        --max_retries 3 \
                        --max_new_tokens "$MAX_TOKENS"
                else
                    python "${base_directory}/clients/azure_openai_api_parallel.py" \
                        --requests_filepath "$split_file" \
                        --save_filepath "$save_filepath" \
                        --request_url "$GPT_ENDPOINT" \
                        --model_name "$model" \
                        --generation_config "$generation_config" \
                        --max_tokens_per_generation "$MAX_TOKENS" \
                        --max_cost_threshold "$expected_cost" \
                        --api_key "$API_KEY_ENV"
                fi
            } >> "$log_file" 2>&1 &
        
        done

        wait
    done

    # Wait for tmux jobs to complete
    for model in "${models[@]}"; do
        model_safe_name=$(echo "$model" | sed 's/\//_/g')
        splits_directory="${splits_directory_base}/${model_safe_name}/splits_${timestamp}"
        final_output_filepath="${save_directory}/final_merged_${model_safe_name}_max${MAX_TOKENS}_${timestamp}.jsonl"

        echo "Merging outputs into ${final_output_filepath}..."
        find "${splits_directory}" -name "merged_${model_safe_name}_split_*.jsonl" -exec cat {} + > "$final_output_filepath"
    done

    echo "Process complete for all models!"
fi

# Run transformers client if selected
if [ "$client" = "transformers" ]; then
    for model in "${models[@]}"; do
        model_safe_name=$(echo "$model" | sed 's/\//_/g')
        save_filepath="${save_directory}/merged_${model_safe_name}_max${MAX_TOKENS}_${timestamp}.jsonl"

        time python ${base_directory}/clients/transformers_api.py \
            --path_to_prompted_dataset "$REQUEST_FILE" \
            --path_to_output_file "$save_filepath" \
            --model_name_or_path "$model" \
            --generation_config "$generation_config" \
            --dynamic_batching 20000 \
            --max_generation_length "$MAX_TOKENS"
    done
fi