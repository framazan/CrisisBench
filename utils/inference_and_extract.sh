#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# Check arguments and show usage instructions
usage() {
    echo "Usage (with CSV input):"
    echo "  $0 <input_csv> <prompt_dir> <column> <output_name> <model> <engine> <max_requests> <max_tokens> <api_url> [api_key_env]"
    echo ""
    echo "Usage (with pre-prompted JSONL):"
    echo "  $0 --jsonl <prompted_jsonl> <output_name> <model> <engine> <max_requests> <max_tokens> <api_url> [api_key_env]"
    exit 1
}

# Check if using pre-prompted JSONL mode
if [ "$1" == "--jsonl" ]; then
    if [ "$#" -lt 8 ]; then
        usage
    fi
    
    # JSONL mode arguments
    PROMPTED_FILE="$2"
    OUTPUT_NAME="$3"
    MODEL="$4"
    ENGINE="$5"
    MAX_REQUESTS="$6"
    MAX_TOKENS="$7"
    API_URL="$8"
    API_KEY_ENV="${9:-OPENAI_API_KEY}"  # Default to OPENAI_API_KEY if not provided
    
    # Check if prompted file exists
    if [ ! -f "$PROMPTED_FILE" ]; then
        echo "Error: Prompted JSONL file not found: $PROMPTED_FILE"
        exit 1
    fi
    
    echo "Using pre-prompted JSONL file: $PROMPTED_FILE"
else
    # CSV mode - Check if required arguments are provided
    if [ "$#" -lt 9 ]; then
        usage
    fi
    
    # Assign input arguments for CSV mode
    INPUT_CSV="$1"
    PROMPT_DIR="$2"
    COLUMN="$3"
    OUTPUT_NAME="$4"
    MODEL="$5"
    ENGINE="$6"
    MAX_REQUESTS="$7"
    MAX_TOKENS="$8"
    API_URL="$9"
    API_KEY_ENV="${10:-OPENAI_API_KEY}"  # Default to OPENAI_API_KEY if not provided
fi

# Debug print API key environment variable
echo "Using API key environment variable: $API_KEY_ENV"
if [ -z "${!API_KEY_ENV}" ]; then
    echo "ERROR: API key environment variable '$API_KEY_ENV' is not set!"
    echo "Please set the API key using: export $API_KEY_ENV='your-api-key'"
    echo "Or add it to your .env file"
    exit 1
else
    echo "API key is set (value hidden for security)"
fi

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:"$(pwd)/evals"
export DATA_ROOT="$(pwd)/data/llm_inference"

# Create necessary directories
mkdir -p "${DATA_ROOT}/completions"
mkdir -p "${DATA_ROOT}/extracted_completions"
mkdir -p "${DATA_ROOT}/prompted"

# CSV-to-JSONL conversion (skip if using pre-prompted JSONL)
if [ "$1" != "--jsonl" ]; then
    # Convert CSV to JSONL format
    PROMPTED_DIR="${DATA_ROOT}/prompted"
    python utils/build_prompted_datasets.py \
        -f "$INPUT_CSV" \
        -p "$PROMPT_DIR" \
        -o "$PROMPTED_DIR" \
        -n "$OUTPUT_NAME" \
        -c "$COLUMN" \
        -m "$MAX_TOKENS"

    # Get the most recent prompted file
    PROMPTED_FILE=$(ls -t "${PROMPTED_DIR}/${OUTPUT_NAME}_"*.jsonl | head -n 1)

    # Check if the prompted file was created
    if [ ! -f "$PROMPTED_FILE" ]; then
        echo "Error: Prompted file not created"
        exit 1
    fi
fi

echo "Using prompted file: $PROMPTED_FILE"

# Run the LLM client script with the JSONL file
bash singleturn/run_llm_client.sh \
    "$PROMPTED_FILE" \
    "$MODEL" \
    "$ENGINE" \
    "$MAX_REQUESTS" \
    "$MAX_TOKENS" \
    "$API_URL" \
    "$API_KEY_ENV"

# Check if the completion file was created
# 1️⃣ try the legacy OUTPUT_NAME pattern
COMPLETION_FILE=$(ls -t "${DATA_ROOT}/completions/final_merged_${OUTPUT_NAME}_max${MAX_TOKENS}_"*.jsonl 2>/dev/null | head -n 1)

# 2️⃣ if not found, try the MODEL‑based pattern used by run_llm_client.sh ≥ v0.3
if [ -z "$COMPLETION_FILE" ]; then
  COMPLETION_FILE=$(ls -t "${DATA_ROOT}/completions/final_merged_${MODEL}_max${MAX_TOKENS}_"*.jsonl 2>/dev/null | head -n 1)
fi

if [ ! -f "$COMPLETION_FILE" ]; then
    echo "Error: Completion file not found"
    echo "Expected pattern: ${DATA_ROOT}/completions/final_merged_${OUTPUT_NAME}_max${MAX_TOKENS}_*.jsonl"
    echo "Available files:"
    ls -l "${DATA_ROOT}/completions/"
    exit 1
fi

# Extract completions
python singleturn/extract_content_from_llm_completion.py \
    -i "$COMPLETION_FILE" \
    -o "${DATA_ROOT}/extracted_completions/${OUTPUT_NAME}_extracted_$(date +%Y%m%d).csv" \
    -c "llm_output_json_raw"

# Check if the extraction was successful
if [ ! -f "${DATA_ROOT}/extracted_completions/${OUTPUT_NAME}_extracted_$(date +%Y%m%d).csv" ]; then
    echo "Error: Extracted completions file not created"
    exit 1
fi

echo "Inference and extraction complete!"
