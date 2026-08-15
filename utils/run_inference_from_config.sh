#!/bin/bash

# Requirements: yq (https://github.com/mikefarah/yq)
CONFIG_FILE=$1
YAML_ELEMENT=${2:-"runs"}  # Default to "runs" if not specified

if [ -z "$CONFIG_FILE" ]; then
  echo "Usage: ./run_inference_from_config.sh <path_to_yaml_config> [yaml_element]"
  echo "  yaml_element: The YAML element to process (default: runs)"
  exit 1
fi

# Check if the specified element exists in the YAML
if ! yq -r ".$YAML_ELEMENT" "$CONFIG_FILE" > /dev/null 2>&1; then
  echo "Error: Element '$YAML_ELEMENT' not found in config file"
  exit 1
fi

NUM_RUNS=$(yq -r ".$YAML_ELEMENT | length" "$CONFIG_FILE")

echo "Processing $NUM_RUNS configurations from element '$YAML_ELEMENT'"

for (( i=0; i<$NUM_RUNS; i++ ))
do
  NAME=$(yq -r ".$YAML_ELEMENT[$i].name" "$CONFIG_FILE")
  INPUT_CSV=$(yq -r ".$YAML_ELEMENT[$i].input_csv" "$CONFIG_FILE")
  PROMPT_DIR=$(yq -r ".$YAML_ELEMENT[$i].prompt_dir" "$CONFIG_FILE")
  COLUMN=$(yq -r ".$YAML_ELEMENT[$i].column" "$CONFIG_FILE")
  MODEL=$(yq -r ".$YAML_ELEMENT[$i].model" "$CONFIG_FILE")
  ENGINE=$(yq -r ".$YAML_ELEMENT[$i].engine" "$CONFIG_FILE")
  MAX_REQUESTS=$(yq -r ".$YAML_ELEMENT[$i].max_requests" "$CONFIG_FILE")
  MAX_TOKENS=$(yq -r ".$YAML_ELEMENT[$i].max_tokens" "$CONFIG_FILE")
  API_URL=$(yq -r ".$YAML_ELEMENT[$i].api_url // \"https://api.openai.com/v1/chat/completions\"" "$CONFIG_FILE")
  API_KEY_ENV=$(yq -r ".$YAML_ELEMENT[$i].api_key_env // \"OPENAI_API_KEY\"" "$CONFIG_FILE")

  # Handle input file patterns
  if [[ "$INPUT_CSV" == *"*"* ]]; then
    # Find the most recent file matching the pattern
    LATEST_INPUT=$(ls -t $INPUT_CSV 2>/dev/null | head -n 1)
    if [ -z "$LATEST_INPUT" ]; then
      echo "Error: No files found matching pattern: $INPUT_CSV"
      exit 1
    fi
    INPUT_CSV="$LATEST_INPUT"
    echo "Using most recent input file: $INPUT_CSV"
  fi

  # Generate timestamp for output
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_NAME="${NAME}_${TIMESTAMP}"

  echo "Running inference and extraction for: $NAME"
  bash utils/inference_and_extract.sh \
    "$INPUT_CSV" \
    "$PROMPT_DIR" \
    "$COLUMN" \
    "$OUTPUT_NAME" \
    "$MODEL" \
    "$ENGINE" \
    "$MAX_REQUESTS" \
    "$MAX_TOKENS" \
    "$API_URL" \
    "$API_KEY_ENV"
done
