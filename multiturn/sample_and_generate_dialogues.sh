#!/bin/bash

# ORIGINAL:
# Directory containing the config files
CONFIG_DIR="multiturn/dialogue_configs/$(date +%m-%d-%Y)"

# Output directory for generated dialogues
OUTPUT_DIR="multiturn/generated_dialogues/$(date +%m-%d-%Y)"


mkdir -p "$OUTPUT_DIR"

# Number of config files to sample
NUM_SAMPLES=$1

# Check if NUM_SAMPLES is provided
if [ -z "$NUM_SAMPLES" ]; then
  echo "Usage: ./sample_and_generate.sh <number_of_samples>"
  exit 1
fi

# Randomly sample config files
SAMPLED_CONFIGS=$(ls "$CONFIG_DIR"/*.yaml | shuf -n "$NUM_SAMPLES")

# Loop through sampled config files and generate dialogues
for CONFIG in $SAMPLED_CONFIGS; do
  # Extract the base name of the config file
  BASENAME=$(basename "$CONFIG" .yaml)
  
  # Define the output file for the generated dialogue
  OUTPUT_FILE="$OUTPUT_DIR"

  # Generate dialogue using the config file
  echo "Generating dialogue for config: $CONFIG"
  python multiturn/generate_dialogue.py --config "$CONFIG" --output "$OUTPUT_FILE"
  
  if [ $? -eq 0 ]; then
    echo "Dialogue generated successfully: $OUTPUT_FILE"
  else
    echo "Error generating dialogue for config: $CONFIG"
  fi
done

echo "All dialogues generated in $OUTPUT_DIR."
