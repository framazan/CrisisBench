#!/bin/bash

# Configuration files
AGENTS_CONFIG="agents_config.yaml"
COUNSELOR_CONFIG_PATH="/Users/akshayswaminathan/vf_copilot/gradio_demo/prompts.yaml"
PATIENT_CONFIG_PATH="patients_config.yaml"

INITIAL_PATIENT_MESSAGE="I feel really overwhelmed and don't know what to do."

# Output directories
DIALOGUE_OUTPUT_DIR="dialogue_output/"
EVAL_OUTPUT_DIR="dialogue_evals_output/"

# Define the number of conversation turns
CONVERSATION_TURNS=15

PATIENT_MODEL="gpt-4o"

# Define arrays for different prompts, phase selection, and evaluator agents
COUNSELOR_PROMPTS=("base_instructions" "base_instructions2")
EDITOR_PROMPTS=("second_call2" "second_call3" "")
PHASE_SELECTION_PROMPTS=("phase_selection1" "phase_selection2" "")  # Add phase selection options
PATIENT_PROMPTS=("suicidal_24m" "anxiety_24f" "relationship_24f")
EVALUATOR_AGENTS=("risk_assessment" "pacing" "forbidden" "multiple_tasks" "quality" "best_practice")
COUNSELOR_MODELS=("gpt-4o")

# Step 1: Generate dialogues by looping over the counselor, editor, and patient prompt combinations
for COUNSELOR_PROMPT in "${COUNSELOR_PROMPTS[@]}"; do
  for EDITOR_PROMPT in "${EDITOR_PROMPTS[@]}"; do
    for PHASE_SELECTION_PROMPT in "${PHASE_SELECTION_PROMPTS[@]}"; do  # Add phase selection loop
      for PATIENT_PROMPT in "${PATIENT_PROMPTS[@]}"; do
        for COUNSELOR_MODEL in "${COUNSELOR_MODELS[@]}"; do
      
        # Generate dialogue configuration file dynamically
        cat <<EOF > temp_dialogue_config.yaml
# Paths to the config files for counselor and patient
counselor_config_path: "$COUNSELOR_CONFIG_PATH"
patient_config_path: "$PATIENT_CONFIG_PATH"

# Initial message from the patient to start the conversation
initial_patient_message: "$INITIAL_PATIENT_MESSAGE"

# Prompts for counselor, phase selection, and patient
prompts:
    counselor_prompt: "$COUNSELOR_PROMPT"
    editor_prompt: "$EDITOR_PROMPT"
    phase_selection_prompt: "$PHASE_SELECTION_PROMPT"  # Phase selection added
    patient_prompt: "$PATIENT_PROMPT"

# Number of conversation turns
conversation_turns: $CONVERSATION_TURNS

# Generation params
counselor_model: "$COUNSELOR_MODEL"
patient_model: "$PATIENT_MODEL"
EOF

            # Generate dialogue for this combination of counselor, editor, phase selection, and patient prompts
            echo "Generating dialogue for Counselor Prompt: $COUNSELOR_PROMPT, Editor Prompt: $EDITOR_PROMPT, Phase Selection Prompt: $PHASE_SELECTION_PROMPT, Patient Prompt: $PATIENT_PROMPT, Counselor model: $COUNSELOR_MODEL"
            python generate_dialogue.py --config temp_dialogue_config.yaml --output_dir $DIALOGUE_OUTPUT_DIR
            
            done
        done
      done
    done
  done
done

# Clean up the temporary config file
rm temp_dialogue_config.yaml


# Step 2: Evaluate dialogues with different evaluators
for EVALUATOR_AGENT in "${EVALUATOR_AGENTS[@]}"; do
    echo "Evaluating dialogue with $EVALUATOR_AGENT"
    python eval_dialogue.py --config $AGENTS_CONFIG --input_dir $DIALOGUE_OUTPUT_DIR --output_dir $EVAL_OUTPUT_DIR --evaluator_prompt $EVALUATOR_AGENT
done
