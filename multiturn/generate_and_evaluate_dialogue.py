import os
import argparse
from multiturn.generate_dialogue import generate_dialogue, load_config
from multiturn.eval_dialogue import evaluate_conversations

def write_pipeline_results(output_path, patient_profile_name, patient_profile, dialogue_text, 
                           patient_eval_results, counselor_unit_test_results, counselor_dialogue_eval_results):
    """
    Write the results of the dialogue generation and evaluation pipeline to a single, well-structured text file.
    """
    with open(output_path, "w") as f:
        f.write("======================= PATIENT PROFILE =======================\n")
        f.write(f"Profile Name: {patient_profile_name}\n")
        f.write(f"Profile Details:\n")
        f.write(f"{patient_profile}\n\n")

        f.write("======================= GENERATED DIALOGUE =======================\n")
        f.write(dialogue_text + "\n\n")

        f.write("======================= PATIENT EVALUATION =======================\n")
        for prompt, result in patient_eval_results.items():
            f.write(f"** {prompt.replace('_', ' ').title()} **\n")
            for key, value in result.items():
                if isinstance(value, dict) and "score" in value:
                    f.write(f"  {key}: {value['score']} - {value['rationale']}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("======================= COUNSELOR UNIT TESTS =======================\n")
        for test, result in counselor_unit_test_results.items():
            f.write(f"** {test.replace('_', ' ').title()} **\n")
            for key, value in result.items():
                if isinstance(value, list):
                    f.write(f"  {key}: {', '.join(value)}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("======================= COUNSELOR EVALUATION =======================\n")
        for eval_name, result in counselor_dialogue_eval_results.items():
            f.write(f"** {eval_name.replace('_', ' ').title()} **\n")
            for key, value in result.items():
                if isinstance(value, dict) and "score" in value:
                    f.write(f"  {key}: {value['score']} - {value['rationale']}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    print(f"Pipeline results written to {output_path}")


def main(dialogue_config_path, patient_eval_config_path, patient_profile_config_path, 
         counselor_unit_test_config_path, counselor_eval_config_path, output_dir, output_file):
    dialogue_config = load_config(dialogue_config_path)
    dialogue_output_dir = os.path.join(output_dir, "generated_dialogues")

    patient_llm_eval_config = load_config(patient_eval_config_path)
    counselor_unit_test_eval_config = load_config(counselor_unit_test_config_path)
    counselor_dialogue_eval_config = load_config(counselor_eval_config_path)

    os.makedirs(dialogue_output_dir, exist_ok=True)
    evals_output_dir = os.path.join(output_dir, "evaluator_outputs")
    os.makedirs(evals_output_dir, exist_ok=True)

    dialogue_text = generate_dialogue(dialogue_config, dialogue_output_dir)
    dialogue_file = os.path.basename(list(dialogue_text.keys())[0])

    patient_llm_eval = evaluate_conversations(
        config=patient_llm_eval_config, 
        input_path=os.path.join(dialogue_output_dir, dialogue_file), #dialogue_output_dir, 
        output_dir=evals_output_dir, 
        patient_config=patient_profile_config_path,
        evaluator_prompts=["language_patterns", "presenting_concern", "realism", "big_five_traits_simple"], 
        model="o1-mini"
    )

    counselor_unit_test_eval = evaluate_conversations(
        config=counselor_unit_test_eval_config,
        input_path=os.path.join(dialogue_output_dir, dialogue_file), #dialogue_output_dir, 
        output_dir=evals_output_dir,
        evaluator_prompts=[
            "risk_assessment", "multiple_tasks", "forbidden", "questions", "empathy_chaining"
            # "multiple_tasks", 
            #                "forbidden", "pacing", "emotional_language", "best_practice",
            #                "build_rapport", "explore_situation", "high_acuity_risk_assessment", "next_steps", 
            #                "supporting_resources", "end_conversation"
                           ],
        model="o1-mini"
    )

    counselor_dialogue_eval = evaluate_conversations(
        config=counselor_dialogue_eval_config,
        input_path=os.path.join(dialogue_output_dir, dialogue_file), #dialogue_output_dir, 
        output_dir=evals_output_dir,
        evaluator_prompts=["productivity", "micro_skills", "context", "style_tone_language"],
        model="o1-mini"
    )

    patient_profile_name = dialogue_config["prompts"]["patient_prompt"]
    patient_profile = load_config(patient_profile_config_path)[patient_profile_name]

    write_pipeline_results(
        output_path=os.path.join(output_dir, output_file),
        patient_profile_name=patient_profile_name,
        patient_profile=patient_profile,
        dialogue_text=dialogue_text[dialogue_file],
        patient_eval_results=patient_llm_eval[dialogue_file],
        counselor_unit_test_results=counselor_unit_test_eval[dialogue_file],
        counselor_dialogue_eval_results=counselor_dialogue_eval[dialogue_file]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dialogue generation and evaluation pipeline.")
    parser.add_argument("--dialogue_config", required=True, help="Path to the dialogue configuration YAML file.")
    parser.add_argument("--patient_eval_config", required=True, help="Path to the patient evaluation configuration YAML file.")
    parser.add_argument("--patient_profile_config", required=True, help="Path to the patient profile configuration YAML file.")
    parser.add_argument("--counselor_unit_test_config", required=True, help="Path to the counselor unit test configuration YAML file.")
    parser.add_argument("--counselor_eval_config", required=True, help="Path to the counselor evaluation configuration YAML file.")
    parser.add_argument("--output_dir", required=True, help="Directory to store generated dialogues and evaluation outputs.")
    parser.add_argument("--output_file", default="pipeline_results.txt", help="Name of the final results file.")
    args = parser.parse_args()

    main(
        dialogue_config_path=args.dialogue_config,
        patient_eval_config_path=args.patient_eval_config,
        patient_profile_config_path=args.patient_profile_config,
        counselor_unit_test_config_path=args.counselor_unit_test_config,
        counselor_eval_config_path=args.counselor_eval_config,
        output_dir=args.output_dir,
        output_file=args.output_file
    )
