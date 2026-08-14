import glob
import json
import os
import sys
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List

import yaml

from crisisbench.multi_turn.generate_dialogue import generate_response
# Constants
DEFAULT_PROMPTS = [
    "What's your name, age, and occupation?",
    "Tell me about what's going on.",
    "What are some of your hobbies?",
    "This seems random, but it's really going to help me with my counseling - what do you usually dream about when you're stressed?",
    "If your anxiety had a voice, what would it say?",
    "How do you usually handle failure or setbacks?",
]

DEFAULT_CONFIG_DIR = "/Users/filipr/vf_copilot/evals/llm_patient/dialogue_configs/5-26"
DEFAULT_RESPONSE_ROLE = "patient"
DEFAULT_N_SAMPLES = 5
DEFAULT_MAX_WORKERS = 20
DEFAULT_JUDGE_PROMPT = 'binary_judge'

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_responses_parallel(prompt: str, config: dict, role: str, n_samples: int, max_workers: int = 5) -> List[str]:
    responses = []

    def single_call():
        return generate_response(
            role=role,
            message=prompt,
            prompt=config['prompts'][f'{role}_prompt'],
            config_path=config[f'{role}_config_path'],
            prompt_suffix=config.get('prompt_suffix'),
            editor_prompt=config['prompts'].get('editor_prompt' if role == 'counselor' else 'patient_editor_prompt'),
            editor_prompt_config=config.get(f'{role}_editor_config_path'),
            model=config[f'{role}_model'],
            temperature=config.get(f'{role}_temperature', 0.7),
            api_url=config.get(f'{role}_api_url'),
            api_key_env=config.get(f'{role}_api_key_env')
        ).strip()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_call) for _ in range(n_samples)]
        for future in as_completed(futures):
            try:
                responses.append(future.result().replace('//', ''))
            except Exception as e:
                responses.append(f"[ERROR] {e}")
                print(f"Error during LLM call: {e}")
    return responses

def judge_response(compare_data: str, config, judge_prompt) -> str:
    return generate_response(
        role='counselor',
        message=compare_data,
        prompt=judge_prompt,
        config_path=os.path.abspath('evals/llm_patient/stability_analysis_config.yaml'),
        prompt_suffix=None,
        editor_prompt=None,
        model=config['patient_model'], #choose only patient model, since to judge patient responses with the model that generated them
        phase_selection_prompt=None,
        editor_prompt_config=None,
        temperature=1,
        api_url=config['patient_api_url'],
        api_key_env=config['patient_api_key_env']
    ).strip()

def stability_analysis(prompts: List[str], config_path: str, response_role: str, n_samples: int, max_workers: int, judge_prompt: str):
    config = load_config(config_path)
    patient_prompts = load_config(config[f'patient_config_path'])
    results = []

    for prompt in prompts:
        responses = get_responses_parallel(prompt, config, response_role, n_samples, max_workers)
        judged_scores = judge_response(
            'PATIENT WAS PROMPTED WITH: ' + prompt + '\n' +
            'PATIENT RESPONSES:' + str([ responses for response in responses if '[error]' not in response.lower()]) + '\n' +
            'PATIENT PROFILE:' + re.search(r"(\*\*\* Personal details \*\*\*.*?)(?=\nMUST FOLLOW THESE)", patient_prompts[config['prompts']['patient_prompt']], re.DOTALL).group(1).strip().replace(
                '\nUse the following language patterns as model responses of what the patient actually sounds like.\nCopy the language in the examples closely, modifying details (names, events, etc.) as needed.',
                ''), config, judge_prompt)
        # Extract JSON content from Markdown-wrapped code block
        try:
            json_str = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', judged_scores, re.DOTALL)
            if json_str is None:
                raise ValueError("No JSON code block found.")
            judged_score_json = json.loads(json_str.group(1))
        except Exception as e:
            print(f"Failed to parse JSON list from judge response: {e}")
            judged_score_json = [{"error": f"Failed to parse JSON: {str(e)}", "raw": judged_scores}]

        results.append({
            "prompt": prompt,
            "responses": responses,
            "judge_score": judged_score_json
        })

    return results

def run_stability_for_config(config_path, prompts, response_role, n_samples, max_workers, judge_prompt):
    try:
        results = stability_analysis(prompts, config_path, response_role, n_samples, max_workers, judge_prompt)
        return os.path.basename(config_path), results
    except Exception as e:
        print(f"Error running analysis for {config_path}: {e}")
        return os.path.basename(config_path), {"error": str(e)}

def run_batch_stability_analysis(
        config_dir: str,
        prompts: List[str],
        response_role: str,
        n_samples: int,
        max_workers: int,
        judge_prompt: str):
    config_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    batch_results = {}

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_stability_for_config, config_path, prompts, response_role, n_samples, max_workers, judge_prompt): config_path
            for config_path in config_files
        }

        for future in as_completed(futures):
            config_name, results = future.result()
            print(f"Finished analysis for config: {config_name}")
            batch_results[config_name] = results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'stability_analysis_batch_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=4, ensure_ascii=False)
    print(f"Batch stability analysis results saved to {output_file}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch stability analysis across LLM configs.")
    parser.add_argument("--config_dir", type=str, required=True, help="Path to directory with YAML config files.")
    parser.add_argument("--response_role", type=str, default=DEFAULT_RESPONSE_ROLE, help="LLM response role (e.g., patient, counselor).")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Number of times prompt is repeated.")
    parser.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS, help="Number of threads for parallelism.")
    parser.add_argument("--judge_prompt", type=str, default=DEFAULT_JUDGE_PROMPT, help="Prompt from stability analysis config file to use for LLM judge.")

    known_args, unknown_args = parser.parse_known_args()

    valid_prompt_pattern = re.compile(r"--prompt_\d+$")
    prompts = []
    unrecognized = []
    for i in range(0, len(unknown_args), 2):
        try:
            key = unknown_args[i]
            value = unknown_args[i + 1]
        except IndexError:
            unrecognized.append(unknown_args[i])
            continue
        if valid_prompt_pattern.match(key):
            prompts.append((int(key.split("_")[1]), value))
        else:
            unrecognized.extend([key, value])

    if unrecognized:
        print(f"Error: Unrecognized arguments: {' '.join(unrecognized)}", file=sys.stderr)
        sys.exit(1)

    prompts = [text for _, text in sorted(prompts)]

    if not prompts:
        prompts = DEFAULT_PROMPTS
    print(f'Beginning batch stability analysis with prompts: \n{prompts}')
    run_batch_stability_analysis(
        config_dir=known_args.config_dir,
        prompts=prompts,
        response_role=known_args.response_role,
        n_samples=known_args.n_samples,
        max_workers=known_args.max_workers,
        judge_prompt=known_args.judge_prompt
    )
