"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python singleturn/clients/azure_openai_api_parallel.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-3-small", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-3-small`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).
- concurrency : int, optional
    - number of requests running at the same time with async
    - if ommited, will default to 25, which is safe for most systems
    - you can choose higher values if you have a good network connection and specs or lots of data     

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 0, 1, 2, ...)
    - Run main()
"""

# constants
APIM_SUBSCRIPTION_HEADER = "Ocp-Apim-Subscription-Key" 

# imports
# print python path
# add /home/stanford/vf_copilot/evals to python path
import sys
sys.path.append("/home/stanford/vf_copilot/evals")
print(sys.path)

from static_evals.estimate_llm_api_cost import estimate_request_limits
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata
from dotenv import load_dotenv
from typing import Dict, Any  # Add this import

load_dotenv()

def build_request_header(request_url: str, api_key: str) -> Dict[str, str]:
    """Return the correct auth header for OpenAI, Azure OpenAI, or SHC‑APIM."""
    base = {"Content-Type": "application/json"}

    if request_url.startswith("https://api.openai.com"):
        base["Authorization"] = f"Bearer {api_key}"

    elif ".openai.azure.com" in request_url:
        # direct Azure OpenAI (not routed through APIM)
        base["api-key"] = api_key

    elif "apim.stanfordhealthcare.org" in request_url:
        # Stanford APIM gateway (Claude, o1/o3, Gemini, etc.)
        base[APIM_SUBSCRIPTION_HEADER] = api_key

    else:
        raise ValueError(f"Don't know how to auth for URL {request_url}")

    return base


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    model_name: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    max_tokens_per_generation: int,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
    generation_params: dict,
    max_concurrent_requests: int
):
    """
    Processes API requests in parallel, throttling to stay under rate limits **and
    always carries the original `metadata` from the input file all the way to the
    saved completion line – no matter what model‐specific transforms happen
    later.  (This is the only change.)**
    """
    # ── unchanged setup ──────────────────────────────────────────────────────
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001
    logging.basicConfig(level=logging_level)
    api_endpoint  = api_endpoint_from_url(request_url)
    request_header = build_request_header(request_url, api_key)

    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator         = task_id_generator_function()
    status_tracker            = StatusTracker()
    next_request              = None

    available_request_capacity = max_requests_per_minute
    available_token_capacity   = max_tokens_per_minute
    last_update_time           = time.time()
    file_not_finished          = True

    # OpenAI rate limit is 1500 per minute, that's 25 per second.
    # If average API latency is ~1s (normal), then 25 concurrent requests should be safe.
    # This should hold for most other APIs
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # ── main loop ───────────────────────────────────────────────────────────
    with open(requests_filepath) as fh:
        async with aiohttp.ClientSession() as session:
            lines = fh.__iter__()

            while True:

                # ---------- pull the next task --------------------------------
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                    elif file_not_finished:
                        try:
                            raw = json.loads(next(lines))

                            # ░░ capture original metadata before any mutation ░░
                            original_metadata = raw.pop("metadata", None)

                            # --- common knobs --------------------------------------------------
                            raw.update(
                                model       = model_name,
                                temperature = generation_params["generation"]["temperature"],
                                top_p       = generation_params["generation"]["top_p"],
                            )

                            # --- model‑specific tweaks -----------------------------------------
                            if model_name in ("o1", "o3") or model_name.startswith(("o1-", "o3-")):
                                raw.pop("max_tokens", None)
                                raw["max_completion_tokens"] = max_tokens_per_generation
                                raw["temperature"] = 1
                                raw["top_p"]       = 1

                            elif "claude" in model_name.lower():
                                raw = {
                                    "model_id"  : f"arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.{model_name}:0",
                                    "prompt_text": raw["messages"][0]["content"],
                                }

                            elif "gemini" in model_name.lower():
                                raw = {
                                    "contents" : {
                                        "role" : "user",
                                        "parts": {"text": raw["messages"][0]["content"]},
                                    },
                                    "safety_settings"  : {
                                        "category"  : "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                        "threshold" : "BLOCK_NONE",
                                    },
                                    "generation_config": {
                                        "temperature": generation_params["generation"]["temperature"],
                                        "topP"       : generation_params["generation"]["top_p"],
                                        "topK"       : 40,
                                    },
                                }

                            elif "phi" in model_name.lower():
                                raw["presence_penalty"]   = 0
                                raw["frequency_penalty"]  = 0

                            elif "deepseek" in model_name.lower():
                                raw["stream"] = False

                            elif "llama" in model_name.lower():
                                raw["max_tokens"] = max_tokens_per_generation
                            else:
                                raw["max_tokens"] = max_tokens_per_generation
                            # -------------------------------------------------------------------

                            next_request = APIRequest(
                                task_id        = next(task_id_generator),
                                request_json   = raw,
                                token_consumption = num_tokens_consumed_from_request(
                                    raw, api_endpoint, token_encoding_name
                                ),
                                attempts_left  = max_attempts,
                                metadata       = original_metadata,   # ← keep it!
                            )
                            status_tracker.num_tasks_started   += 1
                            status_tracker.num_tasks_in_progress += 1
                        except StopIteration:
                            file_not_finished = False

                # ---------- throttle ----------------------------------------------------------
                now = time.time()
                elapsed = now - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * elapsed / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * elapsed / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = now

                # ---------- dispatch ----------------------------------------------------------
                if next_request:
                    if (available_request_capacity >= 1 and
                        available_token_capacity >= next_request.token_consumption):

                        available_request_capacity -= 1
                        available_token_capacity   -= next_request.token_consumption
                        next_request.attempts_left -= 1

                        # Use a semaphore to limit concurrent API requests
                        async def run_with_semaphore(request):
                            async with semaphore:
                                await request.call_api(
                                    session      = session,
                                    request_url  = request_url,
                                    request_header = request_header,
                                    retry_queue  = queue_of_requests_to_retry,
                                    save_filepath= save_filepath,
                                    status_tracker = status_tracker,
                                )

                        asyncio.create_task(run_with_semaphore(next_request))
                        next_request = None

                # ---------- exit check --------------------------------------------------------
                if status_tracker.num_tasks_in_progress == 0:
                    break

                await asyncio.sleep(seconds_to_sleep_each_loop)

                # cool‑down after rate‑limit
                since_rl = time.time() - status_tracker.time_of_last_rate_limit_error
                if since_rl < seconds_to_pause_after_rate_limit_error:
                    await asyncio.sleep(seconds_to_pause_after_rate_limit_error - since_rl)

    # ── final log ───────────────────────────────────────────────────────────
    logging.info(f"Parallel processing complete. Results saved to {save_filepath}")
    if status_tracker.num_tasks_failed:
        logging.warning(
            f"{status_tracker.num_tasks_failed}/{status_tracker.num_tasks_started} "
            f"requests failed.  See error lines in results file."
        )
    if status_tracker.num_rate_limit_errors:
        logging.warning(f"{status_tracker.num_rate_limit_errors} rate‑limit errors encountered.")


# dataclasses


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: "StatusTracker",
    ):
        """Calls the API once, handles HTTP / JSON errors, and writes result."""
        task = self.task_id
        logging.debug(f"[{task}] → POST {request_url}")

        # build request body (already prepared upstream)
        body = self.request_json

        try:
            async with session.post(request_url, headers=request_header, json=body) as resp:
                raw_text = await resp.text()
                logging.debug(f"[{task}] ← {resp.status} {resp.headers.get('Content-Type')}")
                logging.debug(dict(resp.headers))

                # ── 1. HTTP‑level errors ────────────────────────────────────────
                if resp.status >= 400:
                    raise ValueError(f"HTTP {resp.status}: {raw_text[:300]}")

                # ── 2. Decode JSON safely ──────────────────────────────────────
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    raise ValueError(f"Non‑JSON response: {raw_text[:300]}")

                # ── 3. Provider‑specific normalisation ────────────────────────
                model_name = body.get("model", "")
                if "model_id" in body:  # Claude branch
                    model_name = self.metadata.get("model_name", model_name)  # Default to claude-3 for model_id requests
                
                # Parse the response into a consistent format
                content = parse_response(model_name, data)
                
                # Format the response in a consistent structure
                normalized_response = {
                    "choices": [{
                        "message": {
                            "content": content
                        }
                    }]
                }

                # For Claude models, ensure metadata is preserved in a flat format
                if model_name.startswith("claude"):
                    # Extract metadata from the request body
                    metadata = {}
                    if self.metadata:
                        metadata.update(self.metadata)
                    if "metadata" in body:
                        metadata.update(body["metadata"])
                    
                    # Create a flat response structure that matches the expected format
                    flat_response = {
                        "prompt": body.get("prompt_text", ""),
                        "completion": content,
                        "uid": metadata.get("uid", ""),
                        "prompt_template_name": metadata.get("prompt_template_name", ""),
                        "metadata": metadata
                    }
                    payload = flat_response
                else:
                    # For other models, keep the existing format
                    payload = [body, normalized_response] if self.metadata is None else [body, normalized_response, {"metadata": self.metadata}]

                append_to_jsonl(payload, save_filepath)
                status_tracker.num_tasks_succeeded += 1
                logging.debug(f"[{task}] saved")

        except Exception as e:
            logging.warning(f"[{task}] failed: {e}")
            self.attempts_left -= 1
            self.result.append(str(e))

            if self.attempts_left > 0:
                retry_queue.put_nowait(self)
            else:
                payload = [body, self.result] if self.metadata is None else [body, self.result, {"metadata": self.metadata}]
                append_to_jsonl(payload, save_filepath)
                status_tracker.num_tasks_failed += 1
        finally:
            status_tracker.num_tasks_in_progress -= 1



# functions


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    # Try OpenAI/Azure pattern first
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is not None:
        return match[1]
    
    # Try Azure OpenAI deployment pattern
    match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
    if match is not None:
        return match[1]
    
    # Try Claude pattern
    if "anthropic.com" in request_url:
        return "messages"
    
    # Try Gemini pattern
    if "generativelanguage.googleapis.com" in request_url:
        return "generateContent"
    
    # Try DeepInfra pattern
    if "deepinfra.com" in request_url:
        return "inference"
    
    # Try Deepseek pattern
    if "deepseek.com" in request_url:
        return "chat/completions"
    
    # Try Meta pattern
    if "meta.ai" in request_url:
        return "chat/completions"
    
    # If no pattern matches, return a default value
    return "chat/completions"


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file, creating parent directories if necessary."""
    # Ensure parent directory exists
    parent_dir = os.path.dirname(filename)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # Append the JSON data to the file
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    
    # Handle different model request formats
    if "messages" in request_json:  # OpenAI/Azure format
        num_tokens = 0
        for message in request_json["messages"]:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    elif "prompt_text" in request_json:  # Claude format
        return len(encoding.encode(request_json["prompt_text"]))
    elif "contents" in request_json:  # Gemini format
        num_tokens = 0
        contents = request_json["contents"]

        # Gemini API can accept either a single dict or a list of dicts under "contents"
        if isinstance(contents, dict):
            parts = contents.get("parts")
            if isinstance(parts, dict):
                # Newer Gemini spec: single dict
                if "text" in parts:
                    num_tokens += len(encoding.encode(parts["text"]))
            elif isinstance(parts, list):
                # Alternative spec: list of part dicts
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        num_tokens += len(encoding.encode(part["text"]))
        elif isinstance(contents, list):
            for content in contents:
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if isinstance(parts, dict):
                    if "text" in parts:
                        num_tokens += len(encoding.encode(parts["text"]))
                elif isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            num_tokens += len(encoding.encode(part["text"]))
        return num_tokens
    elif "input" in request_json:  # Generic format
        if isinstance(request_json["input"], str):
            return len(encoding.encode(request_json["input"]))
        elif isinstance(request_json["input"], list):
            return sum(len(encoding.encode(i)) for i in request_json["input"])
        else:
            raise TypeError('Expecting either string or list of strings for "input" field')
    else:
        raise ValueError("Unsupported request format")


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def parse_response(model_name: str, response: Dict[str, Any]) -> str:
    """
    Parse the response from different model types into a consistent format.
    
    Supports:
      - Anthropic/Claude via Bedrock ("content": [ … ] or "completion": str)
      - Gemini ("candidates": [ … ])
      - OpenAI‑style chat/completions ("choices": [ { "message": {"content": …} } ])
      - Older OpenAI‑style completions ("choices": [ { "text": … } ])
    """
    try:
        # ────────────────────── Gemini first (may be list) ──────────────────────
        if isinstance(response, list):
            wrappers = response
        else:
            wrappers = [response]

        aggregated = []
        for wrapper in wrappers:
            if not isinstance(wrapper, dict):
                continue
            # Gemini wrapper → candidates list
            candidates = wrapper.get("candidates")
            if not isinstance(candidates, list):
                continue
            for cand in candidates:
                if not isinstance(cand, dict):
                    continue
                content_obj = cand.get("content", {}) or {}
                parts = content_obj.get("parts")
                if isinstance(parts, dict) and "text" in parts:
                    aggregated.append(parts["text"])
                elif isinstance(parts, list):
                    aggregated.extend(
                        p.get("text", "") for p in parts if isinstance(p, dict)
                    )

        if aggregated:
            result = "".join(aggregated)
            logging.debug(
                f"Gemini parse_response extracted text length={len(result)} from {len(aggregated)} chunks"
            )
            return result

        # ── Anthropic / Claude via Bedrock: a list of content parts ─────────────
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )

            # ── Anthropic / Claude via Bedrock: single completion key ────────────
            if isinstance(response.get("completion"), str):
                return response["completion"]

        # ── OpenAI‑style chat completions ────────────────────────────────────────
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                msg = first.get("message", {})
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                if "text" in first and isinstance(first["text"], str):
                    return first["text"]

        # ── nothing matched ───────────────────────────────────────────────────────
        return ""
    except Exception as e:
        logging.error(f"Error parsing response for model {model_name}: {e}")
        logging.debug(f"Response payload was: {response}")
        return ""


# run script


if __name__ == "__main__":

    start_time = time.time()

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests_filepath")
    parser.add_argument("--save_filepath", default=None)
    parser.add_argument("--request_url", default="https://api.openai.com/v1/embeddings")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model_name", type=str, help="model identifier", default=None)
    parser.add_argument("--max_requests_per_minute", type=int, default=3_000 * 0.5)
    parser.add_argument("--max_tokens_per_minute", type=int, default=250_000 * 0.5)
    parser.add_argument("--token_encoding_name", default="cl100k_base")
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument("--max_tokens_per_generation", type=int, default=256, help="max new tokens")
    parser.add_argument("--logging_level", default=logging.INFO)
    parser.add_argument("--generation_config", type=str, help="json file containing generation parameters", default=None)
    parser.add_argument("--max_cost_threshold", type=int, help="max cost per batch in $", default=100)
    parser.add_argument("--concurrency", type=int, help="Max number of concurrent api requests", default=20)
    args = parser.parse_args()

    if args.save_filepath is None:
        args.save_filepath = args.requests_filepath.replace(".jsonl", "_results.jsonl")
        
    generation_params = json.load(open(args.generation_config, "r"))
    
    args.api_key = os.getenv(args.api_key)

    # Estimate the cost and request limits before running the API processing
    # Do a rough estimation by converting all the "messages" to string
    
    with open(args.requests_filepath) as f:
        prompts = []
        for line in f:
            request = json.loads(line)
            # Join all the content from the "messages" into one string
            concatenated_prompt = " ".join([message["content"] for message in request["messages"]])
            prompts.append(concatenated_prompt)

    try:
        estimate_request_limits(
            prompts=prompts,
            user_model_name=args.model_name,
            tokens_per_minute=args.max_tokens_per_minute,
            max_tokens=args.max_tokens_per_generation,
            max_cost_threshold=args.max_cost_threshold  # Pass the cost threshold
        )
    except ValueError as e:
        print(f"Cost estimation error: {e}")
        exit(1) 

    # run script
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=args.requests_filepath,
            save_filepath=args.save_filepath,
            request_url=args.request_url,
            api_key=args.api_key,
            model_name=args.model_name,
            max_requests_per_minute=float(args.max_requests_per_minute),
            max_tokens_per_minute=float(args.max_tokens_per_minute),
            max_tokens_per_generation=args.max_tokens_per_generation,
            token_encoding_name=args.token_encoding_name,
            max_attempts=int(args.max_attempts),
            logging_level=int(args.logging_level),
            generation_params=generation_params,
            max_concurrent_requests=args.concurrency
        )
    )

    # Stop measuring time and print the elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


"""
APPENDIX

The example requests file at openai-cookbook/examples/data/example_requests_to_parallel_process.jsonl contains 10,000 requests to text-embedding-3-small.

It was generated with the following code:

```python
import json

filename = "data/example_requests_to_parallel_process.jsonl"
n_requests = 10_000
jobs = [{"model": "text-embedding-3-small", "input": str(x) + "\n"} for x in range(n_requests)]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
```

As with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically).
"""
