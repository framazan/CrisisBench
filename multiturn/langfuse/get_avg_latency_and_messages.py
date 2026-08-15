import os
from langfuse import Langfuse
import re
from collections import defaultdict

# CONFIGURATION
DATASET_NAME = os.environ.get("DATASET_NAME") or "dynamic-evals-96-final"
RUN_NAMES = ['4.1-realism-200tk-0.5tmp']  # Set to None to auto-discover
TRACE_FETCH_LIMIT = 100  # API max per request
MAX_PAGES = 200  # Safety cap (fetch up to 10,000 traces)

lf = Langfuse()


def _get_response_data(resp):
    return resp.data if hasattr(resp, "data") else resp


def _normalize_batch(resp):
    data = _get_response_data(resp)
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, tuple):
        return list(data)
    if hasattr(data, "data"):
        inner = getattr(data, "data")
        if isinstance(inner, list):
            return inner
    if isinstance(data, dict):
        return []
    try:
        return list(data)
    except TypeError:
        return []


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _obs_item_get(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _extract_turn_number(name, role_prefixes):
    if not name:
        return None
    escaped = "|".join(re.escape(prefix) for prefix in role_prefixes)
    m = re.match(rf"^(?:{escaped})-(\d+)$", name)
    return int(m.group(1)) if m else None


def _extract_embedding_turn_number(name):
    if not name:
        return None

    lower_name = name.lower()
    if "embedding" not in lower_name and "embed" not in lower_name:
        return None

    explicit = _extract_turn_number(
        name,
        [
            "rag-embedding-turn",
            "query-embedding-turn",
            "embedding-turn",
            "rag-retrieve-turn",
            "rag-retrieval-turn",
        ],
    )
    if explicit is not None:
        return explicit

    # Fallback for names like "...turn-3...embedding...".
    matches = re.findall(r"turn-(\d+)", lower_name)
    if matches:
        return int(matches[-1])
    return None


def _trace_turn_count_from_metadata(trace):
    metadata = getattr(trace, "metadata", None)
    if metadata is None:
        return None

    if isinstance(metadata, dict):
        raw = metadata.get("turn_count")
    else:
        raw = getattr(metadata, "turn_count", None)

    if raw is None:
        return None

    try:
        turn_count = int(raw)
    except (TypeError, ValueError):
        return None

    return turn_count if turn_count >= 0 else None


def get_generations_for_trace(trace_id, max_pages=MAX_PAGES):
    """Fetch generation observations and embedding spans for a trace."""
    observations = []
    last_exc = None

    # Primary path for current SDKs.
    endpoint = getattr(lf.api, "observations", None)
    if endpoint is not None and hasattr(endpoint, "get_many"):
        for obs_type in ("GENERATION", "generation", "SPAN", "span"):
            observations = []
            try:
                for page in range(1, max_pages + 1):
                    resp = endpoint.get_many(
                        limit=TRACE_FETCH_LIMIT,
                        page=page,
                        trace_id=trace_id,
                        type=obs_type,
                    )
                    batch = _normalize_batch(resp)
                    if not batch:
                        break
                    observations.extend(batch)
                    if len(batch) < TRACE_FETCH_LIMIT:
                        break
                if observations:
                    break
            except Exception as exc:
                last_exc = exc

    # Fallback path for observations_v_2 cursor pagination.
    if not observations:
        endpoint_v2 = getattr(lf.api, "observations_v_2", None)
        if endpoint_v2 is not None and hasattr(endpoint_v2, "get_many"):
            for obs_type in ("GENERATION", "generation", "SPAN", "span"):
                observations = []
                cursor = None
                try:
                    for _ in range(max_pages):
                        resp = endpoint_v2.get_many(
                            limit=TRACE_FETCH_LIMIT,
                            cursor=cursor,
                            trace_id=trace_id,
                            type=obs_type,
                        )
                        batch = _normalize_batch(resp)
                        if not batch:
                            break
                        observations.extend(batch)

                        meta = getattr(resp, "meta", None)
                        next_cursor = getattr(meta, "next_cursor", None) if meta is not None else None
                        if not next_cursor:
                            break
                        cursor = next_cursor
                    if observations:
                        break
                except Exception as exc:
                    last_exc = exc

    if not observations and last_exc is not None:
        print(f"  [warn] Could not fetch generations for trace {trace_id}: {last_exc}")

    generations = []
    for item in observations:
        item_type = str(_obs_item_get(item, "type", "")).lower()
        name = (_obs_item_get(item, "name", "") or "").lower()
        is_embedding_span = item_type == "span" and ("embedding" in name or "embed" in name)
        if item_type and item_type != "generation" and not is_embedding_span:
            continue
        generations.append(item)

    return sorted(
        generations,
        key=lambda x: (
            _obs_item_get(x, "start_time", None)
            or _obs_item_get(x, "startTime", None)
            or _obs_item_get(x, "created_at", None)
            or _obs_item_get(x, "createdAt", None)
            or ""
        ),
    )


def counselor_latency_stats_for_trace(trace):
    """Return (latencies_per_exchange, inferred_exchange_count, inferred_conversation_turn_count)."""
    generations = get_generations_for_trace(getattr(trace, "id", None))
    if not generations:
        return [], 0, 0

    patient_turns = set()
    counselor_turns = set()
    processed_observation_ids = set()
    patient_latency_by_turn = defaultdict(float)
    counselor_latency_by_turn = defaultdict(float)
    turn_indexes_with_numeric_latency = set()

    for gen in generations:
        observation_id = _obs_item_get(gen, "id", None)
        if observation_id is not None:
            if observation_id in processed_observation_ids:
                continue
            processed_observation_ids.add(observation_id)

        name = _obs_item_get(gen, "name", "") or ""
        latency = _to_float(_obs_item_get(gen, "latency", None))

        patient_turn = _extract_turn_number(name, ["patient-turn"])
        if patient_turn is not None:
            patient_turns.add(patient_turn)
            if latency is not None:
                patient_latency_by_turn[patient_turn] += latency
                turn_indexes_with_numeric_latency.add(patient_turn)
            continue

        embedding_turn = _extract_embedding_turn_number(name)
        if embedding_turn is not None:
            counselor_turns.add(embedding_turn)
            if latency is not None:
                # RAG embedding calls contribute to the exchange latency budget.
                counselor_latency_by_turn[embedding_turn] += latency
                turn_indexes_with_numeric_latency.add(embedding_turn)
            continue

        counselor_turn = _extract_turn_number(name, ["counselor-turn", "counselor-editor-turn"])
        if counselor_turn is not None:
            counselor_turns.add(counselor_turn)
            if latency is not None:
                counselor_latency_by_turn[counselor_turn] += latency
                turn_indexes_with_numeric_latency.add(counselor_turn)

    inferred_conversation_turn_count = 0
    if patient_turns or counselor_turns:
        inferred_conversation_turn_count = len(patient_turns) + len(counselor_turns)

    ordered_turns = sorted(patient_turns | counselor_turns)
    exchange_latencies = [
        patient_latency_by_turn.get(turn_index, 0.0) + counselor_latency_by_turn.get(turn_index, 0.0)
        for turn_index in ordered_turns
        if turn_index in turn_indexes_with_numeric_latency
    ]
    inferred_exchange_count = len(ordered_turns)
    return exchange_latencies, inferred_exchange_count, inferred_conversation_turn_count

def get_all_traces_for_run(dataset_name, run_name, max_pages=MAX_PAGES):
    tag = f"{dataset_name}/{run_name}"
    all_traces = []
    for page in range(1, max_pages+1):
        resp = lf.api.trace.list(limit=TRACE_FETCH_LIMIT, page=page, tags=tag)
        batch = _normalize_batch(resp)
        if not batch:
            break
        all_traces.extend(batch)
        if len(batch) < TRACE_FETCH_LIMIT:
            break
    # Only keep traces named 'generated-conversation'
    gen_traces = [t for t in all_traces if getattr(t, 'name', None) == 'generated-conversation']
    sorted_traces = sorted(gen_traces, key=lambda x: x.tags[0] if x.tags else "")
    return sorted_traces

if RUN_NAMES is None:
    # Auto-discover run names from all traces with dataset_name as prefix
    all_traces = []
    for page in range(1, MAX_PAGES+1):
        resp = lf.api.trace.list(limit=TRACE_FETCH_LIMIT, page=page)
        batch = _normalize_batch(resp)
        if not batch:
            break
        all_traces.extend(batch)
        if len(batch) < TRACE_FETCH_LIMIT:
            break
    run_names = sorted(set(
        tag.split("/", 1)[1]
        for t in all_traces for tag in (t.tags or [])
        if tag.startswith(DATASET_NAME + "/")
    ))
else:
    run_names = RUN_NAMES

print(f"Found runs: {run_names}\n")

for run_name in run_names:
    traces = get_all_traces_for_run(DATASET_NAME, run_name)
    exchange_latencies = []
    counselor_turn_counts = []
    conversation_latencies = []
    conversation_turn_counts = []
    for trace in traces:
        (
            per_exchange_latencies,
            inferred_exchange_count,
            inferred_conversation_turn_count,
        ) = counselor_latency_stats_for_trace(trace)
        exchange_latencies.extend(per_exchange_latencies)

        metadata_turn_count = _trace_turn_count_from_metadata(trace)
        if metadata_turn_count is not None:
            counselor_turn_count = metadata_turn_count
            # In generation code, turn_count is exchange count (patient+counselor pair).
            conversation_turn_count = metadata_turn_count * 2
        else:
            counselor_turn_count = inferred_exchange_count
            conversation_turn_count = inferred_conversation_turn_count

        counselor_turn_counts.append(counselor_turn_count)
        conversation_turn_counts.append(conversation_turn_count)

        total_latency = _to_float(getattr(trace, "latency", None))
        if total_latency is not None:
            conversation_latencies.append(total_latency)

    n = len(traces)
    avg_exchange_latency = (
        sum(exchange_latencies) / len(exchange_latencies)
        if exchange_latencies
        else 0
    )
    avg_counselor_msgs = (
        sum(counselor_turn_counts) / len(counselor_turn_counts)
        if counselor_turn_counts
        else 0
    )
    avg_conversation_latency = (
        sum(conversation_latencies) / len(conversation_latencies)
        if conversation_latencies
        else 0
    )
    avg_conversation_turns = (
        sum(conversation_turn_counts) / len(conversation_turn_counts)
        if conversation_turn_counts
        else 0
    )
    print(f"Run: {run_name}")
    print(f"  Generated conversations: {n}")
    print(f"  Avg exchange latency (patient+counselor+rag-embedding) (s): {avg_exchange_latency:.2f}")
    print(f"  Avg counselor turns/conversation: {avg_counselor_msgs:.2f}\n")
    print(f"  Avg total conversation latency (s): {avg_conversation_latency:.2f}")
    print(f"  Avg conversation turns: {avg_conversation_turns:.2f}\n")
