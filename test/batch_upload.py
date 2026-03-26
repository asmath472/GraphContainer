"""Batch upload helper for six RAG outputs files in ``./outputs``.

Creates query-wise LLM-as-a-Judge batch jobs by comparing every pair of result
files and generating OpenAI batch payloads (and optionally submits them).  For each
query in a pair, both forward and reverse directions are judged.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = PROJECT_ROOT / "batch_jobs" / "judge_ledger.json"
PAIR_BATCH_DIR = PROJECT_ROOT / "batch_jobs" / "judge"
PAIR_METADATA_DIR = PROJECT_ROOT / "batch_jobs" / "judge_metadata"
OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "outputs"


def update_ledger(batch_id: str, info: dict) -> None:
    """Persist batch job metadata in a simple ledger JSON."""
    try:
        with LEDGER_PATH.open("r", encoding="utf-8") as f:
            ledger = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {}

    ledger[batch_id] = info
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def normalize_output(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def parse_answer(record: dict) -> str:
    if "outputs" in record and record["outputs"] is not None:
        text = normalize_output(record["outputs"])
        if text:
            return text

    # Fallback: pick first non-empty string-like value.
    for value in record.values():
        text = normalize_output(value)
        if text:
            return text
    return ""


def load_output_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue

            record = json.loads(raw)
            if not isinstance(record, dict):
                continue

            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                continue

            data[query.strip()] = parse_answer(record)
    return data


def discover_outputs_files(outputs_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in sorted(outputs_dir.glob("*.jsonl")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) >= 2:
            candidates.append(path)

    if len(candidates) < 2:
        raise FileNotFoundError(f"Need at least 2 jsonl files in {outputs_dir}")
    return candidates


def build_judge_prompt(query: str, answer_a: str, answer_b: str) -> Tuple[str, str]:
    system_prompt = """---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness, Diversity, and Empowerment**.

---Goal---
Evaluate two answers based on:
- **Comprehensiveness**: Detail covering all aspects.
- **Diversity**: Variety of perspectives.
- **Empowerment**: Helping the reader make informed judgments.

For each criterion, choose the better answer (Answer 1 or Answer 2) and explain why. Then, select an overall winner."""
    user_prompt = f"""
Here is the question: {query}

Here are the two answers:
**Answer 1**: {answer_a}
**Answer 2**: {answer_b}

Evaluate both answers. Output your evaluation in the following **JSON format**:
{{
  "Comprehensiveness": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "..." }},
  "Diversity": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "..." }},
  "Empowerment": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "..." }},
  "Overall Winner": {{ "Winner": "[Answer 1 or Answer 2]", "Explanation": "..." }}
}}
"""
    return system_prompt, user_prompt


def build_pair_id(left: Path, right: Path) -> str:
    if left.as_posix() < right.as_posix():
        first, second = left.stem, right.stem
    else:
        first, second = right.stem, left.stem
    return f"{first}_vs_{second}"


def create_batch_input(
    left_path: Path,
    right_path: Path,
    left_data: Dict[str, str],
    right_data: Dict[str, str],
    model: str,
    max_queries: int,
) -> Tuple[Path, Path, Path, int]:
    pair_id = build_pair_id(left_path, right_path)
    primary = left_data if len(left_data) <= len(right_data) else right_data
    secondary = right_data if primary is left_data else left_data
    pair_queries = []

    for query in primary:
        if query in secondary:
            pair_queries.append(query)
            if max_queries > 0 and len(pair_queries) >= max_queries:
                break

    timestamp = int(time.time())
    batch_dir = PAIR_BATCH_DIR
    metadata_dir = PAIR_METADATA_DIR
    batch_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    batch_input_path = batch_dir / f"judge_input_{pair_id}_{timestamp}.jsonl"
    metadata_path = metadata_dir / f"judge_metadata_{pair_id}_{timestamp}.jsonl"
    final_outputs_path = PROJECT_ROOT / "outputs" / "judge_results" / f"{pair_id}_results.jsonl"

    request_metadata: Dict[str, Dict[str, str]] = {}
    with batch_input_path.open("w", encoding="utf-8") as batch_file:
        for idx, query in enumerate(pair_queries):
            for direction, a_data, b_data, a_file, b_file in (
                ("forward", left_data[query], right_data[query], left_path, right_path),
                ("reverse", right_data[query], left_data[query], right_path, left_path),
            ):
                direction_suffix = direction[:3]
                system_prompt, user_prompt = build_judge_prompt(query, a_data, b_data)
                custom_id = f"judge-{idx:06d}-{direction_suffix}"
                request_metadata[custom_id] = {
                    "query": query,
                    "direction": direction,
                    "answer_a_file": a_file.name,
                    "answer_b_file": b_file.name,
                    "model_a": a_file.stem,
                    "model_b": b_file.stem,
                }
                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                    },
                }
                batch_file.write(json.dumps(request, ensure_ascii=False) + "\n")

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {"pair_id": pair_id, "requests": request_metadata},
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    return batch_input_path, metadata_path, final_outputs_path, len(pair_queries) * 2


def submit_batch(
    batch_input_path: Path,
    metadata_path: Path,
    final_outputs_path: Path,
    pair_id: str,
    model: str,
) -> str:
    from openai import OpenAI

    client = OpenAI()
    with batch_input_path.open("rb") as batch_file:
        file_obj = client.files.create(file=batch_file, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"judge_{pair_id}", "model": model},
    )

    update_ledger(
        batch_job.id,
        {
            "status": "submitted",
            "pair_id": pair_id,
            "metadata_path": str(metadata_path),
            "final_outputs_path": str(final_outputs_path),
            "batch_id": batch_job.id,
            "created_at": int(time.time()),
        },
    )
    return batch_job.id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max_queries", type=int, default=-1)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Upload and create OpenAI batch jobs; default is only JSONL generation.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = (PROJECT_ROOT / outputs_dir).resolve()
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs dir not found: {outputs_dir}")

        outputs_files = discover_outputs_files(outputs_dir)
    results: List[Tuple[Path, Dict[str, str]]] = [
        (path, load_output_file(path)) for path in outputs_files
    ]

    if not results:
        raise RuntimeError("No usable outputs data files found.")

    pair_count = 0
    total_requests = 0

    for left, right in combinations(results, 2):
        left_path, left_data = left
        right_path, right_data = right
        pair_id = build_pair_id(left_path, right_path)
        batch_input_path, metadata_path, final_outputs_path, n_queries = create_batch_input(
            left_path,
            right_path,
            left_data,
            right_data,
            args.model,
            args.max_queries,
        )
        pair_count += 1
        total_requests += n_queries

        query_count = n_queries // 2
        msg = f"[{pair_id}] common queries: {query_count}, batch requests: {n_queries}"
        if args.submit:
            batch_id = submit_batch(
        batch_input_path,
        metadata_path,
        final_outputs_path,
        pair_id,
        args.model,
        )
        msg += f", batch_id={batch_id}"
        else:
            msg += ", status=draft"

        msg += f"\n  batch_input: {batch_input_path}\n  metadata: {metadata_path}\n  expected_results: {final_outputs_path}"
        print(msg)

    print(f"\nGenerated {pair_count} pairwise batches with {total_requests} requests.")


if __name__ == "__main__":
    main()
