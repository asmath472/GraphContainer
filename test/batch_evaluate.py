"""Evaluate downloaded LLM-as-a-Judge batch outputs and show win rates."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "batch_jobs" / "judge_ledger.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "output" / "judge_results"

CRITERIA = ("Comprehensiveness", "Diversity", "Empowerment", "Overall Winner")


def _safe_path(value: object) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def _extract_content(record: dict) -> Optional[str]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None

    body = response.get("body")
    if not isinstance(body, dict):
        return None

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    return content.strip() if isinstance(content, str) else None


def _strip_fence(text: str) -> str:
    if text.strip().startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _parse_judge_payload(text: str) -> Optional[dict]:
    try:
        data = json.loads(_strip_fence(text))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_winner(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return 1
        if int(value) == 2:
            return 2
        return None

    raw = str(value).strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    if "answer 1" in raw or raw in {"1", "a", "answer1"}:
        return 1
    if "answer 2" in raw or raw in {"2", "b", "answer2"}:
        return 2
    if "tie" in raw or "draw" in raw or raw in {"both", "tie."}:
        return 0
    return None


def _winner_from(payload: str, criterion: str) -> Optional[int]:
    parsed = _parse_judge_payload(payload)
    if parsed:
        block = parsed.get(criterion)
        if isinstance(block, dict):
            return _normalize_winner(block.get("Winner"))

        for key, value in parsed.items():
            if isinstance(key, str) and key.lower() == criterion.lower() and isinstance(value, dict):
                return _normalize_winner(value.get("Winner"))

    pattern = re.compile(
        rf'"{re.escape(criterion)}"\s*:\s*\{{[^}}]*"Winner"\s*:\s*"([^"]+)"',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(payload)
    if not match:
        return None
    return _normalize_winner(match.group(1))


def _normalize_direction(value: object) -> str:
    if not isinstance(value, str):
        return "forward"
    raw = value.strip().lower()
    if raw in {"reverse", "rev", "backward", "back"}:
        return "reverse"
    return "forward"


def _apply_direction(winner: Optional[int], direction: str) -> Optional[int]:
    if direction != "reverse" or winner is None:
        return winner
    if winner == 1:
        return 2
    if winner == 2:
        return 1
    return winner


@dataclass
class Counter:
    answer1: int = 0
    answer2: int = 0
    tie: int = 0
    invalid: int = 0

    def add(self, winner: Optional[int]) -> None:
        if winner == 1:
            self.answer1 += 1
        elif winner == 2:
            self.answer2 += 1
        elif winner == 0:
            self.tie += 1
        else:
            self.invalid += 1

    def total(self) -> int:
        return self.answer1 + self.answer2 + self.tie + self.invalid

    def valid(self) -> int:
        return self.answer1 + self.answer2 + self.tie


@dataclass
class PairReport:
    pair_id: str
    model_a: str
    model_b: str
    metrics: Dict[str, Counter] = field(default_factory=dict)


def _pair_model_names(pair_id: str, metadata: dict) -> tuple[str, str]:
    requests = metadata.get("requests")
    if isinstance(requests, dict) and requests:
        first = next(iter(requests.values()))
        if isinstance(first, dict):
            a = first.get("model_a")
            b = first.get("model_b")
            if isinstance(a, str) and isinstance(b, str):
                return a, b

    if "_vs_" in pair_id:
        return tuple(pair_id.split("_vs_", 1))  # type: ignore[return-value]
    return "answer_1", "answer_2"


def evaluate_one_file(path: Path, metadata: dict) -> PairReport:
    pair_id = path.stem.replace("_results", "")
    model_a, model_b = _pair_model_names(pair_id, metadata)
    report = PairReport(pair_id=pair_id, model_a=model_a, model_b=model_b)
    for c in CRITERIA:
        report.metrics[c] = Counter()
    request_metadata = metadata.get("requests")
    if not isinstance(request_metadata, dict):
        request_metadata = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            content = _extract_content(record)
            custom_id = record.get("custom_id")
            if isinstance(custom_id, str):
                direction_info = request_metadata.get(custom_id)
                direction = _normalize_direction(
                    direction_info.get("direction") if isinstance(direction_info, dict) else None
                )
            else:
                direction = "forward"
            if content is None:
                for c in CRITERIA:
                    report.metrics[c].invalid += 1
                continue

            for c in CRITERIA:
                raw_winner = _winner_from(content, c)
                report.metrics[c].add(_apply_direction(raw_winner, direction))

    return report


def _percent(n: int, d: int) -> float:
    if d <= 0:
        return 0.0
    return round((n / d) * 100, 2)


def print_report(reports: List[PairReport]) -> None:
    overall: Dict[str, Counter] = {c: Counter() for c in CRITERIA}

    for report in reports:
        print(f"\n=== {report.pair_id} ===")
        print(f"Models: {report.model_a} (Answer 1) vs {report.model_b} (Answer 2)")
        for c in CRITERIA:
            cnt = report.metrics[c]
            print(f"\n[{c}]")
            print(f"  total={cnt.total()} valid={cnt.valid()} tie={cnt.tie} invalid={cnt.invalid}")
            print(
                f"  Answer 1: {_percent(cnt.answer1, cnt.valid())}% ({cnt.answer1}) | "
                f"Answer 2: {_percent(cnt.answer2, cnt.valid())}% ({cnt.answer2}) | "
                f"Tie: {_percent(cnt.tie, cnt.valid())}% ({cnt.tie})"
            )
            overall[c].answer1 += cnt.answer1
            overall[c].answer2 += cnt.answer2
            overall[c].tie += cnt.tie
            overall[c].invalid += cnt.invalid

    print("\n=== Overall ===")
    for c in CRITERIA:
        cnt = overall[c]
        print(f"\n[{c}]")
        print(f"  total={cnt.total()} valid={cnt.valid()} tie={cnt.tie} invalid={cnt.invalid}")
        print(
            f"  Answer 1: {_percent(cnt.answer1, cnt.valid())}% ({cnt.answer1}) | "
            f"Answer 2: {_percent(cnt.answer2, cnt.valid())}% ({cnt.answer2}) | "
            f"Tie: {_percent(cnt.tie, cnt.valid())}% ({cnt.tie})"
        )


def collect_result_files(ledger: dict, results_dir: Path, explicit_files: List[str]) -> List[tuple[Path, dict]]:
    metadata_by_path: dict[str, dict] = {}
    for info in ledger.values():
        if not isinstance(info, dict):
            continue
        result_path = _safe_path(info.get("final_output_path"))
        metadata_path = _safe_path(info.get("metadata_path"))
        if result_path is None:
            continue
        if metadata_path is None:
            metadata_by_path[str(result_path)] = {}
        else:
            metadata_by_path[str(result_path)] = _load_json(metadata_path)

    if explicit_files:
        files: List[Path] = []
        for path_str in explicit_files:
            p = _safe_path(path_str)
            if p is not None:
                files.append(p)
    else:
        if not results_dir.exists():
            return []
        files = sorted(results_dir.glob("*_results.jsonl"))

    outputs = []
    for p in files:
        if not p.exists():
            print(f"[skip] missing: {p}")
            continue
        outputs.append((p, metadata_by_path.get(str(p), {})))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--result_file", action="append", default=[])
    args = parser.parse_args()

    ledger = _load_json(Path(args.ledger))
    results = collect_result_files(ledger, Path(args.results_dir), args.result_file)
    if not results:
        print("No judge result file found.")
        return

    reports = [evaluate_one_file(path, metadata) for path, metadata in results]
    print_report(reports)


if __name__ == "__main__":
    main()
