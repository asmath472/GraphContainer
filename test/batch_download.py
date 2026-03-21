"""Download completed OpenAI batch judge outputs from the ledger."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "batch_jobs" / "judge_ledger.json"


@dataclass
class LedgerEntry:
    batch_id: str
    payload: Dict[str, object]


def load_ledger(path: Path) -> Dict[str, LedgerEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected ledger format: expected object at {path}")

    entries: Dict[str, LedgerEntry] = {}
    for batch_id, payload in raw.items():
        if isinstance(batch_id, str) and isinstance(payload, dict):
            entries[batch_id] = LedgerEntry(batch_id=batch_id, payload=payload)
    return entries


def as_dict(obj):
    return obj if isinstance(obj, dict) else getattr(obj, "__dict__", {})


def batch_status(client: OpenAI, batch_id: str) -> str:
    return str(as_dict(client.batches.retrieve(batch_id)).get("status", "unknown"))


def output_file_id(client: OpenAI, batch_id: str) -> str | None:
    raw = as_dict(client.batches.retrieve(batch_id)).get("output_file_id")
    return raw if isinstance(raw, str) and raw else None


def download_batch_output(
    client: OpenAI,
    batch_id: str,
    final_output_path: Path,
    force: bool,
) -> bool:
    status = batch_status(client, batch_id)
    if status != "completed":
        print(f"[{batch_id}] status={status}, skip")
        return False

    out_file_id = output_file_id(client, batch_id)
    if not out_file_id:
        print(f"[{batch_id}] completed but no output_file_id")
        return False

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if final_output_path.exists() and not force:
        print(f"[{batch_id}] output already exists: {final_output_path}")
        return False

    response = client.files.content(out_file_id)
    content = getattr(response, "text", None)
    if content is None:
        raw = getattr(response, "content", b"")
        content = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    final_output_path.write_text(content, encoding="utf-8")
    print(f"[{batch_id}] downloaded -> {final_output_path}")
    return True


def collect_completed(
    client: OpenAI,
    entries: List[LedgerEntry],
    poll_interval_sec: int,
    timeout_sec: int,
) -> List[LedgerEntry]:
    if not entries:
        return []

    deadline = time.time() + timeout_sec
    pending = list(entries)
    completed: List[LedgerEntry] = []

    print(
        f"Wait mode: {len(pending)} jobs pending (timeout={timeout_sec}s, poll={poll_interval_sec}s)"
    )

    while pending and time.time() < deadline:
        next_pending: List[LedgerEntry] = []
        for entry in pending:
            status = batch_status(client, entry.batch_id)
            if status == "completed":
                completed.append(entry)
            else:
                next_pending.append(entry)
        pending = next_pending
        if not pending:
            break
        time.sleep(poll_interval_sec)

    # final sweep for anything that finished at the end of timeout window
    if pending:
        still_waiting: List[LedgerEntry] = []
        for entry in pending:
            if batch_status(client, entry.batch_id) == "completed":
                completed.append(entry)
            else:
                still_waiting.append(entry)
        pending = still_waiting

    if pending:
        print(f"[warn] {len(pending)} batches still not completed")

    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--batch_id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--wait_timeout", type=int, default=3600)
    args = parser.parse_args()

    ledger = load_ledger(Path(args.ledger))
    if not ledger:
        print("No entries in ledger.")
        return

    entries = list(ledger.values())
    if args.batch_id:
        targets = set(args.batch_id)
        entries = [entry for entry in entries if entry.batch_id in targets]
        if not entries:
            raise RuntimeError("No matching batch_id found in ledger.")

    client = OpenAI()
    ready: List[LedgerEntry] = []
    for entry in entries:
        if batch_status(client, entry.batch_id) == "completed":
            ready.append(entry)

    if args.wait:
        pending = [entry for entry in entries if entry.batch_id not in {e.batch_id for e in ready}]
        if pending:
            ready.extend(collect_completed(client, pending, args.poll_interval, args.wait_timeout))

    downloaded = 0
    skipped = 0
    seen = set()
    unique_ready = []
    for item in ready:
        if item.batch_id in seen:
            continue
        seen.add(item.batch_id)
        unique_ready.append(item)

    for entry in unique_ready:
        final_path_raw = entry.payload.get("final_output_path")
        if not isinstance(final_path_raw, str) or not final_path_raw.strip():
            skipped += 1
            print(f"[{entry.batch_id}] missing final_output_path, skip")
            continue

        output_path = Path(final_path_raw)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()

        if download_batch_output(client, entry.batch_id, output_path, force=args.force):
            downloaded += 1
        else:
            skipped += 1

    print(f"Done. downloaded={downloaded}, skipped={skipped}")


if __name__ == "__main__":
    main()
