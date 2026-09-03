"""Message Batches driver shared by the per-card LLM stages (04, 04b).

Batch requests bill at 50% of live rates, which is the whole reason to use them
for a 5,000-card corpus. The price of that discount is asynchrony: a batch may
take up to 24 hours, so a run has to survive being interrupted while it waits.

Submitted batch ids are therefore journaled to disk *before* polling begins. An
interrupted run drains the in-flight batches on its next invocation instead of
resubmitting them and paying for the same work twice. Chunking exists for the
same reason: a batch that errors out costs one chunk, not the whole corpus.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from anthropic.types.messages.batch_create_params import Request as BatchRequest
from config import BATCH_CHUNK_SIZE, BATCH_POLL_SECONDS


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))
        os.replace(tmp_path, path)
    except BaseException:  # a Ctrl-C mid-write should still clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _load_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())["batches"]
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"  (unreadable batch journal at {path}; ignoring it)")
        return []


def _save_journal(path: Path, entries: list[dict]) -> None:
    if entries:
        _write_json_atomic(path, {"batches": entries})
    elif path.exists():
        path.unlink()


def _submit(
    client,
    items: list[tuple[str, Any]],
    build_params: Callable[[str, Any], Any],
    journal: list[dict],
    journal_path: Path,
    label: str,
) -> None:
    """Create one batch per chunk, journaling each id as soon as it exists."""
    for start in range(0, len(items), BATCH_CHUNK_SIZE):
        chunk = items[start : start + BATCH_CHUNK_SIZE]
        # Built in one pass: a custom_id that drifts from the key it was built for
        # would silently attach one dataset's answer to another dataset's row.
        id_map: dict[str, str] = {}
        requests: list[BatchRequest] = []
        for offset, (key, payload) in enumerate(chunk):
            custom_id = f"i{start + offset}"
            id_map[custom_id] = key
            requests.append(BatchRequest(custom_id=custom_id, params=build_params(key, payload)))
        batch = client.messages.batches.create(requests=requests)
        journal.append({"id": batch.id, "map": id_map})
        _save_journal(journal_path, journal)
        print(f"  {label}: submitted {batch.id} ({len(requests)} requests)")


def _poll_until_ended(client, batch_id: str, label: str):
    started = time.time()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return batch
        c = batch.request_counts
        elapsed = (time.time() - started) / 60
        print(
            f"  {label}: {batch_id} {c.succeeded} ok / {c.errored} err / "
            f"{c.processing} processing ({elapsed:.1f}m elapsed)"
        )
        time.sleep(BATCH_POLL_SECONDS)


def run_batches(
    client,
    items: list[tuple[str, Any]],
    build_params: Callable[[str, Any], Any],
    journal_path: Path,
    label: str = "batch",
) -> Iterator[tuple[str, Any, Any, str]]:
    """Run `items` through the Batches API, yielding (key, payload, message, status).

    `items` is a list of (key, payload) pairs and `build_params(key, payload)`
    returns the `MessageCreateParamsNonStreaming` for one request. `message` is
    None unless `status` is "succeeded"; the other statuses ("errored",
    "canceled", "expired") are the caller's to handle.

    Results for keys absent from `items` are skipped rather than yielded: that is
    what a re-drain after an interrupted run looks like, and those keys have
    already been persisted by the run that was interrupted.
    """
    payloads = dict(items)
    journal = _load_journal(journal_path)

    journaled = {key for entry in journal for key in entry["map"].values()}
    if journal:
        print(f"  {label}: resuming {len(journal)} in-flight batch(es), {len(journaled)} requests")

    remaining = [(key, payload) for key, payload in items if key not in journaled]
    if remaining:
        _submit(client, remaining, build_params, journal, journal_path, label)

    for entry in list(journal):
        _poll_until_ended(client, entry["id"], label)
        for result in client.messages.batches.results(entry["id"]):
            key = entry["map"].get(result.custom_id)
            if key is None or key not in payloads:
                continue
            status = result.result.type
            message = result.result.message if status == "succeeded" else None
            yield key, payloads[key], message, status
        # Only now is the batch fully consumed and its output durable.
        journal.remove(entry)
        _save_journal(journal_path, journal)
