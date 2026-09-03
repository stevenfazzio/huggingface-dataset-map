"""Tests for the Message Batches driver (no network, no API keys).

The behaviour worth pinning down here is the journal: it is what stops an
interrupted run from resubmitting batches that are already in flight and paying
for the same 5,000 cards twice.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import batch_runner  # noqa: E402


def _message(text):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
    )


class FakeBatches:
    """Stand-in for client.messages.batches with scriptable per-request outcomes."""

    def __init__(self, statuses=None, on_retrieve=None):
        self.statuses = statuses or {}  # custom_id -> status, default "succeeded"
        self.on_retrieve = on_retrieve
        self.created = []  # [(batch_id, [custom_id, ...])]
        self.retrieved = []

    def create(self, requests):
        batch_id = f"batch_{len(self.created)}"
        self.created.append((batch_id, [r["custom_id"] for r in requests]))
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        self.retrieved.append(batch_id)
        if self.on_retrieve:
            self.on_retrieve(batch_id)
        return SimpleNamespace(id=batch_id, processing_status="ended", request_counts=None)

    def results(self, batch_id):
        custom_ids = dict(self.created).get(batch_id) or self._journaled_ids(batch_id)
        for custom_id in custom_ids:
            status = self.statuses.get(custom_id, "succeeded")
            message = _message(f"result for {custom_id}") if status == "succeeded" else None
            yield SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type=status, message=message))

    def _journaled_ids(self, batch_id):
        return self.resumable.get(batch_id, [])

    resumable: dict = {}


def make_client(**kwargs):
    batches = FakeBatches(**kwargs)
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


def params(key, payload):
    return {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": payload}]}


@pytest.fixture(autouse=True)
def _fast_small_batches(monkeypatch):
    monkeypatch.setattr(batch_runner, "BATCH_CHUNK_SIZE", 2)
    monkeypatch.setattr(batch_runner, "BATCH_POLL_SECONDS", 0)


class TestRunBatches:
    def test_splits_into_chunks_and_yields_every_item(self, tmp_path):
        client, batches = make_client()
        items = [(f"repo/{i}", f"card {i}") for i in range(5)]

        out = list(batch_runner.run_batches(client, items, params, tmp_path / "j.json"))

        assert [cids for _, cids in batches.created] == [["i0", "i1"], ["i2", "i3"], ["i4"]]
        assert [key for key, _, _, _ in out] == [k for k, _ in items]
        assert all(status == "succeeded" for *_, status in out)
        assert [payload for _, payload, _, _ in out] == [p for _, p in items]

    def test_each_request_carries_its_own_key_payload(self, tmp_path):
        """A custom_id that drifts from its key would cross-wire whole rows."""
        built = []
        client, batches = make_client()
        items = [(f"repo/{i}", f"card {i}") for i in range(5)]

        def recording_params(key, payload):
            built.append((key, payload))
            return params(key, payload)

        list(batch_runner.run_batches(client, items, recording_params, tmp_path / "j.json"))

        assert built == items
        submitted = [cid for _, cids in batches.created for cid in cids]
        assert submitted == [f"i{i}" for i in range(5)], "custom_ids must be unique across chunks"

    def test_journal_is_written_before_polling_and_removed_after(self, tmp_path):
        journal = tmp_path / "j.json"
        seen_during_poll = []

        client, _ = make_client(on_retrieve=lambda _: seen_during_poll.append(journal.exists()))
        list(batch_runner.run_batches(client, [("a", "card a")], params, journal))

        assert seen_during_poll == [True], "batch id must be durable before we wait on it"
        assert not journal.exists(), "a fully drained run leaves no journal behind"

    def test_failed_requests_yield_no_message(self, tmp_path):
        client, _ = make_client(statuses={"i1": "expired"})
        items = [("a", "card a"), ("b", "card b")]

        out = {
            key: (message, status)
            for key, _, message, status in batch_runner.run_batches(client, items, params, tmp_path / "j.json")
        }

        assert out["a"][1] == "succeeded" and out["a"][0] is not None
        assert out["b"] == (None, "expired")

    def test_resume_drains_in_flight_batches_without_resubmitting(self, tmp_path):
        journal = tmp_path / "j.json"
        journal.write_text(json.dumps({"batches": [{"id": "batch_old", "map": {"i0": "a", "i1": "b"}}]}))

        client, batches = make_client()
        FakeBatches.resumable = {"batch_old": ["i0", "i1"]}
        items = [("a", "card a"), ("b", "card b"), ("c", "card c")]

        out = list(batch_runner.run_batches(client, items, params, journal))
        FakeBatches.resumable = {}

        submitted = [cid for _, cids in batches.created for cid in cids]
        assert len(submitted) == 1, "only the un-journaled item should be resubmitted"
        assert {key for key, *_ in out} == {"a", "b", "c"}
        assert not journal.exists()

    def test_results_for_already_saved_keys_are_skipped(self, tmp_path):
        """Re-draining after an interrupted run must not re-yield finished work."""
        journal = tmp_path / "j.json"
        journal.write_text(json.dumps({"batches": [{"id": "batch_old", "map": {"i0": "done", "i1": "b"}}]}))

        client, _ = make_client()
        FakeBatches.resumable = {"batch_old": ["i0", "i1"]}
        out = list(batch_runner.run_batches(client, [("b", "card b")], params, journal))
        FakeBatches.resumable = {}

        assert [key for key, *_ in out] == ["b"]

    def test_unreadable_journal_is_ignored(self, tmp_path):
        journal = tmp_path / "j.json"
        journal.write_text("{ this is not json")

        client, batches = make_client()
        out = list(batch_runner.run_batches(client, [("a", "card a")], params, journal))

        assert len(batches.created) == 1
        assert [key for key, *_ in out] == ["a"]
