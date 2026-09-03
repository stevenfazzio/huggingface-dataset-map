"""Regression test for stage-04 aggregation against malformed model output.

A single card returning `"is_benchmark": true` — a bare scalar where the
{value, quote} object belongs — crashed the aggregation of an otherwise
complete 5,000-card run. The API results were already durable, but the stage
exited non-zero and wrote no parquet.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

spec = importlib.util.spec_from_file_location("extract_structured", "pipeline/04_extract_structured.py")
extract = importlib.util.module_from_spec(spec)
sys.modules["extract_structured"] = extract
spec.loader.exec_module(extract)


def _cache_entry(repo_id, payload):
    return {"repo_id": repo_id, "raw_text": json.dumps(payload), "error": None}


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Point the module's cache dir and output parquet at a temp dir."""
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(extract, "STRUCTURED_FIELDS_CACHE_DIR", cache)
    monkeypatch.setattr(extract, "STRUCTURED_FIELDS_PARQUET", tmp_path / "out.parquet")
    return cache


TAXONOMY = {
    "subject_domain": {"type": "single-select", "categories": [{"name": "code-and-software", "description": "x"}]},
    "is_benchmark": {"type": "boolean", "rule": "x"},
}
FIELDS = ["subject_domain", "is_benchmark"]


class TestAggregate:
    def test_bare_scalar_where_object_expected_does_not_crash(self, staged):
        (staged / "good.json").write_text(
            json.dumps(
                _cache_entry(
                    "org/good",
                    {
                        "subject_domain": {"value": "code-and-software", "quote": "a quote"},
                        "is_benchmark": {"value": True, "quote": "a quote"},
                    },
                )
            )
        )
        # The shape that crashed the real run.
        (staged / "bad.json").write_text(
            json.dumps(
                _cache_entry(
                    "org/bad",
                    {
                        "subject_domain": {"value": "code-and-software", "quote": "a quote"},
                        "is_benchmark": True,
                    },
                )
            )
        )

        df = extract.aggregate(TAXONOMY, FIELDS)

        assert len(df) == 2, "one malformed card must not drop the other rows"
        bad = df[df.repo_id == "org/bad"].iloc[0]
        assert bad["is_benchmark"] is None, "the unusable value is dropped, not guessed at"
        assert "is_benchmark:missing" in bad["validation_issues"], "and it is recorded, not hidden"
        good = df[df.repo_id == "org/good"].iloc[0]
        assert bool(good["is_benchmark"]) is True
        assert good["validation_issues"] is None

    def test_false_scalar_also_survives(self, staged):
        """`False or {}` already coerced; keep it that way so both branches are covered."""
        (staged / "f.json").write_text(
            json.dumps(
                _cache_entry(
                    "org/f",
                    {
                        "subject_domain": {"value": "code-and-software", "quote": "q"},
                        "is_benchmark": False,
                    },
                )
            )
        )

        df = extract.aggregate(TAXONOMY, FIELDS)

        assert len(df) == 1
        assert df.iloc[0]["is_benchmark"] is None
        assert pd.notna(df.iloc[0]["validation_issues"])
