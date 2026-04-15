"""Tests for pure functions in 00_fetch_datasets.py."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

spec = importlib.util.spec_from_file_location("fetch_datasets", "pipeline/00_fetch_datasets.py")
fetch_datasets = importlib.util.module_from_spec(spec)
sys.modules["fetch_datasets"] = fetch_datasets
spec.loader.exec_module(fetch_datasets)

_extract_tag_values = fetch_datasets._extract_tag_values
_parse_dataset_info = fetch_datasets._parse_dataset_info
_strip_yaml_frontmatter = fetch_datasets._strip_yaml_frontmatter


class TestExtractTagValues:
    def test_extracts_matching_prefix(self):
        tags = ["task_categories:text-classification", "language:en", "language:fr"]
        assert _extract_tag_values(tags, "language:") == ["en", "fr"]

    def test_empty_when_none(self):
        assert _extract_tag_values(None, "x:") == []
        assert _extract_tag_values([], "x:") == []

    def test_no_match(self):
        assert _extract_tag_values(["a:b", "c:d"], "z:") == []


class TestStripYamlFrontmatter:
    def test_strips_leading_block(self):
        text = "---\nlicense: mit\n---\n\n# Title\nbody"
        assert _strip_yaml_frontmatter(text) == "# Title\nbody"

    def test_leaves_untouched_when_no_frontmatter(self):
        assert _strip_yaml_frontmatter("# Title\nbody") == "# Title\nbody"

    def test_handles_unterminated_frontmatter(self):
        text = "---\nbroken"
        assert _strip_yaml_frontmatter(text) == text


class TestParseDatasetInfo:
    def _make_info(self, **overrides):
        defaults = dict(
            id="org/dataset",
            author="org",
            likes=1234,
            downloads=50000,
            last_modified=None,
            created_at=None,
            private=False,
            disabled=False,
            gated=False,
            tags=[
                "task_categories:text-classification",
                "task_ids:sentiment-classification",
                "language:en",
                "modality:text",
                "size_categories:10K<n<100K",
                "license:mit",
            ],
            card_data={"pretty_name": "My Dataset"},
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_parses_from_tags_when_card_missing(self):
        info = self._make_info(card_data={})
        row = _parse_dataset_info(info)
        assert row["repo_id"] == "org/dataset"
        assert row["likes"] == 1234
        assert row["downloads"] == 50000
        assert row["task_categories"] == "text-classification"
        assert row["task_ids"] == "sentiment-classification"
        assert row["languages"] == "en"
        assert row["modalities"] == "text"
        assert row["size_categories"] == "10K<n<100K"
        assert row["license"] == "mit"

    def test_prefers_card_data_over_tags(self):
        info = self._make_info(card_data={"task_categories": ["translation"], "language": ["de"]})
        row = _parse_dataset_info(info)
        assert row["task_categories"] == "translation"
        assert row["languages"] == "de"

    def test_handles_missing_everything(self):
        info = SimpleNamespace(id="x/y", tags=None, card_data=None)
        row = _parse_dataset_info(info)
        assert row["repo_id"] == "x/y"
        assert row["likes"] == 0
        assert row["downloads"] == 0
        assert row["task_categories"] == ""
        assert row["license"] == ""
