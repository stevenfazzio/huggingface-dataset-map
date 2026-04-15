"""Tests for config.py constants and paths."""

from pathlib import Path

from pipeline.config import (
    ANTHROPIC_API_KEY,
    CARD_MAX_CHARS,
    CO_API_KEY,
    COHERE_BATCH_SIZE,
    COHERE_EMBED_DIMENSION,
    DATA_DIR,
    DATASETS_PARQUET,
    EMBEDDINGS_NPZ,
    FETCH_OVERSHOOT_COUNT,
    HF_TOKEN,
    LABELS_PARQUET,
    MAP_HTML,
    MIN_CARD_CHARS,
    RANK_SORT_KEY,
    TARGET_DATASET_COUNT,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)


def test_data_dir_is_path():
    assert isinstance(DATA_DIR, Path)


def test_all_data_paths_under_data_dir():
    for p in (
        DATASETS_PARQUET,
        EMBEDDINGS_NPZ,
        UMAP_COORDS_NPZ,
        TOPONYMY_MODEL_JOBLIB,
        LABELS_PARQUET,
        MAP_HTML,
    ):
        assert isinstance(p, Path)
        assert p.parts[0] == "data", f"{p} is not under data/"


def test_constants_have_expected_values():
    assert TARGET_DATASET_COUNT == 1_000
    assert FETCH_OVERSHOOT_COUNT == 1_200
    assert COHERE_BATCH_SIZE == 96
    assert COHERE_EMBED_DIMENSION == 512
    assert CARD_MAX_CHARS == 4_000
    assert MIN_CARD_CHARS == 200
    assert UMAP_N_NEIGHBORS == 15
    assert UMAP_MIN_DIST == 0.05
    assert UMAP_RANDOM_STATE == 42
    assert RANK_SORT_KEY in ("likes", "downloads")


def test_api_keys_are_strings():
    for key in (HF_TOKEN, CO_API_KEY, ANTHROPIC_API_KEY):
        assert isinstance(key, str)
