"""Shared paths, constants, and env var loading."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.home() / ".config" / "data-apis" / ".env")
load_dotenv(override=True)

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ── File paths ───────────────────────────────────────────────────────────────
DATASETS_PARQUET = DATA_DIR / "datasets.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
TOPONYMY_MODEL_JOBLIB = DATA_DIR / "toponymy_model.joblib"
LABELS_PARQUET = DATA_DIR / "labels.parquet"
STRUCTURED_FIELDS_PARQUET = DATA_DIR / "structured_fields.parquet"
STRUCTURED_FIELDS_CACHE_DIR = DATA_DIR / "structured_fields_cache"
STRUCTURED_FIELDS_BATCH_JOURNAL = DATA_DIR / "structured_fields_batches.json"
SUMMARIES_PARQUET = DATA_DIR / "summaries.parquet"
SUMMARIES_CACHE_DIR = DATA_DIR / "summaries_cache"
SUMMARIES_BATCH_JOURNAL = DATA_DIR / "summaries_batches.json"
TAXONOMY_JSON = Path(__file__).resolve().parent / "taxonomy.json"
MAP_HTML = DATA_DIR / "huggingface_dataset_map.html"
METHODOLOGY_HTML = DATA_DIR / "methodology.html"

# ── Docs (GitHub Pages) ──────────────────────────────────────────────────────
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(exist_ok=True)
DOCS_INDEX_HTML = DOCS_DIR / "index.html"

# ── API keys ─────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CO_API_KEY = os.environ.get("CO_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_DATASET_COUNT = 5_000
FETCH_OVERSHOOT_COUNT = 7_500  # 6_000 yielded exactly 5_000 in Apr 2026; gated/404/short cards make that margin thin
RANK_SORT_KEY = "likes"  # decided via experiments/rank_signal_analysis.py + _characterization.py
COHERE_BATCH_SIZE = 96
COHERE_EMBED_DIMENSION = 512
CARD_MAX_CHARS = 4_000
MIN_CARD_CHARS = 200
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_RANDOM_STATE = 42
# Toponymy 0.4.0 calls the Messages API with temperature= and reads response.content[0].text,
# so the naming model must accept sampling params and not emit a thinking block first.
# That rules out Sonnet 5 (400s on temperature); 4.6 is the newest Sonnet that works as-is.
ANTHROPIC_MODEL_TOPIC_NAMING = "claude-sonnet-4-6"
ANTHROPIC_MODEL_EXTRACT = "claude-haiku-4-5-20251001"
EXTRACT_CONCURRENCY = 12
EXTRACT_CARD_CHAR_LIMIT = 6_000
EXTRACT_MAX_RETRIES = 4

ANTHROPIC_MODEL_SUMMARIZE = "claude-haiku-4-5-20251001"
SUMMARIZE_CONCURRENCY = 12
SUMMARIZE_CARD_CHAR_LIMIT = 4_000  # TL;DRs come from the opening; shorter than extract
SUMMARIZE_MAX_WORDS = 25
SUMMARIZE_MAX_RETRIES = 4

# ── Message Batches (stages 04 / 04b) ────────────────────────────────────────
BATCH_CHUNK_SIZE = 1_000  # requests per batch; a batch that fails costs one chunk, not the run
BATCH_POLL_SECONDS = 30

# ── Experiments ─────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = DATA_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
