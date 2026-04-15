# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python data pipeline that maps the top HuggingFace datasets by semantic similarity of their cards. It enumerates ranked datasets via the HF Hub API, downloads each card, embeds them, reduces dimensionality, labels topics via LLM-driven hierarchical clustering, and renders an interactive 2D visualization (`data/huggingface_dataset_map.html`).

Sister project: `~/repos/semantic-github-map` (same stack, applied to top GitHub repos).

## Running the Pipeline

```bash
uv sync --extra dev

python pipeline/00_fetch_datasets.py   # HF Hub list + card download → data/datasets.parquet
python pipeline/01_embed_cards.py      # Cohere embed-v4.0 → embeddings.npz
python pipeline/02_reduce_umap.py      # UMAP 512D → 2D → umap_coords.npz
python pipeline/03_label_topics.py     # Toponymy + Claude Sonnet → labels.parquet
python pipeline/04_visualize.py        # DataMapPlot → data/huggingface_dataset_map.html
```

Enumeration is single-stage: `HfApi().list_datasets(sort=..., direction=-1, limit=N, full=True)` already returns ranked, filterable results — no BigQuery-style pre-pass is needed.

## Required Environment Variables

Set in `.env` (loaded by `python-dotenv`):
- `HF_TOKEN` — HuggingFace auth. Raises rate limits; read-only is fine.
- `CO_API_KEY` — Cohere, used by `01_embed_cards.py` and `03_label_topics.py`.
- `ANTHROPIC_API_KEY` — Claude Sonnet, used by `03_label_topics.py` for Toponymy topic naming.

## Architecture

Sequential data pipeline — each script reads outputs from previous stages:

```
datasets.parquet ──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
       │                                                            │
       └────────────────────────────────────────────────────────────┴──> huggingface_dataset_map.html
```

`pipeline/config.py` is the central configuration hub: paths, API keys, constants (target count, batch sizes, UMAP params, rank sort key) are all defined there. Every pipeline script imports from it.

Key technology choices:
- HuggingFace Hub `list_datasets` for ranked enumeration + `hf_hub_download` for card text
- Cohere `embed-v4.0` (512-dim, `input_type="clustering"`) for card embeddings
- UMAP (n_neighbors=15, min_dist=0.05, cosine) for 512D → 2D
- Toponymy library for hierarchical density-based clustering with LLM topic naming
- Claude Sonnet for topic naming inside Toponymy
- DataMapPlot for the final interactive HTML visualization

## Current Scope and Open Decisions

- **Target**: top **1,000** datasets (per-stage constants in `config.py`). Easy to bump.
- **Rank signal open question**: `RANK_SORT_KEY = "likes"` for now, but `downloads` is also stored on every row. Formal comparison lives in `experiments/rank_signal_analysis.py` (to be written) and the decision will land there.
- **Deferred for now**:
  - LLM card summaries (Haiku) and LLM-extracted structured fields — revisit after the first map reveals what's worth extracting.
  - `docs/` GitHub Pages deployment, methodology.html, filter panel, Plausible analytics.
  - GitHub Actions CI.
- **Included from day one**:
  - Claude Sonnet topic naming inside Toponymy (cheap per run, materially improves the map).
  - Hover card with: dataset id, likes, downloads, task_categories, modalities, languages, size_categories, license, last_modified.
  - Colormaps over: first task_category, first modality, license, size_categories, first language, log-bucketed likes, log-bucketed downloads, and topic layers from Toponymy.

## Data Pipeline Rules

- NEVER overwrite existing parquet/data files in-place. Stage 00 uses temp-file + rename.
- For HF Hub calls, implement retry with exponential backoff on 429/5xx (already present in `_fetch_card`).
- Treat fetched cards as expensive/slow; stage 00 is resumable by default and re-uses any existing `datasets.parquet` cards unless `--refresh` is passed.

## Data Directory

All outputs go to `data/` (gitignored). Key files: `datasets.parquet`, `embeddings.npz`, `umap_coords.npz`, `labels.parquet`, `toponymy_model.joblib`, `huggingface_dataset_map.html`.

## Development

Makefile targets: `install`, `lint`, `format`, `test`, `pipeline`, `clean`.

Testing: pytest tests live in `tests/` and cover pure helpers only (no network, no API keys required). `test_fetch_datasets.py` loads the stage-00 module via `importlib.util.spec_from_file_location` because the filename starts with a digit.

Pre-commit hooks: ruff check and ruff format run automatically on commit. Install with `pre-commit install`.
