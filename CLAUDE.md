# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python data pipeline that maps the top HuggingFace datasets by semantic similarity of their cards. It enumerates ranked datasets via the HF Hub API, downloads each card, embeds them, reduces dimensionality, labels topics via LLM-driven hierarchical clustering, and renders an interactive 2D visualization (`data/huggingface_dataset_map.html`).

Sister project: `~/repos/semantic-github-map` (same stack, applied to top GitHub repos).

## Running the Pipeline

```bash
uv sync --extra dev

python pipeline/00_fetch_datasets.py       # HF Hub list + card download → data/datasets.parquet
python pipeline/01_embed_cards.py          # Cohere embed-v4.0 → embeddings.npz
python pipeline/02_reduce_umap.py          # UMAP 512D → 2D → umap_coords.npz
python pipeline/03_label_topics.py         # Toponymy + Claude Sonnet → labels.parquet
python pipeline/04_extract_structured.py   # Claude Haiku per-card extraction → structured_fields.parquet
python pipeline/04b_summarize_cards.py     # Claude Haiku ≤25-word TL;DRs → summaries.parquet
python pipeline/05_visualize.py            # DataMapPlot → data/huggingface_dataset_map.html
```

Enumeration is single-stage: `HfApi().list_datasets(sort=..., direction=-1, limit=N, full=True)` already returns ranked, filterable results — no BigQuery-style pre-pass is needed.

## Required Environment Variables

Set in `.env` (loaded by `python-dotenv`):
- `HF_TOKEN` — HuggingFace auth. Raises rate limits; read-only is fine.
- `CO_API_KEY` — Cohere, used by `01_embed_cards.py` and `03_label_topics.py`.
- `ANTHROPIC_API_KEY` — Claude. Used by `03_label_topics.py` (Sonnet, topic naming), `04_extract_structured.py` (Haiku, per-card field extraction), and `04b_summarize_cards.py` (Haiku, per-card TL;DR summaries).

## Architecture

Sequential data pipeline — each script reads outputs from previous stages:

```
datasets.parquet ──> embeddings.npz ──> umap_coords.npz ──> labels.parquet ──┐
       │                                                                     │
       ├──> structured_fields.parquet ─────────────────────────────────────┐ │
       │                                                                   │ │
       └──> summaries.parquet ────────────────────────────────────────────┐│ │
                                                                          ▼▼ ▼
                                                         huggingface_dataset_map.html
```

`structured_fields.parquet` (stage 04) and `summaries.parquet` (stage 04b) are independently produced from `datasets.parquet` alone (neither depends on embeddings/UMAP/topics, nor on each other). Both are resumable via per-repo JSONs in `data/structured_fields_cache/` and `data/summaries_cache/` respectively.

`pipeline/config.py` is the central configuration hub: paths, API keys, constants (target count, batch sizes, UMAP params, rank sort key) are all defined there. Every pipeline script imports from it.

Key technology choices:
- HuggingFace Hub `list_datasets` for ranked enumeration + `hf_hub_download` for card text
- Cohere `embed-v4.0` (512-dim, `input_type="clustering"`) for card embeddings
- UMAP (n_neighbors=15, min_dist=0.05, cosine) for 512D → 2D
- Toponymy library for hierarchical density-based clustering with LLM topic naming
- Claude Sonnet for topic naming inside Toponymy
- Claude Haiku for per-card structured-field extraction against a constrained schema (`pipeline/taxonomy.json`)
- Claude Haiku for per-card ≤25-word TL;DR summaries (hover-card "what is this?" context)
- DataMapPlot for the final interactive HTML visualization

## Current Scope and Open Decisions

- **Target**: top **5,000** datasets ranked by `likes` (per-stage constants in `config.py`).
- **Rank signal**: decided as `likes` based on `experiments/rank_signal_analysis.py` and `experiments/rank_signal_characterization.py`. Likes-vs-downloads top-1K overlap was only ~17%; downloads-only repos skewed toward newer vision/robotics/pipeline-plumbing data with median 0 likes, while likes-top reflects community-curated, mostly NLP datasets. Revisit if the corpus expands past ~5K, where the bottom slice (likes ~10) starts to be noise-dominated.
- **Deferred for now**:
  - `docs/` GitHub Pages deployment, methodology.html, filter panel, Plausible analytics.
  - GitHub Actions CI.
  - Hover-card visual redesign and field pruning — the current card has all metadata for development convenience; expect trimming and styling passes later.
- **Included**:
  - Claude Sonnet topic naming inside Toponymy (cheap per run, materially improves the map).
  - LLM-extracted structured fields (stage 04, Haiku): provenance_method, subject_domain, training_stage, format_convention, special_characteristics, geo_scope, upstream_models, is_benchmark. Schema lives in `pipeline/taxonomy.json`; iteration history is in `experiments/extract_structured_fields_v{1,2,3}.py` and `experiments/taxonomy_v3_proposed.json`.
  - LLM card summaries (stage 04b, Haiku): ≤25-word self-contained TL;DR per dataset, appears above metadata rows in the hover card. Prompt developed in `experiments/summarize_cards_v1.py` using EVoC-stratified ~150-card trial.
  - Hover card with: dataset id, likes, downloads, LLM TL;DR summary, HF metadata (task, modality, language, size, license, updated), and LLM fields (subject, stage, provenance, format, benchmark).
  - Colormaps over HF metadata (first task_category, first modality, license, size_categories, first language, log-bucketed likes, log-bucketed downloads), Toponymy topic layers, and LLM fields (subject_domain, provenance_method, training_stage primary, format_convention, is_benchmark).

## Data Pipeline Rules

- NEVER overwrite existing parquet/data files in-place. Stage 00 uses temp-file + rename.
- For HF Hub calls, implement retry with exponential backoff on 429/5xx (already present in `_fetch_card`).
- Treat fetched cards as expensive/slow; stage 00 is resumable by default and re-uses any existing `datasets.parquet` cards unless `--refresh` is passed.

## Data Directory

All outputs go to `data/` (gitignored). Key files: `datasets.parquet`, `embeddings.npz`, `umap_coords.npz`, `labels.parquet`, `toponymy_model.joblib`, `structured_fields.parquet`, `structured_fields_cache/`, `summaries.parquet`, `summaries_cache/`, `huggingface_dataset_map.html`. The `*_cache/` directories hold per-repo JSONs and are used for resuming the corresponding LLM stages.

## Development

Makefile targets: `install`, `lint`, `format`, `test`, `pipeline`, `clean`.

Testing: pytest tests live in `tests/` and cover pure helpers only (no network, no API keys required). `test_fetch_datasets.py` loads the stage-00 module via `importlib.util.spec_from_file_location` because the filename starts with a digit.

Pre-commit hooks: ruff check and ruff format run automatically on commit. Install with `pre-commit install`.
