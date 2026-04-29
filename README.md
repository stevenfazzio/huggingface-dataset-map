# Semantic Map of HuggingFace Datasets

**[View the live map](https://stevenfazzio.github.io/huggingface-dataset-map/)**

An interactive 2D map of the top HuggingFace datasets, positioned by semantic similarity of their dataset cards.

The pipeline enumerates the top-ranked datasets directly from the HuggingFace Hub, downloads each card, embeds them with Cohere, reduces to 2D with UMAP, generates hierarchical topic labels with Toponymy + Claude Sonnet, and renders an interactive HTML visualization with DataMapPlot.

## Pipeline

Each stage is a standalone script. Run them in order:

```bash
uv sync --extra dev                          # Install deps

python pipeline/00_fetch_datasets.py         # HF Hub list + card download → datasets.parquet
python pipeline/01_embed_cards.py            # Cohere embed-v4.0 → embeddings.npz
python pipeline/02_reduce_umap.py            # UMAP 512D → 2D → umap_coords.npz
python pipeline/03_label_topics.py           # Toponymy + Claude Sonnet → labels.parquet
python pipeline/04_extract_structured.py     # Claude Haiku per-card extraction → structured_fields.parquet
python pipeline/04b_summarize_cards.py       # Claude Haiku ≤25-word TL;DRs → summaries.parquet
python pipeline/05_visualize.py              # DataMapPlot HTML → huggingface_dataset_map.html (+ docs/index.html, docs/methodology.html)
```

Single-stage enumeration: HuggingFace Hub's `list_datasets(sort=..., direction=-1, limit=N, full=True)` already returns ranked + filterable results, so there's no BigQuery-style pre-enumeration pass.

Data flows through `data/` (gitignored):

```
datasets.parquet ──> embeddings.npz ──> umap_coords.npz ──> labels.parquet ──┐
       │                                                                     │
       ├──> structured_fields.parquet ─────────────────────────────────────┐ │
       │                                                                   │ │
       └──> summaries.parquet ────────────────────────────────────────────┐│ │
                                                                          ▼▼ ▼
                                                         huggingface_dataset_map.html
```

Stages 04 and 04b read only `datasets.parquet` and are independent of each other and of the embed/UMAP/topic chain. Both are resumable via per-repo JSONs in `data/structured_fields_cache/` and `data/summaries_cache/`.

## Environment Variables

Set in `.env`:

| Variable | Used by | Purpose |
|---|---|---|
| `HF_TOKEN` | `pipeline/00_fetch_datasets.py` | HuggingFace auth (raises rate limits) |
| `CO_API_KEY` | `pipeline/01_embed_cards.py`, `pipeline/03_label_topics.py` | Cohere embeddings |
| `ANTHROPIC_API_KEY` | `pipeline/03_label_topics.py` (Sonnet topic naming), `pipeline/04_extract_structured.py` (Haiku field extraction), `pipeline/04b_summarize_cards.py` (Haiku TL;DR) | Claude API |

## Current scope

- **Target**: top **5,000** datasets, ranked by `likes` (see `pipeline/config.py`).
- **Rank signal**: `likes`, chosen via `experiments/rank_signal_analysis.py` and `experiments/rank_signal_characterization.py`. Likes ranks community-curated, mostly NLP datasets; downloads also picks up a lot of vision/robotics/pipeline-plumbing data with median 0 likes (top-1K overlap between the two rankings is only ~17%).
- **LLM augmentations** (all included): Sonnet-named topic clusters via Toponymy; Haiku-extracted structured fields against `pipeline/taxonomy.json` (provenance, subject domain, training stage, format convention, special characteristics, geo scope, upstream models, is-benchmark); Haiku-written ≤25-word TL;DR summaries.
- **Hover card**: org/name, popularity stats (likes/downloads/size with inline bars), the LLM TL;DR, an LLM-classified subject pill, and a 2-column grid of HF metadata (task, modality, language) + LLM-extracted fields (role, stage, provenance, format), with license + last-modified in the footer.
- **Deployment**: `docs/index.html` (the rendered map) and `docs/methodology.html` are committed to `main` and served by GitHub Pages at <https://stevenfazzio.github.io/huggingface-dataset-map/>.
- **Deferred**: Plausible analytics, GitHub Actions CI, social-preview image.

## Technical Details

| Component | Choice |
|---|---|
| Dataset enumeration | HuggingFace Hub `list_datasets` |
| Card text | `README.md` via `hf_hub_download` (YAML frontmatter stripped) |
| Embeddings | Cohere `embed-v4.0` (512-dim, input_type=clustering) |
| Dimensionality reduction | UMAP (n_neighbors=15, min_dist=0.05, cosine) 512D → 2D |
| Topic clustering | [Toponymy](https://github.com/TutteInstitute/Toponymy) (hierarchical density-based) |
| Topic naming | Claude Sonnet |
| Visualization | [DataMapPlot](https://github.com/TutteInstitute/DataMapPlot) |

## Development

```bash
make install        # uv sync --extra dev
make lint           # ruff check + format check
make format         # auto-format
make test           # pytest
make pipeline       # run all stages in order
```

Pre-commit hooks run ruff on every commit. Set up with `pre-commit install`.

## Requirements

Python ≥ 3.10. See `pyproject.toml` for the full dependency list.
