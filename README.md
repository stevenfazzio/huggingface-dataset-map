# Semantic Map of HuggingFace Datasets

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
python pipeline/04_visualize.py              # DataMapPlot HTML → huggingface_dataset_map.html
```

Single-stage enumeration: HuggingFace Hub's `list_datasets(sort=..., direction=-1, limit=N, full=True)` already returns ranked + filterable results, so there's no BigQuery-style pre-enumeration pass.

Data flows through `data/` (gitignored):

```
datasets.parquet ──> embeddings.npz ──> umap_coords.npz ──> labels.parquet
       │                                                            │
       └────────────────────────────────────────────────────────────┴──> huggingface_dataset_map.html
```

## Environment Variables

Set in `.env`:

| Variable | Used by | Purpose |
|---|---|---|
| `HF_TOKEN` | `pipeline/00_fetch_datasets.py` | HuggingFace auth (raises rate limits) |
| `CO_API_KEY` | `pipeline/01_embed_cards.py`, `pipeline/03_label_topics.py` | Cohere embeddings |
| `ANTHROPIC_API_KEY` | `pipeline/03_label_topics.py` | Claude Sonnet topic naming |

## Current scope

- **Target**: top **5,000** datasets, ranked by `likes` (see `pipeline/config.py`).
- **Rank signal**: `likes`, chosen via `experiments/rank_signal_analysis.py` and `experiments/rank_signal_characterization.py`. Likes ranks community-curated, mostly NLP datasets; downloads also picks up a lot of vision/robotics/pipeline-plumbing data with median 0 likes (top-1K overlap between the two rankings is only ~17%).
- **Hover card v1**: dataset id, likes, downloads, task_categories, modalities, languages, size_categories, license, last_modified.
- **Deferred**: LLM card summaries (Haiku) and LLM-extracted structured fields. Topic naming via Claude Sonnet is included from day one.

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
