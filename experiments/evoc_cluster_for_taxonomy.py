"""Cluster the 5K card embeddings with EVoC + Toponymy for taxonomy-gap analysis.

Unlike `pipeline/03_label_topics.py` (which clusters on 2D UMAP coords so the
visual map can label its regions), this script clusters on the full 512D Cohere
embeddings via EVoC. That gives better conceptual fidelity but labels that
don't correspond 1:1 with visual regions on the map — so it's saved to a
separate artifact, not overwriting `labels.parquet`.

Output is intended as input to a subsequent gap-analysis pass that embeds these
EVoC labels alongside all structured-field slugs to surface concepts no
structured field captures.

Run with:
    # Cluster only (no LLM cost) to check layer sizes:
    uv run --with evoc python experiments/evoc_cluster_for_taxonomy.py --cluster-only

    # Full run with Haiku naming (cheap, ~$0.50):
    uv run --with evoc python experiments/evoc_cluster_for_taxonomy.py

    # Upgrade to Sonnet if Haiku names look lazy:
    uv run --with evoc python experiments/evoc_cluster_for_taxonomy.py --model sonnet

Outputs:
    data/experiments/evoc_taxonomy/cluster_layers.npz   # labels per layer per doc
    data/experiments/evoc_taxonomy/topic_names.json     # named topics per layer
    data/experiments/evoc_taxonomy/summary.md           # human-readable report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import nest_asyncio
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    CO_API_KEY,
    DATASETS_PARQUET,
    EMBEDDINGS_NPZ,
    EXPERIMENTS_DIR,
    UMAP_RANDOM_STATE,
)

# fast-hdbscan <0.3 in this project may or may not need the numpy 2.x kdtree
# patch from the sister project. Apply defensively — it's a no-op if internals
# already match.
try:
    import fast_hdbscan.numba_kdtree as _nkd

    def _kdtree_to_numba_patched(sklearn_kdtree):
        data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
        return _nkd.NumbaKDTree(
            data,
            idx_array,
            node_data["idx_start"],
            node_data["idx_end"],
            node_data["radius"],
            node_data["is_leaf"],
            node_bounds,
        )

    _nkd.kdtree_to_numba = _kdtree_to_numba_patched
except Exception:  # noqa: BLE001
    pass

import evoc  # noqa: E402
from toponymy import Toponymy  # noqa: E402
from toponymy.cluster_layer import ClusterLayerText  # noqa: E402
from toponymy.clustering import Clusterer, build_cluster_tree, centroids_from_labels  # noqa: E402
from toponymy.embedding_wrappers import CohereEmbedder  # noqa: E402
from toponymy.llm_wrappers import AsyncAnthropicNamer  # noqa: E402

nest_asyncio.apply()

OUT_DIR = EXPERIMENTS_DIR / "evoc_taxonomy"
LAYERS_NPZ = OUT_DIR / "cluster_layers.npz"
TOPICS_JSON = OUT_DIR / "topic_names.json"
SUMMARY_MD = OUT_DIR / "summary.md"

MODEL_CHOICES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-20250514",
}


class EVoCClusterer(Clusterer):
    """Wrapper so EVoC plugs into Toponymy's Clusterer API."""

    def __init__(self, base_min_cluster_size=25, noise_level=0.3, max_layers=10):
        super().__init__()
        self.evoc_model = evoc.EVoC(
            base_min_cluster_size=base_min_cluster_size,
            noise_level=noise_level,
            max_layers=max_layers,
        )

    def fit(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, **kwargs):
        self.evoc_model.fit(embedding_vectors)
        cluster_labels = self.evoc_model.cluster_layers_
        self.cluster_tree_ = build_cluster_tree(cluster_labels)
        self.cluster_layers_ = [
            layer_class(
                labels,
                centroids_from_labels(labels, embedding_vectors),
                layer_id=i,
            )
            for i, labels in enumerate(cluster_labels)
        ]
        return self

    def fit_predict(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, **kwargs):
        self.fit(clusterable_vectors, embedding_vectors, layer_class=layer_class, **kwargs)
        return self.cluster_layers_, self.cluster_tree_


def _build_document(row: pd.Series) -> str:
    """Same document shape as stage 03 — repo_id + tags + card excerpt."""
    parts = []
    if row.get("pretty_name"):
        parts.append(str(row["pretty_name"]).strip())
    parts.append(str(row["repo_id"]))
    tag_bits = []
    for col in ("task_categories", "modalities", "languages", "size_categories"):
        val = row.get(col) or ""
        if val:
            tag_bits.append(f"{col}: {val}")
    if tag_bits:
        parts.append(" | ".join(tag_bits))
    card = str(row.get("card_text_clean") or "").strip()
    if card:
        parts.append(card[:2000])
    return "\n".join(parts)


def _atomic_write(path: Path, writer):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def cluster_only(embeddings: np.ndarray, min_cluster_size: int, max_layers: int) -> list[np.ndarray]:
    """Run EVoC clustering without naming. Returns cluster_labels per layer."""
    np.random.seed(UMAP_RANDOM_STATE)
    model = evoc.EVoC(
        base_min_cluster_size=min_cluster_size,
        noise_level=0.3,
        max_layers=max_layers,
    )
    model.fit(embeddings)
    return model.cluster_layers_


def report_counts(cluster_layers: list[np.ndarray]) -> list[dict]:
    """Print and return per-layer stats."""
    print("\n── EVoC layer counts (0 = finest) ──")
    stats = []
    for i, labels in enumerate(cluster_layers):
        labels = np.asarray(labels)
        n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        n_total = len(labels)
        n_clustered = n_total - n_noise
        stats.append(
            {
                "layer": i,
                "n_clusters": n_clusters,
                "n_clustered": n_clustered,
                "n_noise": n_noise,
                "coverage_pct": round(100 * n_clustered / n_total, 1),
            }
        )
        print(
            f"  Layer {i}: {n_clusters:>4d} clusters   "
            f"coverage {n_clustered}/{n_total} ({stats[-1]['coverage_pct']}%)"
        )
    return stats


def name_clusters(
    documents: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int,
    max_layers: int,
    model_id: str,
) -> Toponymy:
    """Full Toponymy fit (cluster + name)."""
    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=model_id)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0")
    clusterer = EVoCClusterer(
        base_min_cluster_size=min_cluster_size,
        noise_level=0.3,
        max_layers=max_layers,
    )

    np.random.seed(UMAP_RANDOM_STATE)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="HuggingFace dataset cards",
        corpus_description=(
            "collection of the top 5,000 HuggingFace datasets ranked by likes; "
            "clustered on raw 512-dim embeddings (not 2D UMAP coords) to surface "
            "fine-grained conceptual structure"
        ),
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=embeddings,  # ignored by EVoCClusterer, required by API
    )
    return topic_model


def save_outputs(topic_model: Toponymy, stats: list[dict], model_id: str) -> None:
    # Layer labels as an npz — one array per layer, finest first.
    # np.savez appends .npz to the filename; atomic-write the final path directly.
    layer_arrays = {
        f"layer_{i}": np.asarray(layer.cluster_labels)
        for i, layer in enumerate(topic_model.cluster_layers_)
    }
    tmp_path = LAYERS_NPZ.with_suffix(".npz.tmp.npz")
    try:
        # np.savez wants a stem-or-path; appends .npz. Give it a stem ending in .tmp.
        np.savez(str(LAYERS_NPZ.with_suffix(".npz.tmp")), **layer_arrays)
        os.replace(tmp_path, LAYERS_NPZ)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    print(f"Wrote layer labels to {LAYERS_NPZ}")

    # Topic names per layer as JSON.
    topics = {}
    for i, layer in enumerate(topic_model.cluster_layers_):
        topics[f"layer_{i}"] = list(layer.topic_names)
    _atomic_write(
        TOPICS_JSON,
        lambda p: Path(p).write_text(json.dumps(topics, indent=2, ensure_ascii=False)),
    )
    print(f"Wrote topic names to {TOPICS_JSON}")

    # Human-readable markdown summary.
    lines = [f"# EVoC taxonomy exploration ({model_id})\n"]
    for s in stats:
        lines.append(
            f"- Layer {s['layer']}: {s['n_clusters']} clusters, "
            f"{s['coverage_pct']}% coverage, {s['n_noise']} noise"
        )
    lines.append("")
    for i, layer in enumerate(topic_model.cluster_layers_):
        labels = np.asarray(layer.cluster_labels)
        lines.append(f"\n## Layer {i} — {len(layer.topic_names)} topics\n")
        for tid, name in enumerate(layer.topic_names):
            count = int((labels == tid).sum())
            lines.append(f"- **{name}** ({count})")
    _atomic_write(SUMMARY_MD, lambda p: Path(p).write_text("\n".join(lines)))
    print(f"Wrote summary to {SUMMARY_MD}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-cluster-size", type=int, default=25)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument(
        "--max-clusters-gate",
        type=int,
        default=300,
        help="Abort before naming if finest-layer cluster count exceeds this.",
    )
    parser.add_argument("--cluster-only", action="store_true")
    parser.add_argument("--model", choices=list(MODEL_CHOICES), default="haiku")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed with naming even if finest layer exceeds the gate.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    print(f"Loaded {embeddings.shape} embeddings")

    if args.cluster_only:
        layers = cluster_only(embeddings, args.min_cluster_size, args.max_layers)
        report_counts(layers)
        return

    # Quick dry-run first to check cluster counts before paying for naming.
    print(f"\nDry-run cluster check (min_cluster_size={args.min_cluster_size})...")
    layers = cluster_only(embeddings, args.min_cluster_size, args.max_layers)
    stats = report_counts(layers)
    finest = stats[0]["n_clusters"]
    if finest > args.max_clusters_gate and not args.force:
        print(
            f"\nFinest-layer cluster count ({finest}) exceeds --max-clusters-gate "
            f"({args.max_clusters_gate}). Increase --min-cluster-size to coarsen, "
            f"raise --max-clusters-gate, or pass --force to proceed anyway."
        )
        sys.exit(1)

    # Full Toponymy with naming.
    if not ANTHROPIC_API_KEY or not CO_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY and CO_API_KEY required for naming")

    df = pd.read_parquet(DATASETS_PARQUET)
    documents = [_build_document(row) for _, row in df.iterrows()]
    if len(documents) != len(embeddings):
        raise ValueError(
            f"Row mismatch: {len(documents)} docs vs {len(embeddings)} embeddings"
        )
    print(f"Built {len(documents)} documents for naming")

    model_id = MODEL_CHOICES[args.model]
    print(f"\nRunning Toponymy with EVoC clustering + {args.model} naming ({model_id})...")
    topic_model = name_clusters(
        documents,
        embeddings,
        args.min_cluster_size,
        args.max_layers,
        model_id,
    )

    save_outputs(topic_model, stats, model_id)


if __name__ == "__main__":
    main()
