"""Generate hierarchical topic labels via Toponymy + Claude Sonnet."""

import joblib
import nest_asyncio
import numpy as np
import pandas as pd
from toponymy import Toponymy, ToponymyClusterer
from toponymy.embedding_wrappers import CohereEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_TOPIC_NAMING,
    CO_API_KEY,
    DATASETS_PARQUET,
    EMBEDDINGS_NPZ,
    LABELS_PARQUET,
    TOPONYMY_MODEL_JOBLIB,
    UMAP_COORDS_NPZ,
    UMAP_RANDOM_STATE,
)


def _build_document(row: pd.Series) -> str:
    """Compose per-dataset text for topic naming: id + tags + card excerpt."""
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


def main():
    df = pd.read_parquet(DATASETS_PARQUET)
    embeddings = np.load(EMBEDDINGS_NPZ)["embeddings"]
    coords = np.load(UMAP_COORDS_NPZ)["coords"]

    documents = [_build_document(row) for _, row in df.iterrows()]
    print(f"Loaded {len(documents)} documents")

    llm = AsyncAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_TOPIC_NAMING)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model="embed-v4.0")
    clusterer = ToponymyClusterer(min_clusters=4)

    np.random.seed(UMAP_RANDOM_STATE)

    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="HuggingFace dataset cards",
        corpus_description="collection of the top 5,000 HuggingFace datasets ranked by likes",
        exemplar_delimiters=['    * """', '"""\n'],
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    topic_model.fit(
        objects=documents,
        embedding_vectors=embeddings,
        clusterable_vectors=coords,
    )

    n_layers = len(topic_model.cluster_layers_)
    if n_layers == 0:
        raise ValueError("No cluster layers found")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    labels_dict = {"repo_id": df["repo_id"]}
    for i, layer in enumerate(reversed(topic_model.cluster_layers_)):
        labels_dict[f"label_layer_{i}"] = layer.topic_name_vector

    labels_df = pd.DataFrame(labels_dict)
    labels_df.to_parquet(LABELS_PARQUET, index=False)
    print(f"Saved labels to {LABELS_PARQUET}")

    try:
        joblib.dump(topic_model, TOPONYMY_MODEL_JOBLIB)
        print(f"Saved model to {TOPONYMY_MODEL_JOBLIB}")
    except TypeError:
        print("Skipped saving model (async client is not picklable)")


if __name__ == "__main__":
    main()
