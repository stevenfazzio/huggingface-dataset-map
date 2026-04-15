"""Reduce embeddings to 2D with UMAP."""

import numpy as np
import umap
from config import (
    EMBEDDINGS_NPZ,
    UMAP_COORDS_NPZ,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)


def main():
    data = np.load(EMBEDDINGS_NPZ)
    embeddings = data["embeddings"]
    print(f"Loaded embeddings: {embeddings.shape}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(embeddings)

    np.savez(UMAP_COORDS_NPZ, coords=coords)
    print(f"Saved 2D coords {coords.shape} to {UMAP_COORDS_NPZ}")


if __name__ == "__main__":
    main()
