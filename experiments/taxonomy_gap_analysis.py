"""Find taxonomy gaps by comparing EVoC cluster labels against structured-field slugs.

Embeds all labels — 43 EVoC finest-layer names (from `evoc_cluster_for_taxonomy.py`)
plus every slug from `pipeline/taxonomy.json` plus top HF task/modality tags —
with Cohere embed-v4.0, then produces two views:

(1) GAP LIST — for each EVoC label, find its nearest structured slug and sort
    by distance descending. Labels with large nearest-slug distance are
    candidates for new structured fields/slugs.

(2) ORTHOGONALITY CHECK — top cross-field similar slug pairs. Legitimately
    close pairs (code slug ≈ HF task=text-generation) are expected; duplicate-
    concept pairs across LLM fields suggest redundancy or confusing descriptions.

Sanity-checking the method: the known cross-field bleed concepts from the v3
stage-04 run (`multilingual-parallel` in 3 fields, `vqa` in 2) are highlighted
in the orthogonality table — they SHOULD rank high if embedding similarity
tracks conceptual overlap. If they don't, the method isn't picking up the
confusion that we know exists, and the gap results downstream are suspect.

Outputs:
    data/experiments/taxonomy_gap_analysis/report.html
    data/experiments/taxonomy_gap_analysis/gap_list.json
    data/experiments/taxonomy_gap_analysis/orthogonality_pairs.json
"""

from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from pathlib import Path

import cohere
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import CO_API_KEY, DATASETS_PARQUET, EXPERIMENTS_DIR, TAXONOMY_JSON  # noqa: E402

EVOC_TOPICS_JSON = EXPERIMENTS_DIR / "evoc_taxonomy" / "topic_names.json"
EVOC_LAYERS_NPZ = EXPERIMENTS_DIR / "evoc_taxonomy" / "cluster_layers.npz"
OUT_DIR = EXPERIMENTS_DIR / "taxonomy_gap_analysis"
REPORT_HTML = OUT_DIR / "report.html"
GAP_JSON = OUT_DIR / "gap_list.json"
ORTHO_JSON = OUT_DIR / "orthogonality_pairs.json"

# Known cross-field bleed concepts from stage-04 validation (see the rerun
# checklist in pipeline/04_extract_structured.py). If embedding similarity
# captures conceptual overlap, these terms should appear near the top of the
# orthogonality report — that's our method sanity check.
KNOWN_BLEED_TOKENS = {"multilingual-parallel", "vqa", "raw-corpus", "roleplay"}

# Sentinels are meta-tags, not concepts. Exclude from orthogonality analysis
# so "field-A=not_stated ↔ field-B=not_stated" pairs don't dominate the top.
SENTINEL_SLUGS = {"not_stated", "not_applicable", "other"}


def _embed(strings: list[str]) -> np.ndarray:
    client = cohere.ClientV2(api_key=CO_API_KEY)
    vecs: list[list[float]] = []
    batch = 96
    for i in range(0, len(strings), batch):
        chunk = strings[i : i + batch]
        resp = client.embed(
            texts=chunk,
            model="embed-v4.0",
            input_type="clustering",
            embedding_types=["float"],
            output_dimension=512,
        )
        vecs.extend(resp.embeddings.float_)
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    return arr


def _load_evoc_labels() -> list[dict]:
    topics = json.loads(EVOC_TOPICS_JSON.read_text())
    labels = np.load(EVOC_LAYERS_NPZ)
    # Use finest layer (layer_0) — that's where the most granular concepts live.
    layer0_labels = labels["layer_0"]
    layer0_names = topics["layer_0"]
    out = []
    for tid, name in enumerate(layer0_names):
        count = int((layer0_labels == tid).sum())
        out.append(
            {
                "kind": "evoc",
                "layer": 0,
                "cluster_id": tid,
                "display": name,
                "embed_text_bare": name,
                "embed_text_rich": name,
                "count": count,
            }
        )
    return out


def _load_structured_slugs() -> list[dict]:
    """Each slug gets TWO embed texts — bare (just the slug) and rich (slug + desc).

    We run both variants and compare. Hypothesis: rich descriptions compress
    all slug embeddings toward a 'description-y prose' cluster, flattening
    distances. Bare names should spread out and discriminate better.
    """
    tax = json.loads(TAXONOMY_JSON.read_text())
    out = []
    for field, spec in tax.items():
        if field.startswith("_"):
            continue
        for cat in spec.get("categories", []):
            bare = cat["name"].replace("-", " ").replace("_", " ")
            rich = f"{cat['name']}: {cat['description']}"
            out.append(
                {
                    "kind": "structured",
                    "field": field,
                    "slug": cat["name"],
                    "display": f"{field}={cat['name']}",
                    "embed_text_bare": bare,
                    "embed_text_rich": rich,
                }
            )
    return out


def _load_hf_metadata_slugs(df: pd.DataFrame, top_tasks: int = 20) -> list[dict]:
    """Top HF task_categories and all modalities as comparison axes.

    HF slugs have no description so bare and rich are the same here.
    """
    from collections import Counter

    out = []

    task_counter: Counter = Counter()
    for v in df["task_categories"].fillna(""):
        first = v.split(",")[0].strip()
        if first:
            task_counter[first] += 1
    for task, n in task_counter.most_common(top_tasks):
        bare = task.replace("-", " ")
        out.append(
            {
                "kind": "hf",
                "field": "hf_task",
                "slug": task,
                "display": f"hf_task={task}",
                "embed_text_bare": bare,
                "embed_text_rich": bare,
                "count": n,
            }
        )

    mod_counter: Counter = Counter()
    for v in df["modalities"].fillna(""):
        for m in v.split(","):
            m = m.strip()
            if m:
                mod_counter[m] += 1
    for mod, n in mod_counter.most_common(10):
        out.append(
            {
                "kind": "hf",
                "field": "hf_modality",
                "slug": mod,
                "display": f"hf_modality={mod}",
                "embed_text_bare": mod,
                "embed_text_rich": mod,
                "count": n,
            }
        )
    return out


def gap_list(evoc_items: list[dict], structured_items: list[dict], sim: np.ndarray) -> list[dict]:
    """For each EVoC label, find nearest structured or HF slug."""
    n_evoc = len(evoc_items)
    all_other_start = n_evoc  # structured + HF come after evoc in the matrix
    rows = []
    for i, ev in enumerate(evoc_items):
        # Similarity to every non-EVoC label.
        sims = sim[i, all_other_start:]
        ranked = np.argsort(-sims)
        top5 = [
            {
                "target": structured_items[j]["display"] + (
                    f"  [{structured_items[j].get('field', '?')}]"
                    if False
                    else ""
                ),
                "target_display": structured_items[j]["display"],
                "sim": float(sims[j]),
            }
            for j in ranked[:5]
        ]
        rows.append(
            {
                "evoc_label": ev["display"],
                "count": ev["count"],
                "nearest": top5[0]["target_display"],
                "nearest_sim": top5[0]["sim"],
                "nearest_distance": 1 - top5[0]["sim"],
                "top5": top5,
            }
        )
    rows.sort(key=lambda r: -r["nearest_distance"])
    return rows


def orthogonality_pairs(structured_items: list[dict], sim_ss: np.ndarray, top_k: int = 40) -> list[dict]:
    """Top cross-field similar slug pairs (upper triangle, different fields).

    Skips sentinel slugs (not_stated/not_applicable/other) — these are meta-tags,
    not conceptual, and would otherwise dominate the top of the list because all
    'card is silent' descriptions cluster together.
    """
    pairs = []
    n = len(structured_items)
    for i in range(n):
        if structured_items[i]["slug"] in SENTINEL_SLUGS:
            continue
        for j in range(i + 1, n):
            if structured_items[j]["slug"] in SENTINEL_SLUGS:
                continue
            a, b = structured_items[i], structured_items[j]
            if a["field"] == b["field"]:
                continue
            pairs.append(
                {
                    "a": a["display"],
                    "b": b["display"],
                    "sim": float(sim_ss[i, j]),
                    "flagged": any(
                        tok in a.get("slug", "") or tok in b.get("slug", "")
                        for tok in KNOWN_BLEED_TOKENS
                    ),
                }
            )
    pairs.sort(key=lambda p: -p["sim"])
    return pairs[:top_k]


def _atomic_write(path: Path, content: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _gap_rows_html(gaps: list[dict]) -> str:
    def color_for_distance(d: float) -> str:
        if d > 0.55:
            return "#ffe0e0"
        if d > 0.45:
            return "#fff4cc"
        return "#e7f5e7"

    rows = []
    for g in gaps:
        top5_html = "".join(
            f"<li>{html.escape(t['target_display'])} <span class='sim'>sim={t['sim']:.3f}</span></li>"
            for t in g["top5"]
        )
        rows.append(
            f"""<tr style="background:{color_for_distance(g['nearest_distance'])};">
              <td class="label">{html.escape(g['evoc_label'])}</td>
              <td class="n">{g['count']}</td>
              <td class="sim">{g['nearest_sim']:.3f}</td>
              <td class="dist">{g['nearest_distance']:.3f}</td>
              <td class="near">{html.escape(g['nearest'])}</td>
              <td class="top5"><details><summary>top-5</summary><ol>{top5_html}</ol></details></td>
            </tr>"""
        )
    return "".join(rows)


def _pair_rows_html(pairs: list[dict]) -> str:
    rows = []
    for p in pairs:
        flag = "⚠️" if p["flagged"] else ""
        rows.append(
            f"""<tr>
              <td>{html.escape(p['a'])}</td>
              <td>{html.escape(p['b'])}</td>
              <td class="sim">{p['sim']:.3f}</td>
              <td>{flag}</td>
            </tr>"""
        )
    return "".join(rows)


def _dist_stats(gaps: list[dict]) -> dict:
    dists = [g["nearest_distance"] for g in gaps]
    return {
        "min": min(dists),
        "max": max(dists),
        "mean": sum(dists) / len(dists),
        "p50": sorted(dists)[len(dists) // 2],
        "range": max(dists) - min(dists),
    }


def build_html(
    gaps_bare: list[dict],
    gaps_rich: list[dict],
    pairs_bare: list[dict],
    pairs_rich: list[dict],
) -> str:
    stats_bare = _dist_stats(gaps_bare)
    stats_rich = _dist_stats(gaps_rich)

    def fmt_stats(s):
        return (
            f"min={s['min']:.3f} · p50={s['p50']:.3f} · max={s['max']:.3f} · "
            f"range={s['range']:.3f}"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Taxonomy gap analysis</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto; max-width: 1400px; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 2em; color: #114; }}
  h3 {{ margin-top: 1.3em; color: #335; }}
  p.intro, p.stats {{ color: #555; font-size: 0.95em; line-height: 1.5; }}
  p.stats {{ font-family: 'SF Mono', ui-monospace, monospace; font-size: 0.85em; color: #333; background: #f4f4f7; padding: 0.5em 0.8em; border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; margin-top: 0.5em; }}
  th, td {{ text-align: left; padding: 0.4em 0.55em; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #f4f4f7; position: sticky; top: 0; }}
  td.label {{ font-weight: 600; }}
  td.n, td.sim, td.dist {{ text-align: right; font-variant-numeric: tabular-nums; width: 5em; }}
  td.near {{ color: #333; }}
  td.top5 ol {{ margin: 0; padding-left: 1.4em; font-size: 0.85em; }}
  details summary {{ cursor: pointer; color: #06c; }}
  .sim {{ color: #666; font-variant-numeric: tabular-nums; }}
  .note {{ background: #fff9e6; border-left: 3px solid #f5c518; padding: 0.6em 1em; font-size: 0.9em; margin: 1em 0; }}
</style></head><body>

<h1>Taxonomy gap analysis</h1>
<p class="intro">EVoC finest-layer cluster labels (43 labels, clustered on raw 512D embeddings) embedded alongside every structured-field slug and top HF metadata categories, then compared via cosine similarity. Sentinels (not_stated, not_applicable, other) excluded from orthogonality analysis.</p>

<div class="note">
Two embedding variants compared: <b>bare</b> (just the slug name, e.g. <code>"natural images and video"</code>) vs <b>rich</b> (slug name + full description). Hypothesis: rich descriptions cluster all slug embeddings toward description-prose style, flattening distances.
</div>

<h2>Distance-range diagnostic</h2>
<p class="stats"><b>BARE:</b> {fmt_stats(stats_bare)}<br><b>RICH:</b> {fmt_stats(stats_rich)}</p>
<p class="intro">Larger range = better discrimination. If bare has a wider range than rich, the description-prose hypothesis holds and we should trust bare more.</p>

<h2>1. Gap list — EVoC labels ranked by distance from nearest slug</h2>
<p class="intro">Higher distance = weaker match = better gap candidate. Top-5 nearest shown so you can judge whether the nearest match is capturing the concept or the embedding is just picking up surface form.</p>

<h3>1a. BARE embedding (slug names only)</h3>
<table>
  <thead><tr>
    <th>EVoC label</th><th>n</th><th>sim</th><th>dist</th>
    <th>nearest slug</th><th>top-5</th>
  </tr></thead>
  <tbody>{_gap_rows_html(gaps_bare)}</tbody>
</table>

<h3>1b. RICH embedding (slug + description)</h3>
<table>
  <thead><tr>
    <th>EVoC label</th><th>n</th><th>sim</th><th>dist</th>
    <th>nearest slug</th><th>top-5</th>
  </tr></thead>
  <tbody>{_gap_rows_html(gaps_rich)}</tbody>
</table>

<h2>2. Orthogonality check — cross-field similar slug pairs (sentinels excluded)</h2>
<p class="intro">Pairs of slugs from DIFFERENT fields ranked by cosine similarity. Legitimately close pairs are fine; duplicate-concept pairs across LLM fields suggest redundancy. <b>⚠️</b> marks pairs involving known cross-field bleed terms — they should rank high if the method works.</p>

<h3>2a. BARE embedding</h3>
<table>
  <thead><tr><th>Slug A</th><th>Slug B</th><th>sim</th><th>flag</th></tr></thead>
  <tbody>{_pair_rows_html(pairs_bare)}</tbody>
</table>

<h3>2b. RICH embedding</h3>
<table>
  <thead><tr><th>Slug A</th><th>Slug B</th><th>sim</th><th>flag</th></tr></thead>
  <tbody>{_pair_rows_html(pairs_rich)}</tbody>
</table>

</body></html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CO_API_KEY:
        raise SystemExit("CO_API_KEY required")

    df = pd.read_parquet(DATASETS_PARQUET)

    evoc_items = _load_evoc_labels()
    structured_items = _load_structured_slugs()
    hf_items = _load_hf_metadata_slugs(df)

    all_items = evoc_items + structured_items + hf_items
    n_evoc = len(evoc_items)
    n_struct = len(structured_items)
    print(f"Embedding {len(all_items)} labels "
          f"(EVoC: {n_evoc}, structured: {n_struct}, HF: {len(hf_items)}) — "
          f"twice, bare and rich...")

    vecs_bare = _embed([it["embed_text_bare"] for it in all_items])
    vecs_rich = _embed([it["embed_text_rich"] for it in all_items])
    print(f"Got embeddings shape {vecs_bare.shape} (×2)")

    non_evoc = structured_items + hf_items

    # BARE: gap + orthogonality
    sim_bare = vecs_bare @ vecs_bare.T
    gaps_bare = gap_list(evoc_items, non_evoc, sim_bare)
    sim_ss_bare = vecs_bare[n_evoc : n_evoc + n_struct] @ vecs_bare[n_evoc : n_evoc + n_struct].T
    pairs_bare = orthogonality_pairs(structured_items, sim_ss_bare)

    # RICH: gap + orthogonality
    sim_rich = vecs_rich @ vecs_rich.T
    gaps_rich = gap_list(evoc_items, non_evoc, sim_rich)
    sim_ss_rich = vecs_rich[n_evoc : n_evoc + n_struct] @ vecs_rich[n_evoc : n_evoc + n_struct].T
    pairs_rich = orthogonality_pairs(structured_items, sim_ss_rich)

    _atomic_write(
        GAP_JSON,
        json.dumps({"bare": gaps_bare, "rich": gaps_rich}, indent=2, ensure_ascii=False),
    )
    _atomic_write(
        ORTHO_JSON,
        json.dumps({"bare": pairs_bare, "rich": pairs_rich}, indent=2, ensure_ascii=False),
    )
    _atomic_write(REPORT_HTML, build_html(gaps_bare, gaps_rich, pairs_bare, pairs_rich))

    def _range(gaps):
        ds = [g["nearest_distance"] for g in gaps]
        return min(ds), max(ds), max(ds) - min(ds)

    print(f"\nDistance range BARE: min={_range(gaps_bare)[0]:.3f} max={_range(gaps_bare)[1]:.3f} range={_range(gaps_bare)[2]:.3f}")
    print(f"Distance range RICH: min={_range(gaps_rich)[0]:.3f} max={_range(gaps_rich)[1]:.3f} range={_range(gaps_rich)[2]:.3f}")

    print(f"\nTop 10 gap candidates (BARE):")
    for g in gaps_bare[:10]:
        print(f"  dist={g['nearest_distance']:.3f}  n={g['count']:3d}  {g['evoc_label']!r}")
        print(f"    → nearest: {g['nearest']}")

    print(f"\nTop 10 cross-field similar pairs (BARE, sentinels excluded):")
    for p in pairs_bare[:10]:
        flag = " ⚠️" if p["flagged"] else ""
        print(f"  sim={p['sim']:.3f}{flag}  {p['a']}  ↔  {p['b']}")

    print(f"\nWrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
