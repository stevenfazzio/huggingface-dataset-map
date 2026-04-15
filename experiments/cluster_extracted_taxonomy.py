"""Cluster the open-ended extracted strings to propose a round-2 taxonomy.

Round 1 of extract_structured_fields_v1.py produced ~500 nearly-unique values
for provenance_method and subject_domain. This script:
  1. Embeds those strings with Cohere embed-v4.0 (input_type='clustering').
  2. Agglomerative-clusters with cosine distance (threshold tunable).
  3. Asks Claude Sonnet to propose a short name for each cluster.
  4. Writes per-field JSON + a browsable HTML report for human review.

training_stage is deliberately skipped — its vocabulary is already small
(~6 real values); the open question there is single vs multi-label, which
clustering doesn't answer.

Outputs:
    data/experiments/structured_fields_v1/taxonomy/<field>.json
    data/experiments/structured_fields_v1/taxonomy/report.html
"""

from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cohere
import numpy as np
import pandas as pd
from anthropic import Anthropic
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_TOPIC_NAMING,
    CO_API_KEY,
    EXPERIMENTS_DIR,
)

IN_PARQUET = EXPERIMENTS_DIR / "structured_fields_v1" / "extractions.parquet"
OUT_DIR = EXPERIMENTS_DIR / "structured_fields_v1" / "taxonomy"
REPORT_HTML = OUT_DIR / "report.html"

# Target number of clusters per field. Tuned to produce a reviewable-size
# taxonomy; bumped up from distance-threshold approach which gave near-singletons
# because Cohere embed-v4.0 spreads short paraphrases widely in cosine space.
FIELD_CONFIG = {
    "provenance_method": {"n_clusters": 20, "list_valued": False},
    "subject_domain": {"n_clusters": 25, "list_valued": False},
}

SENTINELS = {"not_stated", "not_applicable"}


def _flatten_values(parsed_jsons: list[dict], field: str, list_valued: bool) -> list[tuple[str, str]]:
    """Return (original_value, repo_id) pairs. Skips sentinels and empty values."""
    out = []
    for p, repo_id in parsed_jsons:
        v = p.get(field, {}).get("value")
        if v is None:
            continue
        items = v if (list_valued and isinstance(v, list)) else [v]
        for item in items:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s or s in SENTINELS:
                continue
            out.append((s, repo_id))
    return out


def _embed_strings(strings: list[str]) -> np.ndarray:
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
    # Cohere returns normalized vectors, but normalize again to be safe (cosine via euclidean).
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
    return arr


def _cluster(vecs: np.ndarray, n_clusters: int) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(vecs)


def _name_clusters(field: str, cluster_exemplars: dict[int, list[str]]) -> dict[int, str]:
    """Ask Sonnet to propose short human-readable names for each cluster at once."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    parts = [
        f"We clustered open-ended extracted values for the `{field}` field from HuggingFace dataset cards.",
        "For each cluster, propose a short (2-5 word) human-readable category name suitable for a taxonomy.",
        "Names should be: concise, parallel in form across clusters, and specific enough to distinguish clusters.",
        "",
        "Clusters (id: up to 8 example strings):",
    ]
    for cid, exemplars in sorted(cluster_exemplars.items()):
        parts.append(f"\n[{cid}]")
        for ex in exemplars[:8]:
            parts.append(f"  - {ex}")
    parts.append("\nReturn JSON: {\"<cluster_id>\": \"<name>\", ...}. No prose outside the JSON.")
    prompt = "\n".join(parts)

    resp = client.messages.create(
        model=ANTHROPIC_MODEL_TOPIC_NAMING,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    # Extract first {...} block.
    import re

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {cid: f"cluster_{cid}" for cid in cluster_exemplars}
    try:
        raw = json.loads(m.group(0))
        return {int(k): str(v) for k, v in raw.items()}
    except Exception:
        return {cid: f"cluster_{cid}" for cid in cluster_exemplars}


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(content)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def cluster_field(parsed_jsons: list[tuple[dict, str]], field: str, cfg: dict) -> dict:
    pairs = _flatten_values(parsed_jsons, field, cfg["list_valued"])
    print(f"\n── {field} ──")
    print(f"  {len(pairs)} non-sentinel values; {len(set(s for s, _ in pairs))} unique")

    # Dedupe strings for embedding — map each unique string to its occurrences.
    unique_strings = sorted({s for s, _ in pairs})
    string_to_repos: dict[str, list[str]] = {}
    for s, repo_id in pairs:
        string_to_repos.setdefault(s, []).append(repo_id)

    vecs = _embed_strings(unique_strings)
    labels = _cluster(vecs, cfg["n_clusters"])

    # Group strings by cluster, sort clusters by size desc.
    cluster_to_strings: dict[int, list[str]] = {}
    for s, lab in zip(unique_strings, labels):
        cluster_to_strings.setdefault(int(lab), []).append(s)

    cluster_sizes = {
        cid: sum(len(string_to_repos[s]) for s in members)
        for cid, members in cluster_to_strings.items()
    }
    ordered = sorted(cluster_to_strings, key=lambda c: -cluster_sizes[c])

    # Re-number clusters 0..N by size-desc for readability.
    remap = {old: new for new, old in enumerate(ordered)}
    cluster_to_strings = {remap[old]: cluster_to_strings[old] for old in ordered}
    cluster_sizes = {remap[old]: cluster_sizes[old] for old in ordered}

    # Exemplars = top few strings by occurrence count.
    exemplars = {
        cid: sorted(members, key=lambda s: -len(string_to_repos[s]))[:8]
        for cid, members in cluster_to_strings.items()
    }

    print(f"  → {len(cluster_to_strings)} clusters")
    print(f"  Top 5 cluster sizes: {[cluster_sizes[i] for i in range(min(5, len(cluster_sizes)))]}")

    print("  Naming clusters with Sonnet...")
    names = _name_clusters(field, exemplars)

    result = {
        "field": field,
        "n_clusters_requested": cfg["n_clusters"],
        "n_clusters": len(cluster_to_strings),
        "n_values": len(pairs),
        "n_unique": len(unique_strings),
        "clusters": [
            {
                "id": cid,
                "name": names.get(cid, f"cluster_{cid}"),
                "size": cluster_sizes[cid],
                "n_unique_strings": len(members),
                "exemplars": exemplars[cid],
                "all_strings": sorted(members, key=lambda s: -len(string_to_repos[s])),
            }
            for cid, members in cluster_to_strings.items()
        ],
    }

    out_path = OUT_DIR / f"{field}.json"
    _atomic_write_text(out_path, json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  → wrote {out_path}")
    return result


def build_html_report(results: list[dict]) -> None:
    sections = []
    for r in results:
        rows = []
        for c in r["clusters"]:
            examples_html = "".join(
                f"<li>{html.escape(s)}</li>" for s in c["exemplars"]
            )
            all_html = "".join(
                f"<li>{html.escape(s)}</li>" for s in c["all_strings"]
            )
            rows.append(
                f"""
                <tr>
                  <td class="id">{c['id']}</td>
                  <td class="name">{html.escape(c['name'])}</td>
                  <td class="size">{c['size']}</td>
                  <td class="nunq">{c['n_unique_strings']}</td>
                  <td class="ex"><ul>{examples_html}</ul></td>
                  <td class="all"><details><summary>{c['n_unique_strings']} strings</summary><ul>{all_html}</ul></details></td>
                </tr>"""
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(r['field'])}</h2>
              <p class="meta">
                {r['n_values']} values · {r['n_unique']} unique · {r['n_clusters']} clusters
              </p>
              <table>
                <thead><tr>
                  <th>#</th><th>Proposed name</th><th>Size</th><th>Unique</th>
                  <th>Exemplars</th><th>All members</th>
                </tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>"""
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Extracted-field taxonomy review</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto; max-width: 1400px; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 2em; color: #114; }}
  .meta {{ color: #666; font-size: 0.9em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th, td {{ text-align: left; padding: 0.5em; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #f4f4f7; position: sticky; top: 0; }}
  td.id {{ width: 3em; color: #888; }}
  td.name {{ font-weight: 600; width: 14em; }}
  td.size, td.nunq {{ width: 4em; text-align: right; font-variant-numeric: tabular-nums; }}
  td.ex ul, td.all ul {{ margin: 0; padding-left: 1.1em; }}
  td.ex li, td.all li {{ margin: 0.15em 0; }}
  details summary {{ cursor: pointer; color: #06c; }}
</style></head><body>
<h1>Extracted-field taxonomy — round 1 clusters</h1>
<p>Auto-clustered from open-ended Haiku extractions over 500 stratified cards.
Cluster names proposed by Sonnet; review and override as needed.</p>
{''.join(sections)}
</body></html>
"""
    _atomic_write_text(REPORT_HTML, html_doc)
    print(f"\nWrote {REPORT_HTML}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(IN_PARQUET)
    ok = df[df.parse_error.isna() & df.error.isna()]
    parsed_jsons = [(json.loads(row.parsed_json), row.repo_id) for _, row in ok.iterrows()]
    print(f"Loaded {len(parsed_jsons)} parsed extractions (dropped {len(df) - len(ok)})")

    results = [cluster_field(parsed_jsons, field, cfg) for field, cfg in FIELD_CONFIG.items()]
    build_html_report(results)


if __name__ == "__main__":
    main()
