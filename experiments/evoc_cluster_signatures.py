"""Compute per-EVoC-cluster signatures across LLM + HF structured fields.

For each cluster at each layer, report the distribution of values within the
cluster across all structured fields. Two goals:

(1) Sanity-check cross-references. If Haiku named a cluster "Medical QA" but
    only 40% of its members have subject_domain=medical-and-biomedical, either
    the clustering drifted semantically or the LLM extraction missed. Either
    way, worth surfacing.

(2) Surface correlations the individual fields don't show. E.g. does
    safety-alignment subject cluster by provenance? Do DPO preference datasets
    share a format convention? Does long-context correlate with domain?

Output:
    data/experiments/evoc_cluster_signatures/signatures.parquet  (long-format)
    data/experiments/evoc_cluster_signatures/report.html        (browsable)
"""

from __future__ import annotations

import html
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (  # noqa: E402
    DATASETS_PARQUET,
    EXPERIMENTS_DIR,
    STRUCTURED_FIELDS_PARQUET,
)

EVOC_LAYERS_NPZ = EXPERIMENTS_DIR / "evoc_taxonomy" / "cluster_layers.npz"
EVOC_TOPICS_JSON = EXPERIMENTS_DIR / "evoc_taxonomy" / "topic_names.json"
OUT_DIR = EXPERIMENTS_DIR / "evoc_cluster_signatures"
SIGNATURES_PARQUET = OUT_DIR / "signatures.parquet"
REPORT_HTML = OUT_DIR / "report.html"

# Fields to profile. Multi-valued fields get JSON-parsed before counting.
SINGLE_VAL_FIELDS = [
    "provenance_method",
    "subject_domain",
    "format_convention",
    "is_benchmark",
]
MULTI_VAL_FIELDS = [
    "training_stage",
    "special_characteristics",
    "geo_scope",
    "upstream_models",
]
HF_SINGLE_VAL_FIELDS = [
    "first_task",
    "first_modality",
    "license_family",
]


def _first_or_none(csv: str) -> str | None:
    if not isinstance(csv, str) or not csv:
        return None
    first = csv.split(",")[0].strip()
    return first or None


def _license_family(raw: str) -> str:
    """Same collapse logic as the visualize stage, trimmed."""
    s = (raw or "").strip().lower()
    if not s or s in ("unknown", "other", "noassertion", "none"):
        return "other"
    if s.startswith("apache"):
        return "apache"
    if s == "mit" or s.startswith("mit-"):
        return "mit"
    if s.startswith("bsd"):
        return "bsd"
    if s.startswith(("gpl", "lgpl", "agpl")):
        return "gpl-family"
    if s.startswith("cc0"):
        return "cc0"
    if s.startswith("cc-by-nc"):
        return "cc-nc"
    if s.startswith("cc-by-sa"):
        return "cc-sa"
    if s.startswith("cc-by"):
        return "cc-by"
    if s.startswith(("cc-", "creativecommons")):
        return "cc-other"
    if s.startswith(("openrail", "bigscience-openrail", "creativeml-openrail", "bigcode-openrail")):
        return "openrail"
    if s.startswith("llama"):
        return "llama"
    if s.startswith("gemma"):
        return "gemma"
    return "other"


def _parse_list(v) -> list[str]:
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    if isinstance(v, str) and v.startswith("["):
        try:
            parsed = json.loads(v)
            return [x for x in parsed if isinstance(x, str)] if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def build_joined_df() -> pd.DataFrame:
    datasets = pd.read_parquet(DATASETS_PARQUET)
    structured = pd.read_parquet(STRUCTURED_FIELDS_PARQUET)
    df = datasets.merge(structured, on="repo_id", how="left")

    df["first_task"] = df["task_categories"].map(_first_or_none)
    df["first_modality"] = df["modalities"].map(_first_or_none)
    df["license_family"] = df["license"].fillna("").map(_license_family)
    return df


def load_cluster_layers() -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    layers = dict(np.load(EVOC_LAYERS_NPZ).items())
    topics = json.loads(EVOC_TOPICS_JSON.read_text())
    return layers, topics


def top_values(values, k: int = 3) -> list[tuple[str, int, float]]:
    """Return top-k (value, count, pct) tuples, excluding None/NaN."""
    clean = [v for v in values if v is not None and v == v]  # drops NaN too
    n_total = len(clean)
    if n_total == 0:
        return []
    counter = Counter(clean)
    return [(str(v), c, round(c / n_total * 100, 1)) for v, c in counter.most_common(k)]


def top_list_values(values: list[list[str]], k: int = 4) -> list[tuple[str, int, float]]:
    """For multi-valued fields: count each element, report top-k by frequency."""
    counter: Counter = Counter()
    n_rows_with_any = 0
    for lst in values:
        if lst:
            n_rows_with_any += 1
            counter.update(lst)
    if n_rows_with_any == 0:
        return []
    return [(v, c, round(c / n_rows_with_any * 100, 1)) for v, c in counter.most_common(k)]


def build_signatures(df: pd.DataFrame, layers: dict, topics: dict) -> pd.DataFrame:
    rows = []
    n_total_corpus = len(df)

    for layer_key in sorted(layers.keys()):  # layer_0 (finest), layer_1, layer_2
        labels = layers[layer_key]
        layer_topics = topics.get(layer_key, [])
        if len(labels) != len(df):
            raise ValueError(f"{layer_key}: {len(labels)} labels vs {len(df)} rows")

        # Unique cluster IDs — include -1 (noise) as its own bucket.
        unique_ids = sorted(set(labels.tolist()))
        for cid in unique_ids:
            mask = labels == cid
            n = int(mask.sum())
            if cid == -1:
                name = "__noise__"
            else:
                name = layer_topics[cid] if cid < len(layer_topics) else f"cluster_{cid}"

            row = {
                "layer": layer_key,
                "cluster_id": int(cid),
                "name": name,
                "size": n,
                "pct_of_corpus": round(n / n_total_corpus * 100, 1),
            }

            sub = df.loc[mask]

            # Single-valued fields — top 3 values.
            for f in SINGLE_VAL_FIELDS + HF_SINGLE_VAL_FIELDS:
                row[f"{f}_top"] = top_values(sub[f].tolist(), k=3)

            # Multi-valued fields — top 4 values across all list elements.
            for f in MULTI_VAL_FIELDS:
                parsed = [_parse_list(v) for v in sub[f].tolist()]
                row[f"{f}_top"] = top_list_values(parsed, k=4)

            # Popularity summary.
            row["likes_median"] = int(sub["likes"].fillna(0).median()) if n else 0
            row["downloads_median"] = int(sub["downloads"].fillna(0).median()) if n else 0

            rows.append(row)

    return pd.DataFrame(rows)


def flag_mismatches(sigs: pd.DataFrame) -> pd.DataFrame:
    """Add a per-cluster flag if the name's apparent subject doesn't align with the dominant subject_domain.

    Simple heuristic: does the cluster name contain any keyword from the top
    subject_domain slug (e.g. name="Medical QA Datasets" contains "Medical",
    subject_domain_top=[medical-and-biomedical, ...])?
    """
    flags = []
    for _, row in sigs.iterrows():
        if row["cluster_id"] == -1:
            flags.append("")
            continue
        name_lower = row["name"].lower()
        subj_top = row["subject_domain_top"]
        if not subj_top:
            flags.append("no-subject")
            continue
        top_subj, top_n, top_pct = subj_top[0]
        # Keywords from top subject slug (minus stopwords).
        subj_words = [w for w in top_subj.replace("-", " ").split() if len(w) > 3 and w not in {"and"}]
        hit = any(w in name_lower for w in subj_words)
        flags.append("" if hit or top_pct >= 50 else "⚠ name/subject mismatch")
    sigs = sigs.copy()
    sigs["mismatch_flag"] = flags
    return sigs


def _fmt_top(top_list: list) -> str:
    if not top_list:
        return '<span class="muted">—</span>'
    parts = []
    for v, c, pct in top_list:
        parts.append(f'<span class="val">{html.escape(str(v))}</span> <span class="pct">{pct}%</span>')
    return " · ".join(parts)


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


def build_html(sigs: pd.DataFrame) -> str:
    sigs_named = sigs[sigs["cluster_id"] != -1].sort_values(
        ["layer", "size"], ascending=[True, False]
    )
    sigs_noise = sigs[sigs["cluster_id"] == -1].sort_values("layer")

    sections = []
    for layer_key in sorted(sigs["layer"].unique()):
        layer_num = int(layer_key.split("_")[1])
        layer_clusters = sigs_named[sigs_named["layer"] == layer_key]
        layer_noise = sigs_noise[sigs_noise["layer"] == layer_key]
        sections.append(
            f'<h2>Layer {layer_num} — {len(layer_clusters)} named clusters'
            f'{" + noise" if len(layer_noise) else ""}</h2>'
        )

        for _, row in layer_clusters.iterrows():
            flag = row["mismatch_flag"]
            flag_html = f' <span class="flag">{flag}</span>' if flag else ""
            sections.append(f'''
<section class="cluster">
  <h3>{html.escape(row["name"])}{flag_html}</h3>
  <p class="meta">
    id={row["cluster_id"]} · <b>{row["size"]}</b> datasets ({row["pct_of_corpus"]}% of corpus)
    · median likes: {row["likes_median"]} · median downloads: {row["downloads_median"]}
  </p>
  <dl>
    <dt>subject</dt><dd>{_fmt_top(row["subject_domain_top"])}</dd>
    <dt>provenance</dt><dd>{_fmt_top(row["provenance_method_top"])}</dd>
    <dt>training_stage</dt><dd>{_fmt_top(row["training_stage_top"])}</dd>
    <dt>format_convention</dt><dd>{_fmt_top(row["format_convention_top"])}</dd>
    <dt>special_characteristics</dt><dd>{_fmt_top(row["special_characteristics_top"])}</dd>
    <dt>is_benchmark</dt><dd>{_fmt_top(row["is_benchmark_top"])}</dd>
    <dt>geo_scope</dt><dd>{_fmt_top(row["geo_scope_top"])}</dd>
    <dt>upstream_models</dt><dd>{_fmt_top(row["upstream_models_top"])}</dd>
    <dt class="hf">hf_task</dt><dd class="hf">{_fmt_top(row["first_task_top"])}</dd>
    <dt class="hf">hf_modality</dt><dd class="hf">{_fmt_top(row["first_modality_top"])}</dd>
    <dt class="hf">hf_license</dt><dd class="hf">{_fmt_top(row["license_family_top"])}</dd>
  </dl>
</section>''')

        for _, row in layer_noise.iterrows():
            sections.append(f'''
<section class="cluster noise">
  <h3>__noise__ (layer {layer_num})</h3>
  <p class="meta">{row["size"]} datasets ({row["pct_of_corpus"]}% of corpus) — unclustered, for reference</p>
  <dl>
    <dt>subject</dt><dd>{_fmt_top(row["subject_domain_top"])}</dd>
    <dt>provenance</dt><dd>{_fmt_top(row["provenance_method_top"])}</dd>
    <dt>training_stage</dt><dd>{_fmt_top(row["training_stage_top"])}</dd>
  </dl>
</section>''')

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EVoC cluster signatures</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto; max-width: 1200px; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 2.2em; padding-bottom: 0.3em; border-bottom: 1px solid #ccc; color: #114; }}
  h3 {{ margin: 0 0 0.3em 0; font-size: 1.05em; color: #223; }}
  section.cluster {{ border-left: 3px solid #d0d7de; padding: 0.6em 0 0.6em 1em; margin: 0.8em 0; background: #fafbfc; }}
  section.cluster.noise {{ border-left-color: #ffc107; background: #fffcf2; }}
  p.meta {{ color: #57606a; margin: 0.1em 0 0.6em 0; font-size: 0.9em; }}
  dl {{ display: grid; grid-template-columns: 11em 1fr; gap: 0.25em 1em; margin: 0; font-size: 0.9em; }}
  dt {{ font-weight: 600; color: #57606a; text-transform: none; }}
  dd {{ margin: 0; color: #1f2328; }}
  dl dt.hf, dl dd.hf {{ color: #7a8590; font-style: italic; }}
  .val {{ font-weight: 500; }}
  .pct {{ color: #6e7781; font-variant-numeric: tabular-nums; }}
  .muted {{ color: #999; }}
  .flag {{ font-size: 0.75em; color: #c9480a; background: #fff4cc; padding: 0.1em 0.4em; border-radius: 3px; margin-left: 0.5em; }}
</style></head><body>

<h1>EVoC cluster signatures</h1>
<p>Each cluster (from raw-embedding EVoC clustering + Haiku naming) profiled against the LLM-extracted structured fields and HF metadata. Looking for: (a) named clusters whose dominant fields don't match the name — candidates for mis-labeled extractions or semantic drift; (b) unexpected correlations that the fields alone don't show. HF metadata rows (greyed) are included as reference; the LLM fields are the primary view.</p>

{"".join(sections)}

</body></html>
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_joined_df()
    layers, topics = load_cluster_layers()
    print(f"Loaded {len(df)} datasets; {len(layers)} cluster layers")

    sigs = build_signatures(df, layers, topics)
    sigs = flag_mismatches(sigs)

    # Save parquet — but convert list-of-tuples columns to JSON strings for portability.
    sigs_to_save = sigs.copy()
    list_cols = [c for c in sigs.columns if c.endswith("_top")]
    for c in list_cols:
        sigs_to_save[c] = sigs_to_save[c].map(lambda v: json.dumps(v, ensure_ascii=False))

    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        sigs_to_save.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(sigs_to_save)
        os.replace(tmp_path, SIGNATURES_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"Wrote {SIGNATURES_PARQUET}")

    _atomic_write(REPORT_HTML, build_html(sigs))
    print(f"Wrote {REPORT_HTML}")

    n_flagged = (sigs["mismatch_flag"] != "").sum()
    print(f"\n{n_flagged} named clusters flagged for name↔subject mismatch.")
    for _, row in sigs[sigs["mismatch_flag"] != ""].iterrows():
        if row["cluster_id"] == -1:
            continue
        top_subj = row["subject_domain_top"][0] if row["subject_domain_top"] else ("—", 0, 0)
        print(
            f"  [{row['layer']} / id={row['cluster_id']}]  "
            f"{row['name']!r}  →  top subject: {top_subj[0]} ({top_subj[2]}%)"
        )


if __name__ == "__main__":
    main()
