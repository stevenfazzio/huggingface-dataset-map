"""Render an interactive DataMapPlot HTML map of HuggingFace datasets."""

import json
import math
import re
from html import escape

import datamapplot
import glasbey
import langcodes
import numpy as np
import pandas as pd
from config import (
    DATASETS_PARQUET,
    EMBEDDINGS_NPZ,
    LABELS_PARQUET,
    MAP_HTML,
    STRUCTURED_FIELDS_PARQUET,
    SUMMARIES_PARQUET,
    UMAP_COORDS_NPZ,
)
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import to_hex


def _first_or_other(series: pd.Series, fallback: str = "Other") -> np.ndarray:
    """Take the first comma-separated value per row, fallback when empty."""
    out = []
    for val in series.fillna("").astype(str):
        first = val.split(",")[0].strip()
        out.append(first if first else fallback)
    return np.array(out)


def _prettify(s: str) -> str:
    """Hyphens/underscores → spaces; title-case words (preserving short acronyms)."""
    if not s:
        return s
    text = s.replace("-", " ").replace("_", " ")
    words = []
    for w in text.split():
        if w.isupper() and len(w) <= 4:
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:] if w else w)
    return " ".join(words)


def _top_n_plus_other(values: np.ndarray, n: int = 9, other_label: str = "Other") -> np.ndarray:
    """Keep the top-N most-frequent values; replace the rest with `other_label`."""
    s = pd.Series(values)
    non_other = s[s != other_label]
    top = non_other.value_counts().head(n).index.tolist()
    return s.where(s.isin(top), other_label).values


def _color_mapping(values: np.ndarray, other_label: str = "Other") -> dict:
    """Glasbey palette keyed by unique value. 'Other' is pinned to a neutral gray."""
    unique = sorted(set(values.tolist()))
    non_other = [v for v in unique if v != other_label]
    palette = glasbey.create_palette(palette_size=len(non_other))
    mapping = dict(zip(non_other, palette))
    if other_label in unique:
        mapping[other_label] = "#bdbdbd"
    return mapping


# Ordinal, canonical size bins (bin label keyed by upper exclusive bound).
_SIZE_BINS = [
    (1_000, "<1K"),
    (10_000, "1K–10K"),
    (100_000, "10K–100K"),
    (1_000_000, "100K–1M"),
    (10_000_000, "1M–10M"),
    (100_000_000, "10M–100M"),
    (1_000_000_000, "100M–1B"),
    (10_000_000_000, "1B–10B"),
    (100_000_000_000, "10B–100B"),
    (1_000_000_000_000, "100B–1T"),
    (float("inf"), ">1T"),
]
SIZE_CATEGORY_ORDER = [label for _, label in _SIZE_BINS]
_SIZE_UNIT_MULT = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
_SIZE_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmbt]?)")


def _bucket_size(n: float) -> str:
    for upper, label in _SIZE_BINS:
        if n < upper:
            return label
    return ">1T"


def _friendly_language(code: str) -> str:
    """Map an ISO 639 / BCP-47 code to a readable name; fall back to the code."""
    code = (code or "").strip()
    if not code:
        return "Unknown"
    try:
        name = langcodes.Language.get(code).display_name()
        if name and "unknown language" not in name.lower() and name.lower() != code.lower():
            return name
    except Exception:
        pass
    return code


def _language_bucket(raw_csv: str, multilinguality_csv: str) -> str:
    """Return a single language label per dataset: Multilingual | Translation | <Language> | Unknown."""
    mult = [x.strip().lower() for x in (multilinguality_csv or "").split(",") if x.strip()]
    if "translation" in mult:
        return "Translation"
    if "multilingual" in mult:
        return "Multilingual"
    langs = [x.strip() for x in (raw_csv or "").split(",") if x.strip()]
    if any(x.lower() in ("multilingual", "multiple") for x in langs):
        return "Multilingual"
    if any(x.lower() in ("translation", "parallel") for x in langs):
        return "Translation"
    if not langs:
        return "Unknown"
    if "monolingual" in mult or len(langs) == 1:
        return _friendly_language(langs[0])
    # 2+ languages with no hint: Multilingual.
    return "Multilingual"


def _license_family(raw: str) -> str:
    """Collapse a raw HF license tag into a display family."""
    s = (raw or "").strip().lower()
    if not s or s in ("unknown", "other", "noassertion", "none"):
        return "Other"
    if s.startswith("apache"):
        return "Apache"
    if s == "mit" or s.startswith("mit-"):
        return "MIT"
    if s.startswith("bsd"):
        return "BSD"
    if s.startswith(("gpl", "lgpl", "agpl")):
        return "GPL Family"
    if s.startswith("mpl"):
        return "MPL"
    if s.startswith("cc0"):
        return "CC0 / Public Domain"
    if s.startswith("cc-by-nc") or "cc-by-nc" in s:
        return "CC NonCommercial"
    if s.startswith("cc-by-sa"):
        return "CC ShareAlike"
    if s.startswith("cc-by"):
        return "CC BY"
    if s.startswith(("cc-", "creativecommons")):
        return "CC (other)"
    if s.startswith(("odc", "odbl", "pddl")):
        return "ODC / ODbL"
    if s in ("isc", "unlicense", "wtfpl", "zlib", "bsl-1.0"):
        return "Other Permissive"
    if s.startswith(("openrail", "bigscience-openrail", "creativeml-openrail", "bigcode-openrail")):
        return "OpenRAIL"
    if s.startswith("llama"):
        return "Llama License"
    if s.startswith("gemma"):
        return "Gemma License"
    return "Other"


def _normalize_size(tag: str) -> str:
    """Map a raw size_categories tag into a canonical ordinal bin."""
    if not tag:
        return "Unknown"
    t = tag.lower().strip()
    matches = _SIZE_NUM_RE.findall(t)
    if not matches:
        return "Unknown"
    values = [float(num) * _SIZE_UNIT_MULT[unit] for num, unit in matches]
    # "n<X" form (upper-bounded only): push into the bucket below X.
    if re.match(r"^n\s*<", t) and "<n" not in t:
        return _bucket_size(max(values) - 1)
    # Otherwise the smallest number represents the lower bound of the range.
    return _bucket_size(min(values))


def _ordinal_color_mapping(values: np.ndarray, order: list[str], cmap_name: str = "viridis") -> dict:
    """Sequential palette for ordinal categories. Values outside `order` get gray."""
    cmap = mpl_colormaps[cmap_name]
    present_in_order = [v for v in order if v in set(values.tolist())]
    n = max(len(present_in_order), 1)
    mapping = {v: to_hex(cmap(i / max(n - 1, 1))) for i, v in enumerate(present_in_order)}
    for v in set(values.tolist()):
        mapping.setdefault(v, "#bdbdbd")
    return mapping


def _esc(values) -> list[str]:
    return [escape(str(v)) for v in values]


def _maybe_json_list(v) -> list:
    """Parse a JSON list stored as a string in the parquet, or pass through a real list."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def _primary_stage(raw, allowed: set[str]) -> str:
    """Pick a single training-stage label per row with a priority order. Invalid slugs coerce to 'other'."""
    items = [v for v in _maybe_json_list(raw) if v in allowed]
    priority = ["preference", "sft", "eval", "domain-finetune", "pretraining", "raw-corpus", "other", "not_stated"]
    for p in priority:
        if p in items:
            return p
    return items[0] if items else "other"


def _slug_or(raw, allowed: set[str], fallback: str = "not_stated") -> str:
    """Pass-through if the value is in the allowed enum; else fall back (default 'not_stated')."""
    if isinstance(raw, str) and raw in allowed:
        return raw
    return fallback


def _load_allowed_slugs() -> dict[str, set[str]]:
    from config import TAXONOMY_JSON

    tax = json.loads(TAXONOMY_JSON.read_text())
    return {
        f: {c["name"] for c in spec.get("categories", [])}
        for f, spec in tax.items()
        if not f.startswith("_") and "categories" in spec
    }


HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;max-width:380px;padding:6px 2px;\">"
    '<div style="font-weight:600;font-size:13px;color:#1f2328;margin-bottom:4px;">{repo_id}</div>'
    '<div style="font-size:11.5px;color:#57606a;margin-bottom:6px;">♥ {likes} &nbsp;·&nbsp; ↓ {downloads}</div>'
    '<div style="font-size:12px;color:#1f2328;line-height:1.4;margin-bottom:8px;font-style:italic;">{summary}</div>'
    '<div style="font-size:11.5px;color:#3d4752;line-height:1.45;">'
    "<div><b>task:</b> {task} &nbsp;·&nbsp; <b>modality:</b> {modality}</div>"
    "<div><b>language:</b> {language} &nbsp;·&nbsp; <b>size:</b> {size}</div>"
    "<div><b>license:</b> {license} &nbsp;·&nbsp; <b>updated:</b> {updated}</div>"
    '<div style="margin-top:4px;border-top:1px solid #eaeef2;padding-top:4px;">'
    "<div><b>subject:</b> {subject} &nbsp;·&nbsp; <b>stage:</b> {stage}</div>"
    "<div><b>provenance:</b> {provenance} &nbsp;·&nbsp; <b>format:</b> {format}</div>"
    "<div><b>benchmark:</b> {benchmark}</div>"
    "</div>"
    "</div>"
    "</div>"
)


def main():
    df = pd.read_parquet(DATASETS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    _ = np.load(EMBEDDINGS_NPZ)["embeddings"]  # reserved for future edge bundling
    labels = pd.read_parquet(LABELS_PARQUET)
    structured = pd.read_parquet(STRUCTURED_FIELDS_PARQUET)
    summaries = pd.read_parquet(SUMMARIES_PARQUET)

    df = df.merge(labels, on="repo_id", how="left")
    structured_cols = [
        "repo_id",
        "provenance_method",
        "subject_domain",
        "training_stage",
        "format_convention",
        "is_benchmark",
    ]
    df = df.merge(structured[structured_cols], on="repo_id", how="left")
    df = df.merge(summaries[["repo_id", "summary"]], on="repo_id", how="left")

    label_cols = sorted(c for c in df.columns if c.startswith("label_layer_"))
    label_layers = [df[c].fillna("Unlabeled").astype(str).values for c in label_cols]

    # Per-dataset pretty values (before top-N bucketing) — used in hover cards.
    tasks_full = np.array([_prettify(v) for v in _first_or_other(df["task_categories"])])
    modalities_full = np.array([_prettify(v) for v in _first_or_other(df["modalities"])])
    licenses_full = np.array([_license_family(x) for x in _first_or_other(df["license"], fallback="")])
    sizes_full = np.array([_normalize_size(s) for s in _first_or_other(df["size_categories"], fallback="")])
    langs_full = np.array(
        [_language_bucket(r, m) for r, m in zip(df["languages"].fillna(""), df["multilinguality"].fillna(""))]
    )

    # Bucketed versions — used for colormaps/legend so the tail folds into "Other".
    tasks = _top_n_plus_other(tasks_full)
    modalities = _top_n_plus_other(modalities_full)
    licenses = _top_n_plus_other(licenses_full)
    sizes = sizes_full  # already ordinal, no bucketing
    langs = _top_n_plus_other(langs_full)

    # LLM-extracted structured fields. Coerce invalid slugs to 'other' so validation
    # leakage (~3% of rows have out-of-enum values from cross-field bleed) doesn't
    # produce phantom legend entries. Truth is preserved in the parquet.
    allowed = _load_allowed_slugs()
    subject_full = np.array(
        [_prettify(_slug_or(v, allowed["subject_domain"], "other")) for v in df["subject_domain"].values]
    )
    provenance_full = np.array(
        [_prettify(_slug_or(v, allowed["provenance_method"], "other")) for v in df["provenance_method"].values]
    )
    stage_full = np.array(
        [_prettify(_primary_stage(v, allowed["training_stage"])) for v in df["training_stage"].values]
    )
    format_full = np.array(
        [_prettify(_slug_or(v, allowed["format_convention"], "other")) for v in df["format_convention"].values]
    )
    benchmark_full = np.array(
        ["Benchmark" if v is True else "Training" if v is False else "Unknown" for v in df["is_benchmark"].values]
    )

    # Joined hover string for training_stage (shows all valid stages, not just primary).
    stage_hover = np.array(
        [
            ", ".join(_prettify(s) for s in _maybe_json_list(v) if s in allowed["training_stage"])
            or _prettify(_primary_stage(v, allowed["training_stage"]))
            for v in df["training_stage"].values
        ]
    )

    # Summary: LLM TL;DR (stage 04b). Rare missing rows fall back to blank.
    summary_hover = [(s if isinstance(s, str) and s else "") for s in df["summary"].values]

    extra_data = pd.DataFrame(
        {
            "repo_id": df["repo_id"].values,
            "likes": [f"{int(x or 0):,}" for x in df["likes"].values],
            "downloads": [f"{int(x or 0):,}" for x in df["downloads"].values],
            "summary": _esc(summary_hover),
            "task": _esc(tasks_full),
            "modality": _esc(modalities_full),
            "language": _esc(langs_full),
            "size": _esc(sizes_full),
            "license": _esc(licenses_full),
            "updated": [(str(v) or "")[:10] for v in df["last_modified"].values],
            "subject": _esc(subject_full),
            "provenance": _esc(provenance_full),
            "stage": _esc(stage_hover),
            "format": _esc(format_full),
            "benchmark": _esc(benchmark_full),
        }
    )

    rawdata = [
        tasks,
        modalities,
        licenses,
        sizes,
        langs,
        np.log10(np.maximum(df["likes"].fillna(0).astype(float).values, 1)),
        np.log10(np.maximum(df["downloads"].fillna(0).astype(float).values, 1)),
        subject_full,
        provenance_full,
        stage_full,
        format_full,
        benchmark_full,
    ]

    def _cat(field, desc, values):
        return {
            "field": field,
            "description": desc,
            "kind": "categorical",
            "color_mapping": _color_mapping(values),
            "show_legend": True,
        }

    metadata = [
        _cat("task", "Task Category", tasks),
        _cat("modality", "Modality", modalities),
        _cat("license", "License", licenses),
        {
            "field": "size",
            "description": "Size Category",
            "kind": "categorical",
            "color_mapping": _ordinal_color_mapping(sizes, SIZE_CATEGORY_ORDER),
            "show_legend": True,
        },
        _cat("language", "Language", langs),
        {"field": "likes", "description": "Likes (log10)", "kind": "continuous", "cmap": "YlOrRd"},
        {"field": "downloads", "description": "Downloads (log10)", "kind": "continuous", "cmap": "viridis"},
        _cat("subject", "Subject Domain (LLM)", subject_full),
        _cat("provenance", "Provenance (LLM)", provenance_full),
        _cat("stage", "Training Stage (LLM)", stage_full),
        _cat("format", "Format Convention (LLM)", format_full),
        _cat("benchmark", "Benchmark vs Training (LLM)", benchmark_full),
    ]

    marker_size = max(3.0, min(10.0, 400.0 / math.sqrt(len(df))))

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=df["repo_id"].tolist(),
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra_data,
        on_click="window.open(`https://huggingface.co/datasets/{repo_id}`, `_blank`)",
        marker_size_array=np.full(len(df), marker_size),
        title="HuggingFace Dataset Map",
        sub_title=f"Top {len(df):,} datasets positioned by semantic similarity of their cards",
        enable_search=True,
        font_family="IBM Plex Sans",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
    )
    plot.save(str(MAP_HTML))
    print(f"Wrote {MAP_HTML}")


if __name__ == "__main__":
    main()
