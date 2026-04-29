"""Render an interactive DataMapPlot HTML map of HuggingFace datasets."""

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import datamapplot
import glasbey
import langcodes
import numpy as np
import pandas as pd
from config import (
    DATASETS_PARQUET,
    DOCS_INDEX_HTML,
    EMBEDDINGS_NPZ,
    LABELS_PARQUET,
    MAP_HTML,
    METHODOLOGY_HTML,
    STRUCTURED_FIELDS_PARQUET,
    SUMMARIES_PARQUET,
    UMAP_COORDS_NPZ,
)
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import to_hex

FILTER_PANEL_HTML = Path(__file__).resolve().parent / "filter_panel.html"
# docs/methodology.html is the hand-authored source for the methodology page.
# _write_methodology() reads it and writes an adjusted copy to data/.
METHODOLOGY_SOURCE_HTML = Path(__file__).resolve().parent.parent / "docs" / "methodology.html"


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


_UNKNOWN_VALUES = {"", "Unknown", "Other", "Not Stated", "not_stated", "other"}


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _dim_cell(value: str) -> str:
    """Render a taxonomy/metadata value; muted gray if it's an unknown/default sentinel."""
    raw = (value or "").strip()
    v = escape(raw)
    if raw in _UNKNOWN_VALUES:
        return f'<span style="color:#a3a9b2;font-weight:normal;">{v or "—"}</span>'
    return v


def _subject_pill(value: str, hex_color: str) -> str:
    """Tinted pill for the subject_domain value, using the subject colormap's palette."""
    raw = (value or "").strip()
    v = escape(raw)
    if raw in _UNKNOWN_VALUES:
        return (
            '<span style="background:#f1f3f5;color:#a3a9b2;padding:1px 8px;'
            f'border-radius:10px;font-size:11px;">{v or "—"}</span>'
        )
    bg = _hex_to_rgba(hex_color, 0.30)
    return (
        f'<span style="background:{bg};color:#1f2328;padding:1px 8px;'
        f'border-radius:10px;font-size:11px;font-weight:500;">{v}</span>'
    )


# Fixed-width card; TL;DR dominant; one tinted pill (subject); stats row with bars; 2-col metadata.
_LBL_HALF = "64px"
_BAR = (
    '<div style="background:#eaeef2;height:2px;width:60px;margin-top:3px;border-radius:1px;">'
    '<div style="background:#b0b6be;height:100%;width:{pct}%;border-radius:1px;"></div>'
    "</div>"
)


def _meta_cell(label: str, placeholder: str) -> str:
    return (
        f'<div style="overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:500;">'
        f'<span style="display:inline-block;width:{_LBL_HALF};color:#8b949e;font-weight:normal;">'
        f"{label}</span>{{{placeholder}}}</div>"
    )


HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;width:380px;padding:8px 10px;"
    'box-sizing:border-box;color:#1f2328;">'
    # Title: org muted, name bold.
    '<div style="font-size:13px;margin-bottom:3px;overflow:hidden;text-overflow:ellipsis;'
    'white-space:nowrap;">'
    '<span style="color:#8b949e;">{repo_org}</span>'
    '<span style="font-weight:600;">{repo_name}</span>'
    "</div>"
    # Stats row: likes, downloads, size — each with label + ordinal bar.
    '<div style="font-size:11px;color:#57606a;display:flex;gap:16px;margin-bottom:10px;">'
    "<div>"
    "<div>♥ {likes}</div>" + _BAR.replace("{pct}", "{likes_pct}") + "</div>"
    "<div>"
    "<div>↓ {downloads}</div>" + _BAR.replace("{pct}", "{downloads_pct}") + "</div>"
    "<div>"
    "<div>▪ {size}</div>" + _BAR.replace("{pct}", "{size_pct}") + "</div>"
    "</div>"
    # TL;DR (dominant).
    '<div style="font-size:14px;line-height:1.45;margin-bottom:10px;">{summary}</div>'
    # Subject: standalone tinted pill (label implied by its unique visual treatment).
    '<div style="font-size:11.5px;line-height:1.4;margin-bottom:2px;">'
    "{subject_pill}"
    "</div>"
    # Remaining metadata: 2-column grid.
    '<div style="font-size:11.5px;line-height:1.9;margin-bottom:8px;'
    'display:grid;grid-template-columns:1fr 1fr;gap:0 20px;">'
    + _meta_cell("Role", "role")
    + _meta_cell("Task", "task")
    + _meta_cell("Stage", "stage")
    + _meta_cell("Modality", "modality")
    + _meta_cell("Language", "language")
    + _meta_cell("Provenance", "provenance")
    + _meta_cell("Format", "format")
    + "</div>"
    # Admin footer: license + updated.
    '<div style="font-size:11px;color:#8b949e;line-height:1.6;">'
    "⚖ {license} &nbsp;·&nbsp; 🕐 {updated}"
    "</div>"
    "</div>"
)


# Injected at render time. DataMapPlot's default scroll-zoom speed (0.01) is
# sluggish for large 2D maps; bump to 0.05 with smooth interpolation.
CUSTOM_JS = """
datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});
"""

# Title styling: bolder weight, tighter kerning, and a looser line-height so
# the title's descenders don't crash into the subtitle below (DataMapPlot's
# default line-height is 0.95, which clips descenders on a 48px display title).
CUSTOM_CSS = """
#main-title { font-weight: 700 !important; letter-spacing: -0.02em; line-height: 1.1 !important; }
"""


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
    role_full = np.array(
        ["Benchmark" if v is True else "Training data" if v is False else "Unknown" for v in df["is_benchmark"].values]
    )

    # Joined hover string for training_stage (shows all valid stages, not just primary).
    stage_hover = np.array(
        [
            ", ".join(_prettify(s) for s in _maybe_json_list(v) if s in allowed["training_stage"])
            or _prettify(_primary_stage(v, allowed["training_stage"]))
            for v in df["training_stage"].values
        ]
    )

    # Summary: LLM TL;DR (stage 04b). Rare missing rows fall back to a muted placeholder.
    summary_hover = []
    for s in df["summary"].values:
        if isinstance(s, str) and s.strip():
            summary_hover.append(escape(s))
        else:
            summary_hover.append('<span style="color:#a3a9b2;">Summary unavailable.</span>')

    # Split repo_id into muted org + bold name. Some ids lack an org prefix.
    repo_org, repo_name = [], []
    for rid in df["repo_id"].astype(str).values:
        if "/" in rid:
            o, n = rid.split("/", 1)
            repo_org.append(escape(o) + "/")
            repo_name.append(escape(n))
        else:
            repo_org.append("")
            repo_name.append(escape(rid))

    # Log-scale percentile rank for the inline popularity bars (0..100).
    likes_pct = (
        pd.Series(np.log10(np.maximum(df["likes"].fillna(0).astype(float).values, 1)))
        .rank(pct=True)
        .mul(100)
        .round(1)
        .tolist()
    )
    downloads_pct = (
        pd.Series(np.log10(np.maximum(df["downloads"].fillna(0).astype(float).values, 1)))
        .rank(pct=True)
        .mul(100)
        .round(1)
        .tolist()
    )

    # Size ordinal bar: map bin index within SIZE_CATEGORY_ORDER to 0..100.
    size_index = {label: i for i, label in enumerate(SIZE_CATEGORY_ORDER)}
    n_bins = len(SIZE_CATEGORY_ORDER)
    size_pct = [round(100 * (size_index.get(v, 0) + 1) / n_bins, 1) for v in sizes_full]

    # Subject pill: tinted with the same palette the subject colormap uses.
    subject_palette = _color_mapping(subject_full)
    subject_pill_html = [_subject_pill(v, subject_palette.get(v, "#bdbdbd")) for v in subject_full]

    # Numeric / temporal arrays for the range filters.
    likes_int = df["likes"].fillna(0).astype(int).values
    downloads_int = df["downloads"].fillna(0).astype(int).values
    created_dt = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    created_year_int = created_dt.dt.year.fillna(created_dt.dt.year.dropna().min()).astype(int).values
    now_utc = datetime.now(tz=timezone.utc)
    modified_dt = pd.to_datetime(df["last_modified"], utc=True, errors="coerce")
    days_since_modified_int = (now_utc - modified_dt).dt.days.fillna(9999).clip(lower=0).astype(int).values

    extra_data = pd.DataFrame(
        {
            "repo_id": df["repo_id"].values,
            "repo_org": repo_org,
            "repo_name": repo_name,
            "likes": [f"{int(x or 0):,}" for x in df["likes"].values],
            "downloads": [f"{int(x or 0):,}" for x in df["downloads"].values],
            "likes_pct": likes_pct,
            "downloads_pct": downloads_pct,
            "size": _esc(sizes_full),
            "size_pct": size_pct,
            "summary": summary_hover,
            "task": [_dim_cell(v) for v in tasks_full],
            "modality": [_dim_cell(v) for v in modalities_full],
            "language": [_dim_cell(v) for v in langs_full],
            "license": [_dim_cell(v) for v in licenses_full],
            "updated": [_dim_cell((str(v) or "")[:10]) for v in df["last_modified"].values],
            "subject_pill": subject_pill_html,
            "provenance": [_dim_cell(v) for v in provenance_full],
            "stage": [_dim_cell(v) for v in stage_hover],
            "format": [_dim_cell(v) for v in format_full],
            "role": [_dim_cell(v) for v in role_full],
            # Plain-text bucketed values for the filter panel — match the colormap
            # legend categories so legend-clicks and checkbox-toggles stay in sync.
            "task_filter": tasks,
            "modality_filter": modalities,
            "license_filter": licenses,
            "size_filter": sizes_full,
            "language_filter": langs,
            "subject_filter": subject_full,
            "provenance_filter": provenance_full,
            "stage_filter": stage_full,
            "format_filter": format_full,
            "role_filter": role_full,
            # Numeric / temporal values for the range filters (parsed as Number() in JS).
            "likes_filter": likes_int.astype(str),
            "downloads_filter": downloads_int.astype(str),
            "created_year_filter": created_year_int.astype(str),
            "days_since_modified_filter": days_since_modified_int.astype(str),
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
        role_full,
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
        _cat("role", "Role: Benchmark vs Training (LLM)", role_full),
    ]

    # Marker sizes scale with log10(likes), normalized to [3, 18]px. log10
    # gives better spread than sqrt on HF's compressed likes distribution
    # (top ~10K likes vs median ~30); sqrt would leave the bulk near the floor.
    likes_for_size = df["likes"].fillna(0).astype(float).values
    marker_sizes = np.log10(np.maximum(likes_for_size, 1))
    denom = (marker_sizes.max() - marker_sizes.min()) or 1.0
    marker_sizes = 3 + 15 * (marker_sizes - marker_sizes.min()) / denom

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=df["repo_id"].tolist(),
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra_data,
        on_click="window.open(`https://huggingface.co/datasets/{repo_id}`, `_blank`)",
        marker_size_array=marker_sizes,
        title="HuggingFace Dataset Map",
        sub_title=f"Top {len(df):,} datasets positioned by semantic similarity of their cards",
        enable_search=True,
        font_family="IBM Plex Sans",
        font_weight=700,
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        custom_js=CUSTOM_JS,
        custom_css=CUSTOM_CSS,
    )
    plot.save(str(MAP_HTML))
    print(f"Wrote {MAP_HTML}")

    filter_config = _build_filter_config(
        n_rows=len(df),
        tasks=tasks,
        modalities=modalities,
        licenses=licenses,
        sizes_full=sizes_full,
        langs=langs,
        subject_full=subject_full,
        provenance_full=provenance_full,
        stage_full=stage_full,
        format_full=format_full,
        role_full=role_full,
        likes_int=likes_int,
        downloads_int=downloads_int,
        created_year_int=created_year_int,
        days_since_modified_int=days_since_modified_int,
    )
    _inject_filters(MAP_HTML, filter_config)
    print(f"Injected filter panel into {MAP_HTML}")

    _inject_nav(MAP_HTML)
    print(f"Injected navigation bar into {MAP_HTML}")

    _inject_map_data_date(MAP_HTML)
    print(f"Injected data-date badge into {MAP_HTML}")

    _write_methodology(METHODOLOGY_HTML)
    print(f"Saved methodology page to {METHODOLOGY_HTML}")

    _write_methodology_docs()
    print("Updated docs/methodology.html with data date")

    _copy_for_docs(MAP_HTML, DOCS_INDEX_HTML)
    print(f"Saved docs/ copy of map to {DOCS_INDEX_HTML}")


# ── Nav bar, methodology, and data-date injection ────────────────────────────


def _data_as_of_date():
    """Return 'Month YYYY' string from DATASETS_PARQUET mtime, or None."""
    try:
        mtime = DATASETS_PARQUET.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%B %Y")
    except OSError:
        return None


def _inject_data_date(html: str) -> str:
    """Fill the data-date slot with current parquet mtime.

    Idempotent: handles both the original `<!-- DATA_AS_OF -->` placeholder and a
    previously-rendered `<p class="data-date">...</p>` paragraph, so re-running
    visualize updates the date even if the placeholder has already been replaced.
    """
    date_str = _data_as_of_date()
    if not date_str:
        return html.replace("<!-- DATA_AS_OF -->", "")
    paragraph = f'<p class="data-date">Data as of {date_str}</p>'
    if "<!-- DATA_AS_OF -->" in html:
        return html.replace("<!-- DATA_AS_OF -->", paragraph)
    return re.sub(
        r'<p\s+class="data-date">[^<]*</p>',
        paragraph,
        html,
        count=1,
    )


def _inject_map_data_date(html_path: Path) -> None:
    """Add a styled data-date badge below the subtitle in the map HTML."""
    date_str = _data_as_of_date()
    if not date_str:
        return

    badge_html = (
        '<br /><span id="data-date-badge" style="'
        "display: inline-block;"
        "margin-top: 8px;"
        "padding: 3px 10px;"
        "font-family: 'IBM Plex Sans', sans-serif;"
        "font-size: 11px;"
        "font-weight: 500;"
        "letter-spacing: 0.04em;"
        "text-transform: uppercase;"
        "color: #636c76;"
        "background: rgba(99, 108, 118, 0.08);"
        "border: 1px solid rgba(99, 108, 118, 0.15);"
        "border-radius: 12px;"
        f'">{date_str}</span>'
    )

    path = Path(html_path)
    html = path.read_text()
    html = re.sub(
        r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
        rf"\1{badge_html}\2",
        html,
        count=1,
        flags=re.DOTALL,
    )
    path.write_text(html)


def _inject_nav(html_path: Path) -> None:
    """Add the site-nav bar to the DataMapPlot-generated HTML."""
    html = Path(html_path).read_text()

    nav_css = """<style>
.site-nav{position:fixed;top:0;left:0;right:0;z-index:200;
  background:rgba(255,255,255,0.85);backdrop-filter:blur(8px);
  -webkit-backdrop-filter:blur(8px);border-bottom:1px solid #e0e0e0;
  padding:0 24px;height:44px;display:flex;align-items:center;gap:24px;
  font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:14px;font-weight:500;pointer-events:auto;}
.site-nav a{color:#333;text-decoration:none;transition:color 0.15s;}
.site-nav a:hover{color:#0d9488;}
.site-nav a.active{color:#0d9488;border-bottom:2px solid #0d9488;line-height:42px;}
</style>"""

    nav_html = """<nav class="site-nav">
  <a href="huggingface_dataset_map.html" class="active">Visualization</a>
  <a href="methodology.html">Methodology</a>
</nav>"""

    # Inject viewport meta for proper mobile rendering
    html = html.replace(
        "</head>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>',
        1,
    )

    # Inject nav bar after <body>
    html = html.replace("<body>", f"<body>{nav_css}{nav_html}", 1)

    # Offset the fixed deck-container below the nav bar
    html = html.replace(
        "position: fixed; z-index: -1; top: 0; left: 0; width: 100%; height: 100%;",
        "position: fixed; z-index: -1; top: 44px; left: 0; width: 100%; height: calc(100% - 44px);",
        1,
    )

    # Shrink body and content to account for nav bar height
    html = html.replace(
        "overflow: hidden;",
        "overflow: hidden; padding-top: 44px; height: calc(100vh - 44px);",
        1,
    )
    html = html.replace(
        "height: 100vh;",
        "height: calc(100vh - 44px);",
        1,
    )
    # Adjust content-wrapper min-height so bottom-left stays in viewport
    html = html.replace(
        "min-height:calc(100vh - 16px)",
        "min-height:calc(100vh - 60px)",
        1,
    )

    Path(html_path).write_text(html)


def _copy_for_docs(src_html_path: Path, dest_html_path: Path) -> None:
    """Copy the map HTML to docs/, rewriting nav links for GitHub Pages."""
    html = Path(src_html_path).read_text()
    html = html.replace('href="huggingface_dataset_map.html"', 'href="index.html"')
    Path(dest_html_path).write_text(html)


def _write_methodology(output_path: Path) -> None:
    """Write a data/ copy of the methodology page with nav links adjusted for local use.

    The source of truth is docs/methodology.html (uses href="index.html" for GitHub Pages).
    This function reads that source and writes a copy with href pointing at the local
    map filename.
    """
    html = METHODOLOGY_SOURCE_HTML.read_text()
    html = html.replace('href="index.html"', 'href="huggingface_dataset_map.html"')
    html = _inject_data_date(html)
    Path(output_path).write_text(html)


def _write_methodology_docs() -> None:
    """Write docs/methodology.html with the data date filled in (keeps index.html links)."""
    html = METHODOLOGY_SOURCE_HTML.read_text()
    html = _inject_data_date(html)
    METHODOLOGY_SOURCE_HTML.write_text(html)


# ── Filter panel injection ───────────────────────────────────────────────────


_TAIL_LABELS = {"Other", "Unknown", "Not Stated", "not_stated", "other"}


def _sorted_with_tail(values: np.ndarray) -> list[str]:
    """Return unique values sorted alphabetically, with Other/Unknown/Not Stated pinned to the end."""
    uniques = sorted({str(v) for v in values})
    head = [v for v in uniques if v not in _TAIL_LABELS]
    tail = [v for v in uniques if v in _TAIL_LABELS]
    # Stable order within tail by the canonical order.
    tail.sort(key=lambda v: ["Unknown", "Not Stated", "not_stated", "Other", "other"].index(v))
    return head + tail


def _ordinal_size_values(sizes_full: np.ndarray) -> list[str]:
    """Return present size buckets in canonical ordinal order, with Unknown last."""
    present = set(sizes_full.tolist())
    ordered = [v for v in SIZE_CATEGORY_ORDER if v in present]
    if "Unknown" in present:
        ordered.append("Unknown")
    return ordered


def _p99_cap(arr: np.ndarray) -> int:
    """99th-percentile cap for slider max so a long tail doesn't compress the meaningful range."""
    return int(np.percentile(arr, 99))


def _build_filter_config(
    *,
    n_rows: int,
    tasks: np.ndarray,
    modalities: np.ndarray,
    licenses: np.ndarray,
    sizes_full: np.ndarray,
    langs: np.ndarray,
    subject_full: np.ndarray,
    provenance_full: np.ndarray,
    stage_full: np.ndarray,
    format_full: np.ndarray,
    role_full: np.ndarray,
    likes_int: np.ndarray,
    downloads_int: np.ndarray,
    created_year_int: np.ndarray,
    days_since_modified_int: np.ndarray,
) -> dict:
    return {
        "totalCount": int(n_rows),
        "tasks": _sorted_with_tail(tasks),
        "modalities": _sorted_with_tail(modalities),
        "licenses": _sorted_with_tail(licenses),
        "sizes": _ordinal_size_values(sizes_full),
        "languages": _sorted_with_tail(langs),
        "subjects": _sorted_with_tail(subject_full),
        "provenances": _sorted_with_tail(provenance_full),
        "stages": _sorted_with_tail(stage_full),
        "formats": _sorted_with_tail(format_full),
        "roles": ["Training data", "Benchmark", "Unknown"],
        "ranges": {
            "created_year": {
                "min": int(created_year_int.min()),
                "max": int(created_year_int.max()),
                "sliderMax": int(created_year_int.max()),
            },
            "days_since_modified": {
                "min": 0,
                "max": int(days_since_modified_int.max()),
                "sliderMax": _p99_cap(days_since_modified_int),
            },
            "likes": {
                "min": int(likes_int.min()),
                "max": int(likes_int.max()),
                "sliderMax": _p99_cap(likes_int),
            },
            "downloads": {
                "min": int(downloads_int.min()),
                "max": int(downloads_int.max()),
                "sliderMax": _p99_cap(downloads_int),
            },
        },
        "colormapFieldToFilterId": {
            "task": "filter-task",
            "modality": "filter-modality",
            "license": "filter-license",
            "size": "filter-size",
            "language": "filter-language",
            "subject": "filter-subject",
            "provenance": "filter-provenance",
            "stage": "filter-stage",
            "format": "filter-format",
            "role": "filter-role",
        },
        "filterIdToColormapField": {
            "filter-task": "task",
            "filter-modality": "modality",
            "filter-license": "license",
            "filter-size": "size",
            "filter-language": "language",
            "filter-subject": "subject",
            "filter-provenance": "provenance",
            "filter-stage": "stage",
            "filter-format": "format",
            "filter-role": "role",
        },
    }


def _inject_filters(html_path: Path, filter_config: dict) -> None:
    """Inject the advanced-filter panel CSS/HTML/JS into a DataMapPlot HTML output."""
    html = Path(html_path).read_text()

    # 1. Build the search index client-side (avoid shipping a pre-concatenated blob)
    #    and dispatch datamapReady so the filter panel can initialize. Matches the
    #    pattern used by the sister GitHub-map project's filter injection.
    build_search_js = (
        r"// Build search index client-side from existing metadata fields\n"
        r"          (function() {\n"
        r"            var md = datamap.metaData, n = md.repo_id.length;\n"
        r"            var sa = new Array(n);\n"
        r"            for (var i = 0; i < n; i++) {\n"
        r"              sa[i] = (md.repo_id[i] + ' ' + md.summary[i] + ' '"
        r" + md.task_filter[i] + ' ' + md.modality_filter[i] + ' '"
        r" + md.language_filter[i] + ' ' + md.subject_filter[i]).toLowerCase();\n"
        r"            }\n"
        r"            datamap.searchArray = sa;\n"
        r"          })();\n"
        r"          "
    )
    html = re.sub(
        r"(updateProgressBar\('meta-data-progress', 100\);\s*)(checkAllDataLoaded\(\);)",
        r"\1" + build_search_js + r"window.dispatchEvent(new CustomEvent('datamapReady', "
        r"{ detail: { datamap, hoverData } }));\n          \2",
        html,
        count=1,
    )

    # 2. Read the template and split by section markers.
    template = FILTER_PANEL_HTML.read_text()
    sections = re.split(r"<!-- SECTION: (\w+) -->", template)
    section_map = {}
    for i in range(1, len(sections), 2):
        section_map[sections[i]] = sections[i + 1].strip()

    # 3. Inject the filter config into the JS section.
    js_section = section_map["js"].replace("__FILTER_CONFIG_JSON__", json.dumps(filter_config))

    # 4. Inject CSS before </head>.
    html = html.replace("</head>", section_map["css"] + "\n</head>", 1)

    # 5. Inject filter HTML after the search-container div.
    search_pattern = re.compile(
        r'(<div id="search-container" class="container-box[^"]*">\s*'
        r"<input[^/]*/>\s*</div>)"
    )
    match = search_pattern.search(html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + "\n      " + section_map["html"] + "\n" + html[insert_pos:]
    else:
        # Fallback: place inside .stack.top-left if search-container layout differs.
        html = html.replace(
            '<div class="stack top-left">',
            '<div class="stack top-left">\n      ' + section_map["html"],
            1,
        )

    # 6. Inject filter JS before </html>.
    html = html.replace("</html>", js_section + "\n</html>", 1)

    Path(html_path).write_text(html)


if __name__ == "__main__":
    main()
