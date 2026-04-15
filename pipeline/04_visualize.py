"""Render an interactive DataMapPlot HTML map of HuggingFace datasets."""

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


HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;max-width:360px;padding:6px 2px;\">"
    '<div style="font-weight:600;font-size:13px;color:#1f2328;margin-bottom:4px;">{repo_id}</div>'
    '<div style="font-size:11.5px;color:#57606a;margin-bottom:6px;">♥ {likes} &nbsp;·&nbsp; ↓ {downloads}</div>'
    '<div style="font-size:11.5px;color:#3d4752;line-height:1.45;">'
    "<div><b>task:</b> {task}</div>"
    "<div><b>modality:</b> {modality}</div>"
    "<div><b>language:</b> {language}</div>"
    "<div><b>size:</b> {size}</div>"
    "<div><b>license:</b> {license}</div>"
    "<div><b>updated:</b> {updated}</div>"
    "</div>"
    "</div>"
)


def main():
    df = pd.read_parquet(DATASETS_PARQUET)
    coords = np.load(UMAP_COORDS_NPZ)["coords"]
    _ = np.load(EMBEDDINGS_NPZ)["embeddings"]  # reserved for future edge bundling
    labels = pd.read_parquet(LABELS_PARQUET)

    df = df.merge(labels, on="repo_id", how="left")

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

    extra_data = pd.DataFrame(
        {
            "repo_id": df["repo_id"].values,
            "likes": [f"{int(x or 0):,}" for x in df["likes"].values],
            "downloads": [f"{int(x or 0):,}" for x in df["downloads"].values],
            "task": _esc(tasks_full),
            "modality": _esc(modalities_full),
            "language": _esc(langs_full),
            "size": _esc(sizes_full),
            "license": _esc(licenses_full),
            "updated": [(str(v) or "")[:10] for v in df["last_modified"].values],
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
