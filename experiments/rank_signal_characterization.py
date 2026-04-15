"""Characterize what kinds of repos likes vs downloads ranking biases toward.

Refetches the top-10K under each sort order with full metadata, then compares
the symmetric difference at top-N (default 1000) across categorical and numeric
axes: task_categories, modalities, languages, size_categories, license, author,
gated, created_at age, card presence.

For each categorical axis it reports the most distinctive feature values for
each side (top by lift = rate_likes_only / rate_downloads_only, with smoothing).

Outputs:
    data/experiments/top10k_by_likes_full.parquet
    data/experiments/top10k_by_downloads_full.parquet
    data/experiments/rank_signal_characterization.html
    data/experiments/rank_signal_characterization.json

Re-run with --refresh to refetch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import EXPERIMENTS_DIR, HF_TOKEN  # noqa: E402

# Reuse the same parsing logic as stage 00.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import importlib.util  # noqa: E402

_stage00_spec = importlib.util.spec_from_file_location(
    "stage00",
    Path(__file__).resolve().parent.parent / "pipeline" / "00_fetch_datasets.py",
)
_stage00 = importlib.util.module_from_spec(_stage00_spec)
_stage00_spec.loader.exec_module(_stage00)
_parse_dataset_info = _stage00._parse_dataset_info

FETCH_LIMIT = 10_000
DEFAULT_TOP_N = 1_000
LIKES_FULL = EXPERIMENTS_DIR / "top10k_by_likes_full.parquet"
DOWNLOADS_FULL = EXPERIMENTS_DIR / "top10k_by_downloads_full.parquet"
REPORT_HTML = EXPERIMENTS_DIR / "rank_signal_characterization.html"
REPORT_JSON = EXPERIMENTS_DIR / "rank_signal_characterization.json"

LIST_COL_AXES = ["task_categories", "modalities", "languages", "size_categories"]
SCALAR_COL_AXES = ["license", "author", "gated"]
SMOOTHING = 0.5


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        verify = pd.read_parquet(tmp)
        assert len(verify) == len(df), f"row count mismatch writing {path}"
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def _fetch_full(sort_key: str, limit: int) -> pd.DataFrame:
    api = HfApi(token=HF_TOKEN or None)
    print(f"Fetching top {limit} by {sort_key} (full metadata)...")
    rows = []
    it = api.list_datasets(sort=sort_key, limit=limit, full=True)
    for info in tqdm(it, total=limit, desc=sort_key):
        rows.append(_parse_dataset_info(info))
    df = pd.DataFrame(rows)
    return df


def _load_or_fetch(path: Path, sort_key: str, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        print(f"Loaded cached {path.name} ({len(df)} rows)")
        return df
    df = _fetch_full(sort_key, FETCH_LIMIT)
    _atomic_write_parquet(df, path)
    print(f"Wrote {path}")
    return df


def _split_csv(s: str) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    return [x for x in s.split(",") if x]


def _value_counts_list_col(df: pd.DataFrame, col: str) -> Counter:
    c: Counter = Counter()
    for v in df[col]:
        for item in _split_csv(v):
            c[item] += 1
    return c


def _value_counts_scalar(df: pd.DataFrame, col: str) -> Counter:
    return Counter(str(v) if v else "(none)" for v in df[col])


def _compare_axis(counts_a: Counter, counts_b: Counter, n_a: int, n_b: int, min_count: int = 10) -> pd.DataFrame:
    """Build a per-value comparison: rate on each side + log-lift."""
    keys = set(counts_a) | set(counts_b)
    rows = []
    for k in keys:
        a = counts_a.get(k, 0)
        b = counts_b.get(k, 0)
        if a + b < min_count:
            continue
        rate_a = (a + SMOOTHING) / (n_a + 2 * SMOOTHING)
        rate_b = (b + SMOOTHING) / (n_b + 2 * SMOOTHING)
        rows.append(
            {
                "value": k,
                "n_likes_only": a,
                "n_downloads_only": b,
                "rate_likes_only": rate_a,
                "rate_downloads_only": rate_b,
                "lift_likes_over_downloads": rate_a / rate_b,
            }
        )
    return pd.DataFrame(rows).sort_values("lift_likes_over_downloads", ascending=False)


def _bucket_age_days(df: pd.DataFrame, ref: pd.Timestamp) -> pd.Series:
    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    return (ref - created).dt.days


def _section_html(title: str, df: pd.DataFrame, top_k: int = 15) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p><em>(no values met min-count threshold)</em></p>"
    likes_side = df.head(top_k)
    dl_side = df.tail(top_k).iloc[::-1]

    def _table(rows: pd.DataFrame, side: str) -> str:
        out = [f"<h4>{side}</h4>"]
        out.append(
            "<table border=1 cellpadding=4><tr>"
            "<th>value</th><th>likes-only</th><th>downloads-only</th>"
            "<th>rate (L)</th><th>rate (D)</th><th>lift L/D</th></tr>"
        )
        for _, r in rows.iterrows():
            out.append(
                f"<tr><td>{r['value']}</td>"
                f"<td>{r['n_likes_only']}</td><td>{r['n_downloads_only']}</td>"
                f"<td>{r['rate_likes_only']:.3f}</td>"
                f"<td>{r['rate_downloads_only']:.3f}</td>"
                f"<td>{r['lift_likes_over_downloads']:.2f}</td></tr>"
            )
        out.append("</table>")
        return "".join(out)

    return (
        f"<h3>{title}</h3>"
        "<div style='display:flex;gap:2em;flex-wrap:wrap'>"
        f"<div>{_table(likes_side, 'Over-represented in likes-only')}</div>"
        f"<div>{_table(dl_side, 'Over-represented in downloads-only')}</div>"
        "</div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    args = parser.parse_args()

    likes_df = _load_or_fetch(LIKES_FULL, "likes", args.refresh)
    dl_df = _load_or_fetch(DOWNLOADS_FULL, "downloads", args.refresh)

    n = args.top_n
    likes_top = likes_df.head(n)
    dl_top = dl_df.head(n)
    likes_ids = set(likes_top["repo_id"])
    dl_ids = set(dl_top["repo_id"])
    likes_only = likes_top[~likes_top["repo_id"].isin(dl_ids)].copy()
    dl_only = dl_top[~dl_top["repo_id"].isin(likes_ids)].copy()
    overlap = likes_top[likes_top["repo_id"].isin(dl_ids)].copy()

    print(f"\nAt N={n}:  likes_only={len(likes_only)}  downloads_only={len(dl_only)}  both={len(overlap)}")

    n_a = len(likes_only)
    n_b = len(dl_only)

    sections: list[str] = []
    json_axes: dict[str, list[dict]] = {}

    # Categorical axes.
    for col in LIST_COL_AXES:
        ca = _value_counts_list_col(likes_only, col)
        cb = _value_counts_list_col(dl_only, col)
        cmp_df = _compare_axis(ca, cb, n_a, n_b)
        sections.append(_section_html(f"{col} (multi-valued)", cmp_df))
        json_axes[col] = cmp_df.to_dict(orient="records")
    for col in SCALAR_COL_AXES:
        ca = _value_counts_scalar(likes_only, col)
        cb = _value_counts_scalar(dl_only, col)
        cmp_df = _compare_axis(ca, cb, n_a, n_b)
        sections.append(_section_html(f"{col}", cmp_df))
        json_axes[col] = cmp_df.to_dict(orient="records")

    # Age (created_at) and card length, as numeric summaries.
    ref = pd.Timestamp.utcnow()
    likes_age = _bucket_age_days(likes_only, ref).dropna()
    dl_age = _bucket_age_days(dl_only, ref).dropna()
    age_summary = {
        "likes_only_age_days": {
            "median": float(likes_age.median()),
            "p25": float(likes_age.quantile(0.25)),
            "p75": float(likes_age.quantile(0.75)),
            "n": int(len(likes_age)),
        },
        "downloads_only_age_days": {
            "median": float(dl_age.median()),
            "p25": float(dl_age.quantile(0.25)),
            "p75": float(dl_age.quantile(0.75)),
            "n": int(len(dl_age)),
        },
    }
    sections.append(
        "<h3>Age (days since created_at)</h3><table border=1 cellpadding=4>"
        "<tr><th>side</th><th>median</th><th>p25</th><th>p75</th><th>n</th></tr>"
        f"<tr><td>likes-only</td><td>{age_summary['likes_only_age_days']['median']:.0f}</td>"
        f"<td>{age_summary['likes_only_age_days']['p25']:.0f}</td>"
        f"<td>{age_summary['likes_only_age_days']['p75']:.0f}</td>"
        f"<td>{age_summary['likes_only_age_days']['n']}</td></tr>"
        f"<tr><td>downloads-only</td><td>{age_summary['downloads_only_age_days']['median']:.0f}</td>"
        f"<td>{age_summary['downloads_only_age_days']['p25']:.0f}</td>"
        f"<td>{age_summary['downloads_only_age_days']['p75']:.0f}</td>"
        f"<td>{age_summary['downloads_only_age_days']['n']}</td></tr>"
        "</table>"
    )

    # Headline numbers.
    headline = {
        "top_n": n,
        "n_likes_only": n_a,
        "n_downloads_only": n_b,
        "n_overlap": len(overlap),
        "median_likes_likes_only": int(likes_only["likes"].median()),
        "median_likes_downloads_only": int(dl_only["likes"].median()),
        "median_downloads_likes_only": int(likes_only["downloads"].median()),
        "median_downloads_downloads_only": int(dl_only["downloads"].median()),
        "age_days": age_summary,
    }
    head_html = (
        f"<h2>Top-{n} symmetric difference</h2>"
        f"<p>likes-only: <b>{n_a}</b> | downloads-only: <b>{n_b}</b> | both: <b>{len(overlap)}</b></p>"
        "<table border=1 cellpadding=4>"
        "<tr><th></th><th>median likes</th><th>median downloads</th></tr>"
        f"<tr><td>likes-only</td><td>{headline['median_likes_likes_only']}</td>"
        f"<td>{headline['median_downloads_likes_only']:,}</td></tr>"
        f"<tr><td>downloads-only</td><td>{headline['median_likes_downloads_only']}</td>"
        f"<td>{headline['median_downloads_downloads_only']:,}</td></tr>"
        "</table>"
    )

    page = (
        "<html><head><meta charset='utf-8'>"
        f"<title>Rank-signal characterization (top {n})</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse;font-size:0.9em} th,td{text-align:right;padding:3px 8px}"
        "th:first-child,td:first-child{text-align:left} h3{margin-top:2em}"
        "</style></head><body>"
        f"<h1>What does each ranking signal bias toward?</h1>{head_html}" + "".join(sections) + "</body></html>"
    )
    REPORT_HTML.write_text(page, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps({"headline": headline, "axes": json_axes}, indent=2))
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
