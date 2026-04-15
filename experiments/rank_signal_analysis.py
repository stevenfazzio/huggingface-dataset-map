"""Compare `likes` vs `downloads` as the ranking signal for the dataset corpus.

Pulls the top 10K HF datasets under each sort order (metadata only, no
README cards), then quantifies how different the resulting top-N sets are
and whether a hybrid score is worth considering.

Outputs:
    data/experiments/top10k_by_likes.parquet      (cached HF list)
    data/experiments/top10k_by_downloads.parquet  (cached HF list)
    data/experiments/rank_signal_analysis.html    (interactive plotly report)
    data/experiments/rank_signal_summary.json     (headline numbers)

Re-run with --refresh to refetch the HF lists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import EXPERIMENTS_DIR, HF_TOKEN  # noqa: E402

FETCH_LIMIT = 10_000
TOP_N_GRID = [100, 250, 500, 1_000, 2_500, 5_000, 10_000]
LIKES_PARQUET = EXPERIMENTS_DIR / "top10k_by_likes.parquet"
DOWNLOADS_PARQUET = EXPERIMENTS_DIR / "top10k_by_downloads.parquet"
REPORT_HTML = EXPERIMENTS_DIR / "rank_signal_analysis.html"
SUMMARY_JSON = EXPERIMENTS_DIR / "rank_signal_summary.json"


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


def _fetch_top(sort_key: str, limit: int) -> pd.DataFrame:
    api = HfApi(token=HF_TOKEN or None)
    print(f"Fetching top {limit} datasets sorted by {sort_key} (metadata only)...")
    rows = []
    it = api.list_datasets(sort=sort_key, limit=limit, full=False)
    for info in tqdm(it, total=limit, desc=sort_key):
        rows.append(
            {
                "repo_id": info.id,
                "likes": int(getattr(info, "likes", 0) or 0),
                "downloads": int(getattr(info, "downloads", 0) or 0),
            }
        )
    df = pd.DataFrame(rows)
    print(f"  got {len(df)} rows; min likes={df['likes'].min()}, min dl={df['downloads'].min()}")
    return df


def _load_or_fetch(path: Path, sort_key: str, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        print(f"Loaded cached {path.name} ({len(df)} rows)")
        return df
    df = _fetch_top(sort_key, FETCH_LIMIT)
    _atomic_write_parquet(df, path)
    print(f"Wrote {path}")
    return df


def _topn_overlap(set_a: pd.DataFrame, set_b: pd.DataFrame, n: int) -> dict:
    a = set(set_a["repo_id"].head(n))
    b = set(set_b["repo_id"].head(n))
    inter = len(a & b)
    union = len(a | b)
    return {
        "n": n,
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else 0.0,
        "overlap_frac": inter / n,
    }


def _hybrid_scores(union_df: pd.DataFrame) -> dict[str, pd.Series]:
    likes = union_df["likes"].astype(float)
    dl = union_df["downloads"].astype(float)
    log_likes = np.log10(likes + 1)
    log_dl = np.log10(dl + 1)
    return {
        "likes": likes,
        "downloads": dl,
        "log_likes_plus_log_downloads": log_likes + log_dl,
        "z_log_likes_plus_z_log_downloads": (
            (log_likes - log_likes.mean()) / log_likes.std() + (log_dl - log_dl.mean()) / log_dl.std()
        ),
        "likes_then_downloads_tiebreak": likes + dl / (dl.max() + 1),
    }


def _build_report(likes_df: pd.DataFrame, dl_df: pd.DataFrame) -> dict:
    union_ids = sorted(set(likes_df["repo_id"]) | set(dl_df["repo_id"]))
    print(f"Union of both top-{FETCH_LIMIT}s: {len(union_ids)} unique repos")

    # Build a union dataframe with whichever stats we have (prefer likes_df row).
    by_likes = likes_df.set_index("repo_id")
    by_dl = dl_df.set_index("repo_id")
    union = pd.DataFrame(index=union_ids)
    union["likes"] = by_likes["likes"].reindex(union_ids).combine_first(by_dl["likes"].reindex(union_ids))
    union["downloads"] = by_likes["downloads"].reindex(union_ids).combine_first(by_dl["downloads"].reindex(union_ids))
    union = union.fillna(0).astype({"likes": int, "downloads": int}).reset_index(names="repo_id")

    # Correlations (use only repos present in BOTH lists so both signals are real).
    both = sorted(set(likes_df["repo_id"]) & set(dl_df["repo_id"]))
    both_df = union.set_index("repo_id").loc[both]
    log_likes = np.log10(both_df["likes"] + 1)
    log_dl = np.log10(both_df["downloads"] + 1)
    pearson_log = pearsonr(log_likes, log_dl)
    spearman_raw = spearmanr(both_df["likes"], both_df["downloads"])

    # Top-N overlaps.
    overlaps = [_topn_overlap(likes_df, dl_df, n) for n in TOP_N_GRID]

    # Hybrid rankings on the union.
    scores = _hybrid_scores(union)
    likes_top1k = set(likes_df["repo_id"].head(1_000))
    dl_top1k = set(dl_df["repo_id"].head(1_000))
    hybrid_summary = {}
    for name, s in scores.items():
        ranked = union.assign(_score=s.values).sort_values("_score", ascending=False)
        top1k = set(ranked["repo_id"].head(1_000))
        hybrid_summary[name] = {
            "overlap_with_likes_top1k": len(top1k & likes_top1k),
            "overlap_with_downloads_top1k": len(top1k & dl_top1k),
            "min_likes_in_top1k": int(ranked.head(1_000)["likes"].min()),
            "min_downloads_in_top1k": int(ranked.head(1_000)["downloads"].min()),
        }

    summary = {
        "fetch_limit": FETCH_LIMIT,
        "n_likes_list": len(likes_df),
        "n_downloads_list": len(dl_df),
        "n_union": len(union_ids),
        "n_intersection_full_lists": len(both),
        "pearson_log10_likes_vs_log10_downloads": {
            "r": float(pearson_log.statistic),
            "p": float(pearson_log.pvalue),
            "n": len(both),
        },
        "spearman_likes_vs_downloads": {
            "rho": float(spearman_raw.statistic),
            "p": float(spearman_raw.pvalue),
            "n": len(both),
        },
        "topn_overlap": overlaps,
        "hybrid_top1k_summary": hybrid_summary,
        "thresholds": {
            "min_likes_in_likes_top1k": int(likes_df["likes"].head(1_000).min()),
            "min_downloads_in_downloads_top1k": int(dl_df["downloads"].head(1_000).min()),
            "min_likes_in_downloads_top1k": int(
                union.set_index("repo_id").loc[list(dl_df["repo_id"].head(1_000)), "likes"].min()
            ),
            "min_downloads_in_likes_top1k": int(
                union.set_index("repo_id").loc[list(likes_df["repo_id"].head(1_000)), "downloads"].min()
            ),
        },
    }
    return {"summary": summary, "union": union, "overlaps": overlaps}


def _build_html(likes_df: pd.DataFrame, dl_df: pd.DataFrame, report: dict) -> None:
    union = report["union"]
    overlaps = report["overlaps"]
    s = report["summary"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"likes vs downloads (log-log, n={s['n_intersection_full_lists']} in both lists)",
            "likes / downloads distributions (log10)",
            "Top-N set overlap (likes vs downloads rankings)",
            "Top-N Jaccard similarity",
        ),
    )

    # 1. log-log scatter on the intersection.
    both = union[(union["likes"] > 0) & (union["downloads"] > 0)]
    fig.add_trace(
        go.Scattergl(
            x=both["downloads"],
            y=both["likes"],
            mode="markers",
            marker=dict(size=4, opacity=0.4),
            text=both["repo_id"],
            hovertemplate="%{text}<br>likes=%{y}<br>downloads=%{x}<extra></extra>",
            name="repos",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(type="log", title_text="downloads", row=1, col=1)
    fig.update_yaxes(type="log", title_text="likes", row=1, col=1)

    # 2. histograms of log10(likes) and log10(downloads) on the union.
    fig.add_trace(
        go.Histogram(
            x=np.log10(union["likes"].clip(lower=1)),
            name="log10(likes)",
            opacity=0.6,
            nbinsx=50,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=np.log10(union["downloads"].clip(lower=1)),
            name="log10(downloads)",
            opacity=0.6,
            nbinsx=50,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="log10(value)", row=1, col=2)
    fig.update_yaxes(title_text="count", row=1, col=2)

    # 3. overlap bar chart.
    ns = [o["n"] for o in overlaps]
    inter = [o["intersection"] for o in overlaps]
    fig.add_trace(
        go.Bar(x=ns, y=inter, name="intersection size", showlegend=False),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="N", type="log", row=2, col=1)
    fig.update_yaxes(title_text="repos in both top-N", row=2, col=1)

    # 4. jaccard line.
    jac = [o["jaccard"] for o in overlaps]
    fig.add_trace(
        go.Scatter(x=ns, y=jac, mode="lines+markers", name="jaccard", showlegend=False),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="N", type="log", row=2, col=2)
    fig.update_yaxes(title_text="jaccard", range=[0, 1], row=2, col=2)

    fig.update_layout(
        title=(
            f"HF dataset rank signal: likes vs downloads (top {FETCH_LIMIT} each)"
            f" &nbsp;|&nbsp; pearson(log10) = {s['pearson_log10_likes_vs_log10_downloads']['r']:.3f}"
            f" &nbsp;|&nbsp; spearman = {s['spearman_likes_vs_downloads']['rho']:.3f}"
        ),
        height=900,
        barmode="overlay",
    )

    # Append a small text block summarizing the hybrid scores.
    hybrid_lines = ["<h3>Top-1K overlap by ranking score</h3>", "<table border=1 cellpadding=4>"]
    hybrid_lines.append(
        "<tr><th>score</th><th>overlap w/ likes-top1k</th><th>overlap w/ downloads-top1k</th>"
        "<th>min likes</th><th>min downloads</th></tr>"
    )
    for name, h in s["hybrid_top1k_summary"].items():
        hybrid_lines.append(
            f"<tr><td>{name}</td><td>{h['overlap_with_likes_top1k']}</td>"
            f"<td>{h['overlap_with_downloads_top1k']}</td>"
            f"<td>{h['min_likes_in_top1k']}</td><td>{h['min_downloads_in_top1k']:,}</td></tr>"
        )
    hybrid_lines.append("</table>")

    thresholds = s["thresholds"]
    threshold_html = (
        "<h3>Top-1K thresholds</h3><ul>"
        f"<li>likes-top1k boundary: min likes = {thresholds['min_likes_in_likes_top1k']}, "
        f"min downloads = {thresholds['min_downloads_in_likes_top1k']:,}</li>"
        f"<li>downloads-top1k boundary: min downloads = {thresholds['min_downloads_in_downloads_top1k']:,}, "
        f"min likes = {thresholds['min_likes_in_downloads_top1k']}</li>"
        "</ul>"
    )

    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    page = (
        "<html><head><meta charset='utf-8'>"
        "<title>HF dataset rank signal analysis</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse} th,td{text-align:right} th:first-child,td:first-child{text-align:left}"
        "</style></head><body>"
        f"{fig_html}{threshold_html}{''.join(hybrid_lines)}"
        "</body></html>"
    )
    REPORT_HTML.write_text(page, encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refetch the HF lists.")
    args = parser.parse_args()

    likes_df = _load_or_fetch(LIKES_PARQUET, "likes", args.refresh)
    dl_df = _load_or_fetch(DOWNLOADS_PARQUET, "downloads", args.refresh)

    report = _build_report(likes_df, dl_df)
    _build_html(likes_df, dl_df, report)
    SUMMARY_JSON.write_text(json.dumps(report["summary"], indent=2))
    print(f"Wrote {SUMMARY_JSON}")

    s = report["summary"]
    print()
    print("=" * 60)
    print(f"Union of both top-{FETCH_LIMIT}: {s['n_union']} repos")
    print(f"Intersection: {s['n_intersection_full_lists']} repos")
    print(f"pearson(log10 likes, log10 downloads) = {s['pearson_log10_likes_vs_log10_downloads']['r']:.3f}")
    print(f"spearman(likes, downloads) = {s['spearman_likes_vs_downloads']['rho']:.3f}")
    print()
    print("Top-N overlap (likes ranking vs downloads ranking):")
    for o in report["overlaps"]:
        print(
            f"  N={o['n']:>5}  intersection={o['intersection']:>5}  "
            f"jaccard={o['jaccard']:.3f}  overlap_frac={o['overlap_frac']:.3f}"
        )


if __name__ == "__main__":
    main()
