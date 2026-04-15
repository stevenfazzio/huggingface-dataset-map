# ruff: noqa: E501  # long lines in the embedded HTML template
"""Cheap pre-rerun validation of the stage-04 prompt/taxonomy changes.

Draws an EVoC-stratified ~130-card sample from the corpus, runs the REVISED
stage-04 extraction on it using the current `pipeline/taxonomy.json` and the
updated system prompt, and diffs the results against the existing
`data/structured_fields.parquet` for those same repo_ids.

Purpose: confirm the rerun checklist changes (orthogonality rule, rag-evaluation
slug, not_applicable in training_stage, structured-record in format_convention,
multi-turn-chat removal) actually move the needle BEFORE paying $30 for a full
5K rerun. If the sample shows:
  (a) RAG-ish cards now pick up `subject_domain=rag-evaluation`,
  (b) multi-turn-chat (now removed) doesn't become an invalid_slug pile-up,
  (c) structured-record absorbs a meaningful chunk of prior 'other' cases,
  (d) validation issue count drops materially on bleed concepts like
      `multilingual-parallel`, `raw-corpus`, `vqa`,
then the full rerun is justified.

Outputs:
    data/experiments/rerun_validation/sample.parquet
    data/experiments/rerun_validation/extractions.parquet   (new extractions)
    data/experiments/rerun_validation/diff_report.html     (side-by-side old vs new)
"""

from __future__ import annotations

import argparse
import asyncio
import html
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    DATASETS_PARQUET,
    EXPERIMENTS_DIR,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_RETRIES,
    STRUCTURED_FIELDS_PARQUET,
    TAXONOMY_JSON,
)

# Import the CURRENT stage-04 module so we use the revised system prompt.
_pipeline_dir = Path(__file__).resolve().parent.parent / "pipeline"
_stage04_spec = importlib.util.spec_from_file_location("stage04", _pipeline_dir / "04_extract_structured.py")
stage04 = importlib.util.module_from_spec(_stage04_spec)
sys.path.insert(0, str(_pipeline_dir))
_stage04_spec.loader.exec_module(stage04)

EVOC_LAYERS_NPZ = EXPERIMENTS_DIR / "evoc_taxonomy" / "cluster_layers.npz"
EVOC_TOPICS_JSON = EXPERIMENTS_DIR / "evoc_taxonomy" / "topic_names.json"
OUT_DIR = EXPERIMENTS_DIR / "rerun_validation"
SAMPLE_PARQUET = OUT_DIR / "sample.parquet"
RESULTS_DIR = OUT_DIR / "results"
NEW_EXTRACTIONS_PARQUET = OUT_DIR / "extractions.parquet"
DIFF_REPORT = OUT_DIR / "diff_report.html"

SAMPLE_SEED = 42
PER_CLUSTER = 3
NOISE_EXTRA = 6
CONCURRENCY = EXTRACT_CONCURRENCY
MAX_RETRIES = EXTRACT_MAX_RETRIES


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


def build_sample(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    if len(cluster_labels) != len(df):
        raise ValueError(f"labels length {len(cluster_labels)} != df length {len(df)}")
    df = df.copy()
    df["_evoc_cluster"] = cluster_labels
    df = df[df["card_text_clean"].notna() & (df["card_text_clean"].str.len() > 100)]

    rng = random.Random(SAMPLE_SEED)
    picks: list[pd.DataFrame] = []
    for cid, group in df.groupby("_evoc_cluster"):
        n = NOISE_EXTRA if cid == -1 else PER_CLUSTER
        n = min(n, len(group))
        idx = rng.sample(list(group.index), n)
        picks.append(group.loc[idx])
    sample = pd.concat(picks).sort_values(["_evoc_cluster", "repo_id"]).reset_index(drop=True)
    print(f"Sample: {len(sample)} rows across {sample['_evoc_cluster'].nunique()} clusters")
    return sample


async def run_extractions(sample: pd.DataFrame, system: str) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY)

    todo = []
    for _, row in sample.iterrows():
        cache_path = RESULTS_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if cache_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))
    print(f"{len(todo)} to extract ({len(sample) - len(todo)} cached)")
    if not todo:
        return

    async def _do(repo_id, card):
        res = await stage04._extract_one(client, sem, system, repo_id, card)
        # Save to local results dir (not the pipeline cache).
        out_path = RESULTS_DIR / f"{_safe_filename(repo_id)}.json"
        tmp_fd, tmp_path = tempfile.mkstemp(dir=RESULTS_DIR, suffix=".json.tmp")
        os.close(tmp_fd)
        try:
            Path(tmp_path).write_text(json.dumps(res, indent=2, ensure_ascii=False))
            os.replace(tmp_path, out_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


def aggregate_new(taxonomy, fields) -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        parsed, parse_err = stage04._parse_json(data.get("raw_text"))
        issues = stage04._validate(parsed, taxonomy, fields) if parsed else []
        row = {
            "repo_id": data["repo_id"],
            "error": data.get("error"),
            "parse_error": parse_err,
            "validation_issues": "; ".join(issues) if issues else None,
        }
        for f in fields:
            entry = (parsed or {}).get(f) or {}
            val = entry.get("value")
            row[f] = json.dumps(val, ensure_ascii=False) if isinstance(val, list) else val
            row[f"{f}_quote"] = entry.get("quote")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(NEW_EXTRACTIONS_PARQUET, index=False)
    print(f"Wrote {NEW_EXTRACTIONS_PARQUET}")
    return df


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


def build_diff_html(
    new_df: pd.DataFrame,
    old_df: pd.DataFrame,
    sample: pd.DataFrame,
    fields: list[str],
) -> str:
    topics = json.loads(EVOC_TOPICS_JSON.read_text())
    layer0 = topics["layer_0"]

    sample_meta = sample[["repo_id", "_evoc_cluster", "card_text_clean"]].rename(columns={"card_text_clean": "card"})
    merged = sample_meta.merge(new_df, on="repo_id", how="left", suffixes=("", "_new"))
    merged = merged.merge(old_df, on="repo_id", how="left", suffixes=("_new", "_old"))

    # Summary stats.
    n = len(merged)
    new_issues = int(merged["validation_issues_new"].notna().sum())
    old_issues = int(merged["validation_issues_old"].notna().sum())
    new_parse = int(merged["parse_error_new"].notna().sum()) if "parse_error_new" in merged else 0
    old_parse = int(merged["parse_error_old"].notna().sum()) if "parse_error_old" in merged else 0

    # Per-field change count.
    field_changes = {}
    for f in fields:
        fnew = f"{f}_new"
        fold = f"{f}_old"
        if fnew not in merged.columns or fold not in merged.columns:
            continue
        changed = (merged[fnew].fillna("∅") != merged[fold].fillna("∅")).sum()
        field_changes[f] = int(changed)

    stats_html = (
        f'<p class="stats">Sample size: <b>{n}</b> · '
        f"Validation issues: old={old_issues}, new={new_issues} · "
        f"Parse errors: old={old_parse}, new={new_parse}</p>"
        '<table class="summary"><thead><tr><th>field</th><th>n changed (old→new)</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{html.escape(f)}</td><td class="num">{c}</td></tr>'
            for f, c in sorted(field_changes.items(), key=lambda kv: -kv[1])
        )
        + "</tbody></table>"
    )

    # Per-row diff rows.
    row_html_parts = []
    for _, row in merged.iterrows():
        cid = row["_evoc_cluster"]
        cluster_name = layer0[cid] if 0 <= cid < len(layer0) else ("__noise__" if cid == -1 else f"id_{cid}")

        field_rows = []
        any_diff = False
        for f in fields:
            ov = row.get(f"{f}_old")
            nv = row.get(f"{f}_new")
            ov_s = "" if ov is None or (isinstance(ov, float) and pd.isna(ov)) else str(ov)
            nv_s = "" if nv is None or (isinstance(nv, float) and pd.isna(nv)) else str(nv)
            diff = ov_s != nv_s
            if diff:
                any_diff = True
            diff_class = " diff" if diff else ""
            field_rows.append(
                f'<tr class="field{diff_class}"><td>{f}</td>'
                f"<td>{html.escape(ov_s)}</td><td>{html.escape(nv_s)}</td></tr>"
            )

        issues_new = row.get("validation_issues_new") or ""
        issues_old = row.get("validation_issues_old") or ""
        flag = ""
        if issues_new and not issues_old:
            flag = ' <span class="badge err">NEW ISSUE</span>'
        elif issues_old and not issues_new:
            flag = ' <span class="badge ok">FIXED</span>'
        elif any_diff:
            flag = ' <span class="badge changed">changed</span>'

        card_preview = (row.get("card") or "")[:500]
        row_html_parts.append(f"""<section class="card">
  <h3>{html.escape(row["repo_id"])} {flag}</h3>
  <p class="meta">cluster {cid}: <em>{html.escape(cluster_name)}</em></p>
  {('<p class="issues-old">old issues: ' + html.escape(issues_old) + "</p>") if issues_old else ""}
  {('<p class="issues-new">new issues: ' + html.escape(issues_new) + "</p>") if issues_new else ""}
  <table class="fields">
    <thead><tr><th>field</th><th>old</th><th>new</th></tr></thead>
    <tbody>{"".join(field_rows)}</tbody>
  </table>
  <details><summary>card excerpt</summary><pre>{html.escape(card_preview)}</pre></details>
</section>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage-04 rerun validation diff</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto; max-width: 1300px; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h3 {{ margin: 0.4em 0 0.2em; font-size: 1.0em; }}
  p.stats {{ color: #333; font-size: 0.95em; background: #f4f4f7; padding: 0.6em 1em; border-radius: 4px; }}
  table.summary {{ font-size: 0.9em; margin-bottom: 2em; }}
  table.summary th, table.summary td {{ padding: 0.3em 0.7em; text-align: left; border-bottom: 1px solid #eee; }}
  table.summary td.num {{ text-align: right; font-variant-numeric: tabular-nums; width: 10em; }}
  section.card {{ border-left: 3px solid #d0d7de; padding: 0.5em 0.9em; margin: 0.6em 0; background: #fafbfc; }}
  table.fields {{ width: 100%; font-size: 0.85em; border-collapse: collapse; }}
  table.fields td, table.fields th {{ padding: 0.25em 0.5em; border-bottom: 1px solid #eee; vertical-align: top; }}
  table.fields tr.diff {{ background: #fff8c5; }}
  table.fields tr.diff td:first-child {{ font-weight: 600; }}
  .badge {{ display: inline-block; padding: 0.1em 0.5em; border-radius: 3px; font-size: 0.7em; font-family: monospace; }}
  .badge.err {{ background: #ffe0e0; color: #c9480a; }}
  .badge.ok {{ background: #e7f5e7; color: #1a7f37; }}
  .badge.changed {{ background: #fff4cc; color: #9a6700; }}
  p.meta {{ color: #57606a; font-size: 0.85em; margin: 0.1em 0 0.5em; }}
  p.issues-old, p.issues-new {{ font-size: 0.85em; font-family: monospace; margin: 0.2em 0; }}
  p.issues-old {{ color: #999; }}
  p.issues-new {{ color: #c9480a; }}
  details summary {{ cursor: pointer; color: #06c; font-size: 0.85em; }}
  pre {{ white-space: pre-wrap; color: #555; font-size: 0.8em; background: #fff; padding: 0.5em; border: 1px solid #eee; }}
</style></head><body>
<h1>Stage-04 rerun validation diff</h1>
<p>Side-by-side comparison of current <code>data/structured_fields.parquet</code> (old) vs new extractions using the revised prompt and taxonomy, over a ~130-card EVoC-stratified sample. Rows flagged <b>NEW ISSUE</b> are regressions; <b>FIXED</b> rows had a validation issue before that's now gone; <b>changed</b> rows have at least one field value difference.</p>
{stats_html}
{"".join(row_html_parts)}
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not EVOC_LAYERS_NPZ.exists():
        raise SystemExit("Need experiments/evoc_taxonomy/cluster_layers.npz — run evoc_cluster_for_taxonomy.py first")

    df = pd.read_parquet(DATASETS_PARQUET)
    cluster_labels = np.load(EVOC_LAYERS_NPZ)["layer_0"]
    sample = build_sample(df, cluster_labels)
    sample.to_parquet(SAMPLE_PARQUET, index=False)
    print(f"Wrote sample to {SAMPLE_PARQUET}")

    taxonomy = json.loads(TAXONOMY_JSON.read_text())
    system, fields = stage04.build_system_prompt(taxonomy)
    print(f"System prompt: {len(system)} chars")

    if not args.aggregate_only:
        if not ANTHROPIC_API_KEY:
            raise SystemExit("ANTHROPIC_API_KEY required")
        asyncio.run(run_extractions(sample, system))

    new_df = aggregate_new(taxonomy, fields)
    old_df = pd.read_parquet(STRUCTURED_FIELDS_PARQUET)

    html_doc = build_diff_html(new_df, old_df, sample, fields)
    _atomic_write(DIFF_REPORT, html_doc)
    print(f"Wrote {DIFF_REPORT}")

    # Console stats.
    merged = sample.merge(new_df, on="repo_id", how="left").merge(
        old_df, on="repo_id", how="left", suffixes=("_new", "_old")
    )
    new_issues = int(merged["validation_issues_new"].notna().sum())
    old_issues = int(merged["validation_issues_old"].notna().sum())
    print(f"\nValidation issues on the {len(sample)}-card sample:")
    print(f"  OLD (current parquet):  {old_issues}")
    print(f"  NEW (revised prompt):   {new_issues}")

    # New-field adoption.
    for slug, field in [
        ("rag-evaluation", "subject_domain"),
        ("structured-record", "format_convention"),
        ("not_applicable", "training_stage"),
    ]:
        col_new = f"{field}_new"
        col = col_new if col_new in merged.columns else field
        if field == "training_stage":
            # Multi-select: parse list values.
            def _has(v):
                if isinstance(v, str) and v.startswith("["):
                    try:
                        return slug in json.loads(v)
                    except Exception:
                        return False
                return False

            count = int(merged[col].map(_has).sum())
        else:
            count = int((merged[col] == slug).sum())
        print(f"  NEW slug '{slug}' in {field}: {count} cards")


if __name__ == "__main__":
    main()
