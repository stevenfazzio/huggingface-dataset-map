"""Round-1 card summarization for hover-card TL;DRs.

Generates a ≤25-word self-contained summary per dataset using Claude Haiku.
Independent of stage 04 (structured fields) — reads only the card text so
summaries stay robust to taxonomy changes and extraction errors.

Trial design mirrors `experiments/extract_structured_fields_v1.py` but
stratifies by EVoC cluster (45 finest-layer clusters + noise bucket) instead
of modality. Concept-space stratification is more meaningful for summary
quality since it controls for card-writing style more than modality does.

Outputs:
    data/experiments/summaries_v1/results/<safe_repo>.json   (per-card raw output)
    data/experiments/summaries_v1/summaries.parquet          (aggregated)
    data/experiments/summaries_v1/sample.parquet             (which 150 cards were used)
    data/experiments/summaries_v1/review.html                (browsable review)

Resumable: per-repo JSON cache files are skipped on rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import html
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
)

EVOC_LAYERS_NPZ = EXPERIMENTS_DIR / "evoc_taxonomy" / "cluster_layers.npz"
EVOC_TOPICS_JSON = EXPERIMENTS_DIR / "evoc_taxonomy" / "topic_names.json"
OUT_DIR = EXPERIMENTS_DIR / "summaries_v1"
RESULTS_DIR = OUT_DIR / "results"
SUMMARIES_PARQUET = OUT_DIR / "summaries.parquet"
SAMPLE_PARQUET = OUT_DIR / "sample.parquet"
REVIEW_HTML = OUT_DIR / "review.html"

MODEL = "claude-haiku-4-5-20251001"
PER_CLUSTER = 3            # 3 × 45 clusters + 3 from noise ≈ 138 cards
NOISE_EXTRA = 6            # oversample noise a touch since it's heterogeneous
SAMPLE_SEED = 42
CARD_CHAR_LIMIT = 4_000    # TL;DRs come from the opening; no need for 6K
CONCURRENCY = 8
MAX_RETRIES = 3
MAX_WORDS = 25

SYSTEM_PROMPT = """You write short, specific TL;DR summaries of HuggingFace datasets.

Your summary MUST:
- Be a single sentence of ≤25 words.
- Describe what the dataset IS directly, using a specific noun phrase as the opening.
- Mention what makes it distinctive: origin, scale, methodology, unique property, or source.
- Be self-contained — a reader should understand the dataset without any other context.

Your summary MUST NOT:
- Start with "This dataset…", "A dataset of…", "This is…", or similar filler openings.
- Exceed 25 words.
- Be generic (e.g. "A text classification dataset" is useless — say what it classifies and why it's interesting).
- Include marketing prose or hedging ("powerful", "comprehensive", "may be useful for…").

Good example summaries:
- "12 million YouTube Music track links auto-discovered by recursively walking 'fans might also like' suggestions from a seed of 45,000 artists."
- "Japanese translation of LLaVA-Instruct-150K via DeepL, for Japanese vision-language instruction tuning."
- "One million anonymized real-estate listings from Divar, Iran's largest classifieds platform, with 57 columns of price and location detail."
- "31 English short-answer questions on communication networks, with reference answers and scored student responses for feedback-generation training."

Output strictly valid JSON: {"summary": "<your sentence>"}. No prose, no markdown fences, nothing outside the JSON."""

USER_TEMPLATE = "Dataset `{repo_id}`:\n---\n{card}\n---"


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


def build_sample(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """Stratify by EVoC finest-layer cluster (0..N) plus noise (-1)."""
    if len(cluster_labels) != len(df):
        raise ValueError(f"EVoC labels length {len(cluster_labels)} != df length {len(df)}")

    df = df.copy()
    df["_evoc_cluster"] = cluster_labels
    df = df[df["card_text_clean"].notna() & (df["card_text_clean"].str.len() > 100)]

    rng = random.Random(SAMPLE_SEED)
    picks: list[pd.DataFrame] = []
    for cluster_id, group in df.groupby("_evoc_cluster"):
        n = NOISE_EXTRA if cluster_id == -1 else PER_CLUSTER
        n = min(n, len(group))
        idx = rng.sample(list(group.index), n)
        picks.append(group.loc[idx])

    sample = pd.concat(picks).sort_values(["_evoc_cluster", "repo_id"]).reset_index(drop=True)
    print(f"Sample size: {len(sample)}")
    print(f"  clusters represented: {sample['_evoc_cluster'].nunique()}")
    print(f"  including noise bucket (-1): {(sample['_evoc_cluster'] == -1).sum()} datasets")
    return sample


async def _extract_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    repo_id: str,
    card: str,
) -> dict:
    truncated = card[:CARD_CHAR_LIMIT]
    user_msg = USER_TEMPLATE.format(repo_id=repo_id, card=truncated)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=256,  # more than enough for 25 words + JSON framing
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {
                "repo_id": repo_id,
                "raw_text": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "card_chars_sent": len(truncated),
                "card_was_truncated": len(card) > CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt * 2)
    return {"repo_id": repo_id, "raw_text": None, "error": f"{type(last_err).__name__}: {last_err}"}


def _save_result(result: dict) -> None:
    out_path = RESULTS_DIR / f"{_safe_filename(result['repo_id'])}.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=RESULTS_DIR, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def run_extractions(sample: pd.DataFrame) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY)

    todo = []
    for _, row in sample.iterrows():
        cache_path = RESULTS_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if cache_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))

    print(f"{len(todo)} to summarize ({len(sample) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(repo_id, card):
        res = await _extract_one(client, sem, repo_id, card)
        _save_result(res)

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="summarizing"):
        await coro


def _parse_summary(raw_text: str | None) -> tuple[str | None, str | None]:
    if raw_text is None:
        return None, "no_raw_text"
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not m:
        return None, "no_json_object"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None, "missing_or_empty_summary"
    return summary.strip(), None


def _word_count(s: str) -> int:
    return len(s.split())


def aggregate(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        summary, parse_err = _parse_summary(data.get("raw_text"))
        wc = _word_count(summary) if summary else None
        over_budget = wc is not None and wc > MAX_WORDS
        bad_opening = summary is not None and bool(
            re.match(r"^(this\s+dataset|a\s+dataset|this\s+is|the\s+dataset)\b", summary.lower())
        )
        rows.append(
            {
                "repo_id": data["repo_id"],
                "summary": summary,
                "raw_text": data.get("raw_text"),
                "error": data.get("error"),
                "parse_error": parse_err,
                "word_count": wc,
                "over_budget": over_budget,
                "bad_opening": bad_opening,
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
                "cache_read_input_tokens": data.get("cache_read_input_tokens"),
            }
        )
    df = pd.DataFrame(rows)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df)
        os.replace(tmp_path, SUMMARIES_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    n = len(df)
    n_err = df["error"].notna().sum()
    n_parse_err = df["parse_error"].notna().sum()
    n_over = df["over_budget"].fillna(False).sum()
    n_bad_open = df["bad_opening"].fillna(False).sum()
    print(f"\nAggregated {n} → {SUMMARIES_PARQUET}")
    print(f"  API errors:    {n_err}")
    print(f"  Parse errors:  {n_parse_err}")
    print(f"  Over {MAX_WORDS} words:  {n_over}")
    print(f"  Bad openings:  {n_bad_open}")
    if n:
        total_in = int(df["input_tokens"].fillna(0).sum())
        total_out = int(df["output_tokens"].fillna(0).sum())
        # Haiku 4.5 rough pricing: $1/MTok in, $5/MTok out
        est_cost = total_in / 1e6 * 1.0 + total_out / 1e6 * 5.0
        print(f"  Tokens in/out: {total_in:,} / {total_out:,}")
        print(f"  Est cost:      ${est_cost:.3f}")
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


def build_review_html(summaries: pd.DataFrame, sample: pd.DataFrame) -> str:
    # Load EVoC cluster names for section headers.
    topics = json.loads(EVOC_TOPICS_JSON.read_text())
    layer0_names = topics["layer_0"]

    # Merge summaries with the sample to get cluster id + card text for context.
    merged = sample.merge(summaries, on="repo_id", how="left")

    sections = []
    # Group by cluster; keep clusters sorted with -1 last.
    cluster_ids = sorted(merged["_evoc_cluster"].unique(), key=lambda c: (c == -1, c))
    for cid in cluster_ids:
        group = merged[merged["_evoc_cluster"] == cid]
        if cid == -1:
            section_name = "__noise__"
        else:
            section_name = layer0_names[cid] if cid < len(layer0_names) else f"cluster_{cid}"

        rows_html = []
        for _, row in group.iterrows():
            summary = row["summary"] or "<em class='err'>NO SUMMARY</em>"
            badges = []
            if row["over_budget"] is True:
                badges.append(f"<span class='badge over'>{row['word_count']}w</span>")
            elif row["word_count"] is not None:
                badges.append(f"<span class='badge ok'>{row['word_count']}w</span>")
            if row["bad_opening"] is True:
                badges.append("<span class='badge bad'>bad-opening</span>")
            if row["parse_error"]:
                badges.append(f"<span class='badge err'>{html.escape(row['parse_error'])}</span>")
            badges_html = " ".join(badges)

            card_text = (row.get("card_text_clean") or "")[:700]
            rows_html.append(
                f'''<tr>
  <td class="repo">
    <a href="https://huggingface.co/datasets/{html.escape(row['repo_id'])}" target="_blank">{html.escape(row['repo_id'])}</a>
    <div class="badges">{badges_html}</div>
  </td>
  <td class="summary">{html.escape(summary) if row['summary'] else summary}</td>
  <td class="card"><details><summary>card (first 700 chars)</summary><pre>{html.escape(card_text)}</pre></details></td>
</tr>'''
            )

        sections.append(f'''
<section>
  <h2>[cluster {cid}] {html.escape(section_name)} <span class="count">({len(group)} samples)</span></h2>
  <table>
    <thead><tr><th>repo_id</th><th>summary</th><th>card</th></tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</section>''')

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Summary v1 review</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em auto; max-width: 1400px; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; }}
  h2 {{ margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #ccc; color: #114; font-size: 1.15em; }}
  h2 .count {{ color: #888; font-size: 0.85em; font-weight: normal; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92em; margin-top: 0.4em; }}
  th, td {{ text-align: left; padding: 0.5em 0.7em; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #f4f4f7; }}
  td.repo {{ width: 14em; font-family: 'SF Mono', ui-monospace, monospace; font-size: 0.85em; }}
  td.repo a {{ color: #0969da; text-decoration: none; }}
  td.repo a:hover {{ text-decoration: underline; }}
  td.summary {{ color: #1f2328; line-height: 1.4; }}
  td.card {{ width: 40%; font-size: 0.85em; }}
  td.card pre {{ white-space: pre-wrap; color: #57606a; font-size: 0.85em; max-height: 30em; overflow-y: auto; background: #fafbfc; padding: 0.5em; border-radius: 3px; }}
  .badges {{ margin-top: 0.3em; display: flex; flex-wrap: wrap; gap: 0.3em; }}
  .badge {{ display: inline-block; padding: 0.1em 0.4em; border-radius: 3px; font-size: 0.75em; font-family: monospace; }}
  .badge.ok {{ background: #e7f5e7; color: #1a7f37; }}
  .badge.over {{ background: #fff4cc; color: #9a6700; }}
  .badge.bad {{ background: #ffe0e0; color: #c9480a; }}
  .badge.err {{ background: #ffe0e0; color: #c9480a; }}
  details summary {{ cursor: pointer; color: #06c; }}
  em.err {{ color: #c9480a; font-style: normal; }}
</style></head><body>

<h1>Summary v1 review</h1>
<p style="color:#555;font-size:0.95em;">Haiku-generated ≤25-word summaries across a ~150-card sample stratified by EVoC finest-layer cluster (45 clusters + noise). Clusters ordered by cluster id, noise last. Badges show word count and flagged issues.</p>

{"".join(sections)}

</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    if not EVOC_LAYERS_NPZ.exists():
        raise SystemExit(f"{EVOC_LAYERS_NPZ} not found — run experiments/evoc_cluster_for_taxonomy.py first")

    df = pd.read_parquet(DATASETS_PARQUET)
    cluster_labels = np.load(EVOC_LAYERS_NPZ)["layer_0"]

    sample = build_sample(df, cluster_labels)

    if not SAMPLE_PARQUET.exists():
        sample.to_parquet(SAMPLE_PARQUET, index=False)
        print(f"Wrote sample snapshot to {SAMPLE_PARQUET}")

    if not args.aggregate_only:
        asyncio.run(run_extractions(sample))

    summaries = aggregate(sample)
    _atomic_write(REVIEW_HTML, build_review_html(summaries, sample))
    print(f"Wrote review to {REVIEW_HTML}")


if __name__ == "__main__":
    main()
