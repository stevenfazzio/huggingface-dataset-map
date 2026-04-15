"""Round-1 open-ended extraction of structured fields from dataset cards.

Purpose: explore the label space for LLM-extracted fields before we commit to
enums. Runs Haiku on a stratified sample of 500 cards, asking for open-ended
values + a short quoted justification for each field. Raw JSON per card is
saved so we can re-derive taxonomies later without re-calling the API.

Fields we're probing (all orthogonal to existing columns):
    - provenance:     how the data was created (human/scrape/synthetic/remix/...)
    - training_stage: pretraining / SFT / preference / eval / domain-FT / raw
    - subject_domain: what the data is ABOUT (noun, not verb)
    - upstream_models: which LLMs generated or were distilled into the data
    - geo_scope:      country/region the content is about (distinct from language)
    - is_benchmark:   eval/benchmark dataset vs training corpus

Stratification: ~100 per modality bucket (text/image/audio/tabular/other),
using whatever's available per bucket (some buckets will be smaller than 100).

Outputs:
    data/experiments/structured_fields_v1/results/<safe_repo_id>.json  (one per card)
    data/experiments/structured_fields_v1/extractions.parquet          (aggregated)
    data/experiments/structured_fields_v1/sample.parquet               (sample snapshot)

Resumable: each card's JSON is written once; reruns skip completed repos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from anthropic import AsyncAnthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import ANTHROPIC_API_KEY, DATASETS_PARQUET, EXPERIMENTS_DIR  # noqa: E402

OUT_DIR = EXPERIMENTS_DIR / "structured_fields_v1"
RESULTS_DIR = OUT_DIR / "results"
EXTRACTIONS_PARQUET = OUT_DIR / "extractions.parquet"
SAMPLE_PARQUET = OUT_DIR / "sample.parquet"

MODEL = "claude-haiku-4-5-20251001"
SAMPLE_SIZE = 500
PER_BUCKET = 100
CARD_CHAR_LIMIT = 6_000  # cards above this get truncated before sending
SAMPLE_SEED = 42
CONCURRENCY = 8
MAX_RETRIES = 3

SYSTEM_PROMPT = """You extract structured metadata from HuggingFace dataset cards.

For every field, return an OPEN-ENDED value in your own words — do not try to fit a fixed vocabulary. We are exploring the label space.

For each field, also return `quote`: a short verbatim span (≤25 words) from the card that justified your answer. If the card does not say, return the sentinel "not_stated". If the field does not apply to this kind of dataset (e.g. upstream_models for a raw web scrape), return "not_applicable". Use these sentinels in the value itself, not just in the quote.

Output strictly valid JSON matching the schema. No prose outside the JSON."""

USER_PROMPT_TEMPLATE = """Extract the following fields from this HuggingFace dataset card.

Schema (return exactly this structure):
{{
  "provenance_method":  {{ "value": "<short phrase: how was this data created?>", "quote": "<≤25-word span or sentinel>" }},
  "provenance_parents": {{ "value": ["<parent dataset name or URL>", ...] or ["not_stated"] or ["not_applicable"], "quote": "<span or sentinel>" }},
  "training_stage":     {{ "value": "<intended role in training: pretraining / SFT / preference / eval / domain-finetune / raw-corpus / other>", "quote": "<span or sentinel>" }},
  "subject_domain":     {{ "value": "<short noun phrase for what the data is ABOUT, e.g. 'real-estate listings', 'academic physics papers', 'general web text'>", "quote": "<span or sentinel>" }},
  "upstream_models":    {{ "value": ["<model name>", ...] or ["not_stated"] or ["not_applicable"], "quote": "<span or sentinel>" }},
  "geo_scope":          {{ "value": "<country/region the content is about, or 'global' or 'not_applicable'>", "quote": "<span or sentinel>" }},
  "is_benchmark":       {{ "value": true or false, "quote": "<span or sentinel>" }}
}}

Dataset card for `{repo_id}`:
---
{card}
---"""


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


def _modality_bucket(modalities: str) -> str:
    """Collapse the comma-joined modalities field into one stratification bucket."""
    m = (modalities or "").lower()
    if not m:
        return "other"
    parts = [p.strip() for p in m.split(",") if p.strip()]
    has = lambda k: any(k in p for p in parts)  # noqa: E731
    if has("image") or has("video"):
        return "vision"
    if has("audio"):
        return "audio"
    if has("tabular"):
        return "tabular"
    if parts == ["text"]:
        return "text"
    return "other"


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_bucket"] = df["modalities"].fillna("").map(_modality_bucket)
    df = df[df["card_text_clean"].notna() & (df["card_text_clean"].str.len() > 0)]

    rng = random.Random(SAMPLE_SEED)
    picks: list[pd.DataFrame] = []
    for bucket, group in df.groupby("_bucket"):
        n = min(PER_BUCKET, len(group))
        idx = rng.sample(list(group.index), n)
        picks.append(group.loc[idx])
    sample = pd.concat(picks).sort_values("repo_id").reset_index(drop=True)

    # Top-up to SAMPLE_SIZE with random others if buckets were short.
    if len(sample) < SAMPLE_SIZE:
        remaining = df.drop(sample.index, errors="ignore")
        top_up_n = min(SAMPLE_SIZE - len(sample), len(remaining))
        top_up_idx = rng.sample(list(remaining.index), top_up_n)
        sample = pd.concat([sample, remaining.loc[top_up_idx]]).reset_index(drop=True)

    print(f"Sample size: {len(sample)}")
    print("Per-bucket counts:")
    print(sample["_bucket"].value_counts().to_string())
    return sample


async def _extract_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    repo_id: str,
    card: str,
) -> dict:
    """Call Haiku for one card, return the raw result envelope (no parsing)."""
    truncated_card = card[:CARD_CHAR_LIMIT]
    user_msg = USER_PROMPT_TEMPLATE.format(repo_id=repo_id, card=truncated_card)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
            raw_text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
            return {
                "repo_id": repo_id,
                "model": MODEL,
                "raw_text": raw_text,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    resp.usage, "cache_creation_input_tokens", 0
                ),
                "card_chars_sent": len(truncated_card),
                "card_was_truncated": len(card) > CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 - one malformed item shouldn't crash the batch
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt * 2)
    return {
        "repo_id": repo_id,
        "model": MODEL,
        "raw_text": None,
        "error": f"{type(last_err).__name__}: {last_err}",
    }


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
        out_path = RESULTS_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if out_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))

    print(f"{len(todo)} to extract ({len(sample) - len(todo)} already done)")
    if not todo:
        return

    async def _do(repo_id: str, card: str) -> None:
        result = await _extract_one(client, sem, repo_id, card)
        _save_result(result)

    tasks = [_do(repo_id, card) for repo_id, card in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


def _parse_result(raw_text: str) -> tuple[dict | None, str | None]:
    """Try to parse the model's JSON output, returning (parsed, parse_error)."""
    if raw_text is None:
        return None, "no_raw_text"
    # Strip ```json fences if present.
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def aggregate() -> None:
    """Walk results/ and build one parquet with raw + parsed fields."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        parsed, parse_err = _parse_result(data.get("raw_text"))
        rows.append(
            {
                "repo_id": data["repo_id"],
                "raw_text": data.get("raw_text"),
                "error": data.get("error"),
                "parse_error": parse_err,
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
                "cache_read_input_tokens": data.get("cache_read_input_tokens"),
                "card_was_truncated": data.get("card_was_truncated"),
                "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed else None,
            }
        )
    df = pd.DataFrame(rows)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df)
        os.replace(tmp_path, EXTRACTIONS_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    n = len(df)
    n_err = df["error"].notna().sum()
    n_parse_err = df["parse_error"].notna().sum()
    n_truncated = df["card_was_truncated"].fillna(False).sum()
    print(f"Aggregated {n} results → {EXTRACTIONS_PARQUET}")
    print(f"  API errors:    {n_err}")
    print(f"  Parse errors:  {n_parse_err}")
    print(f"  Truncated:     {n_truncated}")
    if n_err == 0 and n_parse_err == 0:
        total_in = df["input_tokens"].sum()
        total_out = df["output_tokens"].sum()
        print(f"  Input tokens:  {total_in:,}")
        print(f"  Output tokens: {total_out:,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip API calls, just re-aggregate existing results/*.json into the parquet.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    df = pd.read_parquet(DATASETS_PARQUET)
    sample = build_sample(df)

    # Snapshot the sample so we can reproduce exactly which cards were picked.
    if not SAMPLE_PARQUET.exists():
        sample_cols = [
            c for c in sample.columns if c != "card_text"
        ]  # keep card_text_clean, drop raw
        sample[sample_cols].to_parquet(SAMPLE_PARQUET, index=False)
        print(f"Wrote sample snapshot to {SAMPLE_PARQUET}")

    if not args.aggregate_only:
        asyncio.run(run_extractions(sample))

    aggregate()


if __name__ == "__main__":
    main()
