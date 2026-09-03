"""Generate short TL;DR summaries per dataset via Claude Haiku.

Produces a ≤25-word single-sentence summary of each dataset card, suitable for
hover-card display. Independent of stage 04 (structured fields): reads only
the card text, writes to `data/summaries.parquet`. Resumable via per-repo JSON
files in `data/summaries_cache/`.

Prompt was developed through `experiments/summarize_cards_v1.py` on a
~150-card EVoC-stratified sample; v1 quality was high enough to promote
directly (141/141 clean, 0 bad openings, <4% over-budget on dense cards).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from batch_runner import run_batches
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_SUMMARIZE,
    DATASETS_PARQUET,
    SUMMARIES_BATCH_JOURNAL,
    SUMMARIES_CACHE_DIR,
    SUMMARIES_PARQUET,
    SUMMARIZE_CARD_CHAR_LIMIT,
    SUMMARIZE_CONCURRENCY,
    SUMMARIZE_MAX_RETRIES,
    SUMMARIZE_MAX_WORDS,
)
from tqdm import tqdm

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

Output strictly valid JSON: {"summary": "<your sentence>"}. No prose, no markdown fences, nothing outside the JSON."""  # noqa: E501


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


def _build_user_message(repo_id: str, card: str) -> str:
    return f"Dataset `{repo_id}`:\n---\n{card[:SUMMARIZE_CARD_CHAR_LIMIT]}\n---"


async def _extract_one(client, sem, repo_id: str, card: str) -> dict:
    last_err = None
    for attempt in range(SUMMARIZE_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_SUMMARIZE,
                    max_tokens=256,
                    # NB: this prompt is ~400 tokens, well under Haiku 4.5's 4,096-token
                    # minimum cacheable prefix, so the marker is currently inert. Kept so
                    # caching engages for free if the prompt grows or the minimum drops.
                    system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": _build_user_message(repo_id, card)}],
                )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {
                "repo_id": repo_id,
                "raw_text": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "card_was_truncated": len(card) > SUMMARIZE_CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — one malformed card shouldn't crash the batch
            last_err = e
            if attempt < SUMMARIZE_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    return {"repo_id": repo_id, "raw_text": None, "error": f"{type(last_err).__name__}: {last_err}"}


def _save_result(result: dict) -> None:
    out_path = SUMMARIES_CACHE_DIR / f"{_safe_filename(result['repo_id'])}.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=SUMMARIES_CACHE_DIR, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _pending(rows: pd.DataFrame) -> list[tuple[str, str]]:
    """Cards with no cache file yet, in a deterministic order."""
    todo = []
    for _, row in rows.iterrows():
        cache_path = SUMMARIES_CACHE_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if cache_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))
    return sorted(todo)


async def _run_extractions_live(todo: list[tuple[str, str]]) -> None:
    """Concurrent live calls. Full price, but no batch round-trip — use it to top
    up a handful of stragglers rather than for a full-corpus run."""
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(SUMMARIZE_CONCURRENCY)

    async def _do(repo_id, card):
        res = await _extract_one(client, sem, repo_id, card)
        _save_result(res)

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="summarizing"):
        await coro


def _batch_params(repo_id: str, card: str):
    return MessageCreateParamsNonStreaming(
        model=ANTHROPIC_MODEL_SUMMARIZE,
        max_tokens=256,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": _build_user_message(repo_id, card)}],
    )


def _run_extractions_batch(todo: list[tuple[str, str]]) -> None:
    """Same requests at half price via the Batches API. Failures are deliberately
    not written to the cache dir, so re-running the stage retries them."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    saved = 0
    failures: list[str] = []

    for repo_id, card, message, status in run_batches(
        client,
        todo,
        _batch_params,
        SUMMARIES_BATCH_JOURNAL,
        label="summarize",
    ):
        if message is None:
            failures.append(f"{repo_id} ({status})")
            continue
        raw = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        _save_result(
            {
                "repo_id": repo_id,
                "raw_text": raw,
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0),
                "card_was_truncated": len(card) > SUMMARIZE_CARD_CHAR_LIMIT,
                "error": None,
            }
        )
        saved += 1

    print(f"Batch summarization: {saved} saved, {len(failures)} failed")
    if failures:
        print(f"  first few: {failures[:5]}")
        print("  re-run this stage to retry them")


def _parse_summary(raw_text):
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


def aggregate() -> pd.DataFrame:
    rows = []
    for p in sorted(SUMMARIES_CACHE_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        summary, parse_err = _parse_summary(data.get("raw_text"))
        word_count = len(summary.split()) if summary else None
        over_budget = word_count is not None and word_count > SUMMARIZE_MAX_WORDS
        bad_opening = summary is not None and bool(
            re.match(r"^(this\s+dataset|a\s+dataset|this\s+is|the\s+dataset)\b", summary.lower())
        )
        rows.append(
            {
                "repo_id": data["repo_id"],
                "summary": summary,
                "error": data.get("error"),
                "parse_error": parse_err,
                "word_count": word_count,
                "over_budget": over_budget,
                "bad_opening": bad_opening,
                "card_was_truncated": data.get("card_was_truncated"),
            }
        )

    df = pd.DataFrame(rows)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=SUMMARIES_PARQUET.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"{len(verify)} != {len(df)}"
        os.replace(tmp_path, SUMMARIES_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    n = len(df)
    n_err = int(df["error"].notna().sum())
    n_parse_err = int(df["parse_error"].notna().sum())
    n_over = int(df["over_budget"].fillna(False).sum())
    n_bad = int(df["bad_opening"].fillna(False).sum())
    print(f"\nWrote {n} rows to {SUMMARIES_PARQUET}")
    print(f"  API errors:       {n_err}")
    print(f"  Parse errors:     {n_parse_err}")
    print(f"  Over {SUMMARIZE_MAX_WORDS} words:  {n_over}")
    print(f"  Bad openings:     {n_bad}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="Use live concurrent calls instead of the Batches API (2x the cost; no wait).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N pending cards. Use it to smoke-test end to end before a full run.",
    )
    args = parser.parse_args()

    SUMMARIES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    df = pd.read_parquet(DATASETS_PARQUET)
    df = df[df.card_text_clean.notna() & (df.card_text_clean.str.len() > 0)].reset_index(drop=True)
    print(f"Corpus: {len(df)} rows with cards")

    if not args.aggregate_only:
        todo = _pending(df[["repo_id", "card_text_clean"]])
        print(f"{len(todo)} to summarize ({len(df) - len(todo)} already cached)")
        if args.limit:
            todo = todo[: args.limit]
            print(f"  --limit {args.limit}: processing {len(todo)}")
        if todo:
            if args.no_batch:
                asyncio.run(_run_extractions_live(todo))
            else:
                _run_extractions_batch(todo)

    aggregate()


if __name__ == "__main__":
    main()
