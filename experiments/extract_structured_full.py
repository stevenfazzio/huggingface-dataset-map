"""Scale the v3 structured-field extraction to all ~5K datasets.

Differences from extract_structured_fields_v3.py (besides just running on the
full corpus):
  - Schema is in the SYSTEM prompt (not user) so it caches across all 5K calls.
    Cuts per-call input tokens by ~1K once the cache warms.
  - Output parquet is flat-column (one row per repo, one column per field +
    quote), ready for visualize stage to consume.
  - Concurrency bumped to 12; 5K calls at 8 concurrent was ~30 min, should
    trim closer to 20 min without hitting limits.

Outputs:
    data/experiments/structured_fields_full/results/<safe_repo>.json
    data/experiments/structured_fields_full/extractions.parquet

Resumable: per-repo JSON files are skipped on rerun.

Cost estimate: ~5K cards × ~4K input / ~400 output tokens ≈ 20M in / 2M out.
Haiku 4.5 with prompt caching on the schema portion should land around $15-25.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
from anthropic import AsyncAnthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import ANTHROPIC_API_KEY, DATASETS_PARQUET, EXPERIMENTS_DIR  # noqa: E402

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy_v3_proposed.json"
OUT_DIR = EXPERIMENTS_DIR / "structured_fields_full"
RESULTS_DIR = OUT_DIR / "results"
EXTRACTIONS_PARQUET = OUT_DIR / "extractions.parquet"

MODEL = "claude-haiku-4-5-20251001"
CARD_CHAR_LIMIT = 6_000
CONCURRENCY = 12
MAX_RETRIES = 4


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


# ── Prompt building (schema in system for caching) ──────────────────────────


def _field_block(name: str, spec: dict) -> str:
    t = spec["type"]
    if t == "single-select":
        lines = [f"FIELD: {name} (pick EXACTLY ONE slug)"]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        return "\n".join(lines)
    if t == "multi-select":
        has_ns = any(c["name"] == "not_stated" for c in spec["categories"])
        silence = (
            "If silent, use ['not_stated']."
            if has_ns
            else "If NO slug applies or the card is silent, return an empty list []. Do NOT invent 'not_stated' for this field."
        )
        lines = [f"FIELD: {name} (return a JSON LIST of applicable slugs. {silence})"]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        if "rule" in spec:
            lines.append(f"  EXTRA RULE: {spec['rule']}")
        return "\n".join(lines)
    if t == "single-select-or-list":
        return f"FIELD: {name} (return a JSON LIST)\n  RULE: {spec['rule']}"
    if t == "open-list":
        return f"FIELD: {name} (return a JSON LIST of raw strings)\n  RULE: {spec['rule']}"
    if t == "boolean":
        return f"FIELD: {name} (return true or false)\n  RULE: {spec['rule']}"
    raise ValueError(f"Unknown field type: {t}")


def build_prompts(taxonomy: dict) -> tuple[str, list[str]]:
    fields = [k for k in taxonomy if not k.startswith("_")]
    field_blocks = "\n\n".join(_field_block(f, taxonomy[f]) for f in fields)

    shape = "{\n"
    for f in fields:
        shape += f'  "{f}": {{ "value": <per field rule>, "quote": "<≤25-word span or sentinel>" }},\n'
    shape = shape.rstrip(",\n") + "\n}"

    system = (
        "You extract constrained structured metadata from HuggingFace dataset cards.\n\n"
        "RULES:\n"
        "- For slug-valued fields, return one of the provided slugs verbatim. No paraphrases, no combined values like 'a / b'.\n"
        "- For LIST-typed fields, the `value` MUST be a JSON array even if only one item applies: `[\"item\"]`. Never a bare string.\n"
        "- For each field, include `quote`: a ≤25-word verbatim span from the card that justified your choice. If silent, use the sentinel 'not_stated' for the quote.\n"
        "- Output strictly valid JSON. No prose, no markdown fences, no commentary outside the JSON object.\n\n"
        "FIELD DEFINITIONS:\n\n"
        f"{field_blocks}\n\n"
        "OUTPUT SHAPE (fill in values; do not change the structure):\n"
        f"{shape}"
    )
    return system, fields


def build_user_message(repo_id: str, card: str) -> str:
    truncated = card[:CARD_CHAR_LIMIT]
    return f"Dataset card for `{repo_id}`:\n---\n{truncated}\n---"


# ── Extraction ──────────────────────────────────────────────────────────────


async def _extract_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    system: str,
    repo_id: str,
    card: str,
) -> dict:
    user_msg = build_user_message(repo_id, card)
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=[
                        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                    ],
                    messages=[{"role": "user", "content": user_msg}],
                )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {
                "repo_id": repo_id,
                "model": MODEL,
                "raw_text": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(
                    resp.usage, "cache_creation_input_tokens", 0
                ),
                "card_chars_sent": len(card[:CARD_CHAR_LIMIT]),
                "card_was_truncated": len(card) > CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(2 ** attempt * 2, 30))
    return {"repo_id": repo_id, "model": MODEL, "raw_text": None, "error": f"{type(last_err).__name__}: {last_err}"}


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


async def run_extractions(rows: pd.DataFrame, system: str) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(CONCURRENCY)

    todo = []
    for _, row in rows.iterrows():
        out_path = RESULTS_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if out_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))
    print(f"{len(todo)} to extract ({len(rows) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(repo_id: str, card: str) -> None:
        res = await _extract_one(client, sem, system, repo_id, card)
        _save_result(res)

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


# ── Parsing & validation ────────────────────────────────────────────────────


def _parse_json(raw_text: str | None) -> tuple[dict | None, str | None]:
    if raw_text is None:
        return None, "no_raw_text"
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def _allowed(spec: dict) -> set[str]:
    return {c["name"] for c in spec.get("categories", [])}


def _validate(parsed: dict, taxonomy: dict, fields: list[str]) -> list[str]:
    issues = []
    for f in fields:
        spec = taxonomy[f]
        t = spec["type"]
        entry = parsed.get(f)
        if not isinstance(entry, dict) or "value" not in entry:
            issues.append(f"{f}:missing")
            continue
        val = entry["value"]
        allowed = _allowed(spec)
        if t == "single-select":
            if not isinstance(val, str) or val not in allowed:
                issues.append(f"{f}:invalid_slug:{val!r}")
        elif t == "multi-select":
            if not isinstance(val, list):
                issues.append(f"{f}:not_list")
            elif any(not isinstance(v, str) or v not in allowed for v in val):
                bad = [v for v in val if not isinstance(v, str) or v not in allowed]
                issues.append(f"{f}:invalid_slugs:{bad}")
        elif t in ("single-select-or-list", "open-list"):
            if not isinstance(val, list):
                issues.append(f"{f}:not_list")
        elif t == "boolean":
            if not isinstance(val, bool):
                issues.append(f"{f}:not_bool:{val!r}")
    return issues


# ── Aggregation ─────────────────────────────────────────────────────────────


def aggregate(taxonomy: dict, fields: list[str]) -> pd.DataFrame:
    """Flat-column parquet: one row per repo, one column per field + quote."""
    rows = []
    total_in = total_out = total_cache_read = 0
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        parsed, parse_err = _parse_json(data.get("raw_text"))
        issues = _validate(parsed, taxonomy, fields) if parsed else []

        row = {
            "repo_id": data["repo_id"],
            "error": data.get("error"),
            "parse_error": parse_err,
            "validation_issues": "; ".join(issues) if issues else None,
            "card_was_truncated": data.get("card_was_truncated"),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "cache_read_input_tokens": data.get("cache_read_input_tokens", 0),
        }
        total_in += row["input_tokens"] or 0
        total_out += row["output_tokens"] or 0
        total_cache_read += row["cache_read_input_tokens"] or 0

        for f in fields:
            entry = (parsed or {}).get(f) or {}
            val = entry.get("value")
            quote = entry.get("quote")
            # Serialize list-valued fields as JSON for parquet portability.
            if isinstance(val, list):
                row[f] = json.dumps(val, ensure_ascii=False)
            else:
                row[f] = val
            row[f"{f}_quote"] = quote
        rows.append(row)

    df = pd.DataFrame(rows)

    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"{len(verify)} != {len(df)}"
        os.replace(tmp_path, EXTRACTIONS_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    n = len(df)
    print(f"\nAggregated {n} → {EXTRACTIONS_PARQUET}")
    print(f"  API errors:        {df['error'].notna().sum()}")
    print(f"  Parse errors:      {df['parse_error'].notna().sum()}")
    print(f"  Validation issues: {df['validation_issues'].notna().sum()}")
    print(f"  Cards truncated:   {df['card_was_truncated'].fillna(False).sum()}")
    print(f"  Tokens — input:    {total_in:,}")
    print(f"  Tokens — output:   {total_out:,}")
    print(f"  Tokens — cache hit: {total_cache_read:,}  ({total_cache_read / total_in:.0%} of input)" if total_in else "")
    return df


def print_distributions(df: pd.DataFrame, taxonomy: dict, fields: list[str]) -> None:
    clean = df[df.parse_error.isna() & df.validation_issues.isna() & df.error.isna()]
    print(f"\n── distributions (n={len(clean)} clean) ──")
    for f in fields:
        from collections import Counter

        c = Counter()
        for v in clean[f].dropna():
            if isinstance(v, str) and v.startswith("["):
                try:
                    for item in json.loads(v):
                        c[item] += 1
                except Exception:
                    c[v] += 1
            else:
                c[v] += 1
        print(f"\n  {f}  ({len(c)} unique):")
        for val, n in c.most_common(20):
            print(f"    {n:5d}  {val!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--limit", type=int, help="run on only the first N rows (for testing)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    system, fields = build_prompts(taxonomy)
    print(f"Fields: {fields}")
    print(f"System prompt: {len(system)} chars (≈{len(system)//4} tokens) — cached across calls")

    df = pd.read_parquet(DATASETS_PARQUET)
    df = df[df.card_text_clean.notna() & (df.card_text_clean.str.len() > 0)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"Corpus: {len(df)} rows with cards")

    if not args.aggregate_only:
        asyncio.run(run_extractions(df[["repo_id", "card_text_clean"]], system))

    out = aggregate(taxonomy, fields)
    print_distributions(out, taxonomy, fields)


if __name__ == "__main__":
    main()
