"""Round-2 extraction with the constrained schema.

Loads `taxonomy_v2_proposed.json`, builds a per-field prompt with allowed
values, and re-extracts the same 500-card sample we used in v1. Validates
each response against the schema and reports distributions alongside v1
counts so we can see what the constraints changed.

Outputs:
    data/experiments/structured_fields_v2/results/<safe_repo>.json
    data/experiments/structured_fields_v2/extractions.parquet
    data/experiments/structured_fields_v2/v1_v2_summary.json

Resumable: existing per-repo JSONs are skipped on rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from anthropic import AsyncAnthropic
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import ANTHROPIC_API_KEY, DATASETS_PARQUET, EXPERIMENTS_DIR  # noqa: E402

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy_v2_proposed.json"
V1_DIR = EXPERIMENTS_DIR / "structured_fields_v1"
V1_SAMPLE_PARQUET = V1_DIR / "sample.parquet"
V1_EXTRACTIONS_PARQUET = V1_DIR / "extractions.parquet"

OUT_DIR = EXPERIMENTS_DIR / "structured_fields_v2"
RESULTS_DIR = OUT_DIR / "results"
EXTRACTIONS_PARQUET = OUT_DIR / "extractions.parquet"
SUMMARY_JSON = OUT_DIR / "v1_v2_summary.json"

MODEL = "claude-haiku-4-5-20251001"
CARD_CHAR_LIMIT = 6_000
CONCURRENCY = 8
MAX_RETRIES = 3


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


def _build_field_block(name: str, spec: dict) -> str:
    t = spec["type"]
    if t == "single-select":
        lines = [f"FIELD: {name} (pick EXACTLY ONE of these slugs)"]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        return "\n".join(lines)
    if t == "multi-select":
        lines = [f"FIELD: {name} (return a LIST of one or more of these slugs; most cards will have just one)"]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        return "\n".join(lines)
    if t == "single-select-or-list":
        return f"FIELD: {name} (return a LIST)\n  RULE: {spec['rule']}"
    if t == "open-list":
        return f"FIELD: {name} (return a LIST of raw strings)\n  RULE: {spec['rule']}"
    if t == "boolean":
        return f"FIELD: {name} (return true or false)\n  RULE: {spec['rule']}"
    raise ValueError(f"Unknown field type: {t}")


def build_prompts(taxonomy: dict) -> tuple[str, str, list[str]]:
    """Return (system_prompt, user_prompt_template, ordered_field_names)."""
    fields = [k for k in taxonomy if not k.startswith("_")]

    field_blocks = "\n\n".join(_build_field_block(f, taxonomy[f]) for f in fields)

    shape_lines = ["{"]
    for f in fields:
        shape_lines.append(f'  "{f}": {{ "value": <see field rule>, "quote": "<≤25-word span from card or sentinel>" }},')
    shape_lines[-1] = shape_lines[-1].rstrip(",")
    shape_lines.append("}")
    shape = "\n".join(shape_lines)

    system = (
        "You extract constrained structured metadata from HuggingFace dataset cards. "
        "For slug-valued fields, you MUST return one of the provided slug values verbatim — no paraphrases, no combined values like 'a / b'. "
        "For each field, also return `quote`: a short verbatim span (≤25 words) from the card that justified your choice. "
        "If the card is genuinely silent on a field, return the `not_stated` slug (or its field-specific equivalent) and set quote to 'not_stated'. "
        "Output strictly valid JSON. No prose, no markdown fences, no commentary outside the JSON object."
    )

    user_template = (
        f"Extract the following fields from the dataset card.\n\n"
        f"{field_blocks}\n\n"
        f"Output shape (return EXACTLY this structure with the field values filled in):\n"
        f"{shape}\n\n"
        f"Dataset card for `<<REPO_ID>>`:\n---\n<<CARD>>\n---"
    )

    return system, user_template, fields


async def _extract_one(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
    system: str,
    user_template: str,
    repo_id: str,
    card: str,
) -> dict:
    truncated = card[:CARD_CHAR_LIMIT]
    user_msg = user_template.replace("<<REPO_ID>>", repo_id).replace("<<CARD>>", truncated)

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
                "card_chars_sent": len(truncated),
                "card_was_truncated": len(card) > CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt * 2)
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


async def run_extractions(sample: pd.DataFrame, system: str, user_template: str) -> None:
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
        res = await _extract_one(client, sem, system, user_template, repo_id, card)
        _save_result(res)

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


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


def _allowed_slugs(spec: dict) -> set[str]:
    if "categories" not in spec:
        return set()
    return {c["name"] for c in spec["categories"]}


def _validate(parsed: dict, taxonomy: dict, fields: list[str]) -> list[str]:
    """Return a list of validation issue strings; empty list = clean."""
    issues = []
    for f in fields:
        spec = taxonomy[f]
        t = spec["type"]
        entry = parsed.get(f)
        if entry is None:
            issues.append(f"{f}: missing")
            continue
        if not isinstance(entry, dict) or "value" not in entry:
            issues.append(f"{f}: malformed_entry")
            continue
        val = entry["value"]
        allowed = _allowed_slugs(spec)
        if t == "single-select":
            if not isinstance(val, str) or val not in allowed:
                issues.append(f"{f}: invalid_slug:{val!r}")
        elif t == "multi-select":
            if not isinstance(val, list) or not all(isinstance(v, str) for v in val):
                issues.append(f"{f}: not_list")
            elif any(v not in allowed for v in val):
                bad = [v for v in val if v not in allowed]
                issues.append(f"{f}: invalid_slugs:{bad}")
        elif t in ("single-select-or-list", "open-list"):
            if not isinstance(val, list):
                issues.append(f"{f}: not_list")
        elif t == "boolean":
            if not isinstance(val, bool):
                issues.append(f"{f}: not_bool:{val!r}")
    return issues


def aggregate(taxonomy: dict, fields: list[str]) -> pd.DataFrame:
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        parsed, parse_err = _parse_json(data.get("raw_text"))
        issues = _validate(parsed, taxonomy, fields) if parsed else []
        rows.append(
            {
                "repo_id": data["repo_id"],
                "raw_text": data.get("raw_text"),
                "error": data.get("error"),
                "parse_error": parse_err,
                "validation_issues": "; ".join(issues) if issues else None,
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
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
    print(f"Aggregated {n} results → {EXTRACTIONS_PARQUET}")
    print(f"  API errors:       {df['error'].notna().sum()}")
    print(f"  Parse errors:     {df['parse_error'].notna().sum()}")
    print(f"  Validation issues: {df['validation_issues'].notna().sum()}")
    if n:
        print(f"  Input tokens:    {df['input_tokens'].sum():,}")
        print(f"  Output tokens:   {df['output_tokens'].sum():,}")
    return df


def _counts(parsed_jsons: list[dict], field: str) -> Counter:
    c = Counter()
    for p in parsed_jsons:
        v = p.get(field, {}).get("value")
        if isinstance(v, list):
            c.update(v)
        else:
            c[v] += 1
    return c


def summarize_and_compare(v2_df: pd.DataFrame, fields: list[str]) -> None:
    # v2 parsed (only clean rows)
    v2_clean = v2_df[v2_df.parse_error.isna() & v2_df.validation_issues.isna() & v2_df.error.isna()]
    v2_parsed = [json.loads(p) for p in v2_clean.parsed_json]

    # v1 parsed — reuse the same repo_ids so the comparison is apples-to-apples.
    v1_df = pd.read_parquet(V1_EXTRACTIONS_PARQUET)
    v1_df = v1_df[v1_df.parse_error.isna() & v1_df.error.isna()]
    v1_df = v1_df[v1_df.repo_id.isin(v2_clean.repo_id)]
    v1_parsed = [json.loads(p) for p in v1_df.parsed_json]

    print(f"\n── v1 ↔ v2 distribution comparison (n={len(v2_parsed)}) ──")
    summary = {"n_v2_clean": len(v2_parsed), "n_v1_for_compare": len(v1_parsed), "fields": {}}
    for f in fields:
        v1c = _counts(v1_parsed, f)
        v2c = _counts(v2_parsed, f)
        summary["fields"][f] = {
            "v1_top10": v1c.most_common(10),
            "v2_top10": v2c.most_common(10),
            "v1_unique": len(v1c),
            "v2_unique": len(v2c),
        }
        print(f"\n  {f}:  v1_unique={len(v1c)}  v2_unique={len(v2c)}")
        print(f"    v2 top counts:")
        for val, n in v2c.most_common(10):
            print(f"      {n:4d}  {val!r}")

    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        os.replace(tmp_path, SUMMARY_JSON)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"\nWrote {SUMMARY_JSON}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    system, user_template, fields = build_prompts(taxonomy)
    print(f"Fields: {fields}")
    print(f"System prompt: {len(system)} chars | User template: {len(user_template)} chars")

    # Reuse exactly the v1 sample (keyed by repo_id). Pull card text fresh
    # from datasets.parquet in case sample.parquet is missing a column.
    sample = pd.read_parquet(V1_SAMPLE_PARQUET)
    full = pd.read_parquet(DATASETS_PARQUET)[["repo_id", "card_text_clean"]]
    sample = sample[["repo_id"]].merge(full, on="repo_id", how="left")
    sample = sample[sample.card_text_clean.notna() & (sample.card_text_clean.str.len() > 0)]
    print(f"Sample: {len(sample)} rows")

    if not args.aggregate_only:
        asyncio.run(run_extractions(sample, system, user_template))

    df = aggregate(taxonomy, fields)
    summarize_and_compare(df, fields)


if __name__ == "__main__":
    main()
