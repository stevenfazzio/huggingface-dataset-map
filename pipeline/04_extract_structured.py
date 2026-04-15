"""Extract structured fields from dataset cards via Claude Haiku.

Loads the enum/rule schema from `pipeline/taxonomy.json`, calls Haiku once per
dataset, validates responses against the schema, and emits a flat-column
parquet for the visualize stage. Resumable: per-repo JSON files in
`data/structured_fields_cache/` are skipped on rerun.

Related tooling (mostly in `experiments/`):
- `pipeline/05_visualize.py` coerces invalid slugs to 'other' for rendering,
  so phantom legend entries don't appear; truth stays in the parquet.
- `experiments/rerun_validation_sample.py` runs an EVoC-stratified ~130-card
  sample to preview prompt/taxonomy changes before paying for a full 5K rerun.
- `experiments/taxonomy_gap_analysis.py` and `experiments/evoc_cluster_signatures.py`
  surface candidate taxonomy improvements (new slugs, redundancies, extraction-
  quality issues). Run these first when considering taxonomy edits.
- Post-hoc canonicalization of `upstream_models` happens in `aggregate()`
  (collapses GPT-4 / gpt-4 / GPT4 to a single display form).

Iteration history for the schema itself lives in `pipeline/taxonomy.json`'s
`_comment` field and in `experiments/taxonomy_v{2,3}_proposed.json`.
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
from anthropic import AsyncAnthropic
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_EXTRACT,
    DATASETS_PARQUET,
    EXTRACT_CARD_CHAR_LIMIT,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_RETRIES,
    STRUCTURED_FIELDS_CACHE_DIR,
    STRUCTURED_FIELDS_PARQUET,
    TAXONOMY_JSON,
)
from tqdm import tqdm


def _safe_filename(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", repo_id)


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
            else (
                "If NO slug applies or the card is silent, return an empty list []. "
                "Do NOT invent 'not_stated' for this field."
            )
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


def build_system_prompt(taxonomy: dict) -> tuple[str, list[str]]:
    fields = [k for k in taxonomy if not k.startswith("_")]
    field_blocks = "\n\n".join(_field_block(f, taxonomy[f]) for f in fields)

    shape = "{\n"
    for f in fields:
        shape += f'  "{f}": {{ "value": <per field rule>, "quote": "<≤25-word span or sentinel>" }},\n'
    shape = shape.rstrip(",\n") + "\n}"

    system = (
        "You extract constrained structured metadata from HuggingFace dataset cards.\n\n"
        "RULES:\n"
        "- For slug-valued fields, return one of the provided slugs verbatim. "
        "No paraphrases, no combined values like 'a / b'.\n"
        "- For LIST-typed fields, the `value` MUST be a JSON array even if only one item applies: "
        '`["item"]`. Never a bare string.\n'
        "- Each field captures a DIFFERENT axis. Do NOT reuse a slug from one field as the value "
        "for another field. Axis definitions: `subject_domain` = what the data is ABOUT (noun); "
        "`provenance_method` = HOW the data was created; `training_stage` = what the data is FOR "
        "in the training stack; `format_convention` = the SCHEMA SHAPE of each record; "
        "`special_characteristics` = orthogonal PROPERTIES (long-context, roleplay, "
        "multilingual-parallel, reasoning-traces, etc.). Example: a VQA dataset has "
        "`subject_domain='natural-images-and-video'` AND `format_convention='vqa'` — those are "
        "different fields describing different axes. A multilingual-parallel corpus has "
        "`special_characteristics=['multilingual-parallel']` AND a separate subject slug for the "
        "content topic AND a separate format slug for the record shape — do not put "
        "'multilingual-parallel' in subject_domain or format_convention.\n"
        "- For each field, include `quote`: a ≤25-word verbatim span from the card that justified "
        "your choice. If silent, use the sentinel 'not_stated' for the quote.\n"
        "- Output strictly valid JSON. No prose, no markdown fences, no commentary outside the JSON object.\n\n"
        "FIELD DEFINITIONS:\n\n"
        f"{field_blocks}\n\n"
        "OUTPUT SHAPE (fill in values; do not change the structure):\n"
        f"{shape}"
    )
    return system, fields


def _build_user_message(repo_id: str, card: str) -> str:
    return f"Dataset card for `{repo_id}`:\n---\n{card[:EXTRACT_CARD_CHAR_LIMIT]}\n---"


async def _extract_one(client, sem, system, repo_id, card) -> dict:
    last_err = None
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_EXTRACT,
                    max_tokens=1024,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": _build_user_message(repo_id, card)}],
                )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {
                "repo_id": repo_id,
                "raw_text": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "card_was_truncated": len(card) > EXTRACT_CARD_CHAR_LIMIT,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — one bad card shouldn't crash the batch
            last_err = e
            if attempt < EXTRACT_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    return {"repo_id": repo_id, "raw_text": None, "error": f"{type(last_err).__name__}: {last_err}"}


def _save_result(result: dict) -> None:
    out_path = STRUCTURED_FIELDS_CACHE_DIR / f"{_safe_filename(result['repo_id'])}.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STRUCTURED_FIELDS_CACHE_DIR, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def _run_extractions(rows: pd.DataFrame, system: str) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    todo = []
    for _, row in rows.iterrows():
        cache_path = STRUCTURED_FIELDS_CACHE_DIR / f"{_safe_filename(row['repo_id'])}.json"
        if cache_path.exists():
            continue
        todo.append((row["repo_id"], row["card_text_clean"]))
    print(f"{len(todo)} to extract ({len(rows) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(repo_id, card):
        res = await _extract_one(client, sem, system, repo_id, card)
        _save_result(res)

    tasks = [_do(rid, c) for rid, c in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


def _parse_json(raw_text):
    if raw_text is None:
        return None, "no_raw_text"
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def _validate(parsed, taxonomy, fields):
    issues = []
    for f in fields:
        spec = taxonomy[f]
        t = spec["type"]
        entry = parsed.get(f)
        if not isinstance(entry, dict) or "value" not in entry:
            issues.append(f"{f}:missing")
            continue
        val = entry["value"]
        allowed = {c["name"] for c in spec.get("categories", [])}
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


def _canonicalize_upstream_models(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse case/whitespace variants of upstream model names.

    Conservative: lowercase-key dedupe only — picks the most-frequent original
    casing as the display form. Never merges semantically distinct models
    (e.g. 'GPT-4' and 'GPT-4o' stay separate). Also dedupes within each row
    so `['GPT-4', 'gpt-4']` collapses to `['GPT-4']`.
    """
    from collections import Counter

    col = "upstream_models"
    if col not in df.columns:
        return df

    # Build lowercase → display-form map using corpus-wide frequency.
    lower_to_originals: dict[str, Counter] = {}
    for v in df[col].dropna():
        if not isinstance(v, str) or not v.startswith("["):
            continue
        try:
            items = json.loads(v)
        except json.JSONDecodeError:
            continue
        for item in items:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            lower_to_originals.setdefault(s.lower(), Counter())[s] += 1
    canonical = {k: c.most_common(1)[0][0] for k, c in lower_to_originals.items()}

    def _rewrite(v):
        if not isinstance(v, str) or not v.startswith("["):
            return v
        try:
            items = json.loads(v)
        except json.JSONDecodeError:
            return v
        out = []
        seen = set()
        for item in items:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            canon = canonical.get(s.lower(), s)
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
        return json.dumps(out, ensure_ascii=False)

    before = len(lower_to_originals)  # already the collapsed count
    raw_unique = len(
        {
            item.strip()
            for v in df[col].dropna()
            if isinstance(v, str) and v.startswith("[")
            for item in (json.loads(v) if v.startswith("[") else [])
            if isinstance(item, str) and item.strip()
        }
    )
    df[col] = df[col].apply(_rewrite)
    print(f"  upstream_models:   {raw_unique} raw uniques → {before} canonical")
    return df


def aggregate(taxonomy, fields) -> pd.DataFrame:
    rows = []
    for p in sorted(STRUCTURED_FIELDS_CACHE_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        parsed, parse_err = _parse_json(data.get("raw_text"))
        issues = _validate(parsed, taxonomy, fields) if parsed else []

        row = {
            "repo_id": data["repo_id"],
            "error": data.get("error"),
            "parse_error": parse_err,
            "validation_issues": "; ".join(issues) if issues else None,
            "card_was_truncated": data.get("card_was_truncated"),
        }
        for f in fields:
            entry = (parsed or {}).get(f) or {}
            val = entry.get("value")
            row[f] = json.dumps(val, ensure_ascii=False) if isinstance(val, list) else val
            row[f"{f}_quote"] = entry.get("quote")
        rows.append(row)

    df = pd.DataFrame(rows)
    df = _canonicalize_upstream_models(df)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STRUCTURED_FIELDS_PARQUET.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"{len(verify)} != {len(df)}"
        os.replace(tmp_path, STRUCTURED_FIELDS_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    n = len(df)
    print(f"\nWrote {n} rows to {STRUCTURED_FIELDS_PARQUET}")
    print(f"  API errors:        {df['error'].notna().sum()}")
    print(f"  Parse errors:      {df['parse_error'].notna().sum()}")
    print(f"  Validation issues: {df['validation_issues'].notna().sum()}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    STRUCTURED_FIELDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    taxonomy = json.loads(TAXONOMY_JSON.read_text())
    system, fields = build_system_prompt(taxonomy)
    print(f"Fields: {fields}")

    df = pd.read_parquet(DATASETS_PARQUET)
    df = df[df.card_text_clean.notna() & (df.card_text_clean.str.len() > 0)].reset_index(drop=True)
    print(f"Corpus: {len(df)} rows with cards")

    if not args.aggregate_only:
        asyncio.run(_run_extractions(df[["repo_id", "card_text_clean"]], system))

    aggregate(taxonomy, fields)


if __name__ == "__main__":
    main()
