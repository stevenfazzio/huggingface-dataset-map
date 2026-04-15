"""Fetch top HuggingFace datasets and their README cards.

Single-stage enumerate+fetch: HF Hub's list_datasets already returns
ranked + filterable results, so there's no need for a BigQuery-style
pre-enumeration pass.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from config import (
    CARD_MAX_CHARS,
    DATASETS_PARQUET,
    FETCH_OVERSHOOT_COUNT,
    HF_TOKEN,
    MIN_CARD_CHARS,
    RANK_SORT_KEY,
    TARGET_DATASET_COUNT,
)
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from tqdm import tqdm

CONCURRENT_DOWNLOADS = 8
MODALITY_TAG_PREFIX = "modality:"
TASK_CATEGORY_TAG_PREFIX = "task_categories:"
TASK_ID_TAG_PREFIX = "task_ids:"
LANGUAGE_TAG_PREFIX = "language:"
SIZE_TAG_PREFIX = "size_categories:"
LICENSE_TAG_PREFIX = "license:"


def _extract_tag_values(tags: list[str], prefix: str) -> list[str]:
    """Return values for tags matching `prefix:value`, without the prefix."""
    if not tags:
        return []
    return [t[len(prefix) :] for t in tags if t.startswith(prefix)]


def _parse_dataset_info(info) -> dict:
    """Parse a DatasetInfo object into a flat row dict (no card text yet)."""
    tags = list(info.tags or [])
    card = info.card_data or {}

    def _card_list(key: str) -> list[str]:
        val = card.get(key) if hasattr(card, "get") else None
        if val is None:
            return []
        if isinstance(val, str):
            return [val]
        return list(val)

    task_categories = _card_list("task_categories") or _extract_tag_values(tags, TASK_CATEGORY_TAG_PREFIX)
    task_ids = _card_list("task_ids") or _extract_tag_values(tags, TASK_ID_TAG_PREFIX)
    languages = _card_list("language") or _extract_tag_values(tags, LANGUAGE_TAG_PREFIX)
    size_categories = _card_list("size_categories") or _extract_tag_values(tags, SIZE_TAG_PREFIX)
    license_list = _card_list("license") or _extract_tag_values(tags, LICENSE_TAG_PREFIX)
    license_val = license_list[0] if license_list else ""
    modalities = _extract_tag_values(tags, MODALITY_TAG_PREFIX)
    multilinguality = _card_list("multilinguality")
    source_datasets = _card_list("source_datasets")
    pretty_name = ""
    if hasattr(card, "get"):
        pretty_name = card.get("pretty_name") or ""

    last_modified = getattr(info, "last_modified", None)
    last_modified_str = last_modified.isoformat() if last_modified else ""

    return {
        "repo_id": info.id,
        "pretty_name": pretty_name,
        "author": getattr(info, "author", "") or "",
        "likes": int(getattr(info, "likes", 0) or 0),
        "downloads": int(getattr(info, "downloads", 0) or 0),
        "last_modified": last_modified_str,
        "created_at": (info.created_at.isoformat() if getattr(info, "created_at", None) else ""),
        "private": bool(getattr(info, "private", False)),
        "disabled": bool(getattr(info, "disabled", False)),
        "gated": str(getattr(info, "gated", "") or ""),
        "tags": ",".join(tags),
        "task_categories": ",".join(task_categories),
        "task_ids": ",".join(task_ids),
        "languages": ",".join(languages),
        "multilinguality": ",".join(multilinguality),
        "size_categories": ",".join(size_categories),
        "modalities": ",".join(modalities),
        "license": license_val,
        "source_datasets": ",".join(source_datasets),
    }


def _fetch_card(repo_id: str, token: str, max_retries: int = 4) -> str:
    """Download the README.md card for a dataset. Returns '' on miss."""
    for attempt in range(max_retries):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                repo_type="dataset",
                token=token or None,
            )
            with open(path, encoding="utf-8") as f:
                return f.read()
        except (EntryNotFoundError, RepositoryNotFoundError):
            return ""
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (429, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            return ""
        except Exception:
            time.sleep(2 ** (attempt + 1))
    return ""


def _strip_yaml_frontmatter(text: str) -> str:
    """Drop leading YAML frontmatter block (--- ... ---) from a card."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :].lstrip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Ignore existing cache and refetch all.")
    args = parser.parse_args()

    api = HfApi(token=HF_TOKEN or None)

    print(f"Listing top {FETCH_OVERSHOOT_COUNT} datasets sorted by {RANK_SORT_KEY}...")
    infos = list(
        api.list_datasets(
            sort=RANK_SORT_KEY,
            limit=FETCH_OVERSHOOT_COUNT,
            full=True,
        )
    )
    print(f"Got {len(infos)} DatasetInfo records")

    rows = [_parse_dataset_info(info) for info in infos]

    existing_cards: dict[str, str] = {}
    if not args.refresh and DATASETS_PARQUET.exists():
        cached = pd.read_parquet(DATASETS_PARQUET)
        if "card_text" in cached.columns:
            for repo_id, text in zip(cached["repo_id"], cached["card_text"]):
                if isinstance(text, str) and text:
                    existing_cards[repo_id] = text
        print(f"Reusing {len(existing_cards)} cached cards")

    need_fetch = [r["repo_id"] for r in rows if r["repo_id"] not in existing_cards]
    print(f"Fetching {len(need_fetch)} README cards...")

    fetched: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=CONCURRENT_DOWNLOADS) as ex:
        futures = {ex.submit(_fetch_card, rid, HF_TOKEN): rid for rid in need_fetch}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Cards"):
            rid = futures[fut]
            try:
                fetched[rid] = fut.result()
            except Exception as e:
                print(f"  {rid}: {type(e).__name__}: {e}")
                fetched[rid] = ""

    for r in rows:
        rid = r["repo_id"]
        r["card_text"] = existing_cards.get(rid) or fetched.get(rid, "")
        stripped = _strip_yaml_frontmatter(r["card_text"])
        r["card_text_clean"] = stripped[:CARD_MAX_CHARS]

    df = pd.DataFrame(rows)
    df_kept = df[df["card_text_clean"].str.len() >= MIN_CARD_CHARS].copy()
    df_kept = df_kept.sort_values(RANK_SORT_KEY, ascending=False).head(TARGET_DATASET_COUNT).reset_index(drop=True)

    min_rank = int(df_kept[RANK_SORT_KEY].min()) if len(df_kept) else 0
    print(f"Kept {len(df_kept)} datasets (min {RANK_SORT_KEY}={min_rank})")

    tmp = DATASETS_PARQUET.with_suffix(".parquet.tmp")
    df_kept.to_parquet(tmp, index=False)
    tmp.replace(DATASETS_PARQUET)
    print(f"Wrote {DATASETS_PARQUET}")


if __name__ == "__main__":
    main()
