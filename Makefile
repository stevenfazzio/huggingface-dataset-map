.PHONY: install lint format test pipeline clean

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

pipeline:
	uv run python pipeline/00_fetch_datasets.py
	uv run python pipeline/01_embed_cards.py
	uv run python pipeline/02_reduce_umap.py
	uv run python pipeline/03_label_topics.py
	uv run python pipeline/04_extract_structured.py
	uv run python pipeline/05_visualize.py

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
