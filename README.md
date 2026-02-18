# Regulatory Chatbot (Terminal)

Terminal-first QA over regulatory PDF files organized by authority folders under `data/`.

## What It Does
- Reads PDFs from authority folders (for example: `fda/`, `ich/`, `ema/`).
- Extracts and chunks text with page + section metadata.
- Builds a cached TF-IDF retrieval index.
- Answers questions with grounded citations.
- Evaluates quickly using `golden_set.jsonl`.

## Determinism Policy (LLM Layer)
- Global defaults for all LLM calls:
  - `temperature=0.2`
  - `top_p=1.0`
  - `frequency_penalty=0`
  - `presence_penalty=0`
- Fallback mode:
  - `temperature=0.0` via env `LLM_FORCE_FALLBACK_TEMPERATURE_ZERO=1`
- Deterministic settings are enforced in `src/llm_client.py` and per-call overrides are blocked by default.

## Enforced 3-Phase Architecture
- Phase A Retrieval:
  - `top_k=8`, `max_context_chunks=6`, `min_similarity=0.20`, `use_mmr=true`, `mmr_lambda=0.60`
  - Output is facts only (snippet + `pdf/page/chunk_id/score`)
  - Empty -> `Not found in provided PDFs`
- Phase B Fact Filtering:
  - Deduplicate and keep only directly relevant facts
  - Empty -> `Not found in provided PDFs`
- Phase C Answer Synthesis:
  - Uses only relevant facts
  - Citation per sentence required
  - Max 6 sentences
  - Output includes `ANSWER:` block and `CONFIDENCE: High|Medium|Low`
  - Implemented with LLM calls when `OPENAI_API_KEY` is set and `USE_LLM_PHASES=1` (default on).
  - If no LLM is configured, deterministic local fallback is used.

## Setup
1. Install dependencies.
2. Run a first query with index rebuild.

## Usage
- Ask a question:
  - `python -m src.cli "What is data integrity?"`
- Force fresh index:
  - `python -m src.cli --rebuild-index "What is data integrity?"`
- JSON output:
  - `python -m src.cli --json "How should deviations be handled?"`

## Useful Options
- `--scope MIXED|FDA|EMA|ICH|SOPS`
- `--topk 5`
- `--chunk-chars 1000`
- `--min-chunk-chars 220`

## Evaluate Golden Set
- `python -m src.eval_runner`
