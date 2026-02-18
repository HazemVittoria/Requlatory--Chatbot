# QA Contract v1 (Frozen)

Status: Stable  
Version: `v1`  
Scope: Retrieval + 3-phase QA behavior and CI gates for terminal pipeline.

## 1) Retrieval Rules

### Core retrieval settings
- `top_k = 8`
- `max_context_chunks = 6`
- `min_similarity = 0.20`
- `use_mmr = true`
- `mmr_lambda = 0.60`

### Fallback retrieval
- `fallback_min_similarity = 0.18`
- `fallback_only_if_empty = true`
- Fallback is attempted only when no chunks are returned at `0.20`.
- Fallback acceptance gate:
1. top-1 fallback score must be `>= 0.18`
2. top-1 PDF name must match query keywords/doc hints

### Query expansion rules (Phase A only)
- Always run retrieval with original query.
- Current targeted expansion:
1. If query contains `"continuous processing"`:
2. Add expansion term `"continuous manufacturing"` (if missing)
3. Add expansion term `"ICH Q13"` (if missing)
- Retrieval is run for original + expanded variants; results are merged by `(file, page, chunk_id)` and deduplicated by best score.

### Additional retrieval constraints
- Scope filter by authority is enforced (`MIXED|FDA|EMA|ICH|SOPS` mapping).
- If no accepted chunks after retrieval/fallback flow: return `Not found in provided PDFs`.

## 2) Phase Contracts

## Phase A: Retrieval (facts only)
Input:
- `user_question`
- indexed corpus chunks with metadata (`file/source_pdf_name`, `page`, `chunk_id`, `authority`, `section`)

Output in-memory schema (`Fact`):
- `quote: str`
- `pdf: str`
- `page: int`
- `chunk_id: str`
- `score: float`

Debug text contract:
- `FACTS:`
- `- "<quote>" (pdf="<name>", page=<int>, chunk_id="<id>", score=<float>)`

Hard rule:
- Phase A does not synthesize answers.

## Phase B: Fact filtering
### LLM Phase B JSON schema
```json
{
  "relevant_facts": [
    {"quote": "...", "pdf": "...", "page": 1, "chunk_id": "..."}
  ]
}
```

### Validator checks (`_validate_relevant_facts`)
- `relevant_facts` must be a list.
- Each row must include valid `pdf`, `page` (int), `chunk_id`.
- Fact must map to an existing Phase A fact key `(pdf,page,chunk_id)`.
- Duplicate keys are dropped.
- Output facts inherit original score from Phase A.

### Local (non-LLM) filter behavior
- Remove noisy/artifact quotes.
- Keep only question-relevant facts (token overlap/domain-specific checks).
- Deduplicate by normalized similarity key.
- Keep up to `max_context_chunks` facts.

## Phase C: Answer synthesis
### LLM Phase C JSON schema
```json
{
  "answer_sentences": [
    {"sentence": "...", "pdf": "...", "page": 1, "chunk_id": "..."}
  ],
  "confidence": "High|Medium|Low"
}
```

### Validator checks (`_validate_synthesis`)
- `answer_sentences` must be a list.
- Max 6 accepted sentences.
- Each sentence row must include valid `pdf`, `page` (int), `chunk_id`, `sentence`.
- Sentence must map to Phase B relevant fact key `(pdf,page,chunk_id)`.
- Duplicate keys are dropped.
- If no valid sentence remains: `Not found in provided PDFs`.
- Confidence is normalized to `High|Medium|Low` (invalid -> `Low`).

### Output text contract
- `ANSWER:`
- numbered sentences, each with citation `(pdf, p<page>, <chunk_id>)`
- final line: `CONFIDENCE: High|Medium|Low`

### Refusal contract
- Refusal text is exact: `Not found in provided PDFs`

## 3) CI Gates

Run both:
- baseline dataset: `golden_set.jsonl`
- perturbation dataset: `golden_set_perturb.jsonl`

Required gates:
- Baseline pass rate `>= 0.95`
- Perturb pass rate `>= 0.90`
- Hallucination rate must equal `0.00` (baseline and perturb)
- Fallback usage on answerable subset:
  - `fallback_used_rate_answerable <= 0.15`
- Incorrect refusal thresholds:
  - baseline `<= 0.02`
  - perturb `<= 0.02`

Tracked eval metrics:
- `correct_refusal`
- `incorrect_refusal`
- `hallucination`
- `phase_a_fallback_used_count`
- `phase_a_fallback_used_rate`
- `fallback_used_rate_answerable`

Alert/CI failure message for fallback overuse:
- `Fallback retrieval triggered too often; check embeddings/index changes or similarity calibration.`

## 4) Change Policy

1. Any threshold change requires before/after eval comparison on:
- `golden_set.jsonl`
- `golden_set_perturb.jsonl`

2. Any new query expansion rule requires:
- at least one targeted perturb test proving benefit
- no regression on hallucination or fallback overuse gates

3. Any new document added under `data/` requires:
- index rebuild
- full eval run (baseline + perturb)

4. No direct prompt-only tuning can bypass this contract.
- Retrieval/validation behavior changes must be reflected in this file by version bump (`v1 -> v2`).

