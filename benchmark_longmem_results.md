# LongMemEval_S Benchmark — Cogram vs BM25 vs Full Context (Round 3)

**Date:** 2026-07-19 04:43 UTC  
**Model (answer + judge):** `openai/gpt-4o-mini`  
**Dataset:** `longmemeval_s_cleaned.json` (ICLR 2025 LongMemEval)  
**Questions:** 48 (stratified random: 8 per type × 6 types, seed=42)  
**Retrievers this round:** `full_context`, `bm25`, `cogram_bm25` — `cogram` / `cogram_expand` omitted (diagnostic done at n=24)  
**Budget retrievers:** ~2000 tokens (tiktoken cl100k_base)  
**Cogram graph:** 1 session = 1 tick (novelty+surprise promotion, tick decay)

### Selected per type

| question_type | count |
|---------------|------:|
| single-session-user | 8 |
| multi-session | 8 |
| knowledge-update | 8 |
| temporal-reasoning | 8 |
| single-session-preference | 8 |
| single-session-assistant | 8 |

### Sampling note (vs Round 2)

| Metric | Count |
|--------|------:|
| Round 2 overlap (cache-eligible question IDs) | 6 / 48 |
| Fresh questions (not in Round 2 first-4-per-type) | 42 / 48 |

## Overall results (sorted by accuracy)

| Retriever | Accuracy | Avg ctx tokens | Est. cost/Q (USD) |
|-----------|----------|---------------:|------------------:|
| bm25 | 58.3% (28/48) | 1,993 | $0.0008 |
| cogram_bm25 | 58.3% (28/48) | 1,992 | $0.0008 |
| full_context | 35.4% (17/48) | 89,469 | $0.0270 |

## Per-category accuracy

| question_type | full_context | bm25 | cogram_bm25 |
|---------------|---::---::---:|
| single-session-user | 75% | 100% | 100% |
| multi-session | 38% | 50% | 50% |
| knowledge-update | 62% | 38% | 38% |
| temporal-reasoning | 12% | 50% | 50% |
| single-session-preference | 0% | 25% | 25% |
| single-session-assistant | 25% | 88% | 88% |

## Token economics

| Metric | Value |
|--------|------:|
| Avg full_context tokens | 89,469 |
| Avg 2k-budget retriever tokens | 1,992 |
| Token ratio (full / budget) | 44.9× |
| Cost ratio (full / budget, est.) | 34.7× |
| full_context truncation | **mixed caps** (110,000×39, 8,000×3, 3,000×6 rows) — default 30,000; lower caps used when OpenRouter credit prompt limit hit mid-run |
| questions with head+tail truncate | 48/48 |

## Cache / rerun notes

Round 3: stratified random sample, seed=42, 8 per type × 6 types = 48 questions.
Round 2 overlap: 6 question IDs (may reuse cached retriever rows).
Fresh vs Round 2 first-4-per-type: 42 questions.
Retrievers: full_context, bm25, cogram_bm25 only (cogram/cogram_expand dropped).
- full_context: reused=48, rerun=0
- bm25: reused=48, rerun=0
- cogram_bm25: reused=48, rerun=0

## Honest read

At n=48 (stratified random, seed=42), **cogram_bm25** (58%) ≥ BM25 (58%) overall. Best budget retriever (58%) matches or beats full-context (35%) on this slice. **knowledge-update:** cogram_bm25=38% ties bm25=38%. **multi-session:** full_context=38%, bm25=50%, cogram_bm25=50%. **single-session-preference:** full_context underperforms budget methods (preference questions may need targeted retrieval, not full haystack). Token economics: full-context ~45× more context tokens than budget retrievers (~35× estimated API cost). cogram / cogram_expand omitted in Round 3 (established at n=24). **full_context caveat:** **mixed caps** (110,000×39, 8,000×3, 3,000×6 rows) — default 30,000; lower caps used when OpenRouter credit prompt limit hit mid-run. Compare budget retrievers fairly; full_context is not a uniform ~110k baseline on this run.

Raw per-question data: `benchmark_longmem_raw.json`
