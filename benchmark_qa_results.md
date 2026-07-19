# LoCoMo QA Benchmark — Cogram vs Full Context

**Date:** 2026-07-18 21:29 UTC  
**Model (answer + judge):** `openai/gpt-4o-mini`  
**Conversation:** `conv-26` (419 turns)  
**Questions:** 30 (categories 1–4 only; category 5 adversarial skipped)

### Per-category counts

| Category | Count |
|----------|------:|
| 1 | 8 |
| 2 | 8 |
| 3 | 7 |
| 4 | 7 |

## Results

| Condition | Accuracy | Avg context tokens |
|-----------|----------|-------------------:|
| full_context | 53.3% (16/30) | 15,961 |
| cogram | 16.7% (5/30) | 1,961 |

## Headline

**Cogram uses ~8.1× fewer context tokens at 31% of full-context accuracy** (16.7% vs 53.3%).

## Honest read

Cogram accuracy (16.7%) was substantially lower than full-context (53.3%) despite ~8× token savings. On a single short conversation, statistical concept retrieval misses evidential turns that keyword-sparse questions need; full-context wins trivially when everything fits in window. Cogram's value proposition is accuracy-per-token on corpora too large to stuff.

Raw per-question data: `benchmark_qa_raw.json`
