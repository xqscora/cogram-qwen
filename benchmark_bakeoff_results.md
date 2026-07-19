# LoCoMo Retrieval Bake-off — Matched ~2000-Token Budget

**Date:** 2026-07-18 22:27 UTC  
**Model (answer + judge):** `openai/gpt-4o-mini`  
**Conversation:** `conv-26` (419 turns)  
**Questions:** 15 (categories 1–4; category 5 skipped)  
**Shared context budget:** ~2000 tokens (tiktoken cl100k_base)  
**Cogram promotion:** novelty + surprise-mass (every content token; cap by Σ incident edge weight). Replaces the old frequency gate (count>=2).  
**Cogram decay:** experienced ticks (1 tick per LoCoMo session, tick=19), not wall-clock days.

### Per-category counts

| Category | Count |
|----------|------:|
| 1 | 7 |
| 2 | 7 |
| 3 | 1 |

## Results (sorted by accuracy)

| Retriever | Represents | Accuracy | Avg ctx tokens | Build cost |
|-----------|------------|----------|---------------:|------------|
| cooc_ppr | SPRIG / MemGraph-style PPR | 40.0% (6/15) | 1,996 | 0 (statistics) |
| bm25 | Lexical / agent-knowledge | 33.3% (5/15) | 1,994 | 0 (statistics) |
| cogram_bm25 | Cogram substrate + pluggable BM25 ranker | 26.7% (4/15) | 1,994 | 0 (statistics) |
| cogram | Cogram (zero-LLM concept graph) | 20.0% (3/15) | 1,984 | 0 (statistics) |
| random | Random floor | 20.0% (3/15) | 1,988 | 0 (statistics) |
| recency | Naive recency | 6.7% (1/15) | 1,999 | 0 (statistics) |

## Headline

At equal ~2000-token budget, **Cogram ranks #4 among zero-LLM retrievers** (20.0% accuracy). Best zero-LLM retriever: **cooc_ppr** (40.0%).

## Honest read

The count>=1 promotion fix raised Cogram from 16.7% (old benchmark_qa, count>=2) to 20.0% at matched budget. BM25 (33.3%) beats Cogram (20.0%) at equal budget — lexical matching is strong on LoCoMo factoid QA. Cogram is not the top zero-LLM retriever here; #4 of 6 at 20.0%. Vector/embedding baseline omitted for time; it is also the only baseline with a non-zero build cost.

Raw per-question data: `benchmark_bakeoff_raw.json`
