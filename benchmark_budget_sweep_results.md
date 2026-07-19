# LongMemEval Token-Budget Sweep — bm25 vs cogram_bm25

**Date:** 2026-07-19  
**Dataset:** `longmemeval_s_cleaned.json` (48 questions, seed=42, 8/type × 6 types)  
**Retrievers:** `bm25`, `cogram_bm25` only (no `full_context`)  
**Model (answer + judge):** `openai/gpt-4o-mini`  
**Hypothesis tested:** Cogram concept-activation prior keeps accuracy higher than plain BM25 at low token budgets (500–1000), even if they tie at 2000.

---

## LLM-judged accuracy vs budget

| Budget (tokens) | bm25 acc | cogram_bm25 acc | avg ctx tokens (cogram_bm25) |
|----------------:|---------:|----------------:|-----------------------------:|
| 500 | **N/A** (OpenRouter 402) | **N/A** (OpenRouter 402) | 493 |
| 1000 | **N/A** (OpenRouter 402) | **N/A** (OpenRouter 402) | 992 |
| 1500 | **N/A** (OpenRouter 402) | **N/A** (OpenRouter 402) | 1,492 |
| 2000 | **58.3% (28/48)** | **58.3% (28/48)** | 1,992 |

> **API blocker:** OpenRouter key hit HTTP 402 mid-sweep (`Prompt tokens limit exceeded: ~285–475` depending on remaining credits). New answer+judge calls failed for budgets ≤1500 during this session. **Budget 2000 rows reused** from existing `benchmark_longmem_raw.json` (Round 3, complete 48/48).

---

## Memory-heavy categories — LLM accuracy @ 2000 only

| Category | bm25 | cogram_bm25 |
|----------|-----:|------------:|
| knowledge-update | 37.5% (3/8) | 37.5% (3/8) |
| multi-session | 50.0% (4/8) | 50.0% (4/8) |
| temporal-reasoning | 50.0% (4/8) | 50.0% (4/8) |

**Per-question check:** 0/48 questions differ in correctness between bm25 and cogram_bm25 at budget 2000 — identical preds on every item.

---

## Retrieval-only diagnostics (no LLM; local re-run)

Extractive **gold-in-context** rate = gold answer substring appears in packed context. **Not** LLM accuracy — shown because API sweep blocked.

| Budget | identical line-sets / 48 | gold-in-ctx bm25 | gold-in-ctx cogram | retrieval diffs (non-identical) |
|-------:|-------------------------:|-----------------:|-------------------:|--------------------------------:|
| 500 | 38/48 | 77.1% | 77.1% | 10 |
| 1000 | 36/48 | **81.2%** | 79.2% | 12 |
| 1500 | 38/48 | 87.5% | 87.5% | 10 |
| 2000 | 34/48 | 89.6% | 89.6% | 14 |

Memory categories @ 1000 (extractive gold-in-ctx): multi-session bm25 **87.5%** vs cogram **75.0%** — bm25 retrieval slightly better, not cogram.

---

## Honest read

**cogram_bm25 does not beat bm25 at any budget we could LLM-evaluate.** At 2000 tokens they **tie exactly** (28/48, identical per-question outcomes including knowledge-update, multi-session, and temporal-reasoning). The low-budget hypothesis **cannot be confirmed or refuted on LLM accuracy** because OpenRouter credits exhausted before 500/1000/1500 answer+judge runs completed.

Retrieval-only signals also **do not support** the story: at 500 and 1500 extractive gold-in-context ties; at 1000 bm25 is slightly ahead (81.2% vs 79.2% overall; multi-session 87.5% vs 75.0%). Cogram's prior changes line ranking on 10–14/48 questions per budget but **does not translate into higher judged accuracy** at 2000 where we have ground-truth LLM eval.

**Bottom line for pitch:** No evidence yet that cogram_bm25 saves tokens *and* beats bm25 on LongMemEval_S at matched budgets. They are the same system in practice on this slice at 2000 tok. Re-run 500/1000/1500 LLM eval after topping up OpenRouter credits to close the gap.

---

## Code changes

| File | Change |
|------|--------|
| `benchmark_bakeoff.py` | `TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "2000"))` |
| `benchmark_longmem.py` | Budget-specific raw cache (`benchmark_longmem_raw_b{budget}.json`); `LONGMEM_RETRIEVERS` env; `token_budget` in cache validation + stored per retriever row; skip `full_context` build when not in retrievers; `LONGMEM_ANSWER_MAX_TOKENS` (default 64) |
| `benchmark_budget_sweep.py` | Orchestrator for multi-budget runs |
| `benchmark_budget_sweep_retrieval.py` | Local retrieval-only sweep (no API) |
| `_probe_prompt_limit.py` | OpenRouter prompt-limit probe utility |

Existing `benchmark_longmem_raw.json` (2000-budget, incl. full_context rows) **untouched**.

---

## LLM cost this session

| Item | Calls |
|------|------:|
| Probe tests (250–500 × 1 Q) | ~3 successful answer calls |
| Failed budget=500/300 partial runs | ~40+ failed attempts (402, no charge) |
| **New judged accuracy data** | **0** (2000 reused from cache) |
| Retrieval-only sweep | 0 API calls |

**Estimated spend:** ≈ $0.002 (probe only). Full sweep would need ~192 calls/budget × 3 budgets ≈ 576 calls (~$0.45 at ~2000 ctx) once credits restored.

---

## How to re-run after credits restored

```powershell
cd path/to/cogram
$env:LONGMEM_RETRIEVERS = "bm25,cogram_bm25"
foreach ($b in @(500, 1000, 1500)) {
  $env:TOKEN_BUDGET = "$b"
  python benchmark_longmem.py
}
python benchmark_budget_sweep.py --report-only
```

Raw caches: `benchmark_longmem_raw_b500.json`, `_b1000.json`, `_b1500.json` (separate from 2000 master cache).
