# Procedural recall precision (zero LLM calls)

**Question:** given a new bug report as the only input, does Cogram's concept graph route the citation to the *right project's* history, or does it get confused into an unrelated repo? This is the prerequisite for "recall the fix that generalizes" to mean anything — so it's the number we should be able to show, not just token savings.

**Method** (`benchmark_procedural_precision.py`, fully reproducible, no API calls):
- 600-row SWE-agent-trajectories slice, 35 distinct repos.
- 150 sampled trajectories (seed=42). For each, extract the actual GitHub issue text (stripping the SWE-agent prompt wrapper — a real bug report wouldn't come pre-wrapped in "We're currently solving the following issue... ISSUE:" anyway).
- Run it through the exact same ranking + provenance path `swe_recall_demo.py` uses. Check whether the returned line(s) come from the *same repo* as the query's source trajectory.
- Also reported on the 31 *distinct* issues in the sample (de-duplicating repeated agent attempts at the same GitHub issue), since the 600-row slice has some issues attempted multiple times by different agent/model runs.

**Result:**

| Metric | Cogram | Random-chance baseline |
|---|---:|---:|
| Top-1 same-repo hit rate (n=150) | **43.3%** | 2.9% (1/35 repos) |
| Top-3 same-repo hit rate (n=150) | **48.0%** | 8.6% (3/35 repos) |
| Top-1, distinct issues only (n=31) | **45.2%** | 2.9% |

~15x over chance at top-1, with zero LLM calls involved in either building the graph or answering the query.

## What this does and doesn't prove

- **Does prove:** the concept graph is doing real, repo-specific routing — it is not simply matching on generic English words or SWE-agent's shared prompt scaffolding (see below).
- **Doesn't prove:** generalization to bugs that share *no* vocabulary with anything previously seen. That's a harder claim this benchmark isn't built to test, and we're not claiming it.

## A bug we found and fixed while building this

The first version of this benchmark scored **4.7%** — barely above chance. Root cause: `swe_recall_demo.py`'s pointer-ranking multiplied a concept's raw activation score directly into the final line score, with no discount for how common that concept is. SWE-agent's shared shell-interface tips ("Always make sure to look at the currently open file...") appear in effectively all 600 transcripts, so words like *current*, *check*, *currently*, *runs* had huge activation (their edge weights compound across 600 co-occurrences) and drowned out the actually distinctive, bug-specific terms in the query.

**Fix:** added a standard IDF-style discount — `log((N+1)/df) / log(N+1)` — computed purely from each concept's own measured document frequency vs. corpus size (`swe_recall_demo.py::idf_weight`). No fixed threshold, no blacklist; a concept that's in every document is worth ~0 bits of routing information and gets discounted toward 0 automatically, a concept seen once keeps full weight. This is scoped to the SWE procedural-memory demo path only (`swe_recall_demo.py`) — it does not touch `recall.py`'s shared `seed_activation`/`spread_activation` used by the LongMemEval benchmark retrievers, so the already-reported memory-layer numbers (Cogram+BM25 ties BM25 at 58.3%) are untouched by this change.

That fix took the number from 4.7% → 43.3%.

## Reproduce

```bash
python fetch_swe.py
python swe_to_transcripts.py
python extract_swe.py
python benchmark_procedural_precision.py
```
