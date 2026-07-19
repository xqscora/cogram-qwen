# Cogram Web Demo

Interactive visualization of **Cogram** — a zero-LLM concept-network memory for AI agents. Built from a single [LongMemEval](https://arxiv.org/abs/2410.10824) (ICLR 2025) `knowledge-update` question; fully static, no runtime network calls.

## Quick start

```bash
cd web
python build_web_graph.py   # regenerate graph.json from LongMemEval data
python -m http.server 8000
```

Open **http://localhost:8000**

## What you see

| Control | Behavior |
|---------|----------|
| **Recall query** | Tokenize → seed concepts → 2-hop activation spread with edge pulse animation |
| **Tick slider** | Live decay: `eff = ε + (w−ε)·exp(−Δtick/50)` — edges fade but never hit zero |
| **Provenance** | Activated concepts → source transcript lines; answer lines get amber ring |
| **Knowledge update** | Old “one hour” vs updated “two hours” session lines |
| **Token counter** | ~2,000 budget vs full haystack (tiktoken cl100k_base) |

## Files

| File | Purpose |
|------|---------|
| `build_web_graph.py` | Builds `graph.json` from smallest `knowledge-update` haystack |
| `graph.json` | Exported nodes, edges, lines, pointers, meta |
| `index.html` / `app.js` / `style.css` | Static demo UI |
| `vendor/cytoscape.min.js` | Local Cytoscape (no CDN at runtime) |

## Data

- Source: `../data/longmemeval_s_cleaned.json` (public benchmark)
- Selected question: `cc5ded98` — smallest haystack by character count among `knowledge-update` items
- Graph capped to top ~200 concepts by surprise-mass for legibility

## Benchmark caption

Bottom strip reads numbers from `../benchmark_longmem_results.md` at build time (Round 3, n=48).

## Zero LLM

Graph construction uses statistical tokenization + Hebbian co-occurrence only (`graph_lib.py`, `benchmark_qa.build_graph`). No embeddings, no summarization, no API calls in the demo.
