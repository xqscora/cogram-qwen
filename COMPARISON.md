# Memory Architecture Leaderboard — Cogram vs the field

> Numbers for other systems come from their own papers (different benchmarks — noted per row).
> Cogram numbers are measured live on our corpus (84 transcripts, 4.95M tokens, tiktoken cl100k_base).
> The honest comparison here is **mechanism + cost structure**, not head-to-head accuracy (we did not run LoCoMo/LongMemEval).

### Group A — LLM-built memory (the mainstream)

| System | Year / venue | Memory = | Write cost (LLM tokens) | Read cost / query | Forgetting | Provenance | Runs w/o any LLM |
|---|---|---|---|---|---|---|---|
| Full context | — | nothing (stuff it all) | 0 | ~16k (LoCoMo) / 115k (LongMemEval) / **4.95M ours** | none | trivial | yes |
| Rolling compaction (Claude Code / ChatGPT) | prod standard | LLM-written summary | rewrites each compaction | ~3k | **lossy — gone forever** | no | no |
| MemGPT / Letta (arXiv:2310.08560) | 2023 | OS-style paging, self-edited notes | LLM per memory op | varies | LLM evicts | partial | no |
| Mem0 (arXiv:2504.19413) | 2025 | LLM-extracted facts in vector DB | extraction **every write** | ~1.8k–7k | overwrite | no | no |
| Zep / Graphiti (arXiv:2501.13956) | 2025 | temporal KG, LLM-extracted | entity/edge extraction/write | ~1.6k (LongMemEval 71.2%) | temporal invalidation | episode links | no |
| AriGraph (IJCAI 2025) | 2025 | KG triplets + episodic nodes | LLM triplet extraction | retrieval | **overwrites** edges | episode links | no |
| HeLa-Mem (ACL 2026) | 2026 | Hebbian episodic graph | LLM encode + consolidate | retrieval | **prunes** weak nodes | node content | no |
| **MRAgent** "Reconstructed not Retrieved" (arXiv:2606.06036, NUS, ICML 2026) | 2026 | Cue-Tag-Content graph, coarse→fine | **LLM distillation at build** | **118k** (LongMemEval, beats baselines up to 23%) | n/a | content nodes | no |
| LLMLingua-2 (ACL Findings 2024) | 2024 | compressed text (token classifier) | small local model | 2–5× — **linear in corpus** | n/a | **lossy** | needs own model |

### Group B — zero-LLM statistical memory (the honest peer group — this is where Cogram lives, and it got crowded in 2026)

| System | Year / venue | Memory = | Forgetting | Provenance | Directed | Reported accuracy |
|---|---|---|---|---|---|---|
| **Cogram (ours)** | 2026 hackathon | concept graph, statistics only | **decay→ε>0, never deleted, re-lightable** | **file:line, lossless (gzip raw)** | **yes (`next`)** | measured on LoCoMo — see `benchmark_qa_results.md` |
| **SPRIG** (arXiv:2602.23372) | 2026-02 | NER + co-occurrence graph + Personalized PageRank | none (PPR) | entity spans | no | "matches/exceeds LLM graph build, 0 token cost, +28% faster retrieval" |
| **MemGraph-agent** (GitHub, impl. of SPRIG) | 2026 | NER co-occ graph + PPR, JSON portable, EN+中文 | none | entity spans | no | multi-hop recall +28% vs vector |
| **Mandol** (arXiv:2606.29778, 中科院+MSR) | 2026 | hierarchical SemanticMap+SemanticGraph, basic+abstract layers | agglomerative | traceable abstract mem | partial | LoCoMo/LongMemEval **best overall**, 5.4× retrieval speedup (retrieval LLM-free; **build may use LLM**) |
| **agent-knowledge** (GitHub) | 2026 | BM25 + KG + RRF, no vectors | append-only + contradiction detect | **claim/evidence provenance** | no | **96.6% R@5 LongMemEval-S** |
| **total-agent-memory** (GitHub) | 2026 | deterministic core + async AI enrich | dedup/cache | graph links | no | **96.2% R@5 LongMemEval** (fast mode = 0 LLM) |

## The honest read — Cogram is NOT the first zero-LLM concept-graph memory

Prior art checked 2026-07-18. **SPRIG (arXiv:2602.23372) already proved the core thesis** — statistical (NER + co-occurrence + PageRank) graph memory can match LLM-built graphs at zero token cost — and **MemGraph-agent open-sourced almost exactly this** (co-occurrence graph, frequency/time-weighted edges, JSON-portable, EN+中文, CPU-only). Mandol, agent-knowledge, and total-agent-memory pile on with real LoCoMo/LongMemEval accuracy. So "I invented zero-LLM concept memory" is **false and would be caught** — do not claim it.

### What Cogram still uniquely combines (none of the peers above hold all three)

1. **Forgetting with a floor.** Hebbian decay toward ε>0 — links weaken but are **never deleted, and can be re-lit by neighbors**. MemGraph/SPRIG use PageRank (no decay); agent-knowledge is append-only; Mandol/MRAgent/AriGraph/HeLa-Mem prune or overwrite. A memory that fades but survives, and can be reawakened, is Cogram's alone here.
2. **Surprise-weighted edges.** Edge strength uses an **impressive-surprise + novelty** bonus, not just frequency + timestamp (everyone else's rule). Salience, not just repetition.
3. **Line-level lossless provenance.** `file:line` pointers into **gzip'd bit-exact raw text** — recover the exact original line, never a summary. agent-knowledge has claim provenance but not line-level lossless raw.

### The one column the LLM-based group still can't touch

**Memory that is not made of AI output.** Mem0's facts, Zep's edges, MRAgent's distilled cues, every summary — all *LLM-generated text about your data*. Store AI output, re-process it, drift compounds — the model-collapse loop (Shumailov et al., *Nature* 2024). Cogram (and the Group-B peers) store **the data's own statistics + addresses into ground truth** — nothing generated, nothing to drift. Cogram's edge over Group B is the three-way combo above, not the zero-LLM idea itself.

## Fine print

- Cogram's own LoCoMo QA accuracy (concept-retrieval vs full-context) is measured in `benchmark_qa_results.md` on this corpus with an InsForge-gateway LLM judge. Cross-system accuracy is **not** head-to-head: each row's number is from its own paper on its own split.
- LLMLingua-2 is **lossy** — a model classifier deletes tokens irreversibly. We reject it for the ground-truth layer on principle: a model deciding what to drop is a preset, and deleted tokens are unrecoverable. Cogram's ground truth uses **lossless gzip** instead (`compress_store.py`): ~3-4× smaller on disk, bit-exact recovery, zero models involved. Storage cost and token cost are decoupled axes — lossless compression handles disk, the concept graph handles context.
- Full-context is the accuracy ceiling on small corpora (~73% J on LoCoMo) but is already impossible on ours: 4.95M tokens exceeds every production context window.
