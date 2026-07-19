# Cogram — Procedural Memory for Coding Agents

> **One-liner:** *Every coding agent re-solves problems it already solved last month. Cogram lets it remember how to work — learned with **zero LLM tokens**, recalled in **~200 tokens**, and it **never forgets a fix, only lets old ones fade**.*

---

## The problem (say this first)

Coding agents (SWE-agent, OpenHands, Claude Code, Cursor) are **amnesiacs**. Each run starts cold:

- They re-derive the same fix for the same error class every time (`ModuleNotFoundError`, "how do I run this repo's tests", the same library idiom).
- The "fix" to remembering is usually **stuff the history into context** → tens of thousands of tokens per turn, and it *still* degrades on long context.
- The alternative — **distill experience into skills** — is what everyone's building (SkillX, SkillRL, Trace2Skill, DevMemory)... **but they all use an LLM to do the distillation**, so *learning itself burns tokens*, and the skill libraries only grow (no forgetting, staleness "largely neglected" per the 2026 survey).

## What Cogram is

A **procedural-memory substrate**: it reads an agent's past work (trajectories: reasoning → command → error → fix → patch) and builds a **concept graph with line-level provenance** — **entirely with statistics, no LLM**.

| Property | Cogram | Skill-library systems (SkillX / SkillRL / DevMemory) |
|---|---|---|
| **Cost to learn** | **0 LLM tokens** (pure co-occurrence + surprise) | LLM distillation every time |
| **Recall cost** | **~200 tokens/query** | retrieval + often an LLM selector |
| **Forgetting** | **tick-decay, never delete** — stale fixes fade, re-light on demand | grows unbounded; staleness neglected |
| **Coarse→fine** | native concept hierarchy (the "missing diagonal") | fixed compression level |
| **Provenance** | every concept → exact `file:line` of the original fix | summary blackbox |

> The 2026 survey *Experience Compression Spectrum* (arXiv 2604.15877) names the field's open gap the **"missing diagonal"**: no system does adaptive coarse-to-fine compression. Cogram's concept hierarchy **is** that diagonal.

## The demo (real public data — nobody's private chats)

**Corpus:** 600 *successful* real GitHub-issue fixes from **`nebius/SWE-agent-trajectories`** (CC-BY 4.0) — 184,177 lines, ~2.53M tokens of "how work actually got done."

**Learned with 0 LLM tokens** → concept graph: 2,000 concepts, 34,385 edges, plus 32 level-1 "concept-of-concepts" motifs (recurring problem→fix clusters, reinforced across 87 distinct trajectories — see `CONCEPT_OF_CONCEPTS.md`), **4.4× compression** (index ~569k tokens vs raw 2.53M).

**Then a new problem comes in.** Cogram recalls the past fix:

| New problem (query) | Recall tokens | Points to |
|---|---:|---|
| binary logical operators / cognitive complexity | **~279** | `swe_0017…:367` — the exact file + the B1/B3 spec line |
| dacite `default_factory` on dataclass | **~160** | `swe_0379…:136` + the patch line in `dacite/dataclasses.py` |
| electionguard `words.py` bad input handling | **~371** | `swe_0596–0600…:11` — same issue reopened across 5 agent attempts, all correctly routed to `electionguard-python` |

**~2.5M-token history → ~160–370 tokens per recall = ~7,000–16,000× smaller than re-reading the trajectories**, and it lands on the exact line, not a paraphrase.

Live in the web demo: pull the **tick slider** — a fix untouched for many sessions fades; query it and it **re-lights** (decay-never-delete).

**Is that routing actually reliable, or cherry-picked?** We measured it: 150 sampled queries, zero LLM calls, checking whether the top recall lands in the *same repo* as the bug report. **43.3% top-1 / 48.0% top-3, vs. a 2.9%/8.6% random-chance baseline** (35 distinct repos in the corpus) — about **15× over chance**. Full method, and a real bug we found and fixed while measuring it (a boilerplate-hub-word problem, fixed with a self-computed IDF discount, no fixed threshold), in `benchmark_procedural_precision_results.md`.

## Why this wins the room

1. **It's the judges' own daily pain** — they use coding agents that forget.
2. **0-token learning** is a hard differentiator: every competitor pays an LLM to learn; we don't.
3. **Honest, real data** (CC-BY public trajectories), exact-line provenance = auditable, not vibes.

---

## Appendix — "Are you even good at *memory*?" (the badge)

> We didn't dodge the memory benchmarks — we **chose** the harder, more useful procedural track. On the standard conversational-memory benchmark we're already at parity with the best retriever, for a fraction of the tokens:

**LongMemEval (ICLR 2025), 48 questions stratified random, gpt-4o-mini answer+judge:**

| Retriever | Accuracy | Avg ctx tokens | Cost/Q |
|---|---:|---:|---:|
| **Cogram + BM25** | **58.3%** | **~2,000** | **$0.0008** |
| BM25 alone | 58.3% | ~2,000 | $0.0008 |
| full context (up to ~110k) | 35.4%* | ~89,000 | $0.027 |

*≈45× fewer context tokens than full-context; full-context row ran at mixed truncation caps (API limit) so read it as "token economics + parity with the best budget retriever," not "we beat full context."*

Token-budget sweep: at 2,000 tokens engram_bm25 and bm25 tie on **every one of 48 questions** (0 disagreements), including the memory-heavy categories (knowledge-update, multi-session, temporal). We do **not** claim a low-budget win — lower budgets (500/1000/1500) were blocked by an API credit limit, and the retrieval-only proxy shows no Cogram edge. **Honest read: on this benchmark our concept graph ties the best lexical ranker, it doesn't beat it.**

> Bottom line: **competitive on the memory track, deliberately racing the procedural one.** On conversational retrieval-QA, ranking is at ceiling and memory structure can't add accuracy — the LLM does the reasoning, retrieval only fetches evidence. Cogram's value shows up where that benchmark can't measure it: **0-token learning, tick-decay, and line-level provenance across a working agent's history** — i.e. procedural memory.
