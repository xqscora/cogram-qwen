# Concept-of-concepts: promoting problem→fix motifs into higher-level procedural memory

Status: **mining + storage implemented** (`graph_lib.mine_motifs`, wired into
`extract_swe.py` → `swe_concept_graph.json["motifs"]`, demoed in
`swe_recall_demo.py` Demo 4). On the current 600-trajectory corpus this mines
**32 level-1 motifs, reinforced across 87 distinct-trajectory occurrences**,
after excluding hub/boilerplate concepts (the same `hub` flag already used
by the line-level IDF discount) and capping cluster size at 10 members
(Chase & Simon 1973's ~6-item chunk size) — without those two guards, the
same corpus produces a single 90-member junk motif made of SWE-agent's own
prompt-template boilerplate, which is exactly the failure mode this file
warns about below. **Not yet wired into `recall.py`'s live activation
spreading or the benchmarks** — see "Implementation sketch," step 3, which
is still open. Original design written after the hackathon submission
(2026-07-19) as the next real extension of Cogram, specifically for the
procedural-memory framing (`PITCH_PROCEDURAL.md`).

## The gap this fills

Right now (`extract_swe.py`, `graph_lib.py`) a "concept" is a single token —
a word or a Chinese bigram. Edges are pairwise co-occurrence between two
tokens on the same line, weighted by `hebbian_delta` (surprise on first
co-occurrence, diminishing returns after). That's level 0 of the graph:
individual words, densely cross-linked.

But a *procedure* — "when you see a `KeyError` on a `default_factory` dataclass
field, go check `dataclasses.py`, then patch the `_build_init_for_class`
helper" — is not one token. It's a **recurring ordered cluster** of level-0
concepts that fire together across *different* trajectories, not just
different lines of the same trajectory. That cluster is the actual reusable
unit of "how to work." Right now Cogram can recall the individual lines where
that cluster appeared, but it has no node that *represents* the cluster
itself. There's no "chunk."

This is exactly the chunking problem from cognitive psychology (Miller 1956,
"The Magical Number Seven"; Chase & Simon 1973 on chess chunking, where
experts don't see 32 pieces, they see ~6 recognized configurations) — expert
memory is not "more raw items remembered," it's "raw items pre-grouped into
fewer, larger, meaningful units." A level-1 concept-of-concepts node is
Cogram's version of a chess chunk: not a summary written by an LLM, but a
frozen, reusable pointer to "this specific set of level-0 concepts tends to
co-fire, and here's every place it did."

## What NOT to do (ruled out deliberately)

- **No LLM-written motif labels.** A meta-concept's "name" is just the sorted
  tuple of its constituent concept ids (e.g. `(dacite, dataclass, default,
  default_factory)`), same zero-token philosophy as level 0.
- **No fixed occurrence threshold to "promote" a motif** (not `df >= 2`,
  not `count >= 3`). Cora was explicit about this for level-0 concepts
  already (`extract_swe.py` has no such gate — every token that appears at
  all is a candidate, ranked by surprise mass only when a hard *resource*
  cap like `MAX_CONCEPTS` is hit). The same rule must hold one level up:
  a meta-concept is minted the *first* time a qualifying motif window is
  observed, with whatever weight the surprise formula gives it that instant.
  Whether it survives is entirely a function of the existing tick-decay +
  reinforcement math — never a separate promotion gate.
- **No wall-clock decay for motifs either.** Same `TAU_TICKS` / `EPS` floor
  as everything else — decay is counted in ticks (trajectories/sessions
  actually processed), so an idle Cogram doesn't forget just because time
  passed.

## The mechanism: reuse `hebbian_delta`, one level up

Level-0 surprise already answers "has this *pair* of tokens co-occurred
before, and how many times." A motif is a co-occurring *set* of pairs across
multiple *distinct files* (not just multiple lines in the same file — that's
just local repetition, not evidence of a generalizable procedure). So:

1. **Candidate windows.** For each transcript file, take the set of level-0
   concepts activated within a bounded line-window (e.g. ±15 lines around
   any line that already has ≥3 co-active concepts — this reuses the
   existing `lines_data` structure in `extract_swe.py`, no new tokenizer).
2. **Motif key.** `frozenset` of the level-0 concept ids in that window
   (order-insensitive key for dedup; the *directed* sequence is stored
   separately per occurrence — see "sequence" below).
3. **Motif surprise = mean pairwise surprise of its own edges.** For a
   candidate window with concepts `{a, b, c}`, look up the *already computed*
   edge weights `w(a,b)`, `w(a,c)`, `w(b,c)` from the level-0 graph and take
   their mean. This is not a new surprise formula — it's a direct reuse of
   `hebbian_delta`'s output, just aggregated over a cluster instead of a
   pair. A motif made of concepts that never co-occur elsewhere gets a high
   score (all its pairs were "first co-occurrence" events, i.e. surprising);
   a motif made of concepts that are individually hub-like/generic
   (`generality()` in `graph_lib.py` already computes this) gets discounted
   the same way pairwise edges already are.
4. **Minting, not gating.** As soon as a window's motif-surprise clears
   *whatever the corpus's own surprise distribution says is above-median*
   (a *relative*, self-calibrating cutoff — not a hardcoded number picked by
   a human) a level-1 node is created **the first time it's seen**, exactly
   like a level-0 token would be. Its initial weight is its motif-surprise
   score. No second occurrence required.
5. **Reinforcement = the same tick-decay math.** If the *same* motif key
   (same frozenset of level-0 concepts) shows up again in a *different*
   file later, that's a new `hebbian_delta(is_first=False, n_prev=n)` event
   on the level-1 node itself, using the exact function already in
   `graph_lib.py`. If it never recurs, the node decays toward `EPS` on the
   same tick clock as every other edge — it fades, it does not get deleted,
   and it is not treated any differently from a level-0 concept that turned
   out to be a one-off.

This means "concept-of-concepts" isn't a new subsystem bolted onto Cogram —
it's the *same* `hebbian_delta` / `effective_weight` pair of functions,
called on a different granularity of input. That's the whole design.

## Directed sequence (the "what leads to what")

Cora's original spec (before the pivot to procedural memory) already asked
for *directed* transitions, not just undirected co-occurrence — "a 可能包含
b 但 b 不包含 a." A motif node should store not just its member-concept set
but the **most common line-order** in which those concepts first appeared
within the window (a short ordered tuple, deduped across occurrences by
majority order). That's what turns "these concepts tend to appear together"
into "this is roughly the order you touch them in" — e.g. `grep error string
→ open file → read function → patch` — which is the actually actionable part
of a procedural memory: not just "these are related" but "this is usually
the order that resolves it."

## Provenance stays intact

A level-1 node's provenance list is simply the **union** of the `file:line`
spans of every window that minted or reinforced it. Retrieval never has to
trust the motif blindly — walking from a level-1 hit down to its member
level-0 concepts down to the exact lines is the same coarse-to-fine
traversal `recall.py` already does for level-0 → line. This is what makes it
still zero-hallucination: the meta-concept is a *pointer bundle*, never a
generated description.

## Retrieval benefit (why this is worth building)

Today, a query like "dataclass default_factory bug" has to activate the
individual tokens `dataclass`, `default`, `factory` separately and let them
spread through the level-0 graph to find the relevant lines (this is exactly
what `swe_recall_demo.py`'s Demo 2 shows). With a level-1 motif node for
`{dacite, dataclass, default, default_factory}` already sitting in the
graph, a *matching* new query can activate that single node directly — one
hop instead of averaging four separate token activations — which should
mean **fewer tokens spent per recall and a sharper/less noisy result**,
since the motif node already encodes "these four things go together" instead
of the retriever having to rediscover that at query time. This is the
concrete lever for the token-budget numbers to improve further beyond what's
in `benchmark_budget_sweep_results.md`.

## Implementation sketch (for whoever builds this next — Composer/Codex, not
the orchestrating agent, per the repo's own delegation rules)

1. Add `mine_motifs(lines_data, edge_idx, file_ticks) -> List[MotifCandidate]`
   to `graph_lib.py`, reusing `hebbian_delta`/`effective_weight` — no new
   decay or surprise math.
2. Add a `motifs` section to the graph JSON schema: `{key: frozenset-as-sorted-tuple,
   members: [...], order: [...], w_raw, last_tick, n, provenance: [[file,line],...]}`.
3. Extend `recall.py`'s activation spreading to check motif membership before
   falling back to token-level spreading (coarse-to-fine, motif first).
4. Extend `swe_recall_demo.py` with a 4th demo that shows a motif hit
   specifically, so the difference is visible (token-level vs motif-level
   recall side by side, same query, compare token cost).
5. Re-run `benchmark_budget_sweep.py` with motifs enabled vs disabled as an
   ablation — this is the number that would actually prove the concept, and
   it's a clean, honest ablation (same corpus, same benchmark, one knob).

## Relationship to Aura

This is the same idea as Aura's terrain/scar dynamics one level up: Aura
doesn't have a "personality" field, it has emergent behavior from local
deformation; Cogram shouldn't have a hand-written "motif detector," it
should have emergent chunk nodes from the same surprise math already
driving level-0 concepts. The through-line across both projects is: nothing
gets a privileged, hand-authored existence — everything that persists has to
earn it by being statistically surprising and then re-earn it every tick by
recurring, or it fades (never deletes) toward the floor.
