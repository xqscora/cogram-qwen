# Cogram API recall correction

The Qwen adapter calls `memory_api.recommend()`, whereas the original procedural precision benchmark uses `swe_recall_demo.collect_pointers()`. Their ranking policies were not equivalent. A result from the demo must not be described as a measured result of the API adapter.

## Failure and repair

The actual API returned unrelated electionguard shell-context lines for `dacite default_factory dataclass`. High graph activation drowned out direct query evidence. Source citations were genuine, but relevance was wrong.

The correction:

1. Discounts concepts by their observed document frequency before selecting each source line's route.
2. Prioritizes direct query-concept evidence over indirect graph hops.
3. Keeps direct citation routes explicit, without presenting an indirect predecessor chain as the direct-match explanation.

There is no LLM extraction, new embedding model, command execution, automatic patch application or new cloud deployment in this change. The graph/corpus are not rebuilt or changed.

## Checks

```bash
python -m unittest test_retrieval_ranking -v
```

Four regression tests cover corpus-wide boilerplate, rare but indirect hops, single-document graphs, citation fields and the token budget. They pass on the clean publication branch. Existing locally prepared request-boundary tests were also run, but are not silently included in this change.

The corrected demo returns a dacite source trace, including `items: list[str] = field(default_factory=list)`. Returned text was checked against the public source line after the API's existing whitespace stripping. An unknown canary returns zero recommendations.

## Exploratory API-specific routing proxy

Same 150 public in-corpus queries, random seed 42, budget 220 tokens, old versus repaired API:

| Measure | Before | After |
|---|---:|---:|
| Top-1 same-repository hits | 6/150 (4.0%) | 86/150 (57.3%) |
| First example per distinct issue, top-1 hits | 1/28 (3.6%) | 18/28 (64.3%) |
| All responses within 220-token text budget | Yes | Yes |

These are **not held-out issue-resolution results**, not an end-to-end task-success rate, and not proof of token-cost savings in a production agent. Queries already occur in the indexed corpus and duplicate issues exist. The 150 rows are not 150 independent novel issues. Old benchmark figures for the separate demo ranking path should not be compared as if they were the same API metric.

The measurement used local snapshots with additional previously prepared request/path guards. This publication intentionally includes only the ranking delta and its four new tests, not those unrelated uncommitted changes. The result artifact records the exact measured source/graph hashes and all query rows; it does not assert that the full local file hash equals the clean publication file hash. Both evaluated paths shared the same graph and cached source/token resolvers; no latency claim is made.

Dataset: [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories), public CC-BY-4.0 source described in the repository README. Derived queries/results are an inspectable debugging proxy, not a new private training dataset.

## Competition context

Cogram is an existing project reused transparently for Arbiter Hacks V1's General track. This update adds the API-ranking correction and an evidence walkthrough; it does not claim that the original project was built during Arbiter or addresses its waste-specific Theme track. The new walkthrough uses reformatted actual execution receipts and clearly disclosed synthetic narration, not a simulated live terminal.
