"""Zero-LLM precision check for procedural recall: does a new bug report route
to prior fixes from the *same* repo (correct procedural context), vs a
different, unrelated repo? No API calls — pure graph traversal, same code
path as swe_recall_demo.py.

This is deliberately a narrow, honest metric. It does NOT claim to prove
"solves brand-new bugs never seen anywhere" — the 600-row slice repeats a
number of issues across multiple agent attempts, so some queries are close
to text already in the graph. What it *does* verify: given a bug report as
the only input, does Cogram's concept graph route the citation to the right
project's history, or does it get confused into a random, unrelated repo?
That routing-correctness is the prerequisite for the "recall the fix that
generalizes" claim to mean anything.
"""
from __future__ import annotations

import os
import random
import re
import sys
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import build_edge_index, load_graph, tokenize
from recall import build_adjacency, build_children_map, seed_activation, spread_activation
from swe_recall_demo import collect_pointers

_SCRIPT_DIR = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(_SCRIPT_DIR, "swe_concept_graph.json")
CORPUS_DIR = os.path.join(_SCRIPT_DIR, "swe_corpus")
SEED = 42
N_QUERIES = 150


def repo_of(fname: str) -> str:
    """swe_0379_konradhalas__dacite-216_transcript.md -> konradhalas/dacite"""
    base = fname.replace("_transcript.md", "")
    parts = base.split("_", 2)
    rest = parts[2] if len(parts) > 2 else base
    if "-" in rest:
        owner_repo, _issue = rest.rsplit("-", 1)
        if "__" in owner_repo:
            owner, repo = owner_repo.split("__", 1)
            return f"{owner}/{repo}"
    return rest


def issue_key(fname: str) -> str:
    """Same as repo_of but keeps the issue number -> groups exact-duplicate issues."""
    base = fname.replace("_transcript.md", "")
    parts = base.split("_", 2)
    return parts[2] if len(parts) > 2 else base


_BOILERPLATE_LINE = re.compile(
    r"^(we're currently solving|issue:)\s*$|^we're currently solving the following issue",
    re.IGNORECASE,
)


def extract_query(fpath: str) -> str:
    """Pull the actual issue text SWE-agent was given, skipping (a) the boilerplate
    shell-interface system prompt and (b) the fixed "We're currently solving the
    following issue... ISSUE:" wrapper line every row shares verbatim. A real new
    bug report wouldn't come wrapped in that dataset-specific template, so leaving
    it in would just be testing "can it match the SWE-agent prompt," not the bug."""
    lines: List[str] = []
    capturing = False
    with open(fpath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("## [user-observation]"):
                capturing = True
                continue
            if capturing and line.startswith("## ["):
                break
            if capturing and line.strip() and not _BOILERPLATE_LINE.match(line.strip()):
                lines.append(line.strip())
            if capturing and len(lines) >= 6:
                break
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:400]


def top_k_repos(query: str, nodes: dict, adj: dict, children_map: dict, files_table: List[str], current_tick: int, k: int = 3) -> List[str]:
    """Exact same ranking + provenance-scoring path as swe_recall_demo.py's run_query,
    returning the top-k distinct pointer fnames instead of printing them."""
    qtoks = tokenize(query)
    seed_act = seed_activation(qtoks, nodes)
    if not seed_act:
        return []
    exact_seeds = {c for c, s in seed_act.items() if s >= 1.0}
    act = spread_activation(seed_act, nodes, adj, children_map, max_hops=2, current_tick=current_tick)
    ranked = sorted(act.items(), key=lambda x: (x[0] in exact_seeds, x[1]), reverse=True)[:12]

    pointers = collect_pointers(ranked, nodes, files_table, qtoks, seed_concepts=exact_seeds, max_lines=k)
    return [fname for _score, fname, _lno, _text in pointers]


def main():
    graph = load_graph(GRAPH_PATH)
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})
    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    current_tick = meta.get("tick", 0)
    files_table = meta.get("files", [])

    all_files = sorted(os.listdir(CORPUS_DIR))
    rng = random.Random(SEED)
    sample = rng.sample(all_files, min(N_QUERIES, len(all_files)))

    # Dedup-by-issue subset: one query per distinct (repo, issue) pair, so repeated
    # agent attempts at the identical issue don't inflate the number.
    seen_issue_keys = set()
    total = 0
    top1_hits = 0
    top3_hits = 0
    top1_hits_dedup = 0
    dedup_total = 0
    no_match = 0
    distinct_repos = set(repo_of(f) for f in all_files)

    rows = []
    for fname in sample:
        fpath = os.path.join(CORPUS_DIR, fname)
        query = extract_query(fpath)
        if not query:
            continue
        gold_repo = repo_of(fname)
        hit_fnames = top_k_repos(query, nodes, adj, children_map, files_table, current_tick, k=3)
        total += 1
        is_dedup_new = issue_key(fname) not in seen_issue_keys
        seen_issue_keys.add(issue_key(fname))

        if not hit_fnames:
            no_match += 1
            correct1, correct3 = False, False
        else:
            hit_repos = [repo_of(f) for f in hit_fnames]
            correct1 = hit_repos[0] == gold_repo
            correct3 = gold_repo in hit_repos
            if correct1:
                top1_hits += 1
            if correct3:
                top3_hits += 1

        if is_dedup_new:
            dedup_total += 1
            if correct1:
                top1_hits_dedup += 1

        rows.append((fname, gold_repo, hit_fnames, correct1))

    print("=== Procedural recall: same-repo routing precision (zero LLM calls) ===")
    print(f"Distinct repos in corpus: {len(distinct_repos)}")
    print(f"Sampled queries: {total} (seed={SEED})")
    print(f"No concept match at all: {no_match}")
    print(f"Top-1 same-repo hit rate (all sampled queries): {top1_hits}/{total} = {top1_hits/total:.1%}")
    print(f"Top-3 same-repo hit rate (all sampled queries): {top3_hits}/{total} = {top3_hits/total:.1%}")
    print(f"Top-1 same-repo hit rate (distinct issues only, n={dedup_total}): "
          f"{top1_hits_dedup}/{dedup_total} = {top1_hits_dedup/max(dedup_total,1):.1%}")
    chance = 1.0 / len(distinct_repos)
    chance3 = min(1.0, 3.0 / len(distinct_repos))
    print(f"Random-chance baseline top-1 (1 / distinct repos): {chance:.1%}")
    print(f"Random-chance baseline top-3 (3 / distinct repos): {chance3:.1%}")
    print()
    print("--- sample of top-1 misses (if any) ---")
    misses = [r for r in rows if not r[3]][:8]
    for fname, gold, hits, _ in misses:
        got = [repo_of(h) for h in hits] if hits else None
        print(f"  query={fname}  gold_repo={gold}  got_top3={got}")


if __name__ == "__main__":
    main()
