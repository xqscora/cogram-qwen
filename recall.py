"""Coarse-to-fine concept retrieval — zero LLM, output = memory trace.

Default: ranked concept paths + associates (tiny token count).
--deep N: dereference raw file:line pointers (explicit opt-in).
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import (
    effective_weight,
    estimate_tokens,
    load_graph,
    neighbors_of,
    tokenize,
    build_edge_index,
)

# Prefer real tokenizer so live numbers match benchmark_results.md exactly.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except ImportError:

    def count_tokens(text: str) -> int:
        return estimate_tokens(text)

# Resolve transcript dir same as extract
_SCRIPT_DIR = os.path.dirname(__file__)
_TRANSCRIPT_ROOT = os.environ.get(
    "ENGRAM_TRANSCRIPT_DIR",
    os.path.join(_SCRIPT_DIR, "swe_corpus"),
)
_TRANSCRIPT_ROOT = os.path.normpath(_TRANSCRIPT_ROOT)


def seed_activation(query_tokens: List[str], nodes: dict) -> Dict[str, float]:
    """Exact match 1.0; substring match 0.6 (either direction)."""
    act: Dict[str, float] = {}
    concept_list = list(nodes.keys())
    for qt in query_tokens:
        for c in concept_list:
            if c == qt:
                act[c] = max(act.get(c, 0), 1.0)
            elif qt in c or c in qt:
                act[c] = max(act.get(c, 0), 0.6)
    return act


def spread_activation(
    act: Dict[str, float],
    nodes: dict,
    adj: Dict[str, List[Tuple[str, float, object]]],
    children_map: Dict[str, List[str]],
    max_hops: int = 2,
    current_tick: int = 0,
) -> Dict[str, float]:
    """Coarse-to-fine: hubs/parents first via seeds, then spread down to children/neighbors."""
    # Boost hub parents of seeded concepts (coarse tier)
    boosted = dict(act)
    for c, val in list(act.items()):
        nd = nodes.get(c, {})
        if nd.get("hub"):
            boosted[c] = max(boosted.get(c, 0), val * 1.2)
        for p in nd.get("parents", []):
            boosted[p] = max(boosted.get(p, 0), val * 0.8)

    current = boosted
    for _hop in range(max_hops):
        nxt = dict(current)
        for src, src_act in current.items():
            if src_act < 0.01:
                continue
            # Downward: precomputed children map
            for child in children_map.get(src, []):
                nxt[child] = nxt.get(child, 0) + src_act * 0.5 * 0.7
            # Lateral: neighbors via tick-decayed edges
            for nb, w_raw, last_tick in adj.get(src, []):
                ew = effective_weight(w_raw, last_tick, current_tick)
                nxt[nb] = nxt.get(nb, 0) + src_act * ew * 0.5
        current = nxt
    return current


def build_adjacency(edge_idx: dict) -> Dict[str, List[Tuple[str, float, object]]]:
    """concept -> [(neighbor, w_raw, last_tick), ...]"""
    adj: Dict[str, List[Tuple[str, float, object]]] = {}
    for e in edge_idx.values():
        a, b, w, lt = e["a"], e["b"], e["w_raw"], e["last_tick"]
        adj.setdefault(a, []).append((b, w, lt))
        adj.setdefault(b, []).append((a, w, lt))
    return adj


def build_children_map(nodes: dict) -> Dict[str, List[str]]:
    cm: Dict[str, List[str]] = {}
    for c, nd in nodes.items():
        for p in nd.get("parents", []):
            cm.setdefault(p, []).append(c)
    return cm


def concept_path(concept: str, nodes: dict, edge_idx: dict, current_tick: int) -> str:
    """Build hierarchy path: parent > ... > concept."""
    chain = [concept]
    visited = {concept}
    nd = nodes.get(concept, {})
    for p in nd.get("parents", [])[:2]:
        if p not in visited:
            chain.insert(0, p)
            visited.add(p)
    parts = []
    for c in chain:
        parts.append(c)
    path_str = " > ".join(parts)
    ndc = nodes.get(concept, {})
    # last_seen date kept only as optional display; tick model drives decay.
    ls = ndc.get("last_seen", ndc.get("last_tick", "?"))
    best_w = 0.0
    for nb, wr, lt in neighbors_of(concept, edge_idx):
        best_w = max(best_w, effective_weight(wr, lt, current_tick))
    if ndc.get("hub"):
        best_w = max(best_w, 0.5)
    return f"{path_str}  (w={best_w:.2f}, last {ls})"


def top_associates(concept: str, edge_idx: dict, current_tick: int, k: int = 3) -> str:
    nbs = neighbors_of(concept, edge_idx)
    scored = []
    for nb, wr, lt in nbs:
        scored.append((effective_weight(wr, lt, current_tick), nb))
    scored.sort(reverse=True)
    top = [f"{nb}({ew:.2f})" for ew, nb in scored[:k]]
    return ", ".join(top) if top else "(none)"


def top_next_line(concept: str, nodes: dict, k: int = 2) -> str:
    """Directed transitions: what this concept typically leads to."""
    nxt = nodes.get(concept, {}).get("next", [])
    if not nxt:
        return ""
    labels = [item[0] for item in nxt[:k]]
    return f" → leads to: {', '.join(labels)}"


def suggest_concepts(name: str, nodes: dict, limit: int = 5) -> List[str]:
    """Substring closest matches when concept unknown."""
    hits = []
    for c in nodes:
        if name in c or c in name:
            hits.append(c)
    hits.sort(key=lambda x: (abs(len(x) - len(name)), x))
    return hits[:limit]


def follow_next_chain(start: str, nodes: dict, steps: int = 4) -> None:
    """Greedy walk along strongest outgoing `next` links (skip visited)."""
    if start not in nodes:
        sug = suggest_concepts(start, nodes)
        if sug:
            print(f"Unknown concept '{start}'. Closest matches: {', '.join(sug)}")
        else:
            print(f"Unknown concept '{start}'.")
        return

    chain = [start]
    visited = {start}
    current = start
    for _ in range(steps):
        nxt = nodes.get(current, {}).get("next", [])
        # PMI-style discount: divide by target's global frequency^0.75 so
        # specific transitions beat omnipresent hubs (word2vec's unigram
        # discount exponent). Raw counts alone let mega-hubs swallow chains.
        best, best_score = None, 0.0
        for nb, cnt in nxt:
            if nb in visited:
                continue
            freq = max(nodes.get(nb, {}).get("count", 1), 1)
            score = cnt / (freq ** 0.75)
            if score > best_score:
                best, best_score = nb, score
        if not best:
            break
        chain.append(best)
        visited.add(best)
        current = best

    parts = [chain[0]]
    for i in range(1, len(chain)):
        prev = chain[i - 1]
        cnt = 0
        for nb, n in nodes.get(prev, {}).get("next", []):
            if nb == chain[i]:
                cnt = n
                break
        parts.append(f"{chain[i]}({cnt})" if cnt else chain[i])
    print(" → ".join(parts))


def dereference_pointer(fname: str, lno: int, context: int = 2) -> str:
    fpath = os.path.join(_TRANSCRIPT_ROOT, fname)
    gz_path = os.path.join(_SCRIPT_DIR, "raw_store", f"{fname}.gz")
    from_compressed = False
    if not os.path.isfile(fpath):
        if os.path.isfile(gz_path):
            from_compressed = True
        else:
            return f"[missing: {fname}:{lno}]"
    header = f"\n--- {fname}:{lno}"
    if from_compressed:
        header += " (from compressed store)"
    header += " ---"
    lines_out = [header]
    try:
        if from_compressed:
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        else:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        start = max(0, lno - 1 - context)
        end = min(len(all_lines), lno + context)
        for i in range(start, end):
            marker = ">>" if i == lno - 1 else "  "
            lines_out.append(f"{marker} {i+1}: {all_lines[i].rstrip()}")
    except OSError as e:
        lines_out.append(f"[read error: {e}]")
    return "\n".join(lines_out)


def main():
    parser = argparse.ArgumentParser(description="Engram coarse-to-fine recall")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--next", dest="next_concept", metavar="CONCEPT",
                        help="Follow top outgoing transitions for 4 steps")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--deep", type=int, default=0, help="Dereference top N pointers")
    args = parser.parse_args()

    if not args.query and not args.next_concept:
        parser.error("Provide a query string or --next CONCEPT")

    graph = load_graph()
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})
    if not nodes:
        print("Graph empty — run extract.py first.")
        sys.exit(1)

    if args.next_concept:
        follow_next_chain(args.next_concept.lower(), nodes)
        if not args.query:
            return

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    # Subjective time: current tick = how many interactions/sessions the agent
    # has experienced. Missing on old graphs → 0 → effective_weight no-decay.
    current_tick = meta.get("tick", 0)
    if not args.query:
        return

    qtoks = tokenize(args.query)
    if not qtoks:
        print("No tokens in query.")
        sys.exit(1)

    act = seed_activation(qtoks, nodes)
    if not act:
        print(f"No concept matches for tokens: {qtoks}")
        sys.exit(1)

    act = spread_activation(act, nodes, adj, children_map, max_hops=2, current_tick=current_tick)
    ranked = sorted(act.items(), key=lambda x: x[1], reverse=True)[: args.topk]

    trace_lines = ["=== Memory Trace (concept tier) ==="]
    for c, score in ranked:
        path = concept_path(c, nodes, edge_idx, current_tick)
        assoc = top_associates(c, edge_idx, current_tick)
        leads = top_next_line(c, nodes)
        trace_lines.append(f"  [{score:.2f}] {path}{leads}")
        trace_lines.append(f"         associates: {assoc}")

    trace_text = "\n".join(trace_lines)
    print(trace_text)

    trace_tokens = count_tokens(trace_text)
    # Prefer the tiktoken-measured corpus count written by benchmark.py,
    # so live demo numbers match benchmark_results.md exactly.
    raw_tokens = meta.get("raw_corpus_tokens_cl100k") or estimate_tokens(
        meta.get("raw_total_chars", 0)
    )
    factor = raw_tokens / max(trace_tokens, 1)
    print(
        f"\nContext cost per query: ~{trace_tokens} tokens (concept trace)"
        f"\nvs stuffing full history: ~{raw_tokens:,} tokens"
        f"\n→ {factor:,.0f}x smaller, built with 0 LLM tokens"
    )

    if args.deep > 0:
        files_table = meta.get("files", [])
        deep_text_parts = ["\n=== Deep tier (raw pointers, opt-in) ==="]
        seen_ptrs = set()
        count = 0
        for c, _ in ranked:
            if count >= args.deep:
                break
            for ptr in nodes.get(c, {}).get("pointers", [])[-3:]:
                if count >= args.deep:
                    break
                fidx, lno = ptr[0], ptr[1]
                if (fidx, lno) in seen_ptrs:
                    continue
                seen_ptrs.add((fidx, lno))
                fname = (
                    files_table[fidx]
                    if isinstance(fidx, int) and 0 <= fidx < len(files_table)
                    else str(fidx)
                )
                deep_text_parts.append(dereference_pointer(fname, lno))
                count += 1
        deep_text = "\n".join(deep_text_parts)
        print(deep_text)
        total_tokens = count_tokens(trace_text + deep_text)
        print(
            f"\nWith {args.deep} raw line(s) dereferenced: ~{total_tokens} tokens"
            f" ({raw_tokens / max(total_tokens, 1):,.0f}x smaller than full history)"
        )


if __name__ == "__main__":
    main()
