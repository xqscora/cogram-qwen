"""SWE procedural-memory retrieval demo — zero LLM, concept activation + provenance."""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import build_edge_index, effective_weight, estimate_tokens, load_graph, tokenize
from recall import (
    build_adjacency,
    build_children_map,
    seed_activation,
    spread_activation,
)

_SCRIPT_DIR = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(_SCRIPT_DIR, "swe_concept_graph.json")
TRANSCRIPT_ROOT = os.path.join(_SCRIPT_DIR, "swe_corpus")

# Real task/error phrases from swe_corpus (successful-fix procedural memory)
DEMO_QUERIES = [
    "overriddenSymbolFrom cognitive complexity binary logical operators",
    "dacite default_factory dataclass",
    "electionguard words.py bad input Optional",
]


def read_line(fname: str, lno: int) -> str:
    fpath = os.path.join(TRANSCRIPT_ROOT, fname)
    if not os.path.isfile(fpath):
        return f"[missing: {fname}:{lno}]"
    try:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i == lno:
                    return line.rstrip()
    except OSError as e:
        return f"[read error: {e}]"
    return f"[line {lno} not found in {fname}]"


def pointer_relevance(query_tokens: List[str], text: str) -> float:
    line_tokens = set(tokenize(text))
    hits = [t for t in query_tokens if t in line_tokens or t in text.lower()]
    if len(hits) >= 2:
        return float(len(hits)) * 2.0
    return float(len(hits))


def idf_weight(df: int, n_docs: int) -> float:
    """Standard inverse-document-frequency discount, computed purely from the
    concept's own measured document frequency vs corpus size — no fixed
    cutoff/blacklist. A concept that appears in every trajectory (df==n_docs,
    e.g. SWE-agent's own boilerplate shell-interface tips) carries ~0 bits of
    information about *which* trajectory is relevant, so its contribution to
    ranking is discounted toward 0. A concept seen once keeps full weight.
    Without this, activation score alone (which grows with df via edge
    co-occurrence counts) lets ubiquitous boilerplate concepts drown out the
    rare, bug-specific ones — see benchmark_procedural_precision.py."""
    df = max(df, 1)
    n_docs = max(n_docs, 1)
    import math

    return math.log((n_docs + 1.0) / df) / math.log(n_docs + 1.0) if n_docs > 1 else 1.0


def collect_pointers(
    ranked: List[Tuple[str, float]],
    nodes: dict,
    files_table: List[str],
    query_tokens: List[str],
    seed_concepts: set[str] | None = None,
    max_lines: int = 5,
) -> List[Tuple[float, str, int, str]]:
    """Return (score, fname, lno, text) for top provenance lines."""
    seen = set()
    hits: List[Tuple[float, str, int, str]] = []
    ranked_map = {c: s for c, s in ranked}
    concept_order: List[Tuple[str, float]] = []
    if seed_concepts:
        for c in sorted(seed_concepts):
            if c in nodes:
                concept_order.append((c, ranked_map.get(c, 1.0)))
    for c, s in ranked:
        if seed_concepts and c in seed_concepts:
            continue
        concept_order.append((c, s))

    n_docs = len(files_table) or 1
    for concept, score in concept_order:
        df = nodes.get(concept, {}).get("df", 1)
        idf = idf_weight(df, n_docs)
        for ptr in nodes.get(concept, {}).get("pointers", []):
            fidx, lno = ptr[0], ptr[1]
            key = (fidx, lno)
            if key in seen:
                continue
            seen.add(key)
            fname = (
                files_table[fidx]
                if isinstance(fidx, int) and 0 <= fidx < len(files_table)
                else str(fidx)
            )
            text = read_line(fname, lno)
            rel = pointer_relevance(query_tokens, text)
            fname_hits = sum(1 for t in query_tokens if t in fname.lower())
            if fname_hits == 0 and rel < 2.0:
                continue
            seed_bonus = 2.0 if seed_concepts and concept in seed_concepts else 0.0
            combined = score * idf * (1.0 + seed_bonus + rel + fname_hits * 2.0)
            hits.append((combined, fname, lno, text))
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[:max_lines]


def run_query(
    query: str,
    nodes: dict,
    adj: dict,
    children_map: dict,
    files_table: List[str],
    current_tick: int,
) -> None:
    qtoks = tokenize(query)
    seed_act = seed_activation(qtoks, nodes)
    if not seed_act:
        print(f"  No concept matches for: {qtoks}")
        return

    exact_seeds = {c for c, s in seed_act.items() if s >= 1.0}
    act = spread_activation(seed_act, nodes, adj, children_map, max_hops=2, current_tick=current_tick)
    ranked = sorted(
        act.items(),
        key=lambda x: (x[0] in exact_seeds, x[1]),
        reverse=True,
    )[:12]

    pointers = collect_pointers(
        ranked, nodes, files_table, qtoks, seed_concepts=exact_seeds, max_lines=5
    )
    payload_lines = [f"Query: {query}", "Retrieved provenance:"]
    for score, fname, lno, text in pointers[:5]:
        payload_lines.append(f"  [{fname}:{lno}] {text}")
        print(f"  [{fname}:{lno}] {text}")

    payload = "\n".join(payload_lines)
    tok_count = estimate_tokens(payload)
    print(f"  Returned token estimate: ~{tok_count} tokens")
    print(f"  Top activated concepts: {', '.join(c for c, _ in ranked[:5])}")


def run_motif_demo(motifs: dict, files_table: List[str], query_tokens: List[str]) -> None:
    """Show a level-1 concept-of-concepts hit: one node standing in for a
    whole recurring problem-solution cluster, instead of the retriever
    re-discovering that cluster from separate token activations each time.
    Independent of recall.py's live spreading path — this only demonstrates
    that the motifs exist and generalize across distinct trajectories."""
    if not motifs:
        print("  No motifs mined.")
        return
    qset = set(query_tokens)
    scored = []
    for key, m in motifs.items():
        overlap = len(qset & set(m["members"]))
        if overlap >= 2:
            scored.append((overlap, m["n_files"], key, m))
    if not scored:
        # fall back to the highest-reinforced motif overall, just to show the shape
        key, m = max(motifs.items(), key=lambda kv: kv[1]["n_files"])
        scored = [(0, m["n_files"], key, m)]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _overlap, n_files, key, m = scored[0]

    member_str = " + ".join(m["members"])
    order_str = " -> ".join(m["order"][:6])
    print(f"  Motif: {{{member_str}}}")
    print(f"  Seen together across {n_files} distinct trajectories (not just repeated lines in one file)")
    print(f"  Typical order: {order_str}")
    for fid, lno in m["provenance"][:3]:
        fname = files_table[fid] if isinstance(fid, int) and 0 <= fid < len(files_table) else str(fid)
        text = read_line(fname, lno)
        print(f"    [{fname}:{lno}] {text}")
    single_motif_tokens = estimate_tokens(f"Motif: {member_str} -> {order_str}")
    print(f"  Recalling the whole cluster as ONE node costs ~{single_motif_tokens} tokens")
    print("  (vs. re-discovering the same cluster by spreading through each member token separately)")


def main():
    if not os.path.isfile(GRAPH_PATH):
        print(f"Missing {GRAPH_PATH} — run extract_swe.py first.")
        sys.exit(1)

    graph = load_graph(GRAPH_PATH)
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    motifs = graph.get("motifs", {})
    meta = graph.get("meta", {})
    if not nodes:
        print("Graph empty.")
        sys.exit(1)

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    current_tick = meta.get("tick", 0)
    files_table = meta.get("files", [])

    raw_chars = meta.get("raw_total_chars", 0)
    raw_tokens = estimate_tokens(raw_chars)
    index_json = open(GRAPH_PATH, encoding="utf-8").read()
    index_tokens = estimate_tokens(len(index_json))

    print("=== SWE Procedural Memory Recall Demo ===")
    print(f"Graph: {len(nodes)} concepts, {len(edges)} edges")
    print(f"Corpus: ~{raw_tokens:,} raw tokens | Index: ~{index_tokens:,} tokens")
    print(f"Compression: {raw_tokens / max(index_tokens, 1):.1f}x\n")

    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"--- Demo {i} ---")
        run_query(q, nodes, adj, children_map, files_table, current_tick)
        print()

    print(f"--- Demo 4: concept-of-concepts (level-1 motif, {len(motifs)} mined) ---")
    run_motif_demo(motifs, files_table, tokenize(DEMO_QUERIES[1]))
    print()


if __name__ == "__main__":
    main()
