"""Online concept ingestion — zero LLM, Hebbian update + novelty nodes."""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import (
    GRAPH_PATH,
    edge_key,
    hebbian_delta,
    load_graph,
    save_graph,
    tokenize,
    build_edge_index,
)


def bump_next(nodes: dict, a: str, b: str) -> None:
    """Increment directed transition a→b; keep top-5 by count."""
    nd = nodes[a]
    nxt = nd.get("next", [])
    for item in nxt:
        if item[0] == b:
            item[1] += 1
            nxt.sort(key=lambda x: x[1], reverse=True)
            nd["next"] = nxt[:5]
            return
    nxt.append([b, 1])
    nxt.sort(key=lambda x: x[1], reverse=True)
    nd["next"] = nxt[:5]


def update_next_from_tokens(tokens: List[str], nodes: dict, window: int = 5) -> None:
    """Ordered pairs within sliding token window (same semantics as extract.py)."""
    for i in range(len(tokens)):
        win = tokens[i : i + window]
        first_pos: Dict[str, int] = {}
        for pos, t in enumerate(win):
            if t in nodes and t not in first_pos:
                first_pos[t] = pos
        for a, pa in first_pos.items():
            for b, pb in first_pos.items():
                if a != b and pa < pb:
                    bump_next(nodes, a, b)


def update_edges(
    present: List[str],
    edges: list,
    edge_idx: Dict[Tuple[str, str], dict],
    tick: int,
) -> int:
    """Hebbian strengthen co-present concepts; return count strengthened.

    Strengthened/created edges get last_tick = current tick (subjective time).
    """
    pairs = set()
    uniq = sorted(set(present))
    for i, a in enumerate(uniq):
        for b in uniq[i + 1 :]:
            pairs.add(edge_key(a, b))

    strengthened = 0
    for pair in pairs:
        a, b = pair
        if pair in edge_idx:
            ed = edge_idx[pair]
            n_prev = ed.get("n", 1)
            delta = hebbian_delta(is_first=False, n_prev=n_prev)
            ed["w_raw"] = ed["w_raw"] + delta
            ed["n"] = n_prev + 1
            ed["last_tick"] = tick
            strengthened += 1
        else:
            w = hebbian_delta(is_first=True, n_prev=0)
            edge_idx[pair] = {"a": a, "b": b, "w_raw": w, "last_tick": tick, "n": 1}
            strengthened += 1

    # Rebuild edges list — 4-tuples (a, b, w_raw, last_tick) matching extract.py.
    new_edges = []
    for ed in edge_idx.values():
        lt = ed.get("last_tick", tick)
        new_edges.append([ed["a"], ed["b"], round(ed["w_raw"], 4), lt])
    edges.clear()
    edges.extend(new_edges)
    return strengthened


def main():
    parser = argparse.ArgumentParser(description="Engram online learning")
    parser.add_argument("text", nargs="?", help="Text to ingest")
    parser.add_argument("--file", help="Read text from file")
    parser.add_argument("--source", default="manual", help="Source label for pointers")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            text = f.read()
        source = os.path.basename(args.file)
    elif args.text:
        text = args.text
        source = args.source
    else:
        parser.print_help()
        sys.exit(1)

    graph = load_graph()
    nodes: dict = graph.setdefault("nodes", {})
    edges: list = graph.setdefault("edges", [])
    meta = graph.setdefault("meta", {})
    files_table: list = meta.setdefault("files", [])
    edge_idx = build_edge_index(edges)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # One experienced tick per ingest call (subjective time advances only when
    # the agent actually processes something). Missing/old graph → start at 0.
    tick = int(meta.get("tick", 0)) + 1
    meta["tick"] = tick

    # Intern the source label into the same file table recall.py dereferences.
    if source not in files_table:
        files_table.append(source)
    source_id = files_table.index(source)

    toks = tokenize(text)
    tok_counts = Counter(toks)

    matched: List[str] = []
    new_concepts: List[str] = []

    for tok, cnt in tok_counts.items():
        if tok in nodes:
            nodes[tok]["count"] = nodes[tok].get("count", 0) + cnt
            nodes[tok]["last_seen"] = today
            nodes[tok]["last_tick"] = tick
            matched.append(tok)
        else:
            # NO-PRESET novelty: any never-before-seen content token becomes a
            # node (a novel token is maximally surprising = prediction error).
            nodes[tok] = {
                "count": cnt,
                "df": 1,
                "surprise": 1.0,
                "first_tick": tick,
                "last_tick": tick,
                "first_seen": today,
                "last_seen": today,
                "pointers": [[source_id, 1]],
                "parents": [],
                "hub": False,
                "generality": 0.1,
                "novel": True,
            }
            new_concepts.append(tok)
            matched.append(tok)

    # Pointer for matched existing (interned source id, capped)
    for t in matched:
        if t not in new_concepts:
            pl = nodes[t].setdefault("pointers", [])
            pl.append([source_id, 1])
            if len(pl) > 30:
                nodes[t]["pointers"] = pl[-30:]

    strengthened = update_edges(matched, edges, edge_idx, tick)
    update_next_from_tokens(toks, nodes)

    graph["edges"] = edges
    meta["last_learned"] = today
    save_graph(graph)

    print(f"Edges strengthened: {strengthened}")
    print(f"New concepts created: {len(new_concepts)}")
    if new_concepts:
        print(f"  → {', '.join(new_concepts)}")


if __name__ == "__main__":
    main()
