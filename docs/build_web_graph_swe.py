"""Build the interactive web-demo graph from the SWE procedural-memory corpus.

Reuses the exact same retrieval code as `swe_recall_demo.py` (seed_activation,
spread_activation, collect_pointers) so the web demo can never drift from what
the CLI actually returns. No LLM anywhere in this pipeline — capping is by
surprise mass, same as `extract_swe.py`.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
sys.path.insert(0, str(ROOT))

from graph_lib import build_edge_index, effective_weight, estimate_tokens, load_graph, tokenize  # noqa: E402
from recall import build_adjacency, build_children_map, seed_activation, spread_activation  # noqa: E402
from swe_recall_demo import DEMO_QUERIES, collect_pointers, read_line  # noqa: E402

GRAPH_PATH = ROOT / "swe_concept_graph.json"
CORPUS_DIR = ROOT / "swe_corpus"
OUT_PATH = WEB_DIR / "graph.json"

MAX_NODES = 90
MAX_EDGES_PER_NODE = 3
MAX_DEGREE_OUTLIER = 22
PRIMARY_QUERY_IDX = 1  # "dacite default_factory dataclass" — smallest, cleanest demo
# SWE-agent's own shell-interface template repeats the same instructional
# vocabulary in every single trajectory ("issue", "bash-", "user-observation",
# "reproduce"...). df >= this threshold means "in most of the 600 files" —
# that's corpus-template noise, not a discriminative concept. Cut before
# ranking so the visible graph shows real per-repo vocabulary, not boilerplate.
HUB_DF_CUTOFF = 400


def surprise_mass(nodes: dict, edges: list) -> Counter:
    mass: Counter = Counter()
    for e in edges:
        if len(e) < 3:
            continue
        a, b, w = e[0], e[1], e[2]
        if a in nodes:
            mass[a] += w
        if b in nodes:
            mass[b] += w
    return mass


def pin_nodes_for_queries(queries: list, nodes: dict) -> set:
    pinned = set()
    for q in queries:
        for tok in tokenize(q):
            if tok in nodes:
                pinned.add(tok)
    return pinned


def cap_graph(graph: dict, max_nodes: int, max_edges_per_node: int, pinned: set) -> tuple:
    nodes = graph["nodes"]
    edges = graph["edges"]
    mass = surprise_mass(nodes, edges)

    hub_boilerplate = {c for c, n in nodes.items() if n.get("df", 1) >= HUB_DF_CUTOFF} - pinned
    candidates = [c for c in nodes.keys() if c not in hub_boilerplate]
    ranked = sorted(candidates, key=lambda c: mass.get(c, 0.0), reverse=True)
    keep_nodes = set(ranked[: max(0, max_nodes - len(pinned))]) | (pinned & set(nodes.keys()))

    incident: dict = defaultdict(list)
    for e in edges:
        if len(e) < 4:
            continue
        a, b, w_raw, last_tick = e[0], e[1], e[2], e[3]
        if a not in keep_nodes or b not in keep_nodes:
            continue
        incident[a].append((w_raw, b, last_tick))
        incident[b].append((w_raw, a, last_tick))

    keep_edges: set = set()
    for c in keep_nodes:
        lst = sorted(incident.get(c, []), key=lambda x: x[0], reverse=True)
        for w_raw, nb, last_tick in lst[:max_edges_per_node]:
            pair = tuple(sorted((c, nb)))
            keep_edges.add((pair[0], pair[1], w_raw, last_tick))

    connected: set = set()
    for a, b, _w, _t in keep_edges:
        connected.add(a)
        connected.add(b)
    keep_nodes &= connected

    degree: Counter = Counter()
    for a, b, _w, _t in keep_edges:
        degree[a] += 1
        degree[b] += 1
    outliers = {c for c, d in degree.items() if d > MAX_DEGREE_OUTLIER and c not in pinned}
    if outliers:
        keep_nodes -= outliers
        keep_edges = {(a, b, w, t) for a, b, w, t in keep_edges if a in keep_nodes and b in keep_nodes}
        connected = set()
        for a, b, _w, _t in keep_edges:
            connected.add(a)
            connected.add(b)
        keep_nodes &= connected

    return keep_nodes, keep_edges, mass


def main() -> None:
    if not GRAPH_PATH.is_file():
        raise FileNotFoundError(f"{GRAPH_PATH} — run extract_swe.py first")

    graph = load_graph(str(GRAPH_PATH))
    nodes = graph["nodes"]
    edges = graph["edges"]
    meta = graph["meta"]
    current_tick = meta.get("tick", 0)
    files_table = meta.get("files", [])

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)

    pinned = pin_nodes_for_queries(DEMO_QUERIES, nodes)
    keep_nodes, keep_edges, mass = cap_graph(graph, MAX_NODES, MAX_EDGES_PER_NODE, pinned)
    print(f"Capped graph: {len(keep_nodes)} nodes, {len(keep_edges)} edges (pinned={len(pinned)})")

    # --- Run the real retrieval pipeline for every demo query, exactly like the CLI ---
    query_hits = []
    for q in DEMO_QUERIES:
        qtoks = tokenize(q)
        seed_act = seed_activation(qtoks, nodes)
        if not seed_act:
            query_hits.append((q, [], set()))
            continue
        exact_seeds = {c for c, s in seed_act.items() if s >= 1.0}
        act = spread_activation(seed_act, nodes, adj, children_map, max_hops=2, current_tick=current_tick)
        ranked = sorted(act.items(), key=lambda x: (x[0] in exact_seeds, x[1]), reverse=True)[:12]
        hits = collect_pointers(ranked, nodes, files_table, qtoks, seed_concepts=exact_seeds, max_lines=5)
        query_hits.append((q, hits, exact_seeds))

    primary_query, primary_hits, primary_seeds = query_hits[PRIMARY_QUERY_IDX]

    # --- Export lines: union of (a) kept-node pointers, (b) every query's real hits ---
    line_key_to_idx: dict = {}
    lines_out: list = []
    pointers_out: dict = defaultdict(list)

    def export_line(fidx: int, lno: int, text: str) -> int:
        key = (fidx, lno)
        if key in line_key_to_idx:
            return line_key_to_idx[key]
        idx = len(lines_out)
        line_key_to_idx[key] = idx
        fname = files_table[fidx] if 0 <= fidx < len(files_table) else str(fidx)
        lines_out.append(
            {
                "idx": idx,
                "session": fidx,
                "tick": fidx + 1,
                "date": repo_label(fname),
                "role": "trajectory",
                "text": text,
                "file": fname,
                "line": lno,
            }
        )
        return idx

    def repo_label(fname: str) -> str:
        # swe_0379_konradhalas__dacite-216_transcript.md -> konradhalas/dacite #216
        base = fname.replace("_transcript.md", "")
        parts = base.split("_", 2)
        rest = parts[2] if len(parts) > 2 else base
        if "-" in rest:
            owner_repo, issue = rest.rsplit("-", 1)
            if "__" in owner_repo and issue.isdigit():
                owner, repo = owner_repo.split("__", 1)
                return f"{owner}/{repo} #{issue}"
        return rest

    for c in keep_nodes:
        for ptr in nodes[c].get("pointers", []):
            if len(ptr) < 2:
                continue
            fidx, lno = ptr[0], ptr[1]
            if not isinstance(fidx, int) or not (0 <= fidx < len(files_table)):
                continue
            text = read_line(files_table[fidx], lno)
            if len(text.strip()) < 8:
                continue
            idx = export_line(fidx, lno, text)
            pointers_out[c].append(idx)

    answer_line_idxs = []
    for score, fname, lno, text in primary_hits:
        fidx = files_table.index(fname) if fname in files_table else None
        if fidx is None:
            continue
        idx = export_line(fidx, lno, text)
        answer_line_idxs.append(idx)

    for c in keep_nodes:
        pointers_out[c] = sorted(set(pointers_out.get(c, [])))

    # --- knowledge_update panel repurposed as "bug report -> recalled fix" ---
    old_idx, new_idx = None, None
    if primary_hits:
        top_score, top_fname, top_lno, top_text = primary_hits[0]
        new_idx = export_line(files_table.index(top_fname), top_lno, top_text)
        same_file_lines = [l for l in lines_out if l["file"] == top_fname]
        if same_file_lines:
            earliest = min(same_file_lines, key=lambda l: l["line"])
            if earliest["idx"] != new_idx:
                old_idx = earliest["idx"]

    # --- token accounting (same estimate_tokens heuristic the CLI reports) ---
    raw_tokens = estimate_tokens(meta.get("raw_total_chars", 0))
    payload_lines = [f"Query: {primary_query}", "Retrieved provenance:"]
    for score, fname, lno, text in primary_hits[:5]:
        payload_lines.append(f"  [{fname}:{lno}] {text}")
    budget_tokens = max(estimate_tokens("\n".join(payload_lines)), 1)

    out_nodes = []
    for c in sorted(keep_nodes, key=lambda x: (-mass.get(x, 0), x)):
        nd = nodes[c]
        out_nodes.append(
            {
                "id": c,
                "label": c,
                "df": nd.get("df", 1),
                "first_tick": nd.get("first_tick", 1),
                "last_tick": nd.get("last_tick", current_tick),
            }
        )

    out_edges = []
    seen_e: set = set()
    for a, b, w_raw, last_tick in sorted(keep_edges, key=lambda x: (-x[2], x[0], x[1])):
        key = (a, b) if a <= b else (b, a)
        if key in seen_e:
            continue
        seen_e.add(key)
        eff = round(effective_weight(w_raw, last_tick, current_tick), 4)
        out_edges.append({"source": a, "target": b, "w_raw": round(float(w_raw), 4), "last_tick": last_tick, "eff": eff})

    payload = {
        "meta": {
            "question": primary_query,
            "answer": f"{len(primary_hits)} provenance line(s) recalled",
            "question_type": "procedural-recall",
            "current_tick": current_tick,
            "n_sessions": meta.get("n_files", 0),
            "n_lines": len(lines_out),
            "raw_corpus_tokens": raw_tokens,
            "budget_tokens": budget_tokens,
            "token_ratio": round(raw_tokens / max(budget_tokens, 1), 1),
            "knowledge_update": {"old_line_idx": old_idx, "new_line_idx": new_idx},
            "demo_queries": [q for q, _, _ in query_hits],
        },
        "demo_query": primary_query,
        "answer_line_idxs": answer_line_idxs,
        "nodes": out_nodes,
        "edges": out_edges,
        "lines": lines_out,
        "pointers": {k: v for k, v in pointers_out.items()},
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nExported {OUT_PATH}:")
    print(f"  nodes={len(out_nodes)} edges={len(out_edges)} lines={len(lines_out)}")
    print(f"  raw_corpus_tokens={raw_tokens:,} budget={budget_tokens} ratio~{raw_tokens/budget_tokens:.0f}x")
    print(f"  primary_query={primary_query!r} hits={len(primary_hits)}")
    for score, fname, lno, text in primary_hits:
        print(f"    [{fname}:{lno}] {text[:100]}")


if __name__ == "__main__":
    main()
