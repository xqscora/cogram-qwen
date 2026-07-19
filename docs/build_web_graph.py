"""Build Engram concept graph for interactive web demo (LongMemEval knowledge-update).

Exports web/graph.json from the smallest haystack knowledge-update question.
Reuses benchmark graph build + graph_lib decay — no reinvented tokenization.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
sys.path.insert(0, str(ROOT))

from benchmark_longmem import flatten_haystack  # noqa: E402
from benchmark_qa import build_graph, ensure_tiktoken  # noqa: E402
from graph_lib import effective_weight  # noqa: E402

DATA_PATH = ROOT / "data" / "longmemeval_s_cleaned.json"
OUT_PATH = WEB_DIR / "graph.json"

MAX_NODES = 70
MAX_EDGES_PER_NODE = 3
BUDGET_TOKENS = 2000
QUESTION_TYPE = "knowledge-update"
# Generic role tokens — high degree, no semantic value, create fan edges
EXCLUDE_NODES = frozenset({"assistant", "user"})
MAX_DEGREE_OUTLIER = 24

EN_STOP = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing don down during
    each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just ll me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own re s same she
    should so some such t than that the their theirs them themselves then there these
    they this those through to too under until up ve very was we were what when where
    which while who whom why will with would you your yours yourself yourselves
    also get got like make made use used using one two three new way may well much
    many still even back go going come came take took see saw say said says tell
    told think thought know knew want need let put find give given keep look looking
    something anything everything nothing someone anyone everyone into onto upon""".split()
)


def haystack_chars(item: dict) -> int:
    return sum(
        len((t.get("content") or ""))
        for s in (item.get("haystack_sessions") or [])
        for t in s
    )


def pick_smallest_knowledge_update(dataset: list) -> dict:
    ku = [x for x in dataset if x.get("question_type") == QUESTION_TYPE]
    if not ku:
        raise ValueError(f"No {QUESTION_TYPE} questions in dataset")
    return min(ku, key=haystack_chars)


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


def tokenize_query(text: str) -> list[str]:
    raw = re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
    return [t for t in raw if len(t) >= 3 and t not in EN_STOP]


def query_pin_nodes(question: str, nodes: dict) -> set[str]:
    """Keep demo-query concepts that exist in the full graph."""
    pinned: set[str] = set()
    for tok in tokenize_query(question):
        if tok in nodes:
            pinned.add(tok)
        elif tok.endswith("s") and tok[:-1] in nodes:
            pinned.add(tok[:-1])
        elif f"{tok}s" in nodes:
            pinned.add(f"{tok}s")
    return pinned


def cap_graph(
    graph: dict,
    max_nodes: int,
    max_edges_per_node: int,
    current_tick: int,
    question: str = "",
) -> tuple:
    nodes = graph["nodes"]
    edges = graph["edges"]
    mass = surprise_mass(nodes, edges)
    pinned = query_pin_nodes(question, nodes)

    candidates = [c for c in nodes if c not in EXCLUDE_NODES]
    ranked = sorted(candidates, key=lambda c: mass.get(c, 0.0), reverse=True)
    keep_nodes = set(ranked[: max(0, max_nodes - len(pinned))]) | pinned
    keep_nodes -= EXCLUDE_NODES

    incident: dict[str, list] = defaultdict(list)
    for e in edges:
        if len(e) < 4:
            continue
        a, b, w_raw, last_tick = e[0], e[1], e[2], e[3]
        if a in EXCLUDE_NODES or b in EXCLUDE_NODES:
            continue
        if a not in keep_nodes or b not in keep_nodes:
            continue
        incident[a].append((w_raw, b, last_tick))
        incident[b].append((w_raw, a, last_tick))

    keep_edges: set[tuple] = set()
    for c in keep_nodes:
        lst = sorted(incident.get(c, []), key=lambda x: x[0], reverse=True)
        for w_raw, nb, last_tick in lst[:max_edges_per_node]:
            if nb in EXCLUDE_NODES:
                continue
            pair = tuple(sorted((c, nb)))
            keep_edges.add((pair[0], pair[1], w_raw, last_tick))

    # Drop nodes with zero edges after capping
    connected: set[str] = set()
    for a, b, _w, _t in keep_edges:
        connected.add(a)
        connected.add(b)
    keep_nodes &= connected

    # Drop remaining extreme-degree outliers (fan effect)
    degree: Counter = Counter()
    for a, b, _w, _t in keep_edges:
        degree[a] += 1
        degree[b] += 1
    outliers = {c for c, d in degree.items() if d > MAX_DEGREE_OUTLIER and c not in pinned}
    if outliers:
        keep_nodes -= outliers
        keep_edges = {
            (a, b, w, t)
            for a, b, w, t in keep_edges
            if a in keep_nodes and b in keep_nodes
        }
        connected = set()
        for a, b, _w, _t in keep_edges:
            connected.add(a)
            connected.add(b)
        keep_nodes &= connected

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
    seen: set[tuple] = set()
    for a, b, w_raw, last_tick in sorted(keep_edges, key=lambda x: (-x[2], x[0], x[1])):
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        eff = round(effective_weight(w_raw, last_tick, current_tick), 4)
        out_edges.append(
            {
                "source": a,
                "target": b,
                "w_raw": round(float(w_raw), 4),
                "last_tick": last_tick,
                "eff": eff,
            }
        )

    pointers: dict[str, list] = {}
    for c in keep_nodes:
        ptrs = nodes[c].get("pointers") or []
        pointers[c] = sorted(set(ptrs))

    return out_nodes, out_edges, pointers


def session_tick_map(lines: list) -> dict[int, int]:
    sessions = sorted({ln.get("session", 0) for ln in lines})
    return {s: i for i, s in enumerate(sessions, 1)}


def export_lines(lines: list, session_tick: dict[int, int]) -> list:
    out = []
    for ln in lines:
        sess = ln.get("session", 0)
        out.append(
            {
                "idx": ln["idx"],
                "session": sess,
                "tick": session_tick.get(sess, 1),
                "date": ln.get("session_date") or "",
                "role": ln.get("role", "?"),
                "text": ln.get("content") or ln.get("line_str", ""),
            }
        )
    return out


def answer_line_indices(answer: str, lines: list) -> list[int]:
    """Best-effort: lines whose text contains answer keywords."""
    ans = answer.lower().strip()
    keywords: list[str] = []
    if ans:
        keywords.append(ans)
    # Extract numeric + unit phrases
    for m in re.finditer(r"\b(two|three|four|five|one|\d+)\s+(hours?|minutes?|days?)\b", ans):
        keywords.append(m.group(0))
    for m in re.finditer(r"\babout\s+\w+\s+\w+\b", ans):
        keywords.append(m.group(0))

    seen: set[int] = set()
    hits: list[int] = []
    for kw in keywords:
        kw = kw.strip()
        if len(kw) < 3:
            continue
        for ln in lines:
            if kw in ln["text"].lower() and ln["idx"] not in seen:
                seen.add(ln["idx"])
                hits.append(ln["idx"])
    return sorted(hits)


def knowledge_update_pair(question: str, answer: str, lines: list) -> dict:
    """Find old vs updated info lines for knowledge-update callout."""
    ans_lower = answer.lower()
    # Heuristic: old = "hour" without "two"; new = contains answer phrase
    old_idx = None
    new_idx = None
    for ln in lines:
        t = ln["text"].lower()
        if "coding" in t and "hour" in t:
            if re.search(r"\b(one|1|an)\s+hour", t) and "two" not in t:
                if old_idx is None or ln["idx"] < old_idx:
                    old_idx = ln["idx"]
            if "two hour" in t or "two hours" in t or ans_lower in t:
                if new_idx is None or ln["idx"] > new_idx:
                    new_idx = ln["idx"]
    return {"old_line_idx": old_idx, "new_line_idx": new_idx}


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    count_tokens = ensure_tiktoken()
    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    item = pick_smallest_knowledge_update(dataset)
    qid = item.get("question_id", "?")
    question = item["question"]
    answer = str(item.get("answer") or "").strip()
    question_date = str(item.get("question_date") or "").strip()

    lines_raw = flatten_haystack(
        item.get("haystack_sessions") or [],
        item.get("haystack_dates") or [],
    )
    session_tick = session_tick_map(lines_raw)
    current_tick = len(session_tick) or 1

    print(f"Selected {qid} ({QUESTION_TYPE}), haystack chars={haystack_chars(item):,}")
    print(f"  Q: {question}")
    print(f"  A: {answer}")

    graph = build_graph(lines_raw)
    print(
        f"  Full graph: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
        f"{len(lines_raw)} lines, {current_tick} sessions"
    )

    out_nodes, out_edges, pointers = cap_graph(
        graph, MAX_NODES, MAX_EDGES_PER_NODE, current_tick, question=question
    )

    lines = export_lines(lines_raw, session_tick)
    full_corpus = "\n".join(ln["line_str"] for ln in lines_raw)
    raw_tokens = count_tokens(full_corpus)
    ans_idxs = answer_line_indices(answer, lines)
    ku_pair = knowledge_update_pair(question, answer, lines)

    payload = {
        "meta": {
            "question_id": qid,
            "question": question,
            "answer": answer,
            "question_date": question_date,
            "question_type": QUESTION_TYPE,
            "current_tick": current_tick,
            "n_sessions": len(session_tick),
            "n_lines": len(lines),
            "raw_corpus_tokens": raw_tokens,
            "budget_tokens": BUDGET_TOKENS,
            "token_ratio": round(raw_tokens / max(BUDGET_TOKENS, 1), 1),
            "knowledge_update": ku_pair,
        },
        "demo_query": question,
        "answer_line_idxs": ans_idxs,
        "nodes": out_nodes,
        "edges": out_edges,
        "lines": lines,
        "pointers": pointers,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nExported {OUT_PATH.name}:")
    print(f"  nodes={len(out_nodes)} edges={len(out_edges)} lines={len(lines)}")
    print(f"  raw_corpus_tokens={raw_tokens:,} budget={BUDGET_TOKENS} ratio≈{raw_tokens/BUDGET_TOKENS:.0f}×")
    print(f"  answer_line_idxs={ans_idxs}")
    print(f"  knowledge_update old={ku_pair['old_line_idx']} new={ku_pair['new_line_idx']}")


if __name__ == "__main__":
    main()
