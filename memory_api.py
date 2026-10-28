"""Agent-facing Engram facade — active memory recommender (zero LLM).

Turns the concept graph from a passive index into ranked, path-traced
recommendations an agent can inject each turn. See recommend() / end_turn().
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import (
    GRAPH_PATH,
    build_edge_index,
    effective_weight,
    estimate_tokens,
    load_graph,
    save_graph,
    tokenize,
)
from learn import update_edges, update_next_from_tokens
from recall import build_adjacency, build_children_map

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSCRIPT_ROOT = os.environ.get(
    "ENGRAM_TRANSCRIPT_DIR",
    os.path.join(_SCRIPT_DIR, "swe_corpus"),
)
_TRANSCRIPT_ROOT = os.path.normpath(_TRANSCRIPT_ROOT)

_GAPS_SIDECAR = os.path.join(_SCRIPT_DIR, ".engram_pending_gaps.json")

# In-process cache mirrored to _GAPS_SIDECAR so CLI invocations can close gaps
# after a prior `recommend()` call in a separate process.
_pending_gaps_from_recommend: List[str] = []


def _persist_pending_gaps(gaps: List[str]) -> None:
    global _pending_gaps_from_recommend
    _pending_gaps_from_recommend = list(gaps)
    try:
        with open(_GAPS_SIDECAR, "w", encoding="utf-8") as f:
            json.dump({"gaps": gaps}, f, ensure_ascii=False)
    except OSError:
        pass


def _load_pending_gaps() -> List[str]:
    global _pending_gaps_from_recommend
    if _pending_gaps_from_recommend:
        return list(_pending_gaps_from_recommend)
    if os.path.isfile(_GAPS_SIDECAR):
        try:
            with open(_GAPS_SIDECAR, encoding="utf-8") as f:
                data = json.load(f)
            _pending_gaps_from_recommend = list(data.get("gaps") or [])
        except (OSError, json.JSONDecodeError):
            _pending_gaps_from_recommend = []
    return list(_pending_gaps_from_recommend)


def _clear_pending_gaps() -> None:
    global _pending_gaps_from_recommend
    _pending_gaps_from_recommend = []
    try:
        if os.path.isfile(_GAPS_SIDECAR):
            os.remove(_GAPS_SIDECAR)
    except OSError:
        pass

# (concept, predecessor, eff_weight, edge_age_ticks, seed_query_token)
PredRecord = Tuple[str, str, float, int, str]


def _current_tick(meta: dict) -> int:
    tick = meta.get("tick")
    if tick is None:
        return 0
    try:
        return int(tick)
    except (TypeError, ValueError):
        return 0


def _exact_seed_activation(query_tokens: List[str], nodes: dict) -> Dict[str, float]:
    """Seeds = query tokens present in graph (exact match, no global substring scan)."""
    act: Dict[str, float] = {}
    for qt in query_tokens:
        if qt in nodes:
            act[qt] = max(act.get(qt, 0), 1.0)
    return act


def _query_seed_map(query_tokens: List[str], nodes: dict) -> Dict[str, str]:
    """Map seeded graph concepts to the query token that matched them exactly."""
    return {qt: qt for qt in query_tokens if qt in nodes}


def spread_activation_traced(
    act: Dict[str, float],
    seed_of: Dict[str, str],
    nodes: dict,
    adj: Dict[str, List[Tuple[str, float, object]]],
    children_map: Dict[str, List[str]],
    current_tick: int,
    max_hops: int = 2,
) -> Tuple[Dict[str, float], Dict[str, PredRecord]]:
    """Two-hop spread like recall.py, recording predecessor on every hop."""
    boosted = dict(act)
    for c, val in list(act.items()):
        nd = nodes.get(c, {})
        if nd.get("hub"):
            boosted[c] = max(boosted.get(c, 0), val * 1.2)
        for p in nd.get("parents", []):
            boosted[p] = max(boosted.get(p, 0), val * 0.8)

    pred: Dict[str, PredRecord] = {}
    for c in boosted:
        sq = seed_of.get(c, c)
        pred[c] = (c, "", 1.0, 0, sq)

    current = boosted
    for _hop in range(max_hops):
        nxt = dict(current)
        for src, src_act in current.items():
            if src_act < 0.01:
                continue
            src_seed = pred.get(src, (src, "", 1.0, 0, src))[4]
            for child in children_map.get(src, []):
                contrib = src_act * 0.5 * 0.7
                if contrib > nxt.get(child, 0):
                    nxt[child] = contrib
                    pred[child] = (child, src, 0.0, 0, src_seed)
            for nb, w_raw, last_tick in adj.get(src, []):
                ew = effective_weight(w_raw, last_tick, current_tick)
                if not isinstance(last_tick, int):
                    age = 0
                else:
                    age = max(0, current_tick - last_tick)
                contrib = src_act * ew * 0.5
                if contrib > nxt.get(nb, 0):
                    nxt[nb] = contrib
                    pred[nb] = (nb, src, ew, age, src_seed)
        current = nxt
    return current, pred


def _chain_to_seed(concept: str, pred: Dict[str, PredRecord]) -> List[Tuple[str, str, float]]:
    """Return edge list from seed to concept: [(a, b, w_eff), ...]."""
    if concept not in pred:
        return []
    chain: List[Tuple[str, str, float]] = []
    cur = concept
    visited: Set[str] = set()
    while cur in pred and cur not in visited:
        visited.add(cur)
        _c, parent, ew, _age, seed = pred[cur]
        if not parent or parent == cur:
            break
        chain.append((parent, cur, ew))
        if parent == seed:
            break
        cur = parent
    chain.reverse()
    return chain


def _path_strings(concept: str, line_idx: int, pred: Dict[str, PredRecord]) -> List[str]:
    if concept not in pred:
        return [f"line {line_idx}"]
    _c, _p, _ew, _age, seed = pred[concept]
    parts = [f"query_term '{seed}'"]
    chain = _chain_to_seed(concept, pred)
    for a, b, ew in chain:
        rec = pred.get(b)
        age = rec[3] if rec else 0
        parts.append(f"-> {b} (edge w={ew:.2f}, age={age} ticks)")
    parts.append(f"-> line {line_idx}")
    return parts


def _resolve_line(fidx, lno: int, files_table: list) -> Tuple[str, str]:
    """Resolve interned [file_id, line_no] pointer to (session, text)."""
    if isinstance(fidx, int) and 0 <= fidx < len(files_table):
        fname = files_table[fidx]
    else:
        fname = str(fidx)

    session = fname
    if len(fname) >= 10 and fname[:4].isdigit() and fname[4] == "-":
        session = fname[:10]

    fpath = os.path.join(_TRANSCRIPT_ROOT, fname)
    gz_path = os.path.join(_SCRIPT_DIR, "raw_store", f"{fname}.gz")

    try:
        if os.path.isfile(fpath):
            with open(fpath, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        elif os.path.isfile(gz_path):
            with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        else:
            return session, f"[missing transcript: {fname}:{lno}]"
        if 1 <= lno <= len(lines):
            return session, lines[lno - 1].strip()
        return session, f"[line {lno} out of range in {fname}]"
    except OSError as e:
        return session, f"[read error {fname}:{lno}: {e}]"


def _fit_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return "..."
    if _count_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        chunk = text[:mid].rstrip()
        if _count_tokens(chunk) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    if lo <= 0:
        return "..."
    return text[:lo].rstrip() + "..."


def _dedupe_recommendation_rows(rows: List[dict]) -> List[dict]:
    seen: Set[Tuple[str, int, str]] = set()
    out: List[dict] = []
    for row in rows:
        key = (row.get("session", ""), row.get("line_idx", -1), row.get("text", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _lexical_overlap(query_tokens: List[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    line_tokens = set(tokenize(text))
    if not line_tokens:
        return 0.0
    qset = set(query_tokens)
    hit = len(qset & line_tokens)
    return hit / max(len(qset), 1)


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return estimate_tokens(text)


def recommend(query: str, budget_tokens: int = 2000) -> dict:
    """Ranked memory recommendations with routes, gaps, and token budget."""
    graph = load_graph()
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", [])
    meta = graph.get("meta", {})
    files_table = meta.get("files", [])
    current_tick = _current_tick(meta)

    qtoks = tokenize(query)
    if not qtoks:
        return {
            "recommendations": [],
            "routes": [],
            "gaps": [],
            "stats": {
                "activated_concepts": 0,
                "tokens_used": 0,
                "current_tick": current_tick,
            },
        }

    known_seeds = [t for t in qtoks if t in nodes]
    gap_tokens = list(dict.fromkeys(t for t in qtoks if t not in nodes))
    _persist_pending_gaps(gap_tokens)

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    seed_of = _query_seed_map(qtoks, nodes)

    act = _exact_seed_activation(qtoks, nodes)
    if act:
        act, pred = spread_activation_traced(
            act, seed_of, nodes, adj, children_map, current_tick, max_hops=2
        )
    else:
        pred = {}

    # Discount corpus-wide boilerplate before choosing each line's provenance
    # route, matching the document-frequency principle used by the demo.
    line_scores: Dict[Tuple[int, int], float] = {}
    line_concept: Dict[Tuple[int, int], Tuple[str, float]] = {}
    line_direct: Dict[Tuple[int, int], bool] = {}
    n_docs = max(1, len(files_table))
    for concept, score in act.items():
        df = max(1, min(n_docs, nodes.get(concept, {}).get("df", 1)))
        specificity = math.log((n_docs + 1.0) / df) / math.log(n_docs + 1.0) if n_docs > 1 else 1.0
        weighted_score = score * specificity
        direct = concept in seed_of
        for ptr in nodes.get(concept, {}).get("pointers", []):
            if not ptr or len(ptr) < 2:
                continue
            key = (ptr[0], ptr[1])
            if (direct, weighted_score) > (line_direct.get(key, False), line_scores.get(key, 0)):
                line_scores[key] = weighted_score
                line_concept[key] = (concept, weighted_score)
                line_direct[key] = direct

    rec_rows: List[dict] = []
    for (fidx, lno), base_act in sorted(
        line_scores.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
    ):
        concept, _ = line_concept[(fidx, lno)]
        session, text = _resolve_line(fidx, lno, files_table)
        lex = _lexical_overlap(qtoks, text)
        strength = base_act * (1.0 + 0.25 * lex)
        path = (
            [f"query_term '{concept}'", f"-> line {lno}"]
            if line_direct[(fidx, lno)]
            else _path_strings(concept, lno, pred)
        )
        rec_rows.append(
            {
                "line_idx": lno,
                "text": text,
                "session": session,
                "tick": current_tick,
                "strength": round(strength, 4),
                "path": path,
                "_tokens": _count_tokens(text),
                "_fidx": fidx,
                "_concept": concept,
                "_direct": line_direct[(fidx, lno)],
            }
        )

    rec_rows.sort(key=lambda r: (-r["_direct"], -r["strength"], r["_tokens"], r["line_idx"]))
    rec_rows = _dedupe_recommendation_rows(rec_rows)

    affordable = [r for r in rec_rows if r["_tokens"] <= budget_tokens]
    expensive = [r for r in rec_rows if r["_tokens"] > budget_tokens]
    packing_order = affordable + expensive

    recommendations: List[dict] = []
    tokens_used = 0
    for row in packing_order:
        full_t = row["_tokens"]
        remaining = budget_tokens - tokens_used
        if remaining <= 0:
            break
        out_row = {k: v for k, v in row.items() if not k.startswith("_")}
        if full_t > remaining:
            if recommendations:
                continue
            out_row["text"] = _fit_text_to_tokens(out_row["text"], remaining)
            t = _count_tokens(out_row["text"])
        else:
            t = full_t
        recommendations.append(out_row)
        tokens_used += t
        if tokens_used >= budget_tokens:
            break

    # Routes grouped by seed query token
    seed_lines: Dict[str, Set[int]] = {}
    for row in recommendations:
        if not row["path"]:
            continue
        seed_label = row["path"][0]
        if seed_label.startswith("query_term '"):
            seed = seed_label.split("'", 2)[1]
        else:
            seed = seed_label
        seed_lines.setdefault(seed, set()).add(row["line_idx"])

    routes: List[dict] = []
    seen_routes: Set[str] = set()
    per_seed_limit = 3
    for seed in sorted(set(seed_of.values())):
        if seed not in known_seeds:
            continue
        seed_routes = 0
        seed_concepts = [c for c in pred if pred[c][4] == seed]
        for concept in sorted(seed_concepts, key=lambda c: -act.get(c, 0)):
            if seed_routes >= per_seed_limit:
                break
            chain = _chain_to_seed(concept, pred)
            if not chain:
                if concept != seed:
                    continue
                chain_key = seed + "|direct"
            else:
                chain_key = seed + "|" + "|".join(f"{a}>{b}" for a, b, _ in chain)
            if chain_key in seen_routes:
                continue
            seen_routes.add(chain_key)
            seed_routes += 1
            reached = sorted(
                {
                    lno
                    for (fidx, lno), _ in line_scores.items()
                    if line_concept.get((fidx, lno), ("", 0))[0] == concept
                }
            )
            if not reached:
                reached = sorted(seed_lines.get(seed, set()))
            routes.append(
                {
                    "seed": seed,
                    "chain": [[a, b, round(w, 2)] for a, b, w in chain],
                    "reached_lines": reached[:10],
                }
            )

    # Gaps: unknown tokens + nearest co-activated neighbors from other seeds
    gaps: List[dict] = []
    other_activated = sorted(
        ((c, s) for c, s in act.items() if c in nodes),
        key=lambda x: -x[1],
    )
    for gt in gap_tokens:
        nearest: List[Tuple[str, str]] = []
        for concept, score in other_activated[:12]:
            if concept == gt:
                continue
            nearest.append((concept, f"co-activated neighbor (score={score:.2f})"))
        gaps.append(
            {
                "concept": gt,
                "status": "unknown",
                "nearest_known": nearest[:5],
                "note": "will be learned on next ingest",
            }
        )

    return {
        "recommendations": recommendations,
        "routes": routes[:15],
        "gaps": gaps,
        "stats": {
            "activated_concepts": sum(1 for v in act.values() if v >= 0.01),
            "tokens_used": tokens_used,
            "current_tick": current_tick,
        },
    }


def end_turn(turn_text: str) -> dict:
    """Ingest turn text, advance tick, save graph.

    ``closed_gaps`` compares against concepts recorded as unknown by the most
    recent ``recommend()`` in this process, or the sidecar
    ``.engram_pending_gaps.json`` when using the CLI across separate invocations.
    """
    global _pending_gaps_from_recommend

    graph = load_graph()
    nodes: dict = graph.setdefault("nodes", {})
    edges: list = graph.setdefault("edges", [])
    meta = graph.setdefault("meta", {})
    files_table: list = meta.setdefault("files", [])
    edge_idx = build_edge_index(edges)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    tick = _current_tick(meta) + 1
    meta["tick"] = tick

    source = f"turn_{today}_{tick}"
    if source not in files_table:
        files_table.append(source)
    source_id = files_table.index(source)

    toks = tokenize(turn_text)
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

    for t in matched:
        if t not in new_concepts:
            pl = nodes[t].setdefault("pointers", [])
            pl.append([source_id, 1])
            if len(pl) > 30:
                nodes[t]["pointers"] = pl[-30:]

    reinforced = update_edges(matched, edges, edge_idx, tick)
    update_next_from_tokens(toks, nodes)

    graph["edges"] = edges
    meta["last_learned"] = today
    save_graph(graph)

    pending = _load_pending_gaps()
    closed_gaps = [g for g in pending if g in nodes]
    _clear_pending_gaps()

    return {
        "new_concepts": new_concepts,
        "reinforced_edges": reinforced,
        "tick": tick,
        "closed_gaps": closed_gaps,
    }


def _pretty_print_result(data: dict, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    stats = data.get("stats") or {}
    if stats:
        print(
            f"\n[stats] activated_concepts={stats.get('activated_concepts')} "
            f"tokens_used={stats.get('tokens_used')} "
            f"current_tick={stats.get('current_tick')}"
        )

    gaps = data.get("gaps") or []
    if gaps:
        print("\n--- gaps (unknown query concepts) ---")
        for g in gaps:
            print(f"  ? {g['concept']} — {g['status']}: {g['note']}")
            for nk, reason in g.get("nearest_known", [])[:3]:
                print(f"      near: {nk} ({reason})")

    routes = data.get("routes") or []
    if routes:
        print("\n--- routes (concept chains) ---")
        for r in routes[:8]:
            chain_str = " → ".join(f"{a}—{b}({w})" for a, b, w in r.get("chain", []))
            lines = r.get("reached_lines", [])
            print(f"  seed '{r['seed']}': {chain_str or '(direct)'}")
            if lines:
                print(f"    reached lines: {lines[:8]}")

    recs = data.get("recommendations") or []
    if recs:
        print(f"\n--- recommendations ({len(recs)}) ---")
        for i, rec in enumerate(recs[:12], 1):
            print(f"\n  [{i}] strength={rec['strength']} line={rec['line_idx']} session={rec['session']}")
            print(f"      path: {' '.join(rec.get('path', []))}")
            text = rec.get("text", "")
            if len(text) > 220:
                text = text[:217] + "..."
            print(f"      text: {text}")
    elif not gaps:
        print("\n(no recommendations)")


def _pretty_print_end_turn(data: dict) -> None:
    print(f"\n{'=' * 60}")
    print("end_turn report")
    print("=" * 60)
    print(f"  tick: {data.get('tick')}")
    print(f"  reinforced_edges: {data.get('reinforced_edges')}")
    nc = data.get("new_concepts") or []
    print(f"  new_concepts ({len(nc)}): {', '.join(nc[:20])}{'...' if len(nc) > 20 else ''}")
    cg = data.get("closed_gaps") or []
    if cg:
        print(f"  closed_gaps: {', '.join(cg)}")
    else:
        print("  closed_gaps: (none)")
    if "sync" in data:
        print(f"  sync: {data['sync']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Engram memory API — recommend memories or ingest a turn"
    )
    parser.add_argument("query", nargs="?", help="Query for recommend()")
    parser.add_argument(
        "--end-turn",
        dest="end_turn_text",
        metavar="TEXT",
        help="Ingest turn text via end_turn()",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Token budget for recommendations (default 2000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of pretty sections",
    )
    args = parser.parse_args()

    if args.end_turn_text:
        result = end_turn(args.end_turn_text)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            _pretty_print_end_turn(result)
        return

    if not args.query:
        parser.error("Provide a query string or --end-turn TEXT")

    result = recommend(args.query, budget_tokens=args.budget)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _pretty_print_result(result, f'recommend("{args.query}")')


if __name__ == "__main__":
    main()
