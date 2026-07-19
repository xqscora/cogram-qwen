"""Bootstrap concept graph from chat transcripts — batch, zero LLM.

Memory = concept network extracted statistically, NOT summaries.
Co-occurrence within line + 3-line window → Hebbian edges with surprise bonus.
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_lib import (
    GRAPH_PATH,
    edge_key,
    generality,
    hebbian_delta,
    parse_date_from_filename,
    save_graph,
    estimate_tokens,
    tokenize,
)

TRANSCRIPT_DIR = os.environ.get(
    "ENGRAM_TRANSCRIPT_DIR",
    os.path.join(os.path.dirname(__file__), "swe_corpus"),
)
TRANSCRIPT_DIR = os.path.normpath(TRANSCRIPT_DIR)

MAX_CONCEPTS = 2000
TOP_K_EDGES = 20  # keep each concept's K strongest links (bounds size, no orphans)
MAX_POINTERS = 30
COOC_WINDOW = 3  # lines
PARENT_P_THRESHOLD = 0.6
PARENT_G_RATIO = 1.3
MAX_PARENTS = 3


def iter_transcripts() -> List[str]:
    pattern = os.path.join(TRANSCRIPT_DIR, "*_transcript.md")
    files = []
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        if base == "README.md" or base.startswith("RECENT_"):
            continue
        files.append(p)
    return sorted(files)


def promote_concepts(
    candidates: Set[str],
    surprise_mass: Dict[str, float],
) -> Set[str]:
    """NO-PRESET promotion: every content token is a candidate concept (novelty =
    a never-before-seen token is maximally surprising → encode it). Cap by
    SURPRISE MASS (sum of incident edge weights), NOT frequency, if over the cap.

    surprise_mass rewards concepts that formed NEW connections (first
    co-occurrence = 1.0 via hebbian_delta), not ones that merely repeat.
    """
    if len(candidates) <= MAX_CONCEPTS:
        return set(candidates)
    ranked = sorted(candidates, key=lambda t: surprise_mass.get(t, 0.0), reverse=True)
    return set(ranked[:MAX_CONCEPTS])


def cooc_pairs(tokens: List[str]) -> Set[Tuple[str, str]]:
    uniq = sorted(set(tokens))
    pairs = set()
    for i, a in enumerate(uniq):
        for b in uniq[i + 1 :]:
            pairs.add(edge_key(a, b))
    return pairs


def main():
    files = iter_transcripts()
    if not files:
        print(f"No transcripts found in {TRANSCRIPT_DIR}")
        sys.exit(1)

    print(f"Reading {len(files)} transcript files...")

    # EXPERIENCED TICKS (subjective time): each transcript FILE is one session
    # the agent lived through. file 1 (sorted) → tick 1, ..., file N → tick N.
    file_tick: Dict[str, int] = {os.path.basename(f): i for i, f in enumerate(files, 1)}
    meta_tick = len(files)

    token_totals: Counter = Counter()
    token_file_counts: Dict[str, Counter] = defaultdict(Counter)
    token_first_tick: Dict[str, int] = {}
    token_last_tick: Dict[str, int] = {}
    file_dates: Dict[str, str] = {}
    # line records: (fname, line_no, tick, tokens)
    lines_data: List[Tuple[str, int, int, List[str]]] = []
    n_lines = 0
    raw_chars = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        tick = file_tick[fname]
        file_dates[fname] = parse_date_from_filename(fpath)
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                raw_chars += len(line)
                toks = tokenize(line)
                if not toks:
                    continue
                n_lines += 1
                lines_data.append((fname, i, tick, toks))
                for t in toks:
                    token_totals[t] += 1
                    token_file_counts[t][fpath] += 1
                    if t not in token_first_tick or tick < token_first_tick[t]:
                        token_first_tick[t] = tick
                    if t not in token_last_tick or tick > token_last_tick[t]:
                        token_last_tick[t] = tick

    # NO threshold: every content token is a candidate concept (novelty).
    candidates: Set[str] = set(token_totals.keys())

    # Co-occurrence + directed transitions + per-pair last tick, over ALL
    # candidates BEFORE capping (surprise mass needs the edges to exist first).
    cooc_events: Counter = Counter()
    pair_last_tick: Dict[Tuple[str, str], int] = {}
    trans: Counter = Counter()

    for idx in range(len(lines_data)):
        window_tokens: List[str] = []
        for j in range(idx, min(idx + COOC_WINDOW, len(lines_data))):
            window_tokens.extend(lines_data[j][3])
        tick = lines_data[idx][2]
        present = sorted(set(t for t in window_tokens if t in candidates))
        if len(present) < 2:
            continue
        for pair in cooc_pairs(present):
            cooc_events[pair] += 1
            if tick > pair_last_tick.get(pair, 0):
                pair_last_tick[pair] = tick

        first_pos: Dict[str, int] = {}
        for pos, t in enumerate(window_tokens):
            if t in candidates and t not in first_pos:
                first_pos[t] = pos
        for a, pa in first_pos.items():
            for b, pb in first_pos.items():
                if a != b and pa < pb:
                    trans[(a, b)] += 1

    # Hebbian edge weights from co-occurrence events
    edge_data: Dict[Tuple[str, str], dict] = {}
    for pair, n_events in cooc_events.items():
        w = 0.0
        for i in range(n_events):
            w += hebbian_delta(is_first=(i == 0), n_prev=i)
        edge_data[pair] = {
            "w_raw": w,
            "last_tick": pair_last_tick.get(pair, meta_tick),
            "n": n_events,
        }

    # Surprise mass = sum of incident edge weights (NOT frequency).
    surprise_mass: Counter = Counter()
    for (a, b), ed in edge_data.items():
        surprise_mass[a] += ed["w_raw"]
        surprise_mass[b] += ed["w_raw"]

    # Promote by novelty; cap by surprise mass only if over MAX_CONCEPTS.
    concepts = promote_concepts(candidates, surprise_mass)
    if len(candidates) > len(concepts):
        edge_data = {
            p: ed for p, ed in edge_data.items() if p[0] in concepts and p[1] in concepts
        }
        trans = Counter({k: v for k, v in trans.items() if k[0] in concepts and k[1] in concepts})

    # Build nodes (surprise=1.0 at creation; count kept for display only)
    nodes: Dict[str, dict] = {}
    for c in concepts:
        df = len(token_file_counts[c])
        dates = [file_dates.get(os.path.basename(fp), "") for fp in token_file_counts[c]]
        dates = [d for d in dates if d]
        first_seen = min(dates) if dates else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        last_seen = max(dates) if dates else first_seen
        nodes[c] = {
            "count": token_totals[c],
            "df": df,
            "surprise": 1.0,
            "first_tick": token_first_tick.get(c, 1),
            "last_tick": token_last_tick.get(c, meta_tick),
            "first_seen": first_seen,  # display only
            "last_seen": last_seen,    # display only
            "pointers": [],
            "parents": [],
            "hub": False,
            "generality": 0.0,
            "neighbors": [],
        }

    # Intern filenames: pointers store a small int id, not the ~50-char filename.
    file_list: List[str] = []
    file_id: Dict[str, int] = {}

    def fid(name: str) -> int:
        if name not in file_id:
            file_id[name] = len(file_list)
            file_list.append(name)
        return file_id[name]

    for fname, lno, tick, toks in lines_data:
        present = [t for t in toks if t in concepts]
        if not present:
            continue
        for t in present:
            ptr = [fid(fname), lno]
            pl = nodes[t]["pointers"]
            pl.append(ptr)
            if len(pl) > MAX_POINTERS:
                nodes[t]["pointers"] = pl[-MAX_POINTERS:]

    # Neighbor lists for generality
    neighbor_sets: Dict[str, Set[str]] = defaultdict(set)
    for (a, b) in edge_data:
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)
    max_deg = max((len(v) for v in neighbor_sets.values()), default=1)
    max_df = max((n["df"] for n in nodes.values()), default=1)

    for c, nd in nodes.items():
        nd["neighbors"] = sorted(neighbor_sets.get(c, set()))
        nd["generality"] = generality(nd, max_deg, max_df)

    # Top-5 outgoing transitions per concept (small, brain-like sequence prior)
    outgoing: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (a, b), cnt in trans.items():
        outgoing[a].append((b, cnt))
    for c, nd in nodes.items():
        lst = outgoing.get(c, [])
        lst.sort(key=lambda x: x[1], reverse=True)
        if lst:
            nd["next"] = [[b, cnt] for b, cnt in lst[:5]]

    # Hub tier: top ~10% by generality
    sorted_by_g = sorted(nodes.items(), key=lambda x: x[1]["generality"], reverse=True)
    n_hubs = max(1, len(sorted_by_g) // 10)
    hub_set = {c for c, _ in sorted_by_g[:n_hubs]}
    for c in hub_set:
        nodes[c]["hub"] = True

    # Parent-child subsumption: P(a|b) >= 0.6 and generality(a) > 1.3 * generality(b)
    b_totals: Counter = Counter()
    for (a, b), n in cooc_events.items():
        if a in concepts and b in concepts:
            b_totals[a] += n
            b_totals[b] += n

    for (a, b), n_ab in cooc_events.items():
        if a not in concepts or b not in concepts:
            continue
        for parent, child in ((a, b), (b, a)):
            p_given = n_ab / max(b_totals[child], 1)
            gp = nodes[parent]["generality"]
            gc = nodes[child]["generality"]
            if p_given >= PARENT_P_THRESHOLD and gp > PARENT_G_RATIO * gc:
                parents = nodes[child]["parents"]
                if parent not in parents and len(parents) < MAX_PARENTS:
                    parents.append(parent)
                    nodes[child]["parents"] = parents

    # Keep each concept's TOP_K strongest links instead of a global cap.
    incident: Dict[str, List[Tuple[float, Tuple[str, str]]]] = defaultdict(list)
    for pair, ed in edge_data.items():
        a, b = pair
        incident[a].append((ed["w_raw"], pair))
        incident[b].append((ed["w_raw"], pair))
    keep_pairs: Set[Tuple[str, str]] = set()
    for c, lst in incident.items():
        lst.sort(reverse=True)
        for _, pair in lst[:TOP_K_EDGES]:
            keep_pairs.add(pair)

    edges = []
    for pair in keep_pairs:
        ed = edge_data[pair]
        a, b = pair
        # 4th element is now last_tick (int), not a date string.
        edges.append([a, b, round(ed["w_raw"], 4), ed["last_tick"]])

    # Drop the redundant neighbors list before saving.
    for nd in nodes.values():
        nd.pop("neighbors", None)
        nd["generality"] = round(nd["generality"], 4)

    graph = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "tick": meta_tick,  # current global tick = number of sessions/files
            "n_files": len(files),
            "n_lines": n_lines,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "raw_total_chars": raw_chars,
            "transcript_dir": TRANSCRIPT_DIR,
            "files": file_list,  # pointer id -> filename
        },
    }
    save_graph(graph, GRAPH_PATH)

    index_json = open(GRAPH_PATH, encoding="utf-8").read()
    index_tokens = estimate_tokens(len(index_json))
    raw_tokens = estimate_tokens(raw_chars)
    ratio = raw_tokens / max(index_tokens, 1)

    print("\n=== Engram Extract Stats ===")
    print(f"Files:     {len(files)}")
    print(f"Lines:     {n_lines}")
    print(f"Concepts:  {len(nodes)}")
    print(f"Edges:     {len(edges)}")
    print(f"Hubs:      {n_hubs}")
    print(f"Index:     ~{index_tokens:,} tokens ({len(index_json):,} chars)")
    print(f"Raw corpus:~{raw_tokens:,} tokens ({raw_chars:,} chars)")
    print(f"Compression ratio (raw/index): {ratio:.1f}x")
    print(f"Saved → {GRAPH_PATH}")


if __name__ == "__main__":
    main()
