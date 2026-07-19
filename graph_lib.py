"""Engram shared library — concept-network memory (zero LLM).

Design: memory = statistical concept graph, not summaries.
Forgetting = edge decay toward floor EPS (never zero, never deleted).
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Decay-but-never-death: edges approach EPS, never reach zero.
EPS = 0.05
TAU = 30.0  # days (legacy wall-clock constant; superseded by TAU_TICKS)
# Subjective time: decay is measured in EXPERIENCED ticks (interactions/sessions),
# not calendar days. If the agent is OFF, no ticks pass → memory freezes (no decay).
TAU_TICKS = 50

GRAPH_PATH = os.environ.get(
    "COGRAM_GRAPH",
    os.path.join(os.path.dirname(__file__), "concept_graph.json"),
)

# --- English stopwords (~150) ---
EN_STOP = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing don down during
    each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just ll me more most my myself no nor not
    now of off on once only or other our ours ourselves out over own re s same she
    should so some such t than that the their theirs them themselves then there these
    they this those through to too under until up ve very was we were what when where
    which while who whom why will with would you your yours yourself yourselves
    also get got like make made use used using one two three new way may well much
    many still even back go going come came take took see saw say said says tell
    told think thought know knew want need let put find give given keep let look
    looking look looked help helps helped try tried trying really actually maybe
    something anything everything nothing someone anyone everyone into onto upon
    """.split()
)

# --- Chinese stopwords (~80) ---
ZH_STOP = frozenset(
    """
    的 了 是 我 你 他 她 它 就 都 和 也 这 那 一个 可以 什么 没有 我们 你们 他们
    但是 因为 所以 如果 还是 就是 不是 时候 现在 然后 觉得 知道 一下 这个 那个
    怎么 已经 需要 自己 还有 或者 虽然 不过 可能 应该 这样 那样 一些 这些 那些
    非常 比较 真的 其实 只是 还是 已经 正在 开始 结束 问题 回答 谢谢 好的 嗯 啊
    吧 呢 吗 哦 哈 呀 嘛 被 把 给 让 向 从 到 在 对 与 及 等 之 其 所 以 于 为
    会 能 要 想 说 看 做 用 有 没 不 很 更 最 太 又 再 还 只 才 并 而 且 若 当
    """.split()
)

# English: word tokens (min 3 chars after first letter)
EN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
# CJK single chars / sequences for bigram fallback
CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# Markdown / syntax noise
MD_NOISE = re.compile(
    r"^(\#{1,6}|```|---|\*\*|\*|>|`|\[|\]|\(|\)|\||\.\.\.)$"
)
PURE_NUM = re.compile(r"^\d+$")

_jieba = None
_jieba_tried = False


def _ensure_jieba():
    global _jieba, _jieba_tried
    if _jieba_tried:
        return _jieba
    _jieba_tried = True
    try:
        import jieba  # type: ignore

        _jieba = jieba
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "jieba", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import jieba  # type: ignore

            _jieba = jieba
        except Exception:
            _jieba = None
    return _jieba


def _cjk_bigrams(text: str) -> List[str]:
    chars = CJK.findall(text)
    if len(chars) < 2:
        return chars
    out = []
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        if bg not in ZH_STOP and len(bg) >= 2:
            out.append(bg)
    return out


def tokenize(text: str) -> List[str]:
    """Statistical tokens — English regex + jieba/CJK, zero LLM."""
    tokens: List[str] = []
    seen_pos: Set[Tuple[int, int]] = set()

    for m in EN_WORD.finditer(text):
        w = m.group(0).lower()
        if w in EN_STOP or PURE_NUM.match(w):
            continue
        tokens.append(w)
        seen_pos.add((m.start(), m.end()))

    jb = _ensure_jieba()
    if jb is not None:
        for w in jb.cut(text):
            w = w.strip()
            if not w or len(w) < 2:
                continue
            if w in ZH_STOP or PURE_NUM.match(w):
                continue
            if not CJK.search(w) and not EN_WORD.fullmatch(w):
                continue
            if EN_WORD.fullmatch(w):
                wl = w.lower()
                if wl in EN_STOP:
                    continue
                tokens.append(wl)
            else:
                tokens.append(w)
    else:
        tokens.extend(_cjk_bigrams(text))

    # Drop markdown-syntax tokens
    return [t for t in tokens if not MD_NOISE.match(t) and len(t) >= 2]


def parse_date_from_filename(filename: str) -> str:
    """Extract YYYY-MM-DD from transcript filename prefix."""
    base = os.path.basename(filename)
    m = re.match(r"(\d{4}-\d{2}-\d{2})", base)
    if m:
        return m.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def effective_weight(
    w_raw: float,
    last_tick,
    current_tick,
    tau_ticks: float = TAU_TICKS,
) -> float:
    """Tick-decayed edge weight; floor EPS = decay but never death.

    Decay is by EXPERIENCED ticks (subjective time), not wall-clock days. A tick
    is one interaction/session the agent actually lives through; while off, no
    ticks pass so weights are frozen.

    Backward-compat: if `last_tick`/`current_tick` are not ints (e.g. an old
    graph stored a date string), treat elapsed ticks as 0 → no decay.
    """
    if not isinstance(last_tick, int) or not isinstance(current_tick, int):
        return EPS + (w_raw - EPS)  # == w_raw; no-decay fallback for old format
    dt = max(0, current_tick - last_tick)
    return EPS + (w_raw - EPS) * math.exp(-dt / tau_ticks)


def load_graph(path: str = GRAPH_PATH) -> dict:
    if not os.path.isfile(path):
        return {"nodes": {}, "edges": [], "meta": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_graph(graph: dict, path: str = GRAPH_PATH) -> None:
    # Compact separators: the index is a machine artifact, not for human reading.
    # Whitespace pretty-printing tripled the on-disk footprint.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, separators=(",", ":"))


def edge_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def hebbian_delta(is_first: bool, n_prev: int) -> float:
    """Surprise bonus on first co-occurrence; diminishing returns on repeats."""
    if is_first:
        return 1.0  # statistical prediction error
    return 0.3 / (1.0 + n_prev)


def mine_motifs(
    lines_data: List[Tuple[str, int, int, List[str]]],
    concepts: Set[str],
    edge_data: Dict[Tuple[str, str], dict],
    hub_concepts: Optional[Set[str]] = None,
    window: int = 15,
    min_coactive: int = 3,
    max_coactive: int = 10,
    max_motifs: int = 300,
) -> Dict[Tuple[str, ...], dict]:
    """Level-1 concept-of-concepts: recurring co-active clusters of level-0
    concepts, generalized across DISTINCT files (not just repeated lines in
    one file — that would just be local repetition). Reuses `hebbian_delta`
    for both minting and reinforcement — no new decay or surprise math.

    Design: Cogram's own `CONCEPT_OF_CONCEPTS.md`. Mint threshold is a
    self-calibrating median of THIS corpus's own motif-surprise distribution,
    not a hand-picked number. `max_motifs` is the same kind of hard resource
    cap as `MAX_CONCEPTS` for level-0 — it trims by weight, it never gates
    which motifs get a chance to be minted in the first place.

    Returns: {motif_key (sorted tuple of member concept ids): {
        members, order, w_raw, n_files, first_tick, last_tick, provenance
    }}
    """
    n = len(lines_data)
    hub_concepts = hub_concepts or set()
    # Boilerplate/template words (the same "hub" concepts already excluded by
    # the IDF discount at line-level retrieval, see swe_recall_demo.py) recur
    # on nearly every line and would otherwise glue unrelated windows into one
    # giant meaningless cluster instead of a small, specific problem->fix motif.
    motif_pool = concepts - hub_concepts
    raw_candidates: List[Tuple[float, str, int, int, frozenset, tuple]] = []

    def surprise(members: Iterable[str]) -> float:
        members = sorted(members)
        ws = [
            edge_data[edge_key(a, b)]["w_raw"]
            for i, a in enumerate(members)
            for b in members[i + 1 :]
            if edge_key(a, b) in edge_data
        ]
        return sum(ws) / len(ws) if ws else 0.0

    for idx in range(n):
        fname, lno, tick, toks = lines_data[idx]
        present_here = {t for t in toks if t in motif_pool}
        if len(present_here) < min_coactive:
            continue
        lo, hi = max(0, idx - window), min(n, idx + window + 1)
        order: List[str] = []
        seen: Set[str] = set()
        for j in range(lo, hi):
            f2, _l2, _t2, toks2 = lines_data[j]
            if f2 != fname:
                continue
            for tok in toks2:
                if tok in motif_pool and tok not in seen:
                    seen.add(tok)
                    order.append(tok)
        if len(seen) < min_coactive or len(seen) > max_coactive:
            # Chase & Simon (1973): experts chunk into ~6 recognized configurations,
            # not dozens of loose items — a window this diffuse isn't a clean chunk.
            continue
        raw_candidates.append((surprise(seen), fname, lno, tick, frozenset(seen), tuple(order)))

    if not raw_candidates:
        return {}

    surprises = sorted(c[0] for c in raw_candidates)
    median = surprises[len(surprises) // 2]

    motifs: Dict[Tuple[str, ...], dict] = {}
    files_seen: Dict[Tuple[str, ...], Set[str]] = {}

    for s, fname, lno, tick, fs, order in raw_candidates:
        if s < median:
            continue  # self-calibrating cutoff, derived from this corpus's own distribution
        key = tuple(sorted(fs))
        first_time_overall = key not in motifs
        if first_time_overall:
            motifs[key] = {
                "members": list(key),
                "order": list(order),
                "w_raw": s,  # minted at its own motif-surprise score, not a fixed 1.0
                "n_files": 0,
                "first_tick": tick,
                "last_tick": tick,
                "provenance": [],
            }
            files_seen[key] = set()

        m = motifs[key]
        if fname not in files_seen[key]:
            if not first_time_overall:
                m["w_raw"] += hebbian_delta(is_first=False, n_prev=m["n_files"])
            m["n_files"] += 1
            files_seen[key].add(fname)

        m["last_tick"] = max(m["last_tick"], tick)
        if len(m["provenance"]) < 30:
            m["provenance"].append([fname, lno])

    if len(motifs) > max_motifs:
        ranked = sorted(motifs.items(), key=lambda kv: kv[1]["w_raw"], reverse=True)
        motifs = dict(ranked[:max_motifs])

    return motifs


def estimate_tokens(text_or_len) -> int:
    if isinstance(text_or_len, (int, float)):
        return int(text_or_len) // 4
    return len(str(text_or_len)) // 4


def generality(node: dict, max_degree: int, max_df: int) -> float:
    """Hub score: normalized neighbor diversity + document frequency."""
    deg = len(node.get("neighbors", []))
    df = node.get("df", 1)
    nd = deg / max(max_degree, 1)
    ndf = df / max(max_df, 1)
    return 0.5 * nd + 0.5 * ndf


def build_edge_index(edges: list) -> Dict[Tuple[str, str], dict]:
    idx: Dict[Tuple[str, str], dict] = {}
    for e in edges:
        if len(e) >= 4:
            # 4th element is now last_tick (int). Old graphs stored a date
            # string here; effective_weight()'s guard treats non-ints as no-decay.
            a, b, w, lt = e[0], e[1], e[2], e[3]
        else:
            continue
        idx[edge_key(a, b)] = {"a": a, "b": b, "w_raw": w, "last_tick": lt, "n": e[4] if len(e) > 4 else 1}
    return idx


def neighbors_of(concept: str, edge_idx: Dict[Tuple[str, str], dict]) -> List[Tuple[str, float, object]]:
    """Return (neighbor, w_raw, last_tick) for concept."""
    out = []
    for k, e in edge_idx.items():
        if e["a"] == concept:
            out.append((e["b"], e["w_raw"], e["last_tick"]))
        elif e["b"] == concept:
            out.append((e["a"], e["w_raw"], e["last_tick"]))
    return out
