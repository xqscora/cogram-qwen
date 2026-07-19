"""LoCoMo QA benchmark: Engram concept-graph retrieval vs full-context memory.

Real API calls via OpenRouter; no mocks. Uses graph_lib statistical primitives.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from graph_lib import (  # noqa: E402
    EPS,
    build_edge_index,
    edge_key,
    effective_weight,
    generality,
    hebbian_delta,
    neighbors_of,
    tokenize,
)

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
LOCOMO_PATH = os.path.join(_SCRIPT_DIR, "locomo10.json")
ENV_PATH = os.path.join(_SCRIPT_DIR, ".env.local")
RESULTS_MD = os.path.join(_SCRIPT_DIR, "benchmark_qa_results.md")
RAW_JSON = os.path.join(_SCRIPT_DIR, "benchmark_qa_raw.json")

OPENROUTER_CHAT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

PREFERRED_MODEL = "openai/gpt-4o-mini"
FALLBACK_MODEL_PATTERNS = (
    r"gpt-4o-mini",
    r"haiku",
    r"flash",
    r"mini",
    r"3\.5",
)

COOC_WINDOW = 3
MAX_CONCEPTS = 1500
TOP_K_EDGES = 20
# Index-compression cap disabled for retrieval fidelity — Engram retains all
# line pointers losslessly; capping dropped evidence from associative reach.
MAX_POINTERS = 100000
PARENT_P_THRESHOLD = 0.6
PARENT_G_RATIO = 1.3
MAX_PARENTS = 3
MAX_ENGRAM_LINES = 50
TARGET_QA = 30
RANDOM_SEED = 42


def load_env_key(path: str = ENV_PATH) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    key = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in .env.local")
    return key


def ensure_tiktoken():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda t: len(enc.encode(t))
    except ImportError:
        return lambda t: len(t) // 4


def download_locomo() -> None:
    if os.path.isfile(LOCOMO_PATH):
        print(f"Using cached {LOCOMO_PATH}")
        return
    print(f"Downloading LoCoMo → {LOCOMO_PATH}")
    urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_PATH)
    print("Download complete.")


def flatten_conversation(conv: dict) -> List[dict]:
    """Return chronologically ordered turns with dia_id, speaker, text, line_str."""
    session_nums = []
    for k in conv:
        m = re.match(r"session_(\d+)$", k)
        if m:
            session_nums.append(int(m.group(1)))
    session_nums.sort()

    lines: List[dict] = []
    for sn in session_nums:
        sk = f"session_{sn}"
        turns = conv.get(sk) or []
        for turn in turns:
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "?")
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            line_str = f"[{dia_id}] {speaker}: {text}"
            lines.append(
                {
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "text": text,
                    "line_str": line_str,
                    "idx": len(lines),
                    "session": sn,
                }
            )
    return lines


def cooc_pairs(tokens: List[str]) -> Set[Tuple[str, str]]:
    uniq = sorted(set(tokens))
    pairs: Set[Tuple[str, str]] = set()
    for i, a in enumerate(uniq):
        for b in uniq[i + 1 :]:
            pairs.add(edge_key(a, b))
    return pairs


def build_graph(lines: List[dict]) -> dict:
    """Concept graph from conversation lines — NO-PRESET build.

    Promotion by NOVELTY + SURPRISE, not frequency:
      * Every content token from tokenize() is a candidate concept (a
        never-before-seen token is maximally surprising = prediction error →
        encode it). NO count/df threshold.
      * If candidates exceed MAX_CONCEPTS, cap by SURPRISE MASS =
        sum(w_raw of incident edges), where edge weights come from
        hebbian_delta (first co-occurrence = 1.0, repeats decay). This rewards
        concepts that formed NEW connections, not ones that merely repeat.

    Decay by EXPERIENCED ticks: each LoCoMo session = one tick (session_1 →
    tick 1, ...). Nodes/edges store last_tick; meta.tick = number of sessions.
    """
    # Assign a tick per session in sorted order (order/先后, measured in sessions).
    sessions_sorted = sorted({ln.get("session", 0) for ln in lines})
    session_tick = {s: i for i, s in enumerate(sessions_sorted, 1)}
    meta_tick = len(sessions_sorted)

    token_totals: Counter = Counter()
    token_first_tick: Dict[str, int] = {}
    token_last_tick: Dict[str, int] = {}
    lines_data: List[Tuple[int, List[str], int]] = []  # (line_idx, tokens, tick)

    for ln in lines:
        toks = tokenize(ln["line_str"])
        if not toks:
            continue
        tick = session_tick.get(ln.get("session", 0), meta_tick)
        lines_data.append((ln["idx"], toks, tick))
        for t in toks:
            token_totals[t] += 1
            if t not in token_first_tick or tick < token_first_tick[t]:
                token_first_tick[t] = tick
            if t not in token_last_tick or tick > token_last_tick[t]:
                token_last_tick[t] = tick

    # NO threshold: every content token is a candidate concept.
    candidates: Set[str] = set(token_totals.keys())

    # Co-occurrence events + directed transitions + per-pair last tick,
    # computed over ALL candidates BEFORE any capping (surprise mass needs edges).
    cooc_events: Counter = Counter()
    pair_last_tick: Dict[Tuple[str, str], int] = {}
    trans: Counter = Counter()

    for idx in range(len(lines_data)):
        window_tokens: List[str] = []
        for j in range(idx, min(idx + COOC_WINDOW, len(lines_data))):
            window_tokens.extend(lines_data[j][1])
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

    # Cap by surprise mass if over MAX_CONCEPTS; on one conversation this rarely
    # binds, so effectively all content words become concepts — that's the point.
    if len(candidates) > MAX_CONCEPTS:
        ranked = sorted(candidates, key=lambda t: surprise_mass.get(t, 0.0), reverse=True)
        concepts: Set[str] = set(ranked[:MAX_CONCEPTS])
        edge_data = {
            p: ed for p, ed in edge_data.items() if p[0] in concepts and p[1] in concepts
        }
        trans = Counter({k: v for k, v in trans.items() if k[0] in concepts and k[1] in concepts})
    else:
        concepts = set(candidates)

    nodes: Dict[str, dict] = {}
    for c in concepts:
        nodes[c] = {
            "count": token_totals[c],  # display only, NOT used for gating
            "df": 1,
            "surprise": 1.0,
            "first_tick": token_first_tick.get(c, 1),
            "last_tick": token_last_tick.get(c, meta_tick),
            "pointers": [],
            "parents": [],
            "hub": False,
            "generality": 0.0,
        }

    for line_idx, toks, _tick in lines_data:
        present = [t for t in toks if t in concepts]
        if not present:
            continue
        for t in present:
            pl = nodes[t]["pointers"]
            pl.append(line_idx)
            if len(pl) > MAX_POINTERS:
                nodes[t]["pointers"] = pl[-MAX_POINTERS:]

    neighbor_sets: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edge_data:
        neighbor_sets[a].add(b)
        neighbor_sets[b].add(a)
    max_deg = max((len(v) for v in neighbor_sets.values()), default=1)
    max_df = 1

    for c, nd in nodes.items():
        nd["neighbors"] = sorted(neighbor_sets.get(c, set()))
        nd["generality"] = generality(nd, max_deg, max_df)

    outgoing: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (a, b), cnt in trans.items():
        outgoing[a].append((b, cnt))
    for c, nd in nodes.items():
        lst = outgoing.get(c, [])
        lst.sort(key=lambda x: x[1], reverse=True)
        if lst:
            nd["next"] = [[b, cnt] for b, cnt in lst[:5]]

    sorted_by_g = sorted(nodes.items(), key=lambda x: x[1]["generality"], reverse=True)
    n_hubs = max(1, len(sorted_by_g) // 10)
    for c, _ in sorted_by_g[:n_hubs]:
        nodes[c]["hub"] = True

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
        edges.append([a, b, round(ed["w_raw"], 4), ed["last_tick"]])

    for nd in nodes.values():
        nd.pop("neighbors", None)
        nd["generality"] = round(nd["generality"], 4)

    return {"nodes": nodes, "edges": edges, "meta": {"tick": meta_tick}}


def seed_activation(query_tokens: List[str], nodes: dict) -> Dict[str, float]:
    act: Dict[str, float] = {}
    for qt in query_tokens:
        for c in nodes:
            if c == qt:
                act[c] = max(act.get(c, 0), 1.0)
            elif qt in c or c in qt:
                act[c] = max(act.get(c, 0), 0.6)
    return act


def build_adjacency(edge_idx: dict) -> Dict[str, List[Tuple[str, float, object]]]:
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


def spread_activation(
    act: Dict[str, float],
    nodes: dict,
    adj: Dict[str, List[Tuple[str, float, object]]],
    children_map: Dict[str, List[str]],
    max_hops: int = 2,
    current_tick: int = 0,
) -> Dict[str, float]:
    boosted = dict(act)
    for c, val in list(act.items()):
        nd = nodes.get(c, {})
        if nd.get("hub"):
            boosted[c] = max(boosted.get(c, 0), val * 1.2)
        for p in nd.get("parents", []):
            boosted[p] = max(boosted.get(p, 0), val * 0.8)

    current = boosted
    for _ in range(max_hops):
        nxt = dict(current)
        for src, src_act in current.items():
            if src_act < 0.01:
                continue
            for child in children_map.get(src, []):
                nxt[child] = nxt.get(child, 0) + src_act * 0.5 * 0.7
            for nb, w_raw, last_tick in adj.get(src, []):
                ew = effective_weight(w_raw, last_tick, current_tick)
                nxt[nb] = nxt.get(nb, 0) + src_act * ew * 0.5
        current = nxt
    return current


def engram_retrieve(
    question: str,
    graph: dict,
    lines: List[dict],
    max_lines: int = MAX_ENGRAM_LINES,
) -> Tuple[str, int]:
    """Concept graph → pointers → raw conversation lines."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        return "", 0

    qtoks = tokenize(question)
    if not qtoks:
        return "", 0

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    current_tick = graph.get("meta", {}).get("tick", 0)

    act = seed_activation(qtoks, nodes)
    if not act:
        return "", 0

    act = spread_activation(act, nodes, adj, children_map, max_hops=2, current_tick=current_tick)
    ranked = sorted(act.items(), key=lambda x: x[1], reverse=True)

    # Collect line indices from top concepts' pointers, weighted by activation
    line_scores: Dict[int, float] = {}
    for concept, score in ranked[:20]:
        for ptr in nodes.get(concept, {}).get("pointers", []):
            line_scores[ptr] = max(line_scores.get(ptr, 0), score)

    if not line_scores:
        return "", 0

    ordered = sorted(line_scores.items(), key=lambda x: (-x[1], x[0]))
    selected_idx = [idx for idx, _ in ordered[:max_lines]]
    selected_idx = sorted(set(selected_idx))

    ctx_lines = [lines[i]["line_str"] for i in selected_idx if i < len(lines)]
    context = "\n".join(ctx_lines)
    return context, len(selected_idx)


def select_qa_items(qa_list: List[dict], target: int = TARGET_QA) -> List[dict]:
    """Skip category 5; sample across 1-4."""
    filtered = [q for q in qa_list if q.get("category", 0) != 5]
    if len(filtered) <= target:
        return filtered

    by_cat: Dict[int, List[dict]] = defaultdict(list)
    for q in filtered:
        by_cat[q.get("category", 0)].append(q)

    rng = random.Random(RANDOM_SEED)
    per_cat = max(1, target // max(len(by_cat), 1))
    selected: List[dict] = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        rng.shuffle(pool)
        selected.extend(pool[:per_cat])

    if len(selected) < target:
        remaining = [q for q in filtered if q not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: target - len(selected)])

    return selected[:target]


def gold_answer(q: dict) -> str:
    ans = q.get("answer")
    if ans is None:
        ans = q.get("adversarial_answer")
    if ans is None:
        return ""
    return str(ans).strip()


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model: Optional[str] = None

    def _request(self, payload: dict, timeout: int = 120) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OPENROUTER_CHAT,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/engram-benchmark",
                "X-Title": "Engram LoCoMo Benchmark",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
                if e.code >= 500 or e.code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise last_err from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(str(last_err))

    def resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            self._request(
                {
                    "model": PREFERRED_MODEL,
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "temperature": 0,
                    "max_tokens": 5,
                }
            )
            self.model = PREFERRED_MODEL
            print(f"Using model: {self.model}")
            return self.model
        except Exception as e:
            print(f"{PREFERRED_MODEL} failed ({e}); fetching model list…")

        req = urllib.request.Request(
            OPENROUTER_MODELS,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        models = data.get("data") or data.get("models") or []
        ids = [m.get("id", "") for m in models if isinstance(m, dict)]

        def price_key(mid: str) -> float:
            for m in models:
                if m.get("id") == mid:
                    p = m.get("pricing") or {}
                    try:
                        return float(p.get("prompt", "999")) + float(p.get("completion", "999"))
                    except (TypeError, ValueError):
                        return 999.0
            return 999.0

        chosen = None
        for pat in FALLBACK_MODEL_PATTERNS:
            for mid in sorted(ids, key=price_key):
                if re.search(pat, mid, re.I):
                    chosen = mid
                    break
            if chosen:
                break
        if not chosen and ids:
            chosen = sorted(ids, key=price_key)[0]
        if not chosen:
            raise RuntimeError("No OpenRouter models available")
        self.model = chosen
        print(f"Using fallback model: {self.model}")
        return self.model

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0,
        max_tokens: int = 256,
        timeout: int = 120,
    ) -> str:
        model = self.resolve_model()
        result = self._request(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in response: {result}")
        return (choices[0].get("message") or {}).get("content", "").strip()

    def judge(self, question: str, gold: str, predicted: str) -> bool:
        if not predicted:
            return False
        prompt = (
            "You are a strict QA evaluator for the LoCoMo benchmark.\n"
            "Given a question, gold answer, and predicted answer, decide if the prediction "
            "is semantically correct (contains the gold fact or equivalent meaning).\n"
            "Reply with exactly one word: yes or no.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Predicted answer: {predicted}\n"
        )
        reply = self.chat(
            [{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8,
        ).lower().strip()
        if reply.startswith("yes"):
            return True
        if reply.startswith("no"):
            return False
        return "yes" in reply and "no" not in reply


def answer_question(client: OpenRouterClient, context: str, question: str) -> str:
    system = (
        "You answer questions about a conversation between two people. "
        "Use ONLY the provided conversation context. Be concise — one short sentence or phrase."
    )
    user = f"Conversation:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=128,
    )


def main():
    count_tokens = ensure_tiktoken()
    api_key = load_env_key()
    client = OpenRouterClient(api_key)

    download_locomo()
    with open(LOCOMO_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    sample = dataset[0]
    conv = sample["conversation"]
    conv_id = sample.get("sample_id") or sample.get("id") or "sample_0"
    lines = flatten_conversation(conv)
    full_context = "\n".join(ln["line_str"] for ln in lines)
    full_ctx_tokens = count_tokens(full_context)

    print(f"Conversation {conv_id}: {len(lines)} turns, ~{full_ctx_tokens:,} context tokens")

    graph = build_graph(lines)
    print(
        f"Engram graph: {len(graph['nodes'])} concepts, {len(graph['edges'])} edges"
    )

    qa_items = select_qa_items(sample.get("qa") or [])
    print(f"Selected {len(qa_items)} QA items (skipped category 5)")

    cat_counts: Counter = Counter()
    for q in qa_items:
        cat_counts[q.get("category", 0)] += 1

    results = []
    full_correct = 0
    engram_correct = 0
    full_token_sum = 0
    engram_token_sum = 0
    n = len(qa_items)

    for i, q in enumerate(qa_items, 1):
        question = q["question"]
        gold = gold_answer(q)
        cat = q.get("category", 0)

        rec = {
            "question": question,
            "category": cat,
            "gold": gold,
            "full_pred": None,
            "full_correct": None,
            "engram_pred": None,
            "engram_correct": None,
            "engram_ctx_tokens": None,
            "full_ctx_tokens": full_ctx_tokens,
            "error": None,
        }

        try:
            full_pred = answer_question(client, full_context, question)
            rec["full_pred"] = full_pred
            fc = client.judge(question, gold, full_pred)
            rec["full_correct"] = fc
            if fc:
                full_correct += 1
            full_token_sum += full_ctx_tokens

            engram_ctx, n_lines = engram_retrieve(question, graph, lines)
            engram_ctx_tokens = count_tokens(engram_ctx) if engram_ctx else 0
            rec["engram_ctx_tokens"] = engram_ctx_tokens
            rec["engram_lines"] = n_lines

            if engram_ctx:
                engram_pred = answer_question(client, engram_ctx, question)
            else:
                engram_pred = ""
            rec["engram_pred"] = engram_pred
            ec = client.judge(question, gold, engram_pred)
            rec["engram_correct"] = ec
            if ec:
                engram_correct += 1
            engram_token_sum += engram_ctx_tokens

            print(
                f"[{i}/{n}] cat={cat} full={'Y' if fc else 'N'} "
                f"engram={'Y' if ec else 'N'} "
                f"(ctx {engram_ctx_tokens}/{full_ctx_tokens} tok)"
            )
        except Exception as e:
            rec["error"] = str(e)
            print(f"[{i}/{n}] cat={cat} ERROR: {e}")

        results.append(rec)
        time.sleep(0.3)

    valid_full = sum(1 for r in results if r["full_correct"] is not None)
    valid_engram = sum(1 for r in results if r["engram_correct"] is not None)

    full_acc = full_correct / max(valid_full, 1)
    engram_acc = engram_correct / max(valid_engram, 1)
    avg_full_tok = full_token_sum / max(valid_full, 1)
    avg_engram_tok = engram_token_sum / max(valid_engram, 1)
    token_ratio = avg_full_tok / max(avg_engram_tok, 1)

    model = client.model or PREFERRED_MODEL

    md = f"""# LoCoMo QA Benchmark — Engram vs Full Context

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Model (answer + judge):** `{model}`  
**Conversation:** `{conv_id}` ({len(lines)} turns)  
**Questions:** {n} (categories 1–4 only; category 5 adversarial skipped)

### Per-category counts

| Category | Count |
|----------|------:|
"""
    for c in sorted(cat_counts):
        md += f"| {c} | {cat_counts[c]} |\n"

    md += f"""
## Results

| Condition | Accuracy | Avg context tokens |
|-----------|----------|-------------------:|
| full_context | {full_acc:.1%} ({full_correct}/{valid_full}) | {avg_full_tok:,.0f} |
| engram | {engram_acc:.1%} ({engram_correct}/{valid_engram}) | {avg_engram_tok:,.0f} |

## Headline

**Engram uses ~{token_ratio:.1f}× fewer context tokens at {engram_acc / max(full_acc, 0.001):.0%} of full-context accuracy** ({engram_acc:.1%} vs {full_acc:.1%}).

## Honest read

"""
    if engram_acc >= full_acc * 0.85:
        md += (
            f"Engram retrieval stayed close to full-context on this ~{full_ctx_tokens:,}-token "
            f"conversation while sending only ~{avg_engram_tok:.0f} tokens per question "
            f"(~{token_ratio:.0f}× compression). That is the intended win: near-parity accuracy "
            "at dramatically lower context cost, with zero LLM tokens spent building the graph."
        )
    elif engram_acc >= full_acc * 0.6:
        md += (
            f"Engram underperformed full-context ({engram_acc:.1%} vs {full_acc:.1%}) but still "
            f"achieved meaningful token savings (~{token_ratio:.0f}×). Concept-graph retrieval "
            "likely missed fine-grained details that full context trivially includes on this "
            "small corpus (~9k tokens). The tradeoff may improve on longer histories where "
            "full-context is infeasible."
        )
    else:
        md += (
            f"Engram accuracy ({engram_acc:.1%}) was substantially lower than full-context "
            f"({full_acc:.1%}) despite ~{token_ratio:.0f}× token savings. On a single short "
            "conversation, statistical concept retrieval misses evidential turns that keyword-"
            "sparse questions need; full-context wins trivially when everything fits in window. "
            "Engram's value proposition is accuracy-per-token on corpora too large to stuff."
        )

    md += f"\n\nRaw per-question data: `{os.path.basename(RAW_JSON)}`\n"

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(md)

    raw_out = {
        "model": model,
        "conversation_id": conv_id,
        "n_turns": len(lines),
        "full_ctx_tokens": full_ctx_tokens,
        "n_questions": n,
        "category_counts": dict(cat_counts),
        "summary": {
            "full_accuracy": full_acc,
            "engram_accuracy": engram_acc,
            "avg_full_context_tokens": avg_full_tok,
            "avg_engram_context_tokens": avg_engram_tok,
            "token_ratio": token_ratio,
        },
        "results": results,
    }
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(raw_out, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Model: {model}")
    print(f"| Condition | Accuracy | Avg context tokens |")
    print(f"| full_context | {full_acc:.1%} | {avg_full_tok:,.0f} |")
    print(f"| engram | {engram_acc:.1%} | {avg_engram_tok:,.0f} |")
    print(f"Token ratio: {token_ratio:.1f}x")
    print(f"Wrote {RESULTS_MD} and {RAW_JSON}")


if __name__ == "__main__":
    main()
