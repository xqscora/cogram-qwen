"""Matched-token-budget retrieval bake-off: Engram vs BM25 / PPR / vector / baselines.

Same ~2000-token context budget, same LLM answerer + judge. Fair comparison of
retrieval mechanisms on LoCoMo conv-26. Does NOT modify benchmark_qa.py.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from graph_lib import (  # noqa: E402
    build_edge_index,
    edge_key,
    effective_weight,
    generality,
    hebbian_delta,
    tokenize,
)

# Reuse benchmark_qa infrastructure (the no-preset build_graph lives there).
from benchmark_qa import (  # noqa: E402
    LOCOMO_PATH,
    MAX_ENGRAM_LINES,
    RANDOM_SEED,
    OpenRouterClient,
    answer_question,
    build_adjacency,
    build_children_map,
    build_graph,
    download_locomo,
    ensure_tiktoken,
    flatten_conversation,
    gold_answer,
    load_env_key,
    seed_activation,
    select_qa_items,
    spread_activation,
)

RESULTS_MD = os.path.join(_SCRIPT_DIR, "benchmark_bakeoff_results.md")
RAW_JSON = os.path.join(_SCRIPT_DIR, "benchmark_bakeoff_raw.json")
EMBED_CACHE = os.path.join(_SCRIPT_DIR, "benchmark_bakeoff_embeddings.json")

OPENROUTER_EMBEDDINGS = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
PREFERRED_EMBED_MODEL = "openai/text-embedding-3-small"

TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "2000"))
BM25_K1 = 1.5
BM25_B = 0.75
PPR_ALPHA = 0.85
PPR_ITERATIONS = 30

# Engram substrate + BM25 blend (associative prior is a real contributor).
ENGRAM_TOP_CONCEPTS = 60
ENGRAM_BM25_LAMBDA = 0.5
ENGRAM_EXPAND_TOP = 8
ENGRAM_EXPAND_WEIGHT = 0.3

# Engram uses benchmark_qa.build_graph directly (no-preset: novelty+surprise
# promotion, tick-based decay). No separate bake-off graph builder needed.


def select_lines_under_budget(
    ranked_indices: List[int],
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int, List[int]]:
    """Greedy pack ranked lines until budget; skip lines that do not fit."""
    selected: List[int] = []
    total = 0
    for idx in ranked_indices:
        if idx < 0 or idx >= len(lines):
            continue
        line_str = lines[idx]["line_str"]
        line_tok = count_tokens(line_str)
        sep = 1 if selected else 0
        if total + sep + line_tok > budget:
            continue
        selected.append(idx)
        total += sep + line_tok
    selected.sort()
    ctx = "\n".join(lines[i]["line_str"] for i in selected)
    return ctx, total, selected


def engram_spread_activation(question: str, graph: dict) -> Dict[str, float]:
    """Seed + 2-hop spread on concept graph; includes tick-decay via effective_weight."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        return {}

    qtoks = tokenize(question)
    if not qtoks:
        return {}

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    current_tick = graph.get("meta", {}).get("tick", 0)

    act = seed_activation(qtoks, nodes)
    if not act:
        return {}

    return spread_activation(
        act, nodes, adj, children_map, max_hops=2, current_tick=current_tick
    )


def engram_activation_line_scores(
    activation: Dict[str, float], lines: List[dict]
) -> Dict[int, float]:
    """Score every line: Σ concept activation / (1 + log(1 + token_count))."""
    scores: Dict[int, float] = {}
    for i, ln in enumerate(lines):
        toks = tokenize(ln["line_str"])
        if not toks:
            continue
        act_sum = sum(activation.get(t, 0.0) for t in toks)
        denom = 1.0 + math.log(1.0 + len(toks))
        scores[i] = act_sum / denom
    return scores


def retrieve_engram(
    question: str,
    graph: dict,
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    """Pure graph-signal retrieval: rank all lines by spread activation density."""
    act = engram_spread_activation(question, graph)
    if not act:
        return "", 0

    scores = engram_activation_line_scores(act, lines)
    if not scores:
        return "", 0

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if all(s == 0 for _, s in ordered):
        ranked = list(range(len(lines)))
        ranked.reverse()
    else:
        ranked = [i for i, s in ordered if s > 0]
        ranked += [i for i, s in ordered if s <= 0]
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def bm25_line_scores(
    query_tokens: List[str],
    lines: List[dict],
    extra_term_groups: Optional[List[Tuple[List[str], float]]] = None,
) -> Dict[int, float]:
    """BM25 score per line index — shared by retrieve_bm25, engram_bm25, engram_expand.

    Score = Σ weight_g * BM25(doc, terms_g). Default group is query_tokens @ weight 1.0.
    """
    docs = [tokenize(ln["line_str"]) for ln in lines]
    n_docs = len(docs)
    if n_docs == 0:
        return {}

    term_groups: List[Tuple[List[str], float]] = [(query_tokens, 1.0)]
    if extra_term_groups:
        term_groups.extend(extra_term_groups)

    df: Counter = Counter()
    for doc in docs:
        for t in set(doc):
            df[t] += 1

    avgdl = sum(len(d) for d in docs) / n_docs

    def idf(term: str) -> float:
        n = df.get(term, 0)
        return math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)

    scores: Dict[int, float] = {}
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for terms, weight in term_groups:
            if not terms or weight == 0:
                continue
            sub = 0.0
            for qt in terms:
                if qt not in tf:
                    continue
                f = tf[qt]
                denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl)
                sub += idf(qt) * (f * (BM25_K1 + 1.0)) / denom
            score += weight * sub
        scores[i] = score
    return scores


def retrieve_bm25(
    question: str,
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    scores = bm25_line_scores(tokenize(question), lines)
    if not scores:
        return "", 0

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if all(s == 0 for _, s in ordered):
        ranked = list(range(len(lines)))
        ranked.reverse()
    else:
        ranked = [i for i, s in ordered if s > 0]
        ranked += [i for i, s in ordered if s <= 0]
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def engram_substrate_candidates(
    question: str, graph: dict, top_concepts: int = ENGRAM_TOP_CONCEPTS
) -> Tuple[Set[int], Dict[int, float]]:
    """Candidate lines from Engram substrate + per-line activation priors."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    if not nodes:
        return set(), {}

    qtoks = tokenize(question)
    if not qtoks:
        return set(), {}

    edge_idx = build_edge_index(edges)
    adj = build_adjacency(edge_idx)
    children_map = build_children_map(nodes)
    current_tick = graph.get("meta", {}).get("tick", 0)

    act = seed_activation(qtoks, nodes)
    if not act:
        return set(), {}

    act = spread_activation(
        act, nodes, adj, children_map, max_hops=2, current_tick=current_tick
    )
    ranked = sorted(act.items(), key=lambda x: x[1], reverse=True)

    line_priors: Dict[int, float] = {}
    for concept, score in ranked[:top_concepts]:
        for ptr in nodes.get(concept, {}).get("pointers", []):
            line_priors[ptr] = max(line_priors.get(ptr, 0), score)

    return set(line_priors.keys()), line_priors


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """Map scores to [0, 1]; all-zero → all zeros."""
    if not scores:
        return {}
    mx = max(scores.values())
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def retrieve_engram_bm25(
    question: str,
    graph: dict,
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    candidate_idx, engram_prior = engram_substrate_candidates(question, graph)
    if not candidate_idx:
        candidate_idx = set(range(len(lines)))
        engram_prior = {i: 0.0 for i in candidate_idx}

    bm25_scores = bm25_line_scores(tokenize(question), lines)
    bm25_cand = {idx: bm25_scores.get(idx, 0.0) for idx in candidate_idx}
    prior_cand = {idx: engram_prior.get(idx, 0.0) for idx in candidate_idx}

    bm25_norm = _normalize_scores(bm25_cand)
    prior_norm = _normalize_scores(prior_cand)

    combined: Dict[int, float] = {}
    for idx in candidate_idx:
        combined[idx] = bm25_norm.get(idx, 0.0) + ENGRAM_BM25_LAMBDA * prior_norm.get(
            idx, 0.0
        )

    ranked = sorted(candidate_idx, key=lambda idx: (-combined[idx], idx))
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def retrieve_engram_expand(
    question: str,
    graph: dict,
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    """BM25 with associative query expansion from 2-hop graph activation."""
    qtoks = tokenize(question)
    qtok_set = set(qtoks)
    act = engram_spread_activation(question, graph)

    expansion: List[str] = []
    for concept, _score in sorted(act.items(), key=lambda x: (-x[1], x[0])):
        if concept in qtok_set:
            continue
        expansion.append(concept)
        if len(expansion) >= ENGRAM_EXPAND_TOP:
            break

    extra = [(expansion, ENGRAM_EXPAND_WEIGHT)] if expansion else None
    scores = bm25_line_scores(qtoks, lines, extra_term_groups=extra)
    if not scores:
        return "", 0

    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if all(s == 0 for _, s in ordered):
        ranked = list(range(len(lines)))
        ranked.reverse()
    else:
        ranked = [i for i, s in ordered if s > 0]
        ranked += [i for i, s in ordered if s <= 0]
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def build_line_term_sets(lines: List[dict]) -> List[Set[str]]:
    return [set(tokenize(ln["line_str"])) for ln in lines]


def build_cooc_graph(term_sets: List[Set[str]], window: int = 3) -> Dict[str, Set[str]]:
    """Undirected co-occurrence adjacency from sliding line windows."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    n = len(term_sets)
    for start in range(n):
        window_terms: Set[str] = set()
        for j in range(start, min(start + window, n)):
            window_terms |= term_sets[j]
        present = sorted(window_terms)
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                adj[a].add(b)
                adj[b].add(a)
    return dict(adj)


def personalized_pagerank(
    adj: Dict[str, Set[str]],
    restart: Dict[str, float],
    alpha: float = PPR_ALPHA,
    iterations: int = PPR_ITERATIONS,
) -> Dict[str, float]:
    nodes = set(adj.keys()) | set(restart.keys())
    if not nodes:
        return {}

    rsum = sum(restart.values())
    if rsum <= 0:
        uniform = 1.0 / len(nodes)
        restart = {n: uniform for n in nodes}
    else:
        restart = {k: v / rsum for k, v in restart.items() if v > 0}
        for n in nodes:
            restart.setdefault(n, 0.0)

    rank = dict(restart)
    for _ in range(iterations):
        nxt: Dict[str, float] = {}
        for n in nodes:
            nxt[n] = (1.0 - alpha) * restart.get(n, 0.0)

        for src, nbrs in adj.items():
            if not nbrs:
                continue
            share = alpha * rank.get(src, 0.0) / len(nbrs)
            if share == 0:
                continue
            for nb in nbrs:
                nxt[nb] = nxt.get(nb, 0.0) + share

        rank = nxt
    return rank


def retrieve_cooc_ppr(
    question: str,
    lines: List[dict],
    term_sets: List[Set[str]],
    cooc_adj: Dict[str, Set[str]],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    qtoks = tokenize(question)
    restart: Dict[str, float] = Counter()
    for qt in qtoks:
        restart[qt] += 1.0
        for term in cooc_adj:
            if qt in term or term in qt:
                restart[term] += 0.5

    ppr = personalized_pagerank(cooc_adj, dict(restart))

    line_scores: List[Tuple[int, float]] = []
    for i, terms in enumerate(term_sets):
        score = sum(ppr.get(t, 0.0) for t in terms)
        line_scores.append((i, score))

    line_scores.sort(key=lambda x: (-x[1], x[0]))
    ranked = [i for i, _ in line_scores]
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingClient:
    """OpenRouter embeddings with local JSON cache."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model: Optional[str] = None
        self.available = True
        self.skip_reason: Optional[str] = None

    def _post(self, payload: dict, timeout: int = 120) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            OPENROUTER_EMBEDDINGS,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/engram-benchmark",
                "X-Title": "Engram Bakeoff Benchmark",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {err_body[:400]}")
                if e.code >= 500 or e.code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise last_err from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(str(last_err))

    def resolve_model(self) -> Optional[str]:
        if self.model:
            return self.model
        try:
            self._post({"model": PREFERRED_EMBED_MODEL, "input": "test"})
            self.model = PREFERRED_EMBED_MODEL
            return self.model
        except Exception as e:
            print(f"Embedding model {PREFERRED_EMBED_MODEL} failed: {e}")

        try:
            req = urllib.request.Request(
                OPENROUTER_MODELS,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data") or data.get("models") or []
            for m in models:
                mid = m.get("id", "")
                arch = m.get("architecture") or {}
                if arch.get("modality") == "embeddings" or "embed" in mid.lower():
                    self.model = mid
                    return self.model
                pricing = m.get("pricing") or {}
                if "embedding" in str(pricing).lower():
                    self.model = mid
                    return self.model
        except Exception as e:
            self.available = False
            self.skip_reason = f"Could not list embedding models: {e}"
            return None

        self.available = False
        self.skip_reason = "No embedding-capable model found via OpenRouter gateway"
        return None

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        model = self.resolve_model()
        if not model:
            raise RuntimeError(self.skip_reason or "Embeddings unavailable")
        result = self._post({"model": model, "input": texts})
        items = result.get("data") or []
        items.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]


def load_or_build_line_embeddings(
    embed_client: EmbeddingClient,
    lines: List[dict],
    conv_id: str,
) -> Optional[List[List[float]]]:
    cache_key = f"{conv_id}|{len(lines)}|{PREFERRED_EMBED_MODEL}"
    if os.path.isfile(EMBED_CACHE):
        with open(EMBED_CACHE, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("key") == cache_key and len(cache.get("embeddings", [])) == len(lines):
            print(f"Loaded {len(lines)} line embeddings from cache")
            return cache["embeddings"]

    model = embed_client.resolve_model()
    if not model:
        return None

    print(f"Embedding {len(lines)} lines with {model}…")
    embeddings: List[List[float]] = []
    batch_size = 32
    for start in range(0, len(lines), batch_size):
        batch = [lines[i]["line_str"] for i in range(start, min(start + batch_size, len(lines)))]
        vecs = embed_client.embed_batch(batch)
        embeddings.extend(vecs)
        time.sleep(0.2)

    with open(EMBED_CACHE, "w", encoding="utf-8") as f:
        json.dump(
            {"key": cache_key, "model": model, "embeddings": embeddings},
            f,
        )
    print(f"Cached embeddings → {EMBED_CACHE}")
    return embeddings


def retrieve_vector(
    question: str,
    lines: List[dict],
    line_embeddings: List[List[float]],
    embed_client: EmbeddingClient,
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    q_vec = embed_client.embed_batch([question])[0]
    scores = [(i, cosine_sim(q_vec, line_embeddings[i])) for i in range(len(lines))]
    scores.sort(key=lambda x: (-x[1], x[0]))
    ranked = [i for i, _ in scores]
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def retrieve_recency(
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
) -> Tuple[str, int]:
    ranked = list(range(len(lines) - 1, -1, -1))
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


def retrieve_random(
    lines: List[dict],
    count_tokens: Callable[[str], int],
    budget: int = TOKEN_BUDGET,
    seed: int = RANDOM_SEED,
) -> Tuple[str, int]:
    rng = random.Random(seed)
    ranked = list(range(len(lines)))
    rng.shuffle(ranked)
    ctx, tok, _ = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return ctx, tok


# Trimmed for hackathon time pressure: vector/embedding baseline dropped
# (slowest — one embedding call per line, and the only non-zero build cost).
RETRIEVERS = [
    ("engram", "Engram concept graph (count>=1 promotion)"),
    ("engram_bm25", "Engram substrate + BM25 ranker (pluggable)"),
    ("bm25", "Lexical / agent-knowledge BM25"),
    ("cooc_ppr", "Co-occurrence graph + PPR (SPRIG / MemGraph)"),
    ("recency", "Recency baseline"),
    ("random", "Random baseline"),
]

REPRESENTS = {
    "engram": "Engram (zero-LLM concept graph)",
    "engram_bm25": "Engram substrate + pluggable BM25 ranker",
    "bm25": "Lexical / agent-knowledge",
    "cooc_ppr": "SPRIG / MemGraph-style PPR",
    "recency": "Naive recency",
    "random": "Random floor",
}

BUILD_COST = {
    "engram": "0 (statistics)",
    "engram_bm25": "0 (statistics)",
    "bm25": "0 (statistics)",
    "cooc_ppr": "0 (statistics)",
    "recency": "0 (statistics)",
    "random": "0 (statistics)",
}

MAX_QUESTIONS = 15


def run_retriever(
    name: str,
    question: str,
    lines: List[dict],
    count_tokens: Callable[[str], int],
    graph: dict,
    term_sets: List[Set[str]],
    cooc_adj: Dict[str, Set[str]],
    line_embeddings: Optional[List[List[float]]],
    embed_client: Optional[EmbeddingClient],
) -> Tuple[str, int]:
    if name == "engram":
        return retrieve_engram(question, graph, lines, count_tokens)
    if name == "engram_bm25":
        return retrieve_engram_bm25(question, graph, lines, count_tokens)
    if name == "bm25":
        return retrieve_bm25(question, lines, count_tokens)
    if name == "cooc_ppr":
        return retrieve_cooc_ppr(question, lines, term_sets, cooc_adj, count_tokens)
    if name == "vector":
        if line_embeddings is None or embed_client is None:
            return "", 0
        return retrieve_vector(
            question, lines, line_embeddings, embed_client, count_tokens
        )
    if name == "recency":
        return retrieve_recency(lines, count_tokens)
    if name == "random":
        return retrieve_random(lines, count_tokens)
    raise ValueError(f"Unknown retriever: {name}")


def main() -> None:
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

    print(
        f"Conversation {conv_id}: {len(lines)} turns | "
        f"shared budget ~{TOKEN_BUDGET} tokens | "
        f"Engram promotion = novelty+surprise (no frequency gate)"
    )

    graph = build_graph(lines)
    meta_tick = graph.get("meta", {}).get("tick", 0)
    print(
        f"Engram graph (no-preset): {len(graph['nodes'])} concepts, "
        f"{len(graph['edges'])} edges, tick={meta_tick}"
    )

    term_sets = build_line_term_sets(lines)
    cooc_adj = build_cooc_graph(term_sets)

    # Vector baseline dropped for time (see RETRIEVERS note above).
    line_embeddings: Optional[List[List[float]]] = None
    embed_client: Optional[EmbeddingClient] = None
    active_retrievers = [r[0] for r in RETRIEVERS]

    qa_items = select_qa_items(sample.get("qa") or [])[:MAX_QUESTIONS]
    print(f"Selected {len(qa_items)} QA items (trimmed to first {MAX_QUESTIONS})")

    cat_counts: Counter = Counter(q.get("category", 0) for q in qa_items)

    model = client.resolve_model()

    per_retriever: Dict[str, dict] = {
        name: {"correct": 0, "valid": 0, "token_sum": 0} for name in active_retrievers
    }
    raw_results: List[dict] = []

    n = len(qa_items)
    for i, q in enumerate(qa_items, 1):
        question = q["question"]
        gold = gold_answer(q)
        cat = q.get("category", 0)

        rec: dict = {
            "question": question,
            "category": cat,
            "gold": gold,
            "retrievers": {},
        }

        status_parts: List[str] = []
        for rname in active_retrievers:
            rrec: dict = {
                "pred": None,
                "correct": None,
                "ctx_tokens": 0,
                "error": None,
            }
            try:
                ctx, ctx_tok = run_retriever(
                    rname,
                    question,
                    lines,
                    count_tokens,
                    graph,
                    term_sets,
                    cooc_adj,
                    line_embeddings,
                    embed_client if line_embeddings else None,
                )
                rrec["ctx_tokens"] = ctx_tok
                if ctx:
                    pred = answer_question(client, ctx, question)
                else:
                    pred = ""
                rrec["pred"] = pred
                ok = client.judge(question, gold, pred)
                rrec["correct"] = ok
                per_retriever[rname]["valid"] += 1
                per_retriever[rname]["token_sum"] += ctx_tok
                if ok:
                    per_retriever[rname]["correct"] += 1
                status_parts.append(f"{rname}={'Y' if ok else 'N'}")
            except Exception as e:
                rrec["error"] = str(e)
                status_parts.append(f"{rname}=ERR")
            rec["retrievers"][rname] = rrec

        print(f"[{i}/{n}] cat={cat} " + " ".join(status_parts))
        raw_results.append(rec)
        time.sleep(0.25)

    summary_rows: List[dict] = []
    for rname in active_retrievers:
        st = per_retriever[rname]
        acc = st["correct"] / max(st["valid"], 1)
        avg_tok = st["token_sum"] / max(st["valid"], 1)
        summary_rows.append(
            {
                "retriever": rname,
                "represents": REPRESENTS[rname],
                "accuracy": acc,
                "correct": st["correct"],
                "valid": st["valid"],
                "avg_ctx_tokens": avg_tok,
                "build_cost": BUILD_COST[rname],
            }
        )

    summary_rows.sort(key=lambda x: (-x["accuracy"], -x["correct"]))

    zero_llm = [r for r in summary_rows if r["retriever"] != "vector"]
    engram_row = next((r for r in summary_rows if r["retriever"] == "engram"), None)
    engram_rank = 1 + sum(
        1 for r in zero_llm if r["accuracy"] > (engram_row or {}).get("accuracy", -1)
    )

    old_engram_acc = 0.167  # benchmark_qa_results.md count>=2, ~1961 tok avg

    md = f"""# LoCoMo Retrieval Bake-off — Matched ~{TOKEN_BUDGET}-Token Budget

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Model (answer + judge):** `{model}`  
**Conversation:** `{conv_id}` ({len(lines)} turns)  
**Questions:** {len(qa_items)} (categories 1–4; category 5 skipped)  
**Shared context budget:** ~{TOKEN_BUDGET} tokens (tiktoken cl100k_base)  
**Engram promotion:** novelty + surprise-mass (every content token; cap by Σ incident edge weight). Replaces the old frequency gate (count>=2).  
**Engram decay:** experienced ticks (1 tick per LoCoMo session, tick={meta_tick}), not wall-clock days.

### Per-category counts

| Category | Count |
|----------|------:|
"""
    for c in sorted(cat_counts):
        md += f"| {c} | {cat_counts[c]} |\n"

    md += """
## Results (sorted by accuracy)

| Retriever | Represents | Accuracy | Avg ctx tokens | Build cost |
|-----------|------------|----------|---------------:|------------|
"""
    for row in summary_rows:
        md += (
            f"| {row['retriever']} | {row['represents']} | "
            f"{row['accuracy']:.1%} ({row['correct']}/{row['valid']}) | "
            f"{row['avg_ctx_tokens']:,.0f} | {row['build_cost']} |\n"
        )

    engram_acc = engram_row["accuracy"] if engram_row else 0.0
    best_zero = zero_llm[0] if zero_llm else None
    best_overall = summary_rows[0]

    md += "\n## Headline\n\n"
    if engram_row:
        md += (
            f"At equal ~{TOKEN_BUDGET}-token budget, **Engram ranks #{engram_rank} "
            f"among zero-LLM retrievers** ({engram_acc:.1%} accuracy). "
        )
        if best_zero and best_zero["retriever"] == "engram":
            md += "Engram is the **best zero-LLM method** in this bake-off."
        elif best_zero:
            md += (
                f"Best zero-LLM retriever: **{best_zero['retriever']}** "
                f"({best_zero['accuracy']:.1%})."
            )
        md += "\n"

    md += "\n## Honest read\n\n"

    parts: List[str] = []
    if engram_row:
        if engram_acc > old_engram_acc + 0.01:
            parts.append(
                f"The count>=1 promotion fix raised Engram from {old_engram_acc:.1%} "
                f"(old benchmark_qa, count>=2) to {engram_acc:.1%} at matched budget."
            )
        elif engram_acc < old_engram_acc - 0.01:
            parts.append(
                f"Engram at matched budget ({engram_acc:.1%}) differs from the old "
                f"benchmark_qa run ({old_engram_acc:.1%}); budget capping and bake-off "
                "retriever parity may explain the gap."
            )
        else:
            parts.append(
                f"Engram accuracy ({engram_acc:.1%}) is similar to the old count>=2 run "
                f"({old_engram_acc:.1%}) despite count>=1 promotion."
            )

    if best_overall["retriever"] == "engram":
        parts.append(
            "Engram wins overall at this token budget among all retrievers tested."
        )
    elif best_zero and best_zero["retriever"] != "engram":
        bm25_or = next((r for r in summary_rows if r["retriever"] == "bm25"), None)
        if bm25_or and bm25_or["accuracy"] > engram_acc:
            parts.append(
                f"BM25 ({bm25_or['accuracy']:.1%}) beats Engram ({engram_acc:.1%}) "
                "at equal budget — lexical matching is strong on LoCoMo factoid QA."
            )
        if engram_rank > 1:
            parts.append(
                f"Engram is not the top zero-LLM retriever here; #{engram_rank} of "
                f"{len(zero_llm)} at {engram_acc:.1%}."
            )

    parts.append(
        "Vector/embedding baseline omitted for time; it is also the only baseline "
        "with a non-zero build cost."
    )

    md += " ".join(parts) if parts else "See table above."
    md += f"\n\nRaw per-question data: `{os.path.basename(RAW_JSON)}`\n"

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(md)

    raw_out = {
        "model": model,
        "conversation_id": conv_id,
        "n_turns": len(lines),
        "token_budget": TOKEN_BUDGET,
        "promotion": "novelty+surprise-mass (no frequency gate)",
        "engram_tick": meta_tick,
        "n_questions": len(qa_items),
        "category_counts": dict(cat_counts),
        "vector_skipped": True,
        "vector_skip_reason": "omitted for time (also only non-zero build-cost baseline)",
        "summary": summary_rows,
        "engram_rank_zero_llm": engram_rank,
        "old_engram_accuracy_count_ge_2": old_engram_acc,
        "results": raw_results,
    }
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(raw_out, f, ensure_ascii=False, indent=2)

    print("\n=== BAKE-OFF SUMMARY ===")
    print(f"| Retriever | Accuracy | Avg ctx tokens |")
    for row in summary_rows:
        print(
            f"| {row['retriever']} | {row['accuracy']:.1%} | "
            f"{row['avg_ctx_tokens']:,.0f} |"
        )
    if engram_row:
        print(f"Engram rank (zero-LLM): #{engram_rank}/{len(zero_llm)}")
    print(f"Wrote {RESULTS_MD} and {RAW_JSON}")


if __name__ == "__main__":
    main()
