"""LongMemEval_S benchmark: Engram vs BM25 vs full-context on long chat histories.

Each question carries its own ~115k-token haystack (30–40 sessions).
Reuses bake-off infrastructure (OpenRouter, BM25, budget packing, judge).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from benchmark_bakeoff import (  # noqa: E402
    ENGRAM_BM25_LAMBDA,
    ENGRAM_EXPAND_TOP,
    ENGRAM_EXPAND_WEIGHT,
    TOKEN_BUDGET,
    bm25_line_scores,
    retrieve_bm25,
    retrieve_engram,
    retrieve_engram_bm25,
    retrieve_engram_expand,
    select_lines_under_budget,
)
from benchmark_qa import (  # noqa: E402
    OpenRouterClient,
    build_graph,
    ensure_tiktoken,
    load_env_key,
)

DATA_PATH = os.path.join(_SCRIPT_DIR, "data", "longmemeval_s_cleaned.json")
RESULTS_MD = os.path.join(_SCRIPT_DIR, "benchmark_longmem_results.md")
_DEFAULT_RAW_JSON = os.path.join(_SCRIPT_DIR, "benchmark_longmem_raw.json")
_LEGACY_TOKEN_BUDGET = 2000


def _resolve_raw_json_path() -> str:
    """Separate raw cache per token budget so sweeps do not clobber 2000-budget rows."""
    override = os.environ.get("LONGMEM_RAW_JSON", "").strip()
    if override:
        return override
    budget_env = os.environ.get("TOKEN_BUDGET", str(_LEGACY_TOKEN_BUDGET)).strip()
    try:
        budget = int(budget_env)
    except ValueError:
        budget = _LEGACY_TOKEN_BUDGET
    if budget == _LEGACY_TOKEN_BUDGET:
        return _DEFAULT_RAW_JSON
    return os.path.join(_SCRIPT_DIR, f"benchmark_longmem_raw_b{budget}.json")


RAW_JSON = _resolve_raw_json_path()

# gpt-4o-mini approximate OpenRouter pricing (USD per 1M tokens)
PRICE_INPUT_PER_M = 0.15
PRICE_OUTPUT_PER_M = 0.60

TARGET_TYPES = [
    "single-session-user",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
    "single-session-preference",
    "single-session-assistant",
]
PER_TYPE = 8
SAMPLING_SEED = 42
# OpenRouter credit-based per-request prompt limit (~34k OR tokens on this key at
# healthy balance). 110k ctx ≈ 80k OR prompt → HTTP 402. Probe 2026-07-19:
# 30k cap OK, 35k fails; tiktoken→OR ratio ~1.1× on these payloads.
_DEFAULT_FULL_CTX_TRUNCATE = 30_000
FULL_CTX_TRUNCATE = int(os.environ.get("FULL_CTX_TRUNCATE", str(_DEFAULT_FULL_CTX_TRUNCATE)))
PROMPT_OVERHEAD_TOKENS = 800

# Round 3: pitch-relevant retrievers only (engram / engram_expand dropped — diagnostic done at n=24).
_DEFAULT_RETRIEVERS = ["full_context", "bm25", "engram_bm25"]
_env_retrievers = os.environ.get("LONGMEM_RETRIEVERS", "").strip()
if _env_retrievers:
    RETRIEVERS = [r.strip() for r in _env_retrievers.split(",") if r.strip()]
else:
    RETRIEVERS = list(_DEFAULT_RETRIEVERS)

RETRIEVER_IMPL_VERSION: Dict[str, int] = {}
BENCHMARK_ROUND = 3


def flatten_haystack(
    haystack_sessions: List[list],
    haystack_dates: List[str],
) -> List[dict]:
    """Flatten sessions to dated lines for graph build + retrieval."""
    lines: List[dict] = []
    for session_idx, session in enumerate(haystack_sessions):
        date = ""
        if session_idx < len(haystack_dates):
            date = str(haystack_dates[session_idx] or "").strip()
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "?")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            prefix = f"[{date} S{session_idx}:{turn_idx}]" if date else f"[S{session_idx}:{turn_idx}]"
            line_str = f"{prefix} {role}: {content}"
            lines.append(
                {
                    "session": session_idx,
                    "session_date": date,
                    "turn_idx": turn_idx,
                    "role": role,
                    "content": content,
                    "line_str": line_str,
                    "idx": len(lines),
                }
            )
    return lines


def select_questions(dataset: List[dict]) -> Tuple[List[dict], Dict[str, int], List[str]]:
    """Stratified random sample: PER_TYPE per question_type; skip _abs types."""
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for item in dataset:
        qtype = item.get("question_type") or "unknown"
        if "_abs" in qtype:
            continue
        by_type[qtype].append(item)

    selected: List[dict] = []
    counts: Dict[str, int] = {}
    rng = random.Random(SAMPLING_SEED)
    for qtype in TARGET_TYPES:
        pool = list(by_type.get(qtype, []))
        if len(pool) < PER_TYPE:
            raise ValueError(
                f"Not enough questions for {qtype}: need {PER_TYPE}, have {len(pool)}"
            )
        pick = rng.sample(pool, PER_TYPE)
        counts[qtype] = len(pick)
        selected.extend(pick)

    round2_ids = _round2_question_ids(dataset)
    overlap = [q.get("question_id") for q in selected if q.get("question_id") in round2_ids]
    return selected, counts, overlap


def _round2_question_ids(dataset: List[dict]) -> set:
    """First 4 per type (Round 2 selection) for overlap reporting."""
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for item in dataset:
        qtype = item.get("question_type") or "unknown"
        if "_abs" in qtype:
            continue
        by_type[qtype].append(item)
    ids: set = set()
    for qtype in TARGET_TYPES:
        for item in by_type.get(qtype, [])[:4]:
            qid = item.get("question_id")
            if qid:
                ids.add(qid)
    return ids


def gold_answer(item: dict) -> str:
    ans = item.get("answer")
    return str(ans).strip() if ans is not None else ""


def build_full_context(
    lines: List[dict],
    count_tokens: Callable[[str], int],
    max_tokens: int = FULL_CTX_TRUNCATE,
) -> Tuple[str, int, bool]:
    """Return full haystack context; truncate middle if over max_tokens."""
    ctx = "\n".join(ln["line_str"] for ln in lines)
    tok = count_tokens(ctx)
    if tok <= max_tokens:
        return ctx, tok, False

    # Binary search how many lines fit in head+tail halves
    n = len(lines)
    head_target = max_tokens // 2
    tail_target = max_tokens - head_target

    head_count = 0
    head_tok = 0
    for i in range(n):
        lt = count_tokens(lines[i]["line_str"]) + (1 if i else 0)
        if head_tok + lt > head_target:
            break
        head_tok += lt
        head_count += 1

    tail_count = 0
    tail_tok = 0
    for j in range(n - 1, head_count - 1, -1):
        lt = count_tokens(lines[j]["line_str"]) + (1 if tail_count else 0)
        if tail_tok + lt > tail_target:
            break
        tail_tok += lt
        tail_count += 1

    head_lines = lines[:head_count]
    tail_lines = lines[n - tail_count :] if tail_count else []
    omitted = n - head_count - tail_count
    marker = f"\n[... {omitted} lines omitted for token limit ...]\n"
    ctx = "\n".join(ln["line_str"] for ln in head_lines)
    if omitted:
        ctx += marker
    if tail_lines:
        ctx += "\n" + "\n".join(ln["line_str"] for ln in tail_lines)
    tok = count_tokens(ctx)
    return ctx, tok, True


def answer_question_longmem(
    client: OpenRouterClient,
    context: str,
    question: str,
    question_date: str,
    timeout: int = 120,
) -> str:
    system = (
        "You answer questions about a user's long chat history with an assistant. "
        "Use ONLY the provided conversation context. Session dates are prefixed on each line. "
        "The question may refer to a specific date — use temporal cues carefully. "
        "Be concise — one short sentence or phrase."
    )
    qdate_line = f"Question date: {question_date}\n\n" if question_date else ""
    user = (
        f"{qdate_line}Conversation history:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=int(os.environ.get("LONGMEM_ANSWER_MAX_TOKENS", "64")),
        timeout=timeout,
    )


def estimate_cost(input_tokens: int, output_tokens: int = 150) -> float:
    """Rough USD cost for answer + judge (2 calls)."""
    per_q_in = input_tokens * 2  # answer + judge
    per_q_out = output_tokens * 2
    return (per_q_in / 1_000_000) * PRICE_INPUT_PER_M + (per_q_out / 1_000_000) * PRICE_OUTPUT_PER_M


def run_retriever(
    name: str,
    question: str,
    lines: List[dict],
    graph: dict,
    count_tokens: Callable[[str], int],
    full_ctx_cache: Tuple[str, int, bool],
) -> Tuple[str, int]:
    if name == "full_context":
        ctx, tok, _ = full_ctx_cache
        return ctx, tok
    if name == "bm25":
        return retrieve_bm25(question, lines, count_tokens, TOKEN_BUDGET)
    if name == "engram":
        return retrieve_engram(question, graph, lines, count_tokens, TOKEN_BUDGET)
    if name == "engram_bm25":
        return retrieve_engram_bm25(question, graph, lines, count_tokens, TOKEN_BUDGET)
    if name == "engram_expand":
        return retrieve_engram_expand(question, graph, lines, count_tokens, TOKEN_BUDGET)
    raise ValueError(f"Unknown retriever: {name}")


def retriever_cache_valid(name: str, r: dict, budget: int = TOKEN_BUDGET) -> bool:
    """True if cached result can be reused for this retriever."""
    if r.get("error") or r.get("correct") is None:
        return False
    ver = RETRIEVER_IMPL_VERSION.get(name)
    if ver is not None and r.get("impl_version", 0) != ver:
        return False
    cached_budget = r.get("token_budget")
    if cached_budget is not None:
        if cached_budget != budget:
            return False
    elif budget != _LEGACY_TOKEN_BUDGET:
        # Legacy rows (no token_budget field) are only valid at 2000-token budget.
        return False
    return True


def load_partial_raw() -> Optional[dict]:
    if not os.path.isfile(RAW_JSON):
        return None
    with open(RAW_JSON, encoding="utf-8") as f:
        return json.load(f)


def save_raw(payload: dict) -> None:
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_summaries(results: List[dict]) -> dict:
    overall: Dict[str, dict] = {
        name: {"correct": 0, "valid": 0, "token_sum": 0, "cost_sum": 0.0}
        for name in RETRIEVERS
    }
    by_type: Dict[str, Dict[str, dict]] = defaultdict(
        lambda: {name: {"correct": 0, "valid": 0} for name in RETRIEVERS}
    )

    for rec in results:
        qtype = rec.get("question_type", "unknown")
        for name in RETRIEVERS:
            r = rec.get("retrievers", {}).get(name, {})
            if r.get("correct") is None and not r.get("error"):
                continue
            if r.get("error"):
                continue
            overall[name]["valid"] += 1
            overall[name]["token_sum"] += r.get("ctx_tokens", 0)
            overall[name]["cost_sum"] += r.get("est_cost_usd", 0)
            if r.get("correct"):
                overall[name]["correct"] += 1
            by_type[qtype][name]["valid"] += 1
            if r.get("correct"):
                by_type[qtype][name]["correct"] += 1

    summary_rows = []
    for name in RETRIEVERS:
        st = overall[name]
        acc = st["correct"] / max(st["valid"], 1)
        avg_tok = st["token_sum"] / max(st["valid"], 1)
        avg_cost = st["cost_sum"] / max(st["valid"], 1)
        summary_rows.append(
            {
                "retriever": name,
                "accuracy": acc,
                "correct": st["correct"],
                "valid": st["valid"],
                "avg_ctx_tokens": avg_tok,
                "avg_cost_usd": avg_cost,
            }
        )

    summary_rows.sort(key=lambda x: (-x["accuracy"], -x["correct"]))

    cat_rows = []
    for qtype in TARGET_TYPES:
        if qtype not in by_type:
            continue
        row = {"question_type": qtype}
        for name in RETRIEVERS:
            st = by_type[qtype][name]
            row[name] = st["correct"] / max(st["valid"], 1) if st["valid"] else None
            row[f"{name}_n"] = st["valid"]
        cat_rows.append(row)

    return {"overall": summary_rows, "by_type": cat_rows}


def write_results_md(payload: dict) -> None:
    model = payload.get("model", "openai/gpt-4o-mini")
    n_q = payload.get("n_questions", 0)
    summaries = payload.get("summary", {})
    overall = summaries.get("overall", [])
    by_type = summaries.get("by_type", [])
    type_pick = payload.get("type_pick_counts", {})
    cap_notes = full_context_cap_notes(payload.get("results", []))
    cache_notes = payload.get("cache_reuse_notes", "")
    overlap = payload.get("round2_overlap", {})
    seed = payload.get("sampling_seed", SAMPLING_SEED)

    retriever_hdr = " | ".join(RETRIEVERS)
    incomplete = payload.get("incomplete", {})
    inc_banner = ""
    if incomplete.get("error_questions"):
        inc_banner = (
            f"\n> ⚠️ **INCOMPLETE RUN:** {incomplete.get('valid_questions', n_q)}/"
            f"{incomplete.get('target_questions', n_q)} questions with valid results. "
            f"{incomplete.get('error_questions', 0)} question(s) failed API calls "
            f"({incomplete.get('error_reason', 'see raw JSON')}). "
            "Re-run `benchmark_longmem.py` when API quota resets to fill gaps.\n"
        )

    md = f"""# LongMemEval_S Benchmark — Engram vs BM25 vs Full Context (Round {BENCHMARK_ROUND})
{inc_banner}
**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Model (answer + judge):** `{model}`  
**Dataset:** `longmemeval_s_cleaned.json` (ICLR 2025 LongMemEval)  
**Questions:** {n_q} (stratified random: {PER_TYPE} per type × {len(TARGET_TYPES)} types, seed={seed})  
**Retrievers this round:** {", ".join(f"`{r}`" for r in RETRIEVERS)} — `engram` / `engram_expand` omitted (diagnostic done at n=24)  
**Budget retrievers:** ~{TOKEN_BUDGET} tokens (tiktoken cl100k_base)  
**Engram graph:** 1 session = 1 tick (novelty+surprise promotion, tick decay)

### Selected per type

| question_type | count |
|---------------|------:|
"""
    for t in TARGET_TYPES:
        md += f"| {t} | {type_pick.get(t, 0)} |\n"

    md += f"""
### Sampling note (vs Round 2)

| Metric | Count |
|--------|------:|
| Round 2 overlap (cache-eligible question IDs) | {overlap.get('question_overlap', 0)} / {n_q} |
| Fresh questions (not in Round 2 first-4-per-type) | {overlap.get('fresh_questions', 0)} / {n_q} |

## Overall results (sorted by accuracy)

| Retriever | Accuracy | Avg ctx tokens | Est. cost/Q (USD) |
|-----------|----------|---------------:|------------------:|
"""
    for row in overall:
        md += (
            f"| {row['retriever']} | {row['accuracy']:.1%} "
            f"({row['correct']}/{row['valid']}) | "
            f"{row['avg_ctx_tokens']:,.0f} | ${row['avg_cost_usd']:.4f} |\n"
        )

    md += f"""
## Per-category accuracy

| question_type | {retriever_hdr} |
|---------------|{':'.join(['---:'] * len(RETRIEVERS))}|
"""
    for row in by_type:
        cells = []
        for name in RETRIEVERS:
            acc = row.get(name)
            n = row.get(f"{name}_n", 0)
            target_n = type_pick.get(row["question_type"], PER_TYPE)
            if acc is None or n == 0:
                cells.append("—")
            elif n < target_n:
                cells.append(f"{acc:.0%} ({n}/{target_n})")
            else:
                cells.append(f"{acc:.0%}")
        md += f"| {row['question_type']} | {' | '.join(cells)} |\n"

    econ = payload.get("token_economics", {})
    md += f"""
## Token economics

| Metric | Value |
|--------|------:|
| Avg full_context tokens | {econ.get('avg_full_tokens', 0):,.0f} |
| Avg 2k-budget retriever tokens | {econ.get('avg_budget_tokens', 0):,.0f} |
| Token ratio (full / budget) | {econ.get('token_ratio', 0):.1f}× |
| Cost ratio (full / budget, est.) | {econ.get('cost_ratio', 0):.1f}× |
| full_context truncation | {cap_notes} |
| questions with head+tail truncate | {payload.get('full_context_truncated_count', 0)}/{n_q} |

## Cache / rerun notes

{cache_notes or 'All retriever results computed in this run.'}

## Honest read

{payload.get('honest_read', 'See table above.')}

Raw per-question data: `{os.path.basename(RAW_JSON)}`
"""

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(md)


_LEGACY_FULL_CTX_TRUNCATE = 110_000


def full_context_cap_notes(results: List[dict]) -> str:
    """Describe truncation caps actually used in full_context rows."""
    caps: Counter = Counter()
    for rec in results:
        fc = rec.get("retrievers", {}).get("full_context", {})
        if fc.get("error") or fc.get("correct") is None:
            continue
        cap = fc.get("trunc_cap")
        if cap is None:
            cap = (
                _LEGACY_FULL_CTX_TRUNCATE
                if fc.get("ctx_tokens", 0) > 20_000
                else FULL_CTX_TRUNCATE
            )
        caps[cap] += 1
    if not caps:
        return f"default cap {FULL_CTX_TRUNCATE:,} tiktoken (head+tail truncate)"
    if len(caps) == 1:
        only = next(iter(caps))
        return f"{only:,} tiktoken cap (head+tail truncate)"
    parts = [f"{cap:,}×{n}" for cap, n in sorted(caps.items(), reverse=True)]
    return (
        f"**mixed caps** ({', '.join(parts)} rows) — "
        f"default {FULL_CTX_TRUNCATE:,}; lower caps used when OpenRouter credit prompt limit hit mid-run"
    )


def build_honest_read(overall: List[dict], by_type: List[dict], econ: dict, n_q: int, results: Optional[List[dict]] = None) -> str:
    by_name = {r["retriever"]: r for r in overall}
    full = by_name.get("full_context", {})
    bm25 = by_name.get("bm25", {})
    hybrid = by_name.get("engram_bm25", {})

    parts: List[str] = []

    full_acc = full.get("accuracy", 0)
    bm25_acc = bm25.get("accuracy", 0)
    hybrid_acc = hybrid.get("accuracy", 0)

    if hybrid_acc >= bm25_acc:
        parts.append(
            f"At n={n_q} (stratified random, seed={SAMPLING_SEED}), **engram_bm25** "
            f"({hybrid_acc:.0%}) ≥ BM25 ({bm25_acc:.0%}) overall."
        )
    else:
        parts.append(
            f"At n={n_q}, BM25 ({bm25_acc:.0%}) beats engram_bm25 ({hybrid_acc:.0%}) — "
            "Round 2 hybrid advantage did not hold at larger random sample."
        )

    if full_acc > max(bm25_acc, hybrid_acc):
        parts.append(
            f"Full-context ({full_acc:.0%}) still leads budget retrievers "
            f"(best budget {max(bm25_acc, hybrid_acc):.0%}); "
            f"~{econ.get('avg_full_tokens', 0):,.0f} tok avg vs ~{econ.get('avg_budget_tokens', 0):,.0f}."
        )
    elif max(bm25_acc, hybrid_acc) >= full_acc:
        parts.append(
            f"Best budget retriever ({max(bm25_acc, hybrid_acc):.0%}) matches or beats "
            f"full-context ({full_acc:.0%}) on this slice."
        )

    ku_row = next((r for r in by_type if r["question_type"] == "knowledge-update"), None)
    if ku_row:
        ku_h = ku_row.get("engram_bm25")
        ku_b = ku_row.get("bm25")
        ku_f = ku_row.get("full_context")
        if ku_h is not None and ku_b is not None:
            ku_f_s = f", full_context={ku_f:.0%}" if ku_f is not None else ""
            if ku_h > ku_b:
                parts.append(
                    f"**knowledge-update:** engram_bm25={ku_h:.0%} > bm25={ku_b:.0%}{ku_f_s} "
                    "— tick-decay prior + BM25 blend still helps."
                )
            elif ku_h == ku_b:
                parts.append(
                    f"**knowledge-update:** engram_bm25={ku_h:.0%} ties bm25={ku_b:.0%}."
                )
            else:
                parts.append(
                    f"**knowledge-update:** bm25={ku_b:.0%} > engram_bm25={ku_h:.0%} — "
                    "hybrid advantage regressed vs Round 2."
                )

    ms_row = next((r for r in by_type if r["question_type"] == "multi-session"), None)
    if ms_row:
        ms_parts = []
        for n in RETRIEVERS:
            a = ms_row.get(n)
            if a is not None:
                ms_parts.append(f"{n}={a:.0%}")
        if ms_parts:
            parts.append(f"**multi-session:** {', '.join(ms_parts)}.")

    pref_row = next(
        (r for r in by_type if r["question_type"] == "single-session-preference"), None
    )
    if pref_row:
        pref_f = pref_row.get("full_context")
        pref_b = pref_row.get("bm25")
        if pref_f is not None and pref_b is not None and pref_f < pref_b:
            parts.append(
                "**single-session-preference:** full_context underperforms budget methods "
                "(preference questions may need targeted retrieval, not full haystack)."
            )

    parts.append(
        f"Token economics: full-context ~{econ.get('token_ratio', 0):.0f}× more context tokens "
        f"than budget retrievers (~{econ.get('cost_ratio', 0):.0f}× estimated API cost). "
        "engram / engram_expand omitted in Round 3 (established at n=24)."
    )

    if results:
        cap_note = full_context_cap_notes(results)
        if "mixed caps" in cap_note:
            parts.append(
                f"**full_context caveat:** {cap_note}. "
                "Compare budget retrievers fairly; full_context is not a uniform ~110k baseline on this run."
            )

    return " ".join(parts)


def main() -> None:
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    count_tokens = ensure_tiktoken()
    api_key = load_env_key()
    client = OpenRouterClient(api_key)
    model = client.resolve_model()

    print(f"Loading {DATA_PATH}…")
    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    questions, type_pick, round2_overlap_ids = select_questions(dataset)
    round2_overlap_n = len(round2_overlap_ids)
    print(f"Selected {len(questions)} questions (seed={SAMPLING_SEED}):")
    for t, c in type_pick.items():
        print(f"  {t}: {c}")
    print(f"  Round 2 overlap: {round2_overlap_n} / {len(questions)} question IDs")

    partial = load_partial_raw()
    results_by_id: Dict[str, dict] = {}
    cache_reused: Counter = Counter()
    cache_rerun: Counter = Counter()
    if partial and partial.get("results"):
        for rec in partial["results"]:
            results_by_id[rec["question_id"]] = rec
        print(f"Resuming: {len(results_by_id)} questions already in {RAW_JSON}")

    n = len(questions)
    graph_build_times: List[float] = []
    if partial and partial.get("avg_graph_build_sec"):
        graph_build_times = [partial["avg_graph_build_sec"]]

    def question_needs_work(qid: str) -> bool:
        rec = results_by_id.get(qid)
        if rec is None:
            return True
        for rname in RETRIEVERS:
            prev = rec.get("retrievers", {}).get(rname, {})
            if not retriever_cache_valid(rname, prev):
                return True
        return False

    def build_cache_notes() -> str:
        fresh_n = len(questions) - round2_overlap_n
        lines = [
            f"Round {BENCHMARK_ROUND}: stratified random sample, seed={SAMPLING_SEED}, "
            f"{PER_TYPE} per type × {len(TARGET_TYPES)} types = {len(questions)} questions.",
            f"Round 2 overlap: {round2_overlap_n} question IDs (may reuse cached retriever rows).",
            f"Fresh vs Round 2 first-4-per-type: {fresh_n} questions.",
            "Retrievers: full_context, bm25, engram_bm25 only (engram/engram_expand dropped).",
        ]
        for rname in RETRIEVERS:
            lines.append(f"- {rname}: reused={cache_reused[rname]}, rerun={cache_rerun[rname]}")
        return "\n".join(lines)

    def count_incomplete() -> dict:
        err_q = 0
        valid_q = 0
        for rec in results_by_id.values():
            has_err = any(
                rec.get("retrievers", {}).get(rn, {}).get("error")
                for rn in RETRIEVERS
            )
            all_ok = all(
                retriever_cache_valid(rn, rec.get("retrievers", {}).get(rn, {}))
                for rn in RETRIEVERS
            )
            if all_ok:
                valid_q += 1
            elif has_err:
                err_q += 1
        return {
            "target_questions": len(questions),
            "valid_questions": valid_q,
            "error_questions": err_q,
            "error_reason": "OpenRouter API error (see raw JSON retriever error fields)",
        }

    def save_progress() -> None:
        results = [
            results_by_id[q.get("question_id", f"q{i}")]
            for i, q in enumerate(questions, 1)
            if q.get("question_id", f"q{i}") in results_by_id
        ]
        summaries = compute_summaries(results)
        full_rows = [r for r in summaries["overall"] if r["retriever"] == "full_context"]
        budget_rows = [r for r in summaries["overall"] if r["retriever"] != "full_context"]
        avg_full = full_rows[0]["avg_ctx_tokens"] if full_rows and full_rows[0]["valid"] else 0
        avg_budget = sum(r["avg_ctx_tokens"] for r in budget_rows) / max(len(budget_rows), 1)
        avg_full_cost = full_rows[0]["avg_cost_usd"] if full_rows and full_rows[0]["valid"] else 0
        avg_budget_cost = sum(r["avg_cost_usd"] for r in budget_rows) / max(len(budget_rows), 1)
        econ = {
            "avg_full_tokens": avg_full,
            "avg_budget_tokens": avg_budget,
            "token_ratio": avg_full / max(avg_budget, 1),
            "cost_ratio": avg_full_cost / max(avg_budget_cost, 1e-9),
        }
        honest = build_honest_read(
            summaries["overall"], summaries["by_type"], econ, len(results), results
        )
        incomplete = count_incomplete()
        if incomplete["error_questions"]:
            honest = (
                f"⚠️ Partial run ({incomplete['valid_questions']}/{incomplete['target_questions']} "
                f"questions valid; {incomplete['error_questions']} API failures). "
                + honest
            )
        payload = {
            "model": model,
            "dataset": os.path.basename(DATA_PATH),
            "benchmark_round": BENCHMARK_ROUND,
            "sampling_seed": SAMPLING_SEED,
            "n_questions": len(results),
            "target_types": TARGET_TYPES,
            "per_type": PER_TYPE,
            "type_pick_counts": type_pick,
            "round2_overlap": {
                "question_overlap": round2_overlap_n,
                "fresh_questions": len(questions) - round2_overlap_n,
                "overlap_ids": round2_overlap_ids,
            },
            "token_budget": TOKEN_BUDGET,
            "engram_bm25_lambda": ENGRAM_BM25_LAMBDA,
            "retriever_impl_version": RETRIEVER_IMPL_VERSION,
            "full_context_trunc_cap": FULL_CTX_TRUNCATE,
            "full_context_truncated_count": sum(
                1 for r in results if r.get("full_context_truncated")
            ),
            "avg_graph_build_sec": sum(graph_build_times) / max(len(graph_build_times), 1),
            "question_ids": [r["question_id"] for r in results],
            "summary": summaries,
            "token_economics": econ,
            "honest_read": honest,
            "cache_reuse_notes": build_cache_notes(),
            "cache_reused": dict(cache_reused),
            "cache_rerun": dict(cache_rerun),
            "incomplete": incomplete,
            "results": results,
        }
        save_raw(payload)

    for i, item in enumerate(questions, 1):
        qid = item.get("question_id", f"q{i}")
        existing = results_by_id.get(qid)
        if existing and not question_needs_work(qid):
            for rname in RETRIEVERS:
                prev = existing.get("retrievers", {}).get(rname, {})
                if retriever_cache_valid(rname, prev):
                    cache_reused[rname] += 1
            print(f"[{i}/{n}] {qid} — skipped (all retrievers complete)")
            continue

        qtype = item.get("question_type", "?")
        question = item["question"]
        gold = gold_answer(item)
        qdate = str(item.get("question_date") or "").strip()

        print(f"[{i}/{n}] {qid} type={qtype} — flattening haystack…")
        lines = flatten_haystack(
            item.get("haystack_sessions") or [],
            item.get("haystack_dates") or [],
        )
        print(f"  {len(lines)} lines, {len(item.get('haystack_sessions') or [])} sessions")

        rec = existing or {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "question_date": qdate,
            "gold": gold,
            "retrievers": {},
        }
        if "retrievers" not in rec:
            rec["retrievers"] = {}

        needs_graph = any(
            rname == "engram_bm25"
            and not retriever_cache_valid(rname, rec["retrievers"].get(rname, {}))
            for rname in RETRIEVERS
        )

        graph: dict = {"nodes": {}, "edges": [], "meta": {"tick": 0}}
        gtime = 0.0
        if needs_graph:
            t0 = time.time()
            graph = build_graph(lines)
            gtime = time.time() - t0
            graph_build_times.append(gtime)
            meta_tick = graph.get("meta", {}).get("tick", 0)
            print(
                f"  graph: {len(graph['nodes'])} concepts, {len(graph['edges'])} edges, "
                f"tick={meta_tick}, build={gtime:.2f}s"
            )
        else:
            print("  graph: skipped (engram-family retrievers cached)")

        full_ctx, full_tok, truncated = ("", 0, False)
        if "full_context" in RETRIEVERS:
            full_ctx, full_tok, truncated = build_full_context(lines, count_tokens)
        full_cache = (full_ctx, full_tok, truncated)

        rec.update(
            {
                "question_type": qtype,
                "question": question,
                "question_date": qdate,
                "gold": gold,
                "n_lines": len(lines),
                "n_sessions": len(item.get("haystack_sessions") or []),
                "graph_concepts": len(graph["nodes"]),
                "graph_edges": len(graph["edges"]),
                "graph_build_sec": round(gtime, 3),
                "full_context_truncated": truncated,
            }
        )

        status_parts: List[str] = []
        for rname in RETRIEVERS:
            prev = rec["retrievers"].get(rname, {})
            if retriever_cache_valid(rname, prev):
                cache_reused[rname] += 1
                status_parts.append(f"{rname}={'Y' if prev['correct'] else 'N'}*")
                continue

            cache_rerun[rname] += 1
            rrec: dict = {
                "pred": None,
                "correct": None,
                "ctx_tokens": 0,
                "est_cost_usd": 0.0,
                "error": None,
                "token_budget": TOKEN_BUDGET,
            }
            if rname in RETRIEVER_IMPL_VERSION:
                rrec["impl_version"] = RETRIEVER_IMPL_VERSION[rname]
            if rname == "full_context":
                rrec["trunc_cap"] = FULL_CTX_TRUNCATE
            try:
                ctx, ctx_tok = run_retriever(
                    rname, question, lines, graph, count_tokens, full_cache
                )
                rrec["ctx_tokens"] = ctx_tok
                api_timeout = 300 if rname == "full_context" else 120
                pred = ""
                for attempt in range(2):
                    try:
                        pred = (
                            answer_question_longmem(
                                client, ctx, question, qdate, timeout=api_timeout
                            )
                            if ctx
                            else ""
                        )
                        break
                    except Exception as api_err:
                        if attempt == 0:
                            time.sleep(2)
                            continue
                        raise api_err
                rrec["pred"] = pred
                ok = client.judge(question, gold, pred)
                rrec["correct"] = ok
                rrec["est_cost_usd"] = estimate_cost(ctx_tok)
                status_parts.append(f"{rname}={'Y' if ok else 'N'}")
            except Exception as e:
                err_msg = str(e)
                rrec["error"] = err_msg
                status_parts.append(f"{rname}=ERR")
                print(f"  ERROR [{rname}] {err_msg[:800]}")
            rec["retrievers"][rname] = rrec

        print(f"  {' '.join(status_parts)} | full_tok={full_tok:,}")
        results_by_id[qid] = rec
        save_progress()
        time.sleep(0.25)

    results = [
        results_by_id[q.get("question_id", f"q{i}")]
        for i, q in enumerate(questions, 1)
        if q.get("question_id", f"q{i}") in results_by_id
    ]
    summaries = compute_summaries(results)
    full_row = next(r for r in summaries["overall"] if r["retriever"] == "full_context")
    budget_rows = [r for r in summaries["overall"] if r["retriever"] != "full_context"]
    avg_budget_tok = sum(r["avg_ctx_tokens"] for r in budget_rows) / max(len(budget_rows), 1)
    avg_budget_cost = sum(r["avg_cost_usd"] for r in budget_rows) / max(len(budget_rows), 1)
    econ = {
        "avg_full_tokens": full_row["avg_ctx_tokens"],
        "avg_budget_tokens": avg_budget_tok,
        "token_ratio": full_row["avg_ctx_tokens"] / max(avg_budget_tok, 1),
        "cost_ratio": full_row["avg_cost_usd"] / max(avg_budget_cost, 1e-9),
    }
    honest = build_honest_read(
        summaries["overall"], summaries["by_type"], econ, len(results), results
    )
    incomplete = count_incomplete()
    if incomplete["error_questions"]:
        honest = (
            f"⚠️ Partial run ({incomplete['valid_questions']}/{incomplete['target_questions']} "
            f"questions valid; {incomplete['error_questions']} API failures). "
            + honest
        )

    final_payload = {
        "model": model,
        "dataset": os.path.basename(DATA_PATH),
        "benchmark_round": BENCHMARK_ROUND,
        "sampling_seed": SAMPLING_SEED,
        "n_questions": len(results),
        "target_types": TARGET_TYPES,
        "per_type": PER_TYPE,
        "type_pick_counts": type_pick,
        "round2_overlap": {
            "question_overlap": round2_overlap_n,
            "fresh_questions": len(questions) - round2_overlap_n,
            "overlap_ids": round2_overlap_ids,
        },
        "token_budget": TOKEN_BUDGET,
        "engram_bm25_lambda": ENGRAM_BM25_LAMBDA,
        "retriever_impl_version": RETRIEVER_IMPL_VERSION,
        "full_context_trunc_cap": FULL_CTX_TRUNCATE,
        "full_context_truncated_count": sum(
            1 for r in results if r.get("full_context_truncated")
        ),
        "avg_graph_build_sec": sum(graph_build_times) / max(len(graph_build_times), 1),
        "question_ids": [r["question_id"] for r in results],
        "summary": summaries,
        "token_economics": econ,
        "honest_read": honest,
        "cache_reuse_notes": build_cache_notes(),
        "cache_reused": dict(cache_reused),
        "cache_rerun": dict(cache_rerun),
        "incomplete": incomplete,
        "results": results,
    }
    save_raw(final_payload)
    write_results_md(final_payload)

    print("\n=== LONGMEM SUMMARY ===")
    print("| Retriever | Accuracy | Avg ctx tokens | Est cost/Q |")
    for row in summaries["overall"]:
        print(
            f"| {row['retriever']} | {row['accuracy']:.1%} | "
            f"{row['avg_ctx_tokens']:,.0f} | ${row['avg_cost_usd']:.4f} |"
        )
    print(f"\nToken ratio full/budget: {econ['token_ratio']:.1f}×")
    print(f"Wrote {RESULTS_MD} and {RAW_JSON}")


if __name__ == "__main__":
    main()
