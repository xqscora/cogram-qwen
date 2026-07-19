"""Retrieval-only token-budget sweep (no OpenRouter). Compares bm25 vs engram_bm25 ctx packing."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from benchmark_bakeoff import retrieve_bm25, retrieve_engram_bm25, select_lines_under_budget
from benchmark_longmem import (
    DATA_PATH,
    SAMPLING_SEED,
    TARGET_TYPES,
    flatten_haystack,
    gold_answer,
    select_questions,
)

MEMORY_TYPES = ["knowledge-update", "multi-session", "temporal-reasoning"]
from benchmark_qa import build_graph, ensure_tiktoken

BUDGETS = [500, 1000, 1500, 2000]
OUT_JSON = os.path.join(_SCRIPT_DIR, "benchmark_budget_sweep_retrieval.json")


def retrieve_line_indices(
    name: str,
    question: str,
    lines: List[dict],
    graph: dict,
    count_tokens,
    budget: int,
) -> Tuple[List[int], int]:
    if name == "bm25":
        from benchmark_bakeoff import bm25_line_scores, tokenize

        scores = bm25_line_scores(tokenize(question), lines)
        ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        ranked = [i for i, s in ordered if s > 0] + [i for i, s in ordered if s <= 0]
    else:
        from benchmark_bakeoff import (
            ENGRAM_BM25_LAMBDA,
            bm25_line_scores,
            engram_substrate_candidates,
            tokenize,
            _normalize_scores,
        )

        candidate_idx, engram_prior = engram_substrate_candidates(question, graph)
        if not candidate_idx:
            candidate_idx = set(range(len(lines)))
            engram_prior = {i: 0.0 for i in candidate_idx}
        bm25_scores = bm25_line_scores(tokenize(question), lines)
        bm25_cand = {idx: bm25_scores.get(idx, 0.0) for idx in candidate_idx}
        prior_cand = {idx: engram_prior.get(idx, 0.0) for idx in candidate_idx}
        bm25_norm = _normalize_scores(bm25_cand)
        prior_norm = _normalize_scores(prior_cand)
        combined = {
            idx: bm25_norm.get(idx, 0.0) + ENGRAM_BM25_LAMBDA * prior_norm.get(idx, 0.0)
            for idx in candidate_idx
        }
        ranked = sorted(candidate_idx, key=lambda idx: (-combined[idx], idx))

    _, tok, selected = select_lines_under_budget(ranked, lines, count_tokens, budget)
    return selected, tok


def gold_in_context(gold: str, ctx: str) -> bool:
    g = gold.strip().lower()
    if not g:
        return False
    c = ctx.lower()
    return g in c or any(part in c for part in g.split() if len(part) > 3)


def main() -> None:
    count_tokens = ensure_tiktoken()
    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    questions, type_pick, _ = select_questions(dataset)

    # Load 2000 LLM-judged accuracy from cache
    llm_2000 = None
    raw_2000 = os.path.join(_SCRIPT_DIR, "benchmark_longmem_raw.json")
    if os.path.isfile(raw_2000):
        with open(raw_2000, encoding="utf-8") as f:
            llm_2000 = json.load(f)

    results_by_budget: Dict[int, dict] = {
        b: {
            "budget": b,
            "n_questions": len(questions),
            "bm25_ctx_sum": 0,
            "engram_ctx_sum": 0,
            "bm25_gold_hits": 0,
            "engram_gold_hits": 0,
            "identical_line_sets": 0,
            "by_type": defaultdict(
                lambda: {
                    "bm25_gold_hits": 0,
                    "engram_gold_hits": 0,
                    "n": 0,
                    "retrieval_diffs": 0,
                }
            ),
        }
        for b in BUDGETS
    }

    n = len(questions)
    for i, item in enumerate(questions, 1):
        qtype = item.get("question_type", "?")
        question = item["question"]
        gold = gold_answer(item)
        lines = flatten_haystack(
            item.get("haystack_sessions") or [],
            item.get("haystack_dates") or [],
        )
        graph = build_graph(lines)

        per_budget_indices: Dict[int, Tuple[List[int], List[int]]] = {}
        for budget in BUDGETS:
            b_idx, b_tok = retrieve_line_indices(
                "bm25", question, lines, graph, count_tokens, budget
            )
            e_idx, e_tok = retrieve_line_indices(
                "engram_bm25", question, lines, graph, count_tokens, budget
            )
            per_budget_indices[budget] = (b_idx, e_idx)
            row = results_by_budget[budget]
            row["bm25_ctx_sum"] += b_tok
            row["engram_ctx_sum"] += e_tok
            b_ctx = "\n".join(lines[j]["line_str"] for j in sorted(b_idx))
            e_ctx = "\n".join(lines[j]["line_str"] for j in sorted(e_idx))
            if gold_in_context(gold, b_ctx):
                row["bm25_gold_hits"] += 1
                row["by_type"][qtype]["bm25_gold_hits"] += 1
            if gold_in_context(gold, e_ctx):
                row["engram_gold_hits"] += 1
                row["by_type"][qtype]["engram_gold_hits"] += 1
            row["by_type"][qtype]["n"] += 1
            if set(b_idx) == set(e_idx):
                row["identical_line_sets"] += 1
            else:
                row["by_type"][qtype]["retrieval_diffs"] += 1

        if i % 4 == 0 or i == n:
            print(f"  [{i}/{n}] …")

    final_rows: Dict[int, dict] = {}
    for budget in BUDGETS:
        row_in = results_by_budget[budget]
        row = {
            "budget": budget,
            "n_questions": n,
            "bm25_avg_ctx": row_in["bm25_ctx_sum"] / n,
            "engram_avg_ctx": row_in["engram_ctx_sum"] / n,
            "bm25_gold_in_ctx_rate": row_in["bm25_gold_hits"] / n,
            "engram_gold_in_ctx_rate": row_in["engram_gold_hits"] / n,
            "identical_line_sets": row_in["identical_line_sets"],
            "by_type": {},
        }

        if budget == 2000 and llm_2000:
            summ = {r["retriever"]: r for r in llm_2000["summary"]["overall"]}
            row["bm25_llm_accuracy"] = summ.get("bm25", {}).get("accuracy")
            row["engram_llm_accuracy"] = summ.get("engram_bm25", {}).get("accuracy")
            row["bm25_llm_correct"] = summ.get("bm25", {}).get("correct")
            row["engram_llm_correct"] = summ.get("engram_bm25", {}).get("correct")
            by_type_llm = {
                r["question_type"]: r for r in llm_2000["summary"]["by_type"]
            }
            for qtype in MEMORY_TYPES:
                tr = by_type_llm.get(qtype, {})
                bt = row_in["by_type"].get(qtype, {})
                nn = max(bt.get("n", 1), 1)
                row["by_type"][qtype] = {
                    "bm25_llm": tr.get("bm25"),
                    "engram_llm": tr.get("engram_bm25"),
                    "bm25_gold_in_ctx": bt.get("bm25_gold_hits", 0) / nn,
                    "engram_gold_in_ctx": bt.get("engram_gold_hits", 0) / nn,
                    "retrieval_diffs": bt.get("retrieval_diffs", 0),
                }
        else:
            for qtype in TARGET_TYPES:
                bt = row_in["by_type"].get(qtype)
                if not bt:
                    continue
                nn = max(bt.get("n", 1), 1)
                row["by_type"][qtype] = {
                    "bm25_gold_in_ctx": bt.get("bm25_gold_hits", 0) / nn,
                    "engram_gold_in_ctx": bt.get("engram_gold_hits", 0) / nn,
                    "retrieval_diffs": bt.get("retrieval_diffs", 0),
                }

        final_rows[budget] = row
        print(
            f"budget={budget}: avg ctx bm25={row['bm25_avg_ctx']:.0f} "
            f"engram={row['engram_avg_ctx']:.0f} | identical={row['identical_line_sets']}/{n} | "
            f"gold-in-ctx bm25={row['bm25_gold_in_ctx_rate']:.1%} "
            f"engram={row['engram_gold_in_ctx_rate']:.1%}"
        )

    results_by_budget = final_rows

    out = {
        "sampling_seed": SAMPLING_SEED,
        "n_questions": len(questions),
        "budgets": BUDGETS,
        "note": "gold_in_ctx is extractive proxy, NOT LLM-judged accuracy",
        "llm_accuracy_available_budgets": [2000],
        "results": results_by_budget,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
