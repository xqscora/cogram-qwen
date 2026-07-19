"""Token-budget sweep: bm25 vs engram_bm25 on LongMemEval (no full_context).

Runs benchmark_longmem.py at TOKEN_BUDGET ∈ {500, 1000, 1500, 2000} and writes
benchmark_budget_sweep_results.md. Reuses benchmark_longmem_raw.json for budget=2000.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_BUDGETS = [500, 1000, 1500, 2000]
SWEEP_RETRIEVERS = ["bm25", "engram_bm25"]
MEMORY_TYPES = ["knowledge-update", "multi-session", "temporal-reasoning"]
DEFAULT_RAW_2000 = os.path.join(_SCRIPT_DIR, "benchmark_longmem_raw.json")
OUT_MD = os.path.join(_SCRIPT_DIR, "benchmark_budget_sweep_results.md")


def raw_path_for_budget(budget: int) -> str:
    if budget == 2000:
        return DEFAULT_RAW_2000
    return os.path.join(_SCRIPT_DIR, f"benchmark_longmem_raw_b{budget}.json")


def load_payload(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_budget(payload: dict, budget: int) -> dict:
    """Extract overall + memory-type accuracies for bm25 / engram_bm25."""
    summary = payload.get("summary", {})
    overall_rows = {r["retriever"]: r for r in summary.get("overall", [])}
    by_type = {r["question_type"]: r for r in summary.get("by_type", [])}

    row = {
        "budget": budget,
        "bm25_acc": overall_rows.get("bm25", {}).get("accuracy"),
        "bm25_correct": overall_rows.get("bm25", {}).get("correct"),
        "bm25_valid": overall_rows.get("bm25", {}).get("valid"),
        "engram_acc": overall_rows.get("engram_bm25", {}).get("accuracy"),
        "engram_correct": overall_rows.get("engram_bm25", {}).get("correct"),
        "engram_valid": overall_rows.get("engram_bm25", {}).get("valid"),
        "engram_avg_ctx": overall_rows.get("engram_bm25", {}).get("avg_ctx_tokens"),
        "bm25_avg_ctx": overall_rows.get("bm25", {}).get("avg_ctx_tokens"),
        "by_type": {},
    }
    for qtype in MEMORY_TYPES:
        tr = by_type.get(qtype, {})
        row["by_type"][qtype] = {
            "bm25": tr.get("bm25"),
            "engram_bm25": tr.get("engram_bm25"),
        }
    return row


def run_budget(budget: int) -> None:
    env = os.environ.copy()
    env["TOKEN_BUDGET"] = str(budget)
    env["LONGMEM_RETRIEVERS"] = ",".join(SWEEP_RETRIEVERS)
    print(f"\n=== Running budget={budget} → {raw_path_for_budget(budget)} ===")
    subprocess.run(
        [sys.executable, os.path.join(_SCRIPT_DIR, "benchmark_longmem.py")],
        cwd=_SCRIPT_DIR,
        env=env,
        check=True,
    )


def count_api_calls(payload: dict) -> int:
    """Answer + judge per retriever row computed in this payload."""
    n = 0
    for rec in payload.get("results", []):
        for name in SWEEP_RETRIEVERS:
            r = rec.get("retrievers", {}).get(name, {})
            if r.get("correct") is not None and not r.get("error"):
                n += 2  # answer + judge
    return n


def build_honest_read(rows: List[dict]) -> str:
    parts: List[str] = []
    wins = []
    for r in rows:
        b = r["budget"]
        ba = r.get("bm25_acc")
        ea = r.get("engram_acc")
        if ba is None or ea is None:
            continue
        if ea > ba + 1e-9:
            wins.append(b)
        elif abs(ea - ba) < 1e-9:
            parts.append(f"At {b} tokens, bm25 and engram_bm25 tie ({ba:.0%}).")

    if wins:
        parts.insert(
            0,
            f"engram_bm25 beats bm25 at budget(s): {', '.join(str(w) for w in wins)}.",
        )
    else:
        parts.insert(
            0,
            "engram_bm25 does **not** beat bm25 at any tested budget — they tie or bm25 leads.",
        )

    r2000 = next((r for r in rows if r["budget"] == 2000), None)
    if r2000:
        ba = r2000.get("bm25_acc", 0)
        ea = r2000.get("engram_acc", 0)
        parts.append(f"At 2000 tokens (baseline): both {ba:.0%} ({r2000.get('bm25_correct')}/{r2000.get('bm25_valid')}).")

    for qtype in MEMORY_TYPES:
        type_bits = []
        for r in rows:
            bt = r.get("by_type", {}).get(qtype, {})
            ba = bt.get("bm25")
            ea = bt.get("engram_bm25")
            if ba is None or ea is None:
                continue
            if ea > ba + 1e-9:
                type_bits.append(f"budget {r['budget']}: engram {ea:.0%} > bm25 {ba:.0%}")
            elif abs(ea - ba) < 1e-9:
                type_bits.append(f"budget {r['budget']}: tie {ea:.0%}")
            else:
                type_bits.append(f"budget {r['budget']}: bm25 {ba:.0%} > engram {ea:.0%}")
        if type_bits:
            parts.append(f"**{qtype}:** " + "; ".join(type_bits) + ".")

    return " ".join(parts)


def write_report(rows: List[dict], total_calls: int, skip_run: bool) -> None:
    md = f"""# LongMemEval Token-Budget Sweep — bm25 vs engram_bm25

**Date:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Dataset:** longmemeval_s_cleaned.json (48 questions, seed=42, 8/type × 6 types)  
**Retrievers:** `bm25`, `engram_bm25` only (no `full_context`)  
**Model:** openai/gpt-4o-mini (answer + judge)  
**Hypothesis tested:** Engram concept-activation prior keeps accuracy higher than plain BM25 at low token budgets.

## Accuracy vs budget

| Budget (tokens) | bm25 acc | engram_bm25 acc | avg ctx tokens (engram_bm25) |
|----------------:|---------:|----------------:|-----------------------------:|
"""
    for r in rows:
        ba = r.get("bm25_acc")
        ea = r.get("engram_acc")
        avg = r.get("engram_avg_ctx")
        ba_s = f"{ba:.1%} ({r.get('bm25_correct')}/{r.get('bm25_valid')})" if ba is not None else "—"
        ea_s = f"{ea:.1%} ({r.get('engram_correct')}/{r.get('engram_valid')})" if ea is not None else "—"
        avg_s = f"{avg:,.0f}" if avg is not None else "—"
        md += f"| {r['budget']} | {ba_s} | {ea_s} | {avg_s} |\n"

    md += f"""
## Memory-heavy categories (per budget)

| Budget | knowledge-update | multi-session | temporal-reasoning |
|-------:|-----------------:|--------------:|-------------------:|
"""
    for r in rows:
        cells = []
        for qtype in MEMORY_TYPES:
            bt = r.get("by_type", {}).get(qtype, {})
            ba = bt.get("bm25")
            ea = bt.get("engram_bm25")
            if ba is None or ea is None:
                cells.append("—")
            elif abs((ea or 0) - (ba or 0)) < 1e-9:
                cells.append(f"{ea:.0%} (tie)")
            elif (ea or 0) > (ba or 0):
                cells.append(f"E {ea:.0%} > B {ba:.0%}")
            else:
                cells.append(f"B {ba:.0%} > E {ea:.0%}")
        md += f"| {r['budget']} | {' | '.join(cells)} |\n"

    honest = build_honest_read(rows)
    md += f"""
## Honest read

{honest}

## Run notes

- Budget 2000 rows reused from `{os.path.basename(DEFAULT_RAW_2000)}` (existing Round 3 cache).
- Other budgets write separate raw files: `benchmark_longmem_raw_b{{budget}}.json`.
- Approx LLM calls this sweep (answer+judge, bm25+engram only): **{total_calls}** (~${total_calls * 0.00078:.2f} est. at ~2000 ctx).
- skip_run={skip_run}
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote {OUT_MD}")


def main() -> None:
    skip_run = "--report-only" in sys.argv
    rows: List[dict] = []
    total_calls = 0

    for budget in SWEEP_BUDGETS:
        path = raw_path_for_budget(budget)
        if budget == 2000:
            payload = load_payload(path)
            if payload is None:
                raise FileNotFoundError(f"Missing 2000-budget cache: {path}")
            rows.append(summarize_budget(payload, budget))
            continue

        if not skip_run:
            run_budget(budget)

        payload = load_payload(path)
        if payload is None:
            print(f"WARNING: no results at {path} after run")
            continue
        rows.append(summarize_budget(payload, budget))
        total_calls += count_api_calls(payload)

    rows.sort(key=lambda r: r["budget"])
    write_report(rows, total_calls, skip_run)


if __name__ == "__main__":
    main()
