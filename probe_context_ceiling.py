"""One-off probe: find OpenRouter per-request prompt token ceiling for full_context."""
from __future__ import annotations

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from benchmark_longmem import (  # noqa: E402
    DATA_PATH,
    answer_question_longmem,
    build_full_context,
    flatten_haystack,
    select_questions,
)
from benchmark_qa import OpenRouterClient, ensure_tiktoken, load_env_key  # noqa: E402


def probe_caps(caps: list[int]) -> None:
    count_tokens = ensure_tiktoken()
    client = OpenRouterClient(load_env_key())

    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    questions, _, _ = select_questions(dataset)
    item = questions[0]
    lines = flatten_haystack(
        item.get("haystack_sessions") or [],
        item.get("haystack_dates") or [],
    )
    question = item["question"]
    qdate = str(item.get("question_date") or "").strip()

    print(f"Probe question: {item.get('question_id')} ({len(lines)} lines)")
    print(f"Model: {client.resolve_model()}\n")

    for cap in caps:
        ctx, ctx_tok, truncated = build_full_context(lines, count_tokens, max_tokens=cap)
        system = (
            "You answer questions about a user's long chat history with an assistant. "
            "Use ONLY the provided conversation context. Session dates are prefixed on each line. "
            "The question may refer to a specific date — use temporal cues carefully. "
            "Be concise — one short sentence or phrase."
        )
        qdate_line = f"Question date: {qdate}\n\n" if qdate else ""
        user = (
            f"{qdate_line}Conversation history:\n{ctx}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        total_tok = count_tokens(system) + count_tokens(user) + 50
        status = "?"
        detail = ""
        try:
            pred = answer_question_longmem(client, ctx, question, qdate, timeout=300)
            status = "OK"
            detail = pred[:80].replace("\n", " ")
        except Exception as e:
            status = "FAIL"
            detail = str(e)
        print(
            f"cap={cap:>6,} ctx_tok={ctx_tok:>7,} est_total={total_tok:>7,} "
            f"trunc={truncated} -> {status}"
        )
        if status == "FAIL":
            print(f"         {detail[:600]}")
        else:
            print(f"         pred: {detail}")


if __name__ == "__main__":
    caps = [110_000, 50_000, 40_000, 35_000, 30_000, 25_000, 20_000]
    if len(sys.argv) > 1:
        caps = [int(x) for x in sys.argv[1:]]
    probe_caps(caps)
