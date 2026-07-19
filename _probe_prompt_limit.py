"""Probe OpenRouter prompt limit vs TOKEN_BUDGET (no full benchmark)."""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

os.environ.setdefault("LONGMEM_RETRIEVERS", "bm25")
os.environ["TOKEN_BUDGET"] = os.environ.get("TOKEN_BUDGET", "300")

from benchmark_bakeoff import TOKEN_BUDGET, retrieve_bm25  # noqa: E402
from benchmark_longmem import (  # noqa: E402
    DATA_PATH,
    answer_question_longmem,
    flatten_haystack,
    select_questions,
)
from benchmark_qa import OpenRouterClient, ensure_tiktoken, load_env_key  # noqa: E402


def main() -> None:
    import json

    count_tokens = ensure_tiktoken()
    client = OpenRouterClient(load_env_key())
    with open(DATA_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    item = select_questions(dataset)[0][0]
    lines = flatten_haystack(item.get("haystack_sessions") or [], item.get("haystack_dates") or [])
    ctx, ctx_tok = retrieve_bm25(item["question"], lines, count_tokens, TOKEN_BUDGET)
    qdate = str(item.get("question_date") or "").strip()
    system = (
        "You answer questions about a user's long chat history with an assistant. "
        "Use ONLY the provided conversation context. Session dates are prefixed on each line. "
        "The question may refer to a specific date — use temporal cues carefully. "
        "Be concise — one short sentence or phrase."
    )
    user = (
        f"Question date: {qdate}\n\nConversation history:\n{ctx}\n\n"
        f"Question: {item['question']}\n\nAnswer:"
    )
    prompt_tok = count_tokens(system) + count_tokens(user)
    print(f"budget={TOKEN_BUDGET} ctx_tok={ctx_tok} est_prompt_tok={prompt_tok}")
    try:
        ans = answer_question_longmem(client, ctx, item["question"], qdate)
        print("answer OK:", ans[:60])
    except Exception as e:
        print("answer FAIL:", str(e)[:200])


if __name__ == "__main__":
    main()
