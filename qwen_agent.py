"""Cogram MemoryAgent for Qwen Cloud (Global AI Hackathon Series, Track 1).

This is the ONE file that actually calls Qwen. Everything memory-related —
concept graph, decay, provenance, procedural motifs — is still zero-LLM,
built entirely in `graph_lib.py` / `extract_swe.py` / `memory_api.py`, none
of which import an LLM SDK. What Qwen is for here is the *last* step: turning
a zero-token recall (a ranked list of exact file:line fixes) into a natural-
language answer for the user.

Why split it this way instead of asking Qwen to "remember" directly:
  - Cogram's memory budget is 0 LLM tokens to build/update (extract, decay,
    motif mining are pure Python/statistics — see graph_lib.py). Only the
    final answer-generation step spends Qwen tokens, and it spends them on
    a ~150-400 token context instead of re-feeding raw history.
  - This is the same split every mature memory framework uses (Mem0, Zep/
    Graphiti): a cheap non-LLM retrieval/storage layer, and the LLM is only
    in the generation loop, not the write path.

Usage:
    export DASHSCOPE_API_KEY=sk-...
    python qwen_agent.py "dacite default_factory dataclass bug"

Or as a library:
    from qwen_agent import answer_with_memory
    result = answer_with_memory("new bug report text here")
"""
from __future__ import annotations

import os
import sys
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from memory_api import recommend

QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")


def _qwen_client():
    """Lazily construct an OpenAI-SDK client pointed at Qwen Cloud (DashScope).

    Kept lazy so the zero-LLM memory layer (memory_api.recommend) can be
    imported and unit-tested with no API key and no network access at all —
    only calling answer_with_memory() actually needs credentials.
    """
    from openai import OpenAI

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY not set. Get one from Alibaba Cloud Model "
            "Studio (https://www.alibabacloud.com/help/en/model-studio/get-api-key) "
            "and `export DASHSCOPE_API_KEY=sk-...`"
        )
    return OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)


def answer_with_memory(query: str, budget_tokens: int = 400, model: Optional[str] = None) -> dict:
    """Zero-LLM recall from the Cogram concept graph, then one Qwen call to
    turn that recall into a natural-language answer.

    Returns a dict with the recall (memory_tokens, recommendations, routes)
    and the Qwen-generated answer, so a judge / caller can see exactly how
    many tokens the memory layer spent vs. the generation call.
    """
    recall = recommend(query, budget_tokens=budget_tokens)
    recs = recall.get("recommendations", [])

    if not recs:
        context_block = "(no prior fix found in memory for this query)"
    else:
        lines = []
        for r in recs:
            lines.append(f"[{r['session']}:{r['line_idx']}] {r['text']}")
        context_block = "\n".join(lines)

    system_prompt = (
        "You are a coding agent's procedural-memory assistant. You are given "
        "a NEW bug report and a set of EXACT lines recalled from the agent's "
        "own past trajectories (file:line, verbatim, not paraphrased) by a "
        "zero-LLM concept-graph memory layer called Cogram. Use only these "
        "recalled lines as your evidence. If they show a matching prior fix, "
        "say what it was and where it came from (session:line). If nothing "
        "relevant was recalled, say so plainly instead of guessing."
    )
    user_prompt = f"NEW bug report:\n{query}\n\nRecalled from memory:\n{context_block}"

    client = _qwen_client()
    completion = client.chat.completions.create(
        model=model or QWEN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    answer_text = completion.choices[0].message.content

    return {
        "query": query,
        "memory_tokens_used": recall.get("stats", {}).get("tokens_used", 0),
        "activated_concepts": recall.get("stats", {}).get("activated_concepts", 0),
        "recalled_lines": recs,
        "routes": recall.get("routes", []),
        "answer": answer_text,
        "qwen_model": model or QWEN_MODEL,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python qwen_agent.py "your query here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    result = answer_with_memory(query)
    print(f"\n=== Cogram recall ({result['memory_tokens_used']} tokens, "
          f"{result['activated_concepts']} concepts activated) ===")
    for r in result["recalled_lines"][:5]:
        print(f"  [{r['session']}:{r['line_idx']}] {r['text'][:120]}")
    print(f"\n=== Qwen ({result['qwen_model']}) answer ===")
    print(result["answer"])


if __name__ == "__main__":
    main()
