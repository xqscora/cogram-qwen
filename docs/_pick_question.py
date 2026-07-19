"""One-off helper to pick smallest knowledge-update question with graph stats."""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from benchmark_longmem import flatten_haystack
from benchmark_qa import build_graph

DATA = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "longmemeval_s_cleaned.json"
data = json.loads(DATA.read_text(encoding="utf-8"))
ku = {x["question_id"]: x for x in data if x.get("question_type") == "knowledge-update"}


def hay_chars(item):
    return sum(
        len((t.get("content") or ""))
        for s in (item.get("haystack_sessions") or [])
        for t in s
    )


ids = sorted(ku.keys(), key=lambda i: hay_chars(ku[i]))[:10]
for qid in ids:
    x = ku[qid]
    lines = flatten_haystack(
        x.get("haystack_sessions") or [],
        x.get("haystack_dates") or [],
    )
    g = build_graph(lines)
    ans = str(x.get("answer", ""))[:60]
    print(
        f"{qid}: chars={hay_chars(x)} nodes={len(g['nodes'])} "
        f"edges={len(g['edges'])} lines={len(lines)} "
        f"sessions={len(x.get('haystack_sessions') or [])}"
    )
    print(f"  Q: {x['question'][:90]}")
    print(f"  A: {ans}")
