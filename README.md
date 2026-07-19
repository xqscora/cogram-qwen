# Cogram MemoryAgent — Procedural Memory for Coding Agents, powered by Qwen Cloud

**Submitted to the Global AI Hackathon Series with Qwen Cloud — Track 1: MemoryAgent.**
This is an Apache-2.0 fork of [Cogram](https://github.com/xqscora/cogram) (canonical repo, CC BY-NC-SA 4.0)
prepared specifically for this hackathon's open-source-license requirement.

![Architecture](docs/architecture.png)

- **Memory layer (`graph_lib.py`, `extract_swe.py`, `memory_api.py`): zero LLM calls.** Concept extraction, edge weighting, tick-based decay, and procedural motif mining are pure statistics — no Qwen, no any LLM, in the write path.
- **Generation layer (`qwen_agent.py`): one Qwen Cloud call per query.** Takes the zero-token recall and turns it into a grounded natural-language answer via Qwen Cloud's OpenAI-compatible API (`qwen-plus`).
- **Deployment (`deploy/`): runs on Alibaba Cloud Function Compute.** See `deploy/README.md` for the one-command deploy and the proof-of-deployment file (`deploy/fc_handler.py`).

Coding agents (SWE-agent, OpenHands, Claude Code, Cursor, etc.) don't remember what they did last week. Every run starts cold, so the same bug gets re-diagnosed and the same fix gets re-derived over and over. The usual workaround is stuffing old transcripts back into the context window, which is slow, expensive, and still degrades once the history gets long.

Cogram is a small memory layer that sits on top of an agent's own past transcripts (reasoning → command → error → fix → patch) and turns them into a concept graph, with every concept pointing back to the exact file and line it came from. Building the graph costs **zero LLM tokens** — it's built with statistics (co-occurrence + a novelty/surprise signal), not with an LLM summarizing anything. Recalling from it later costs on the order of **~200 tokens** instead of thousands.

Old, unused parts of the graph fade over time (measured in "ticks" — actual agent interactions, not wall-clock time, so memory doesn't decay just because the agent was idle) but nothing is ever fully deleted. A fix that hasn't been touched in months can still be found; it just has to be "re-lit" by being relevant again.

## What's in this repo

| File / folder | What it does |
|---|---|
| `extract_swe.py`, `fetch_swe.py`, `swe_to_transcripts.py` | Build the concept graph from a slice of public SWE-agent trajectories |
| `graph_lib.py` | Core graph math: edge weights, tick-based decay, tokenization |
| `recall.py` | Zero-LLM retrieval: given a query, walk the graph and return the most relevant lines |
| `learn.py` | Online learning — reinforces known concepts, creates new ones as they show up |
| `memory_api.py` | Agent-facing API: `recommend()` and `end_turn()`, meant to be called from inside an agent loop |
| `swe_recall_demo.py` | Runnable demo — shows a new problem being recalled against the past-fix graph, zero LLM calls |
| `qwen_agent.py` | The Qwen Cloud generation step — recall (zero LLM) + one Qwen call to produce the final answer |
| `deploy/` | Alibaba Cloud Function Compute deployment (`fc_handler.py`, `s.yaml`) — see `deploy/README.md` |
| `benchmark_longmem.py`, `benchmark_qa.py`, `benchmark_bakeoff.py`, `benchmark_budget_sweep.py`, `benchmark_procedural_precision.py` | Benchmarks against BM25 and full-context baselines (see results below) |
| `docs/` | Interactive graph visualization (concept map + query trace), served live at [cogram.cora.zone](https://cogram.cora.zone) via GitHub Pages |

## Install as a package

Cogram is pip-installable straight from this repo — no code changes needed to use it inside your own agent:

```bash
pip install git+https://github.com/xqscora/cogram-qwen.git
```

Then from any Python agent loop — this part is zero LLM tokens, no API key needed:

```python
import os
os.environ["COGRAM_GRAPH"] = "path/to/swe_concept_graph.json"  # or your own graph
os.environ["ENGRAM_TRANSCRIPT_DIR"] = "path/to/your/transcripts"

import memory_api
result = memory_api.recommend("dacite default_factory dataclass bug")
# result["recommendations"] -> ranked file:line pointers with concept paths
# result["gaps"]            -> concepts the memory doesn't know yet
memory_api.end_turn("what actually happened this turn")  # learn from this turn, advance one tick
```

A `cogram-demo` console command is also installed (same as `python swe_recall_demo.py`).

## Qwen Cloud usage (the one place an LLM is called)

```bash
export DASHSCOPE_API_KEY=sk-...          # from Alibaba Cloud Model Studio
python qwen_agent.py "dacite default_factory dataclass bug"
```

This calls `memory_api.recommend()` (zero LLM tokens) and then one Qwen Cloud
`chat.completions.create()` (`qwen-plus`, OpenAI-compatible endpoint) to turn
the recalled file:line evidence into a grounded answer. See `qwen_agent.py`
for the full flow and `deploy/README.md` to run the same thing as an HTTP
function on Alibaba Cloud Function Compute.

## Try it

The dataset itself is **not** vendored in this repo — it's downloaded fresh from the source below, so nothing here is a private or pre-baked copy of anyone's data.

- **Data source**: [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) on Hugging Face (CC-BY-4.0). `fetch_swe.py` pulls a 600-row slice of *solved* trajectories directly from there — swap in any other slice/dataset if you want to run it on something else.

```bash
pip install -r requirements.txt   # tiktoken, jieba, pyarrow, requests
python fetch_swe.py               # downloads a slice of nebius/SWE-agent-trajectories (CC-BY 4.0) -> swe_slice.jsonl
python swe_to_transcripts.py      # converts it to line-based transcripts -> swe_corpus/
python extract_swe.py             # builds the concept graph, zero LLM tokens -> swe_concept_graph.json
python swe_recall_demo.py         # asks a new question, shows what it recalls and from where
```

`swe_slice.jsonl` and `swe_corpus/` are gitignored on purpose (they're just a local cache of the public dataset above, regenerated by the first two commands). `swe_concept_graph.json` — Cogram's own derived output, not the raw dataset — is committed so you can run the demo/benchmarks without re-downloading anything first.

`extract.py`, `recall.py`, and `memory_api.py` default to reading from `swe_corpus/` (the public data above). To point them at your own agent's transcripts instead, set `ENGRAM_TRANSCRIPT_DIR` to your transcript folder. (Internally the code still says "engram" in places — that's just the original working name; the project is now called Cogram.)

## Results, briefly

- 600 real SWE-agent trajectories, 184k lines, ~2.5M tokens → concept graph is 4.4x smaller than the raw text (including 32 level-1 "concept-of-concepts" motifs — recurring problem→fix clusters seen across distinct trajectories, not just repeated lines — see `CONCEPT_OF_CONCEPTS.md`), and costs 0 LLM tokens to build.
- Recalling a fix for a new problem costs ~150–400 tokens and points to the exact `file:line` of the original fix — about 7,000–16,000x smaller than re-reading the whole history.
- That routing is measured, not cherry-picked: 150 sampled queries, checking whether the top recall lands in the *same repo* as the bug report — **43.3% top-1 vs. a 2.9% random-chance baseline** (~15x over chance), zero LLM calls. See `benchmark_procedural_precision_results.md`, including a real bug (a boilerplate hub-word problem) we found and fixed while measuring it.
- On LongMemEval (ICLR 2025, a standard long-term conversational memory benchmark, not our home turf), a BM25-ranked version of this graph ties the best lexical retriever on accuracy at ~45x fewer tokens. Full numbers in `benchmark_longmem_results.md` and `benchmark_budget_sweep_results.md` — including where it does *not* win, we left that in.

## License

CC BY-NC-SA 4.0 — free to use, run, and modify for personal or research use, with attribution, as long as anything you build on top of it stays open under the same terms. Not for commercial use without asking first. See `LICENSE`.

## Author

Built by Cora Zeng ([@xqscora](https://github.com/xqscora)) for AGI Summit 2026 Hackathon.
