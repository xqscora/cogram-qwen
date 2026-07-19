# Cogram Demo

Video: [`docs/demo.mp4`](docs/demo.mp4) — ~35s, no audio, screen-recording style. GitHub renders it inline if you open that link.

It's a real run of the commands below, captured and played back as a terminal recording — not staged footage. Here's how to reproduce it yourself in under a minute:

```bash
pip install -r requirements.txt
python fetch_swe.py          # downloads a slice of nebius/SWE-agent-trajectories (CC-BY 4.0)
python swe_to_transcripts.py # converts it into line-based transcripts
python extract_swe.py        # builds the concept graph, zero LLM tokens
python swe_recall_demo.py    # asks a new question, shows what it recalls and from where
```

`swe_recall_demo.py` takes a new problem description, walks the concept graph built from 600 past SWE-agent trajectories, and returns:

- the most relevant past `file:line` a similar bug/fix was handled at
- the token cost of that recall (~150-280 tokens)
- a comparison against the token cost of replaying the full original trajectory (thousands to tens of thousands of tokens)

No LLM calls are used to build the graph or to do this recall — it's pure statistics (co-occurrence + novelty/surprise scoring). See `README.md` for the full writeup and `benchmark_longmem_results.md` / `benchmark_budget_sweep_results.md` for the conversational-memory benchmark numbers.
