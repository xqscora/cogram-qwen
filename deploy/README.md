# Deploying Cogram MemoryAgent to Alibaba Cloud Function Compute

This is the Alibaba Cloud deployment for the **Global AI Hackathon Series
with Qwen Cloud — Track 1: MemoryAgent** submission.

## What actually runs where

| Layer | Where it runs | Uses an LLM? |
|---|---|---|
| Concept graph build (`extract_swe.py`, `graph_lib.py`) | Offline, one-time, anywhere | **No — zero LLM calls** |
| Memory recall (`memory_api.recommend`) | Inside the Function Compute function, per request | **No — zero LLM calls** |
| Answer generation (`qwen_agent.answer_with_memory`) | Inside the same function, per request | **Yes — one call to Qwen Cloud (DashScope)** |

Cogram's entire value proposition is that steps 1 and 2 cost 0 LLM tokens.
Only step 3 — turning a recalled fix into a natural-language answer — spends
a Qwen Cloud call, and it spends it on a ~150-400 token context instead of
re-feeding the raw trajectory history (2.5M+ tokens for the full corpus).

## One-time setup

1. **Alibaba Cloud account** — sign up at https://www.alibabacloud.com/
   (free trial credits are usually available; the hackathon also distributes
   Qwen Cloud vouchers separately, see the hackathon's Devpost page).
2. **Get a Model Studio / DashScope API key**:
   https://www.alibabacloud.com/help/en/model-studio/get-api-key
   ```bash
   export DASHSCOPE_API_KEY=sk-...
   ```
3. **Install Serverless Devs** (Alibaba's deployment CLI):
   ```bash
   npm install -g @serverless-devs/s
   s config add --AccessKeyID <your-access-key-id> --AccessKeySecret <your-access-key-secret>
   ```

## Deploy

```bash
cd deploy
s deploy
```

This creates a Function Compute HTTP function named `cogram-memoryagent` in
the `ap-southeast-1` region (Singapore — matches Qwen Cloud's international
DashScope endpoint). Serverless Devs prints the invocation URL after deploy,
e.g.:

```
https://<account-id>.ap-southeast-1.fc.aliyuncs.com/2016-08-15/proxy/cogram-memoryagent-app/cogram-memoryagent/
```

## Test it

```bash
curl -X POST "<the URL above>" \
  -H "Content-Type: application/json" \
  -d '{"query": "dacite default_factory dataclass bug"}'
```

Expected response shape:

```json
{
  "query": "dacite default_factory dataclass bug",
  "memory_tokens_used": 187,
  "activated_concepts": 6,
  "recalled_lines": [
    {"session": "swe_0381_...", "line_idx": 55, "text": "..."}
  ],
  "routes": [...],
  "answer": "Qwen's natural-language answer, grounded in the recalled lines above",
  "qwen_model": "qwen-plus"
}
```

## Why this satisfies "Proof of Alibaba Cloud Deployment"

- `deploy/fc_handler.py` is the real Function Compute entrypoint (WSGI
  handler per Alibaba Cloud's custom-runtime HTTP contract) — this file is
  what actually executes on Alibaba Cloud, not a mock.
- `qwen_agent.py` calls `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
  — Alibaba Cloud Model Studio's Qwen API — via the OpenAI-compatible SDK.
- `deploy/s.yaml` is the Serverless Devs config that provisions the Function
  Compute resource itself (`component: fc3`), not just an API call from a
  laptop.
