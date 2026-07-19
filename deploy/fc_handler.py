"""Alibaba Cloud Function Compute (FC) HTTP handler for the Cogram MemoryAgent.

This is the file that satisfies the hackathon's "Proof of Alibaba Cloud
Deployment" requirement: it is the actual backend entrypoint that runs on
Alibaba Cloud Function Compute and calls Qwen Cloud (DashScope) — an
Alibaba Cloud AI service — for the generation step, while the memory layer
itself (graph_lib / memory_api) stays zero-LLM and runs locally inside the
same function.

Deploy with Serverless Devs (see deploy/s.yaml in this directory):
    npm install -g @serverless-devs/s
    s config add --AccessKeyID <your-key> --AccessKeySecret <your-secret>
    cd deploy && s deploy

Function Compute's HTTP trigger calls `handler(environ, start_response)`
(WSGI-style) for a plain HTTP function, per Alibaba Cloud's custom runtime
contract: https://www.alibabacloud.com/help/en/functioncompute/
"""
from __future__ import annotations

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_agent import answer_with_memory  # noqa: E402


def handler(environ, start_response):
    """WSGI entrypoint for Alibaba Cloud Function Compute's custom HTTP runtime.

    Expects a POST body of {"query": "<bug report / question text>"} and
    returns the Cogram recall + Qwen-generated answer as JSON.
    """
    try:
        request_method = environ.get("REQUEST_METHOD", "GET")
        if request_method != "POST":
            body = json.dumps(
                {
                    "service": "cogram-memoryagent",
                    "status": "ready",
                    "usage": 'POST {"query": "..."}',
                }
            ).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json")])
            return [body]

        try:
            length = int(environ.get("CONTENT_LENGTH", 0) or 0)
        except ValueError:
            length = 0
        raw = environ["wsgi.input"].read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        query = payload.get("query", "")
        budget = int(payload.get("budget_tokens", 400))

        if not query:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error": "missing 'query'"}).encode("utf-8")]

        result = answer_with_memory(query, budget_tokens=budget)
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(result, ensure_ascii=False).encode("utf-8")]

    except Exception as e:  # noqa: BLE001 — FC needs a JSON error, not a raw 500
        start_response("500 Internal Server Error", [("Content-Type", "application/json")])
        err = {"error": str(e), "trace": traceback.format_exc()}
        return [json.dumps(err).encode("utf-8")]
