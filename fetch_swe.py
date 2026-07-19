"""Download SWE-agent-trajectories slice (successful fixes only) — no datasets/pandas."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pyarrow.parquet as pq

DATASET = "nebius/SWE-agent-trajectories"
LICENSE = "CC-BY-4.0"
TARGET_ROWS = 600
FIELDS = (
    "instance_id",
    "model_name",
    "target",
    "trajectory",
    "exit_status",
    "generated_patch",
    "eval_logs",
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "swe_slice.jsonl")


def hf_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "engram-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def list_parquet_shards() -> list[str]:
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/main?recursive=true"
    data = json.loads(hf_get(url).decode("utf-8"))
    paths = []
    for entry in data:
        path = entry.get("path", "")
        if path.endswith(".parquet"):
            paths.append(path)
    paths.sort()
    return paths


def download_shard(rel_path: str, dest: str) -> None:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{rel_path}"
    print(f"Downloading {rel_path} …")
    req = urllib.request.Request(url, headers={"User-Agent": "engram-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  → {dest} ({size_mb:.1f} MB)")


def row_to_dict(batch, idx: int) -> dict:
    rec = {}
    for col in FIELDS:
        if col not in batch.column_names:
            continue
        val = batch.column(col)[idx].as_py()
        rec[col] = val
    return rec


def collect_solved_from_parquet(path: str, need: int, collected: list) -> int:
    pf = pq.ParquetFile(path)
    print(f"Reading {path} ({pf.metadata.num_rows} rows, {pf.num_row_groups} row groups)")
    for rg in range(pf.num_row_groups):
        if len(collected) >= need:
            break
        batch = pf.read_row_group(rg)
        if "target" not in batch.column_names:
            print("  WARN: no 'target' column, skipping row group")
            continue
        targets = batch.column("target").to_pylist()
        for i, t in enumerate(targets):
            if len(collected) >= need:
                break
            if t is True:
                collected.append(row_to_dict(batch, i))
    return len(collected)


def main():
    shards = list_parquet_shards()
    if not shards:
        print("No parquet shards found.")
        sys.exit(1)
    print(f"Found {len(shards)} parquet shard(s); using up to 2 for {TARGET_ROWS} solved rows.")
    print(f"Dataset: {DATASET}  License: {LICENSE}")

    collected: list[dict] = []
    used_shards: list[str] = []

    with tempfile.TemporaryDirectory(prefix="swe_parquet_") as tmp:
        for shard in shards[:2]:
            local = os.path.join(tmp, os.path.basename(shard))
            download_shard(shard, local)
            used_shards.append(shard)
            collect_solved_from_parquet(local, TARGET_ROWS, collected)
            print(f"  Collected {len(collected)} solved rows so far")
            if len(collected) >= TARGET_ROWS:
                break

    collected = collected[:TARGET_ROWS]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in collected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n=== fetch_swe summary ===")
    print(f"Shards used: {used_shards}")
    print(f"Rows saved:  {len(collected)} (target=True)")
    print(f"Output:      {OUT_PATH}")
    print(f"License:     {LICENSE} (nebius/SWE-agent-trajectories)")


if __name__ == "__main__":
    main()
