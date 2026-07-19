"""Convert swe_slice.jsonl → swe_corpus/*_transcript.md for Engram extract."""
from __future__ import annotations

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SLICE_PATH = os.path.join(os.path.dirname(__file__), "swe_slice.jsonl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "swe_corpus")
MAX_LINES = 400
PATCH_LINES = 40
EVAL_LINES = 20

# Strip long base64 / hex blobs
B64_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
HEX_RE = re.compile(r"\b[0-9a-fA-F]{64,}\b")
ROLE_MAP = {
    "system": "system",
    "ai": "ai",
    "user": "user-observation",
    "assistant": "ai",
}


def sanitize_instance_id(iid: str) -> str:
    out = []
    for ch in iid:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s[:80] if s else "unknown"


def strip_blobs(text: str) -> str:
    text = B64_RE.sub("[base64-omitted]", text)
    text = HEX_RE.sub("[hex-omitted]", text)
    return text


def split_meaningful_lines(content: str) -> list[str]:
    """Split content into semantically useful lines for line-level pointers."""
    content = strip_blobs(content)
    lines: list[str] = []
    for raw in content.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # Long single lines: try to break on sentence / shell boundaries
        if len(line) > 240:
            parts = re.split(r"(?<=[.!?])\s+|(?<=\$)\s+|(?<=\\n)", line)
            for p in parts:
                p = p.strip()
                if p:
                    lines.append(p[:240])
        else:
            lines.append(line)
    return lines


def message_content(msg: dict) -> str:
    """SWE-agent rows use text / system_prompt, not always content."""
    for key in ("text", "content", "system_prompt"):
        val = msg.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            parts = []
            for block in val:
                if isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            val = "\n".join(parts)
        if str(val).strip():
            return str(val)
    return ""


def parse_trajectory(traj) -> list:
    if isinstance(traj, list):
        return traj
    if isinstance(traj, str):
        try:
            parsed = json.loads(traj)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def flatten_trajectory(traj) -> list[str]:
    messages = parse_trajectory(traj)
    if not messages:
        return [strip_blobs(str(traj)[:2000])]

    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = ROLE_MAP.get(str(msg.get("role", "unknown")).lower(), "unknown")
        content = message_content(msg)
        if not content.strip():
            continue
        # Skip boilerplate SWE-agent system prompt (same on every trajectory).
        if role == "system" and "SETTING: You are an autonomous programmer" in content:
            out.append("## [system]")
            out.append("SWE-agent shell interface (boilerplate omitted)")
            continue
        out.append(f"## [{role}]")
        out.extend(split_meaningful_lines(content))
    return out


def append_result_section(lines: list[str], row: dict) -> None:
    exit_status = row.get("exit_status", "")
    lines.append(f"RESULT exit_status: {exit_status}")

    patch = row.get("generated_patch") or ""
    if patch.strip():
        lines.append("PATCH:")
        for pl in patch.splitlines()[:PATCH_LINES]:
            lines.append(strip_blobs(pl.rstrip()))

    eval_logs = row.get("eval_logs") or ""
    if eval_logs.strip():
        lines.append("EVAL:")
        for el in eval_logs.splitlines()[:EVAL_LINES]:
            lines.append(strip_blobs(el.rstrip()))


def main():
    if not os.path.isfile(SLICE_PATH):
        print(f"Missing {SLICE_PATH} — run fetch_swe.py first.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    n_written = 0
    with open(SLICE_PATH, encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            iid = sanitize_instance_id(str(row.get("instance_id", f"row{idx}")))
            fname = f"swe_{idx:04d}_{iid}_transcript.md"
            fpath = os.path.join(OUT_DIR, fname)

            body_lines = flatten_trajectory(row.get("trajectory", ""))
            append_result_section(body_lines, row)
            body_lines = body_lines[:MAX_LINES]

            header = (
                f"# SWE trajectory {idx}\n"
                f"instance_id: {row.get('instance_id', '')}\n"
                f"model: {row.get('model_name', '')}\n"
                f"target: {row.get('target', '')}\n"
            )
            with open(fpath, "w", encoding="utf-8") as out:
                out.write(header)
                out.write("\n".join(body_lines))
                out.write("\n")
            n_written += 1

    print(f"Wrote {n_written} transcripts → {OUT_DIR}")


if __name__ == "__main__":
    main()
