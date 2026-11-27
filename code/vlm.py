"""E4: zero-shot solving with a locally served vision-language model (Ollama).

Usage: python vlm.py <model> <n_per_condition> [conditions...]
Writes ../results/e4_<model>.json incrementally (resumable).
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

import gen
from solver import cer

R = Path(__file__).resolve().parent.parent / "results"
URL = "http://127.0.0.1:11434/api/chat"
PROMPT = ("The image shows a distorted text CAPTCHA with 4 to 6 characters "
          "(uppercase letters A-Z and digits 2-9). Transcribe the characters from "
          "left to right. Reply with the characters only, no spaces, no explanation.")


def ask(model: str, png: bytes):
    body = {"model": model, "stream": False,
            "options": {"temperature": 0, "num_predict": 32, "seed": 0},
            "messages": [{"role": "user", "content": PROMPT,
                          "images": [base64.b64encode(png).decode()]}]}
    if "qwen" in model:  # thinking models: disable reasoning for a direct transcription
        body["think"] = False
    t = time.time()
    r = requests.post(URL, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"], time.time() - t


def norm(s: str) -> str:
    s = s.strip().splitlines()[0] if s.strip() else ""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def main():
    model, n = sys.argv[1], int(sys.argv[2])
    conds = sys.argv[3:] or ["L0", "L1", "L2", "L3", "G1"]
    out_p = R / f"e4_{model.replace(':', '_').replace('/', '_')}.json"
    rows = json.loads(out_p.read_text()) if out_p.exists() else []
    done = {(r["cond"], r["i"]) for r in rows}
    for c in conds:
        X, Y = (gen.make_set("G1", n, 920000) if c == "G1"
                else gen.make_set("G2", n, 900000 + int(c[1]), int(c[1])))
        for i in range(n):
            if (c, i) in done:
                continue
            raw, dt = ask(model, gen.to_png_bytes(X[i]))
            p = norm(raw)
            rows.append(dict(cond=c, i=i, truth=Y[i], raw=raw, pred=p, ok=p == Y[i],
                             cer=cer(p, Y[i]), latency_s=dt))
            if i % 25 == 0:
                out_p.write_text(json.dumps(rows, indent=0))
                acc = sum(r["ok"] for r in rows if r["cond"] == c) / max(1, sum(r["cond"] == c for r in rows))
                print(time.strftime("%H:%M:%S"), c, i, f"acc={acc:.3f}", f"{dt:.2f}s", repr(raw[:30]), flush=True)
        out_p.write_text(json.dumps(rows, indent=0))


if __name__ == "__main__":
    main()
