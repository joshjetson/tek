#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How long does each model take, and how much does it actually say?

Two numbers, because they trade against each other and the interesting
question is what the trade costs. Latency is what makes the box feel alive;
depth is what makes it worth talking to.

Also reports time-to-first-token under streaming, which is the number that
actually matters once replies are spoken as they arrive: a model that takes
12 s to finish but starts talking at 3 s feels far faster than one that
finishes in 8 s in silence.

    tools/brain_bench.py [--models haiku,sonnet,opus] [--runs 2]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tekdromo.voice import agent

QUESTIONS = [
    "what day of the week is it",
    "why is the sky blue",
    "tell me something interesting about the moon",
]


def run(model, prompt, stream=False):
    """-> (first_token_s, total_s, text)"""
    cmd = [agent._find_claude(), "-p", prompt,
           "--allowed-tools", "Read",
           "--no-session-persistence", "--disable-slash-commands",
           "--model", model]
    if stream:
        cmd += ["--output-format", "stream-json", "--verbose"]
    t0 = time.time()
    first = None
    out = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=agent.BRAIN_CWD)
    if stream:
        for line in p.stdout:
            try:
                msg = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if msg.get("type") == "assistant":
                for blk in msg.get("message", {}).get("content", []):
                    if blk.get("type") == "text" and blk.get("text", "").strip():
                        if first is None:
                            first = time.time() - t0
                        out.append(blk["text"])
        p.wait()
        text = "".join(out).strip()
    else:
        so, _ = p.communicate()
        text = so.decode("utf-8", "replace").strip()
        first = time.time() - t0
    return first, time.time() - t0, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="haiku,sonnet,opus")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--stream", action="store_true",
                    help="also measure time-to-first-token")
    a = ap.parse_args()

    b = agent.ClaudeBrain()
    print("%-8s %-34s %7s %7s %6s  %s"
          % ("model", "question", "first", "total", "chars", "reply"))
    print("-" * 110)
    for model in a.models.split(","):
        model = model.strip()
        for q in QUESTIONS:
            for _ in range(a.runs):
                prompt = b.build_prompt({"kind": "speech", "heard": q,
                                         "what": 'Someone spoke to you and '
                                                 'said: "%s"' % q})
                try:
                    first, total, text = run(model, prompt, a.stream)
                except Exception as e:
                    print("%-8s %-34s  FAILED %s" % (model, q[:34], e))
                    continue
                print("%-8s %-34s %6.2fs %6.2fs %6d  %s"
                      % (model, q[:34], first or -1, total, len(text),
                         text[:56].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
