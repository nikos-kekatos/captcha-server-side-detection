"""E5 matched-FPR comparison (external review 2026-10-21, paper #8 item 3).

Question: does the dispersion bound P6 add detection, or does it only spend more of the
false-positive budget? Both monitors (P1-P6, and P1-P5 without P6) are calibrated at a sweep
of nominal budgets with the same splits, per-split sigma and seeds as run_monitor.py. Each
point records the realised held-out human false-positive rate and the detection of the
attacker profiles, pooled over splits, at n = 50. Writes ../results/e5_matched.json.
"""
from __future__ import annotations

import json

import numpy as np

import monitor as M
import run_monitor as RM

N = 50
BUDGETS = [0.01, 0.02, 0.04, 0.08, 0.16]
SPLITS = 20
N_EVAL = 1000
OFF = dict(tau1=-1.0, tau2=-1.0, c3=-1.0, k4=10 ** 9, a5=10 ** 9, s6=1e9)


def calibrate_subset(human, n, fpr, rng, props):
    th = M.calibrate(human, n, fpr * len(M.PROPS) / len(props), rng)  # q = fpr / len(props)
    key = {"P1": "tau1", "P2": "tau2", "P3": "c3", "P4": "k4", "P5": "a5", "P6": "s6"}
    for p in M.PROPS:
        if p not in props:
            th[key[p]] = OFF[key[p]]
    return th


def main():
    T = M.load_humans()
    crnn_acc, crnn_ms, e2acc, vlm = RM.measured()
    import sys
    variants = {"P1-P6": M.PROPS, "P1-P5": ["P1", "P2", "P3", "P4", "P5"], "P4": ["P4"]}
    if len(sys.argv) > 1:  # run a subset and merge into the existing file
        variants = {k: v for k, v in variants.items() if k in sys.argv[1:]}
    agg = {}
    for sp in range(SPLITS):
        perm = np.random.default_rng(1000 + sp).permutation(len(T))
        Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
        sigma, sigma_test = M.within_sigma(Tcal), M.within_sigma(Ttest)
        for vname, props in variants.items():
            for b in BUDGETS:
                th = calibrate_subset(RM.humans(Tcal, sigma), N, b, np.random.default_rng(10 * sp + 1), props)
                rows = {"Human, held-out": RM.humans(Ttest, sigma_test)}
                for p in RM.bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    if p.name in ("Pooled mimic, full accuracy", "Farm relay, human gap",
                                  "Fixed delay, human gap", "Person mimic, full accuracy"):
                        rows[p.name] = p
                for name, p in rows.items():
                    r = M.run_profile(p, th, N, np.random.default_rng(10 * sp + 2 + (name != "Human, held-out")), N_EVAL)
                    a = agg.setdefault((vname, b, name), [0, 0])
                    a[0] += r["k"]
                    a[1] += r["n"]
        print("split", sp, flush=True)
    out = [dict(variant=v, budget=b, profile=name, k=k, n=n, rate=k / n)
           for (v, b, name), (k, n) in agg.items()]
    old = json.loads((M.R / "e5_matched.json").read_text()) if (M.R / "e5_matched.json").exists() else []
    keep = [x for x in old if x["variant"] not in variants]
    (M.R / "e5_matched.json").write_text(json.dumps(keep + out, indent=1))
    for v in variants:
        for b in BUDGETS:
            r = {x["profile"]: round(100 * x["rate"], 1) for x in out if x["variant"] == v and x["budget"] == b}
            print(v, b, r)


if __name__ == "__main__":
    main()
