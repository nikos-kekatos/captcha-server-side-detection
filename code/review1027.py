"""Runs for the external review of 2026-10-27 (reviews/REVIEW_2026-10-27_external.md).

A  realised-1% operating point: the full monitor P1-P6 calibrated at a sweep of nominal budgets
   below 1%, so that the realised held-out human rate reaches 1%; same splits, sigma and seeds as
   matched_fpr.py (n = 50, 20 splits, 1,000 evaluation streams per profile and split).
C  telemetry faults of a SIEM integration, at the nominal 1% calibration of P1-P6 and of P1-P5:
   missing events (20% dropped at random), delayed ingestion (exponential delay of mean 30 s added
   to each timestamp, events re-sorted), shared keys (2 or 5 people behind one key).
D  distinguishability: total-variation distance between the human and the attacker distributions
   of each per-key statistic at n = 50 (histogram estimate over 20,000 streams per profile). The TV
   distance of any statistic is a lower bound on the TV distance of the full streams.
Writes ../results/review1027_<parts>.json (one file per run, so A, C and D can run in parallel).
"""
from __future__ import annotations

import json
import sys

import numpy as np

import monitor as M
import run_monitor as RM
from matched_fpr import calibrate_subset

N = 50
SPLITS = 20
N_EVAL = 1000
BUDGETS_A = [0.002, 0.003, 0.004, 0.005, 0.006]
BUDGETS_A2 = [0.0005, 0.001, 0.0015, 0.002]
NCAL_A2 = 20000   # A2: 20,000 calibration streams, so that quantiles below 1/4,000 exist


def calibrate_big(human, n, fpr, rng):
    return M.calibrate(human, n, fpr, rng, n_streams=NCAL_A2)
KEEP = ("CRNN, direct", "VLM qwen3.5_9b, direct", "Fixed delay, human gap", "Farm relay, human gap",
        "Pooled mimic, full accuracy", "Person mimic, full accuracy")


class Faulty:
    """Wraps a profile and applies a telemetry fault to its stream."""

    def __init__(self, base, kind, name):
        self.base, self.kind, self.name = base, kind, name

    def stream(self, n, rng):
        if self.kind == "missing":
            t, L, c = self.base.stream(3 * n, rng)
            keep = rng.random(len(t)) >= 0.2
            return t[keep][:n], L[keep][:n], c[keep][:n]
        if self.kind == "delay":
            t, L, c = self.base.stream(n, rng)
            t2 = t + rng.exponential(30.0, len(t))
            o = np.argsort(t2, kind="stable")
            return t2[o], L[o], c[o]
        k = int(self.kind[-1])          # shared2 / shared5
        parts = [self.base.stream(n, rng) for _ in range(k)]
        t = np.concatenate([p[0] for p in parts])
        o = np.argsort(t, kind="stable")[:n]
        return (t[o], np.concatenate([p[1] for p in parts])[o], np.concatenate([p[2] for p in parts])[o])


def stats(p, rng, m):
    out = {k: [] for k in ("min_latency", "min_median", "min_cv", "max_rate", "correct", "sd_log_latency")}
    for _ in range(m):
        t, L, c = p.stream(N, rng)
        med, cv, cnt, acc, sd = M.signals(t, L, c, N)
        out["min_latency"].append(L.min()); out["min_median"].append(np.nanmin(med))
        out["min_cv"].append(np.nanmin(cv)); out["max_rate"].append(cnt.max())
        out["correct"].append(acc); out["sd_log_latency"].append(sd)
    return {k: np.array(v, float) for k, v in out.items()}


def tv(a, b, bins=60):
    if a.min() == a.max() == b.min() == b.max():
        return 0.0
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if np.all(np.mod(np.concatenate([a, b]), 1) == 0) and hi - lo < 400:
        edges = np.arange(lo - 0.5, hi + 1.5)
    else:
        edges = np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1))
        edges = np.unique(edges)
        edges[0] -= 1e-9; edges[-1] += 1e-9
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def main():
    parts = sys.argv[1:] or ["A", "C", "D"]  # also "A2"
    T = M.load_humans()
    crnn_acc, crnn_ms, e2acc, vlm = RM.measured()
    path = M.R / ("review1027_%s.json" % "".join(parts))
    res = {}
    if "A" in parts:
        agg = {}
        for sp in range(SPLITS):
            perm = np.random.default_rng(1000 + sp).permutation(len(T))
            Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
            sigma, sigma_test = M.within_sigma(Tcal), M.within_sigma(Ttest)
            for b in BUDGETS_A:
                th = calibrate_subset(RM.humans(Tcal, sigma), N, b, np.random.default_rng(10 * sp + 1), M.PROPS)
                rows = {"Human, held-out": RM.humans(Ttest, sigma_test)}
                for p in RM.bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    if p.name in KEEP:
                        rows[p.name] = p
                for name, p in rows.items():
                    r = M.run_profile(p, th, N, np.random.default_rng(10 * sp + 2 + (name != "Human, held-out")), N_EVAL)
                    a = agg.setdefault(f"{b}|{name}", [0, 0]); a[0] += r["k"]; a[1] += r["n"]
            print("A split", sp, flush=True)
        res["A"] = [dict(budget=float(k.split("|")[0]), profile=k.split("|")[1], k=v[0], n=v[1], rate=v[0] / v[1])
                    for k, v in agg.items()]
        path.write_text(json.dumps(res, indent=1))
        for b in BUDGETS_A:
            print(b, {x["profile"]: round(100 * x["rate"], 2) for x in res["A"] if x["budget"] == b})
    if "A2" in parts:
        agg = {}
        for sp in range(SPLITS):
            perm = np.random.default_rng(1000 + sp).permutation(len(T))
            Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
            sigma, sigma_test = M.within_sigma(Tcal), M.within_sigma(Ttest)
            for b in BUDGETS_A2:
                th = calibrate_big(RM.humans(Tcal, sigma), N, b, np.random.default_rng(10 * sp + 1))
                rows = {"Human, held-out": RM.humans(Ttest, sigma_test)}
                for p in RM.bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    if p.name in KEEP:
                        rows[p.name] = p
                for name, p in rows.items():
                    r = M.run_profile(p, th, N, np.random.default_rng(10 * sp + 2 + (name != "Human, held-out")), N_EVAL)
                    a = agg.setdefault(f"{b}|{name}", [0, 0]); a[0] += r["k"]; a[1] += r["n"]
            print("A2 split", sp, flush=True)
        res["A2"] = [dict(budget=float(k.split("|")[0]), profile=k.split("|")[1], k=v[0], n=v[1], rate=v[0] / v[1],
                          calibration_streams=NCAL_A2) for k, v in agg.items()]
        path.write_text(json.dumps(res, indent=1))
        for b in BUDGETS_A2:
            print(b, {x["profile"]: round(100 * x["rate"], 2) for x in res["A2"] if x["budget"] == b})
    if "C" in parts:
        agg = {}
        for sp in range(SPLITS):
            perm = np.random.default_rng(1000 + sp).permutation(len(T))
            Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
            sigma, sigma_test = M.within_sigma(Tcal), M.within_sigma(Ttest)
            for vname, props in (("P1-P6", M.PROPS), ("P1-P5", M.PROPS[:5])):
                th = calibrate_subset(RM.humans(Tcal, sigma), N, 0.01, np.random.default_rng(10 * sp + 1), props)
                base = {"Human, held-out": RM.humans(Ttest, sigma_test)}
                for p in RM.bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
                    if p.name in ("CRNN, direct", "Pooled mimic, full accuracy"):
                        base[p.name] = p
                for fault in ("none", "missing", "delay", "shared2", "shared5"):
                    for name, p in base.items():
                        if fault.startswith("shared") and name != "Human, held-out":
                            continue
                        q = p if fault == "none" else Faulty(p, fault, name)
                        r = M.run_profile(q, th, N, np.random.default_rng(10 * sp + 5 + (name != "Human, held-out")), N_EVAL)
                        a = agg.setdefault(f"{vname}|{fault}|{name}", [0, 0]); a[0] += r["k"]; a[1] += r["n"]
            print("C split", sp, flush=True)
        res["C"] = [dict(variant=k.split("|")[0], fault=k.split("|")[1], profile=k.split("|")[2], k=v[0], n=v[1],
                         rate=v[0] / v[1]) for k, v in agg.items()]
        path.write_text(json.dumps(res, indent=1))
        for x in res["C"]:
            print(x["variant"], x["fault"], x["profile"], round(100 * x["rate"], 2))
    if "D" in parts:
        perm = np.random.default_rng(1000).permutation(len(T))
        Tcal, Ttest = T[perm[: len(T) // 2]], T[perm[len(T) // 2:]]
        sigma = M.within_sigma(Tcal)
        th = M.calibrate(RM.humans(Tcal, sigma), N, 0.01, np.random.default_rng(1))
        h = stats(RM.humans(Tcal, sigma), np.random.default_rng(21), 20000)
        out = {}
        for p in RM.bots(Tcal, sigma, crnn_acc, crnn_ms, e2acc, vlm, th):
            s = stats(p, np.random.default_rng(22), 20000)
            out[p.name] = {k: round(tv(h[k], s[k]), 4) for k in h}
            print(p.name, out[p.name], flush=True)
        # reference: two independent human samples, the estimation noise floor of the TV estimate
        h2 = stats(RM.humans(Tcal, sigma), np.random.default_rng(23), 20000)
        out["Human (second sample)"] = {k: round(tv(h[k], h2[k]), 4) for k in h}
        print("noise floor", out["Human (second sample)"])
        res["D"] = out
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
