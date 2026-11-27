"""Figures for the paper, from ../results/*.json only. Usage: python plots.py"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R, F = ROOT / "results", ROOT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 4,
                     "pdf.fonttype": 42})
COL = 3.45  # IEEE column width (in)


def load(n):
    p = R / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_e1(e1):
    cols = ["L0", "L1", "L2", "L3", "L0_heldout_fonts", "L1_heldout_fonts",
            "L2_heldout_fonts", "L3_heldout_fonts", "G1"]
    lab = ["L0", "L1", "L2", "L3", "L0*", "L1*", "L2*", "L3*", "G1"]
    M = np.zeros((4, len(cols)))
    for lv in range(4):
        rs = [r for r in e1 if r["train_level"] == lv]
        for j, c in enumerate(cols):
            M[lv, j] = np.mean([r["test"][c]["acc"] for r in rs])
    fig, ax = plt.subplots(figsize=(COL, 1.42))
    ax.grid(False)
    im = ax.imshow(M * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(len(cols)):
            v = M[i, j] * 100
            ax.text(j, i, f"{v:.0f}" if v >= 0.5 else "0", ha="center", va="center",
                    color="white" if v > 60 else INK, fontsize=6.5)
    ax.set_xticks(range(len(cols)), lab)
    ax.set_yticks(range(4), [f"train L{i}" for i in range(4)])
    ax.axvline(3.5, color="white", lw=2)
    ax.axvline(7.5, color="white", lw=2)
    ax.set_xlabel("test set (* = held-out fonts; G1 = other generator)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("exact-match acc. (%)")
    cb.outline.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(F / "e1_heatmap.pdf")
    return M


def fig_e1b_e2(e1b, e2):
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.65))
    ax = axs[0]
    g = defaultdict(list)
    for r in e1b:
        g[r["n_train"]].append(r["acc"])
    xs = sorted(g)
    m = [np.mean(g[x]) * 100 for x in xs]
    lo = [min(g[x]) * 100 for x in xs]
    hi = [max(g[x]) * 100 for x in xs]
    ax.fill_between(xs, lo, hi, color=C[0], alpha=0.18, lw=0)
    ax.plot(xs, m, marker=MK[0], color=C[0])
    ax.set_xscale("log")
    ax.set_xlabel("training labels (L3)")
    ax.set_ylabel("acc. (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) sample efficiency", fontsize=7.5)
    ax = axs[1]
    i = 0
    for src, tgt, nm in [("G2mix", "G1", "G2$\\to$G1"), ("G1", "G2L2_heldout", "G1$\\to$G2")]:
        for kind, ls in [("finetune", "-"), ("scratch", ":")]:
            g = defaultdict(list)
            for r in e2:
                if r["src"] == src and r.get(kind) is not None:
                    g[r["k"]].append(r[kind]["acc"])
            xs = sorted(g)
            if not xs:
                continue
            xp = [max(x, 30) * (1.06 if kind == "scratch" else 1.0) for x in xs]  # small offset
            m = np.array([np.mean(g[x]) * 100 for x in xs])
            lo = m - np.array([min(g[x]) * 100 for x in xs])
            hi = np.array([max(g[x]) * 100 for x in xs]) - m
            ax.errorbar(xp, m, yerr=[lo, hi], ls=ls, marker=MK[i // 2], color=C[i // 2],
                        elinewidth=0.7, capsize=1.5, capthick=0.7,
                        label=f"{nm} {'fine-tune' if kind == 'finetune' else 'scratch'}")
            i += 1
    ax.set_xscale("log")
    ax.set_xlabel("target labels $k$")
    ax.set_ylim(0, 100)
    ax.set_title("(b) cross-scheme adaptation", fontsize=7.5)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, loc="lower center", ncol=2, handlelength=2.2, frameon=False)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0.22, 1, 1))
    fig.savefig(F / "e1b_e2.pdf")


def fig_e3(e3):
    keys = [("surrogate", "defender surrogate (white-box)"), ("transfer", "independent solver (transfer)"),
            ("transfer_median", "+ 3$\\times$3 median filter"), ("transfer_quant", "+ 8-level quantisation"),
            ("adaptive_ft", "+ fine-tune on 1k adv. labels")]
    fig, ax = plt.subplots(figsize=(COL, 1.6))
    eps = sorted({r["eps255"] for r in e3})
    for i, (k, nm) in enumerate(keys):
        ys, xs = [], []
        for e in eps:
            v = [r[k]["acc"] for r in e3 if r["eps255"] == e and k in r]
            if not v and k == "adaptive_ft" and e == 0:
                v = [r["transfer"]["acc"] for r in e3 if r["eps255"] == 0]
            if v:
                xs.append(e)
                ys.append(np.mean(v) * 100)
        ax.plot(xs, ys, marker=MK[i], color=C[i], label=nm)
    ax.set_xlabel("$L_\\infty$ perturbation budget $\\epsilon$ ($\\times$1/255)")
    ax.set_ylabel("acc. on L2 (%)")
    ax.set_xticks(eps)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2)
    fig.tight_layout(pad=0.2)
    fig.savefig(F / "e3_adversarial.pdf")


def fig_e5(e5):
    main = {r["horizon"]: r for r in e5["main"]}
    names = list(main[20]["results"].keys())
    short = {n: n.replace("human (held-out)", "human, held-out") for n in names}
    fig, ax = plt.subplots(figsize=(COL, 2.35))
    hs = sorted(main)
    for i, n in enumerate(names):
        y = [main[h]["results"][n]["rate"] * 100 for h in hs]
        ax.plot(hs, y, marker=MK[i % 8], color=C[i % 8], label=short[n],
                ls="--" if n.startswith("human") else "-")
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.text(hs[-1], 2.5, "FPR target 1%", ha="right", fontsize=5.5, color=MUTED)
    ax.set_xticks(hs, [str(h) for h in hs])
    ax.set_xlabel("challenges observed per client key")
    ax.set_ylabel("keys flagged (%)")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
    fig.tight_layout(pad=0.2)
    fig.savefig(F / "e5_detection.pdf")


if __name__ == "__main__":
    F.mkdir(exist_ok=True)
    if (e1 := load("e1")):
        fig_e1(e1)
    if (e1b := load("e1b")) and (e2 := load("e2")):
        for r in (load("e2s") or []):  # small-k scratch baselines run separately
            e2.append(dict(src=r["src"], tgt=r["tgt"], k=r["k"], seed=r["seed"], finetune=None,
                           scratch=r["scratch"]))
        fig_e1b_e2(e1b, e2)
    elif e1b:
        fig_e1b_e2(e1b, [])
    if (e3 := load("e3")):
        fig_e3(e3)
    if False:  # v1 E5 figure retired; E5 v2+ is reported in Table IV (tab_e5.tex)
        fig_e5(e5)
    print("figures written")
