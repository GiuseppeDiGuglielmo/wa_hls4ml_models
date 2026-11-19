#!/usr/bin/env python3
"""Summarize active-learning replay curves: tables and one figure.

    python summarize.py --curves $SCRATCH/al_replay/A/A60/curves.jsonl \
        --ceiling $SCRATCH/al_replay/A/ceiling/curves.jsonl --out $SCRATCH/al_replay/A/A60

Prints, per rule (mean ± std over seeds):
  - Catapult-hours until R² first comes within each gap of the full-pool ceiling
    (ceiling mean minus gap), per test group and target;
  - the last warm-started round vs the cold retrain on the same labels;
and writes <out>/curves_A.png: R² against cumulative Catapult-hours (log x),
latency and area rows, one column per test group, ceiling band in gray.
"""
import argparse
import collections
import json
import os

import numpy as np

RULE_STYLE = {  # default dataviz palette slots 1-3; marker + dash as secondary encoding
    "random":    ("#2a78d6", "o", "-"),
    "coverage":  ("#eb6834", "s", "--"),
    "mcdropout": ("#1baf7a", "^", ":"),
    "mcdropout_hour": ("#eda100", "D", "-."),
}
TARGETS = ["LATENCY", "AREA"]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def curves(rows):
    """{(rule, seed): sorted list of warm rows} and {(rule, seed): cold row}."""
    warm, cold = collections.defaultdict(list), {}
    for r in rows:
        if r["round"] == "final_cold":
            cold[(r["rule"], r["seed"])] = r
        else:
            warm[(r["rule"], r["seed"])].append(r)
    for v in warm.values():
        v.sort(key=lambda r: r["round"])
    return warm, cold


def hours_to(rows, group, target, thr):
    for r in rows:
        if r["test"][group][target]["r2"] >= thr:
            return r["hours"]
    return np.nan


def fmt(vals):
    v = np.array(vals, float)
    ok = v[~np.isnan(v)]
    if not len(ok):
        return "not reached"
    s = f"{ok.mean():7.1f} ± {ok.std():5.1f} h"
    return s + (f"  ({len(ok)}/{len(v)} seeds)" if len(ok) < len(v) else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--curves", required=True)
    ap.add_argument("--ceiling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=["L3", "L6", "L10"])
    ap.add_argument("--title", default="Experiment A: pool L1+L2")
    ap.add_argument("--gaps", type=float, nargs="+", default=[0.05, 0.02, 0.01, 0.001])
    a = ap.parse_args()

    warm, cold = curves(load(a.curves))
    rules = [r for r in RULE_STYLE if any(k[0] == r for k in warm)]
    seeds = sorted({k[1] for k in warm})
    ceil = {}
    if a.ceiling and os.path.exists(a.ceiling):
        crow = [r for r in load(a.ceiling) if r["round"] == 0]
        for g in a.groups:
            for t in TARGETS:
                v = [r["test"][g][t]["r2"] for r in crow]
                ceil[(g, t)] = (np.mean(v), np.std(v), len(v))
    print(f"rules {rules}, seeds {seeds}")
    if ceil:
        print("\nCeiling (full pool):")
        for (g, t), (m, s, n) in ceil.items():
            print(f"  {g:4s} {t:8s} R² {m:.4f} ± {s:.4f}  ({n} seeds)")

    for t in TARGETS if ceil else []:
        print(f"\n{t}: Catapult-hours until R² >= ceiling - gap (mean ± std over seeds)")
        for g in a.groups:
            for gap in a.gaps:
                thr = ceil[(g, t)][0] - gap
                line = "  ".join(f"{rule}: {fmt([hours_to(warm[(rule, s)], g, t, thr) for s in seeds if (rule, s) in warm])}"
                                 for rule in rules)
                print(f"  {g:4s} gap {gap:<5g} (R² >= {thr:.4f})   {line}")

    if any("by_group" in run[-1] for run in warm.values()):
        print("\nDepth mix of the labeled set at the last round (mean over seeds):")
        for rule in rules:
            last = [warm[(rule, s)][-1] for s in seeds if (rule, s) in warm]
            groups = list(last[0]["by_group"])
            mix = "  ".join(f"{g} {np.mean([r['by_group'][g] for r in last]):7.0f}" for g in groups)
            print(f"  {rule:15s} n {np.mean([r['n_labeled'] for r in last]):7.0f}  "
                  f"hours {np.mean([r['hours'] for r in last]):7.0f}  {mix}")

    if cold:
        print("\nWarm-started last round vs cold retrain on the same labels (R², mean over seeds):")
        for rule in rules:
            for g in a.groups:
                for t in TARGETS:
                    w = [warm[(rule, s)][-1]["test"][g][t]["r2"] for s in seeds if (rule, s) in cold]
                    c = [cold[(rule, s)]["test"][g][t]["r2"] for s in seeds if (rule, s) in cold]
                    if w:
                        print(f"  {rule:15s} {g:4s} {t:8s} warm {np.mean(w):.4f}  cold {np.mean(c):.4f}  "
                              f"diff {np.mean(c) - np.mean(w):+.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    fig, axes = plt.subplots(2, len(a.groups), figsize=(4.2 * len(a.groups), 7), sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for i, t in enumerate(TARGETS):
        for j, g in enumerate(a.groups):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            if (g, t) in ceil:
                m, s, _ = ceil[(g, t)]
                ax.axhspan(m - s, m + s, color=grid, lw=0, zorder=0)
                ax.axhline(m, color=muted, lw=1, zorder=1)
            for rule in rules:
                col, mk, ls = RULE_STYLE[rule]
                runs = [warm[(rule, s)] for s in seeds if (rule, s) in warm]
                n = min(len(r) for r in runs)
                h = np.mean([[r["hours"] for r in run[:n]] for run in runs], axis=0)
                y = np.array([[r["test"][g][t]["r2"] for r in run[:n]] for run in runs])
                ax.fill_between(h, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.15, lw=0)
                ax.plot(h, y.mean(0), color=col, ls=ls, lw=2, marker=mk, ms=4, markevery=max(1, n // 8), label=rule)
            ax.set_xscale("log")
            ax.grid(True, color=grid, lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(muted)
            ax.tick_params(colors=muted, labelsize=9)
            lo = min(0.75, ax.get_ylim()[0])
            ax.set_ylim(max(lo, 0.5), 1.0)
            ax.set_title(f"{t.lower()} · test {g}", color=ink, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel("test R²", color=ink)
            if i == 1:
                ax.set_xlabel("cumulative Catapult-hours (log)", color=ink)
    axes[0][0].legend(frameon=False, fontsize=9, labelcolor=ink, loc="lower right")
    fig.suptitle(f"{a.title}: {len(seeds)} seed{'s' if len(seeds) != 1 else ''}, mean ± std; gray = full-pool ceiling",
                 color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    png = os.path.join(a.out, "curves.png")
    fig.savefig(png, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nfigure: {png}")


if __name__ == "__main__":
    main()
