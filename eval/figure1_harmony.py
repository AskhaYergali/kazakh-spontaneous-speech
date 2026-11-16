#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure1_harmony.py -- Figure 1 of Section 6.3.

Share of the front (і-) form of filled pauses per recording session, reference
transcripts against recognizer output, for the sessions carrying at least 15
fillers on each side.

Input is the <out>_sessions.csv written by harmony_fillers.py.

    python3 figure1_harmony.py --sessions out/harmony_sessions.csv \
                               --out Figure1.png

Design notes.  Subcorpus is encoded twice, by hue and by marker shape, so the
figure survives grayscale printing and colour-vision deficiency; the palette is
Okabe-Ito.  Error bars are Wilson 95 % intervals on both axes.  Point area is
proportional to the number of reference fillers, so small cells read as small.
"""

import argparse
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito, colour-vision-deficiency safe; shape carries the same information
STYLE = {
    "parliament": ("#0072B2", "o"),
    "radio":      ("#D55E00", "s"),
    "podcasts":   ("#009E73", "^"),
    "talkshow":   ("#CC79A7", "D"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True, help="<out>_sessions.csv from harmony_fillers.py")
    ap.add_argument("--out", default="Figure1.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.sessions, encoding="utf-8"))
            if r["in_correlation"] == "1"]
    if not rows:
        raise SystemExit("no sessions with in_correlation=1")

    fig, ax = plt.subplots(figsize=(5.1, 4.6))
    ax.plot([0, 100], [0, 100], color="#9a9a9a", lw=1.0, ls="--", zorder=1)

    for r in rows:
        col, mark = STYLE.get(r["subcorpus"], ("#555555", "o"))
        x, y = 100 * float(r["rate_reference"]), 100 * float(r["rate_model"])
        xerr = [[x - 100 * float(r["wilson_lo_reference"])],
                [100 * float(r["wilson_hi_reference"]) - x]]
        yerr = [[y - 100 * float(r["wilson_lo_model"])],
                [100 * float(r["wilson_hi_model"]) - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none",
                    ecolor=col, elinewidth=0.9, capsize=1.8, alpha=0.55, zorder=2)
        ax.scatter([x], [y], s=14 * math.sqrt(int(r["n_reference"])),
                   facecolor=col, edgecolor="white", linewidth=0.8,
                   marker=mark, zorder=3)

    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    ax.set_xlabel("Front (і-) form in the reference transcripts, % of fillers")
    ax.set_ylabel("Front (і-) form in the model output, % of fillers")
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, color="#e8e8e8", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#444444", labelsize=9, length=3)

    present = [k for k in STYLE if any(r["subcorpus"] == k for r in rows)]
    ax.legend(handles=[Line2D([], [], marker=STYLE[k][1], color="none",
                              markerfacecolor=STYLE[k][0], markeredgecolor="white",
                              markersize=7, label=k) for k in present],
              loc="upper left", frameon=False, fontsize=8.5, handletextpad=0.4,
              borderaxespad=0.3, labelcolor="#333333")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("wrote %s  (%d sessions)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
