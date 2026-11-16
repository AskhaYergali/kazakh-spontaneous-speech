#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
supp_figures.py -- Supplementary figures.

  S1  corpus-weighted precision against the volume selected, for the hand-set
      score and for the logistic regression of Section 5.3, plus the volume
      selected as a function of the threshold
  S2  distribution of the spontaneity score S over the corpus, by subcorpus
  S3  per-session word error rate, spontaneous against session-matched control,
      for both Kazakh-adapted models

    python3 supp_figures.py --manifest manifest/manifest.csv.gz \
        --validation-dir annotation --runs-dir results --out supp/
"""
import argparse, csv, gzip, math, os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito; shape repeats the information carried by hue
STYLE = {"parliament": ("#0072B2", "o", "parliament"),
         "radio":      ("#D55E00", "s", "radio"),
         "podcasts":   ("#009E73", "^", "podcasts"),
         "talkshow":   ("#CC79A7", "D", "talk shows"),
         "tv_news":    ("#56B4E9", "v", "TV news"),
         "crowdsourced": ("#E69F00", "P", "crowdsourced")}
INK, GRID, MUTED = "#333333", "#e8e8e8", "#777777"
FEATURES = ["strong_filler_per_1000", "weak_filler_per_1000", "discourse_per_1000",
            "repair_per_1000", "repetition_per_1000", "russian_mixed_per_1000"]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#444444", labelsize=9, length=3)


def load_validation(vdir, manifest):
    """ground truth, stratum weight, score and the six live features per item"""
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(vdir, "validation_key.csv"), encoding="utf-8"))}
    g = lambda r: (r.get("ОЦЕНКА") or "").strip().upper()
    l1 = {r["id"]: g(r) for r in csv.DictReader(open(os.path.join(vdir, "validation_listener_1.csv"), encoding="utf-8"))}
    l2 = {r["id"]: g(r) for r in csv.DictReader(open(os.path.join(vdir, "validation_listener_2.csv"), encoding="utf-8"))}
    adj = {r["id"]: r["СОГЛАСОВАННАЯ"].strip().upper()
           for r in csv.DictReader(open(os.path.join(vdir, "validation_disagreements.csv"), encoding="utf-8"))}
    want = {r["record_id"]: i for i, r in key.items() if "record_id" in r}
    feats = {}
    with gzip.open(manifest, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = want.get(r["record_id"])
            if i is not None:
                feats[i] = [float(r[c]) for c in FEATURES]
    # Horvitz-Thompson weight: stratum size in the released manifest divided by
    # the number of items drawn from that stratum that proved DECIDABLE by ear.
    # The nine undecidable items are dropped and their strata reweighted.
    import collections
    N = collections.Counter()
    with gzip.open(manifest, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            N[(r["domain"].split("/")[-1], r["label"])] += 1
    import os as _os
    p3 = _os.path.join(vdir, "validation_listener_3.csv")
    l3 = {r["id"]: g(r) for r in csv.DictReader(open(p3, encoding="utf-8-sig"))} \
         if _os.path.exists(p3) else {}
    decidable = []
    for i, r in key.items():
        votes = [v for v in (l1.get(i, ""), l2.get(i, ""), l3.get(i, "")) if v in ("S", "P")]
        t = ""
        if votes:
            top, n = collections.Counter(votes).most_common(1)[0]
            if n >= 2 and n > len(votes) - n:
                t = top
        if t in ("S", "P") and i in feats:
            decidable.append((i, r, 1 if t == "S" else 0))
    ndec = collections.Counter(r["stratum"] for _, r, _ in decidable)
    items = []
    for i, r, y in decidable:
        sub, lab = r["stratum"].split("|")
        items.append({"id": i, "y": y, "w": N[(sub, lab)] / ndec[r["stratum"]],
                      "s": float(r["score"]), "x": feats[i], "stratum": r["stratum"]})
    return items


def oof_probabilities(items, folds=5, seed=20261014, weighted=True):
    """out-of-fold predictions of the logistic regression of Section 5.3.

    weighted=False trains without the stratum weights, which is the fit the
    unweighted AUC of Section 5.3 refers to; weighted=True trains with them and
    is the fit evaluated against the corpus."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception:
        return None
    X = np.array([it["x"] for it in items], float)
    y = np.array([it["y"] for it in items], int)
    w = np.array([it["w"] for it in items], float)
    p = np.zeros(len(items))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = LogisticRegression(max_iter=2000, C=1.0)
        m.fit(X[tr], y[tr], sample_weight=(w[tr] if weighted else None))
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def report_auc(items, seeds=40):
    """reproduce the four AUCs of Section 5.3 and print them"""
    try:
        import numpy as np
        from sklearn.metrics import roc_auc_score
    except Exception:
        return
    y = np.array([it["y"] for it in items]); w = np.array([it["w"] for it in items])
    s = np.array([it["s"] for it in items])
    a = {"unw/unw": [], "wt/unw": [], "wt/wt": []}
    for sd in range(seeds):
        pu = oof_probabilities(items, seed=sd, weighted=False)
        pw = oof_probabilities(items, seed=sd, weighted=True)
        a["unw/unw"].append(roc_auc_score(y, pu))
        a["wt/unw"].append(roc_auc_score(y, pu, sample_weight=w))
        a["wt/wt"].append(roc_auc_score(y, pw, sample_weight=w))
    print("Section 5.3, out-of-fold AUC over %d fold splits (mean [min, max])" % seeds)
    print("  logistic, trained unweighted, evaluated unweighted : %.3f [%.3f, %.3f]"
          % (np.mean(a["unw/unw"]), min(a["unw/unw"]), max(a["unw/unw"])))
    print("  logistic, trained unweighted, evaluated weighted   : %.3f [%.3f, %.3f]"
          % (np.mean(a["wt/unw"]), min(a["wt/unw"]), max(a["wt/unw"])))
    print("  logistic, trained weighted,   evaluated weighted   : %.3f [%.3f, %.3f]"
          % (np.mean(a["wt/wt"]), min(a["wt/wt"]), max(a["wt/wt"])))
    print("  hand-set score: unweighted %.3f, weighted %.3f (no fold split involved)"
          % (roc_auc_score(y, s), roc_auc_score(y, s, sample_weight=w)))
    print("  (weighted = stratum weights as sample weights, the weights used throughout)")


HAND_WEIGHTS = {"strong_filler_per_1000": 0.06, "weak_filler_per_1000": 0.02,
                "discourse_per_1000": 0.015, "repair_per_1000": 0.04,
                "repetition_per_1000": 0.025, "russian_mixed_per_1000": 0.01}


def report_weights(items, lexicons=None):
    """compare the hand-set weights with the fitted coefficients (Section 5.3).

    The six features are per-1000-token rates with standard deviations that
    differ by almost an order of magnitude, so the comparison is only defined
    once a scale is fixed, and the two coherent choices disagree:

      raw scale    both sets as they multiply the per-1000 counters
      per SD       both sets multiplied by the standard deviation of their
                   feature, i.e. the effect of a one-SD change

    The paper quotes the PER-SD figure, because cross-feature statements such
    as "the same dominant feature" are meaningful only on that scale. Mixing
    the scales -- raw hand weights against standardized coefficients -- is not
    a correlation between comparable quantities; it is printed here only
    because an earlier version of the paper reported it.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return
    hw = HAND_WEIGHTS
    if lexicons:
        import json
        key = {"strong_fillers": "strong_filler_per_1000", "weak_fillers": "weak_filler_per_1000",
               "discourse_markers": "discourse_per_1000", "repair_phrases": "repair_per_1000",
               "repetitions": "repetition_per_1000", "russian_mixed": "russian_mixed_per_1000"}
        w = json.load(open(lexicons, encoding="utf-8"))["weights"]
        hw = {key[k]: v for k, v in w.items() if k in key}
    hand = np.array([hw[f] for f in FEATURES], float)
    X = np.array([it["x"] for it in items], float)
    y = np.array([it["y"] for it in items], int)
    sw = np.array([it["w"] for it in items], float)
    sd = X.std(0); sd = np.where(sd == 0, 1.0, sd)
    Z = (X - X.mean(0)) / sd
    b_raw = LogisticRegression(max_iter=2000, C=1.0).fit(X, y, sample_weight=sw).coef_[0]
    b_sd = LogisticRegression(max_iter=2000, C=1.0).fit(Z, y, sample_weight=sw).coef_[0]
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print("Section 5.3, hand-set weights against the corpus-weighted coefficients")
    print("  %-26s %9s %9s %9s %9s" % ("feature", "hand", "hand/SD", "fit", "fit/SD"))
    for f, h, a, b in zip(FEATURES, hand, b_raw, b_sd):
        print("  %-26s %9.3f %9.3f %+9.4f %+9.4f" % (f, h, h * sd[FEATURES.index(f)], a, b))
    print("  r, raw scale   (both as they multiply the counters) : %.3f" % r(hand, b_raw))
    print("  r, per SD      (both per standard deviation)        : %.3f   <- quoted in the paper"
          % r(hand * sd, b_sd))
    print("  r, mixed scale (raw hand vs standardized fit)       : %.3f   (not comparable; "
          "reported in the submitted version)" % r(hand, b_sd))
    print("  dominant feature: hand/SD = %s, fit/SD = %s; all fitted signs positive: %s"
          % (FEATURES[int((hand * sd).argmax())], FEATURES[int(b_sd.argmax())],
             bool(np.all(b_sd > 0))))


def curve(items, values):
    """Weighted precision against weighted selected volume, taking the items in
    descending order of `values`.  Ranking rather than thresholding, so that a
    ranker whose top item carries a large stratum weight still produces points
    at small volumes."""
    order = sorted(range(len(items)), key=lambda k: -values[k])
    out, tp, n = [], 0.0, 0.0
    for k in order:
        n += items[k]["w"]
        tp += items[k]["w"] * items[k]["y"]
        out.append((n, tp / n))
    return out


def fig_s1(items, out, seed=20261014):
    s_vals = [it["s"] for it in items]
    cs = curve(items, s_vals)

    vol2 = sum(it["w"] for it in items if it["s"] >= 3.0)

    def at(c, target):
        hit = [p for n, p in c if n >= target]
        return hit[0] if hit else float("nan")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 6.2))
    ax1.axvline(vol2, color=MUTED, lw=0.9, ls=":", zorder=0)
    series = [("hand-set score S", cs, "#0072B2", "-", 2.0, "o")]
    for lab, wtd, col, mk in (("logistic fit, trained unweighted", False, "#D55E00", "s"),
                              ("logistic fit, trained weighted", True, "#009E73", "^")):
        p_oof = oof_probabilities(items, seed=seed, weighted=wtd)
        if p_oof is not None:
            series.append((lab, curve(items, list(p_oof)), col, "--", 1.8, mk))
    for lab, c, col, ls, lw, mk in series:
        # the curve is a step function: selecting any volume in (n[k-1], n[k]] takes
        # exactly the first k ranked items, so the precision there is p[k]
        ax1.plot([v for v, _ in c], [p for _, p in c], lw=lw, ls=ls, color=col,
                 label=lab, drawstyle="steps-pre")
        p2 = at(c, vol2)
        if lab.startswith("hand-set"):
            ax1.scatter([vol2], [p2], s=52, facecolor=col, edgecolor="white", linewidth=1.3,
                        marker=mk, zorder=5)
            ax1.annotate("%.2f" % p2, (vol2, p2), textcoords="offset points", xytext=(9, -3),
                         fontsize=8.5, color=col, fontweight="bold")
    ax1.annotate("volume selected at θ₂", (vol2, 1.03), textcoords="offset points",
                 xytext=(-5, 0), ha="right", fontsize=8.5, color=INK)
    ax1.set_xscale("log")
    ax1.set_xlabel("utterances selected from the corpus (extrapolated, log scale)")
    ax1.set_ylabel("corpus-weighted precision")
    ax1.set_ylim(0, 1.12)
    style(ax1); ax1.legend(frameon=False, fontsize=8, loc=(0.02, 0.10), labelcolor=INK)

    smax = max(s_vals)
    grid_s = [x / 20 for x in range(0, int(smax * 20) + 1)]
    vols = [sum(it["w"] for it in items if it["s"] >= t) for t in grid_s]
    ax2.plot(grid_s, vols, lw=2, color="#0072B2")
    for th, lab in ((3.0, "θ₂"), (5.0, "θ₁")):
        ax2.axvline(th, color=MUTED, lw=1.0, ls=":")
        ax2.annotate(lab, (th, 3.2e5), ha="center", fontsize=9, color=INK)
    ax2.set_yscale("log"); ax2.set_xlim(0, min(smax, 12))
    ax2.set_xlabel("threshold on the spontaneity score S")
    ax2.set_ylabel("utterances selected (log scale)")
    style(ax2)
    fig.tight_layout(); fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def fig_s2(manifest, out):
    buckets = {}
    with gzip.open(manifest, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dom = r["domain"].split("/")[-1]
            b = min(int(float(r["score"]) * 2), 39)
            buckets.setdefault(dom, [0] * 40)[b] += 1
    x = [i / 2 for i in range(40)]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for dom in [d for d in STYLE if d in buckets]:
        col, _, lab = STYLE[dom]
        ax.step(x, [max(v, 0.6) for v in buckets[dom]], where="post", lw=1.8, color=col, label=lab)
    for th, lab in ((3.0, "θ₂"), (5.0, "θ₁")):
        ax.axvline(th, color=MUTED, lw=1.0, ls=":")
        ax.annotate(lab, (th, 2.6e5), ha="center", fontsize=9, color=INK)
    ax.set_yscale("log"); ax.set_xlim(0, 20); ax.set_ylim(0.6, 5e5)
    ax.set_xlabel("spontaneity score S"); ax.set_ylabel("utterances (log scale)")
    style(ax); ax.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print("wrote", out)


def fig_s3(runs_dir, models, out):
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_sets import load, tally, rate, session_of
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.5), sharex=True, sharey=True)
    if len(models) == 1: axes = [axes]
    lim, doms = 0, set()
    panels = []
    for m, _ in models:
        sp = load(os.path.join(runs_dir, m + "__spont_eval.csv"))
        pr = load(os.path.join(runs_dir, m + "__prep_eval_matched.csv"))
        ts, tp = tally(sp), tally(pr)
        dom = {session_of(r["record_id"]): r.get("domain", "NA") for r in sp}
        pts = []
        for s in sorted(set(ts) & set(tp)):
            a, b = rate(tp, [s], "a"), rate(ts, [s], "a")
            pts.append((a, b, ts[s]["a"][1], dom.get(s, "")))
            lim = max(lim, a, b); doms.add(dom.get(s, ""))
        panels.append(pts)
    lim = 5 * math.ceil(lim / 5) + 2
    for ax, pts, (_, label) in zip(axes, panels, models):
        ax.plot([0, lim], [0, lim], color="#9a9a9a", lw=1.0, ls="--", zorder=1)
        for a, b, n, dm in pts:
            col, mk, _ = STYLE.get(dm, ("#555555", "o", dm))
            ax.scatter([a], [b], s=12 * math.sqrt(max(n, 1)), facecolor=col,
                       edgecolor="white", linewidth=0.8, marker=mk, zorder=3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(label, fontsize=9.5, color=INK, pad=6)
        ax.set_xlabel("session-matched control WER, %")
        style(ax)
    axes[0].set_ylabel("spontaneous WER, %")
    present = [d for d in STYLE if d in doms]
    axes[0].legend(handles=[Line2D([], [], marker=STYLE[d][1], color="none",
                                   markerfacecolor=STYLE[d][0], markeredgecolor="white",
                                   markersize=7, label=STYLE[d][2]) for d in present],
                   loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(); fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--validation-dir", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", default="supp")
    ap.add_argument("--lexicons", default=None,
                    help="filter/lexicons.json; read the hand-set weights from it")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    items = load_validation(a.validation_dir, a.manifest)
    print("validation items with features:", len(items))
    report_auc(items)
    report_weights(items, a.lexicons)
    fig_s1(items, os.path.join(a.out, "FigureS1.png"))
    print("corrected weights in use: N(manifest stratum) / decidable items drawn")
    fig_s2(a.manifest, os.path.join(a.out, "FigureS2.png"))
    fig_s3(a.runs_dir, [("abilmansplus__whisper-turbo-ksc2", "whisper-turbo-ksc2"),
                        ("abilmansplus__whisper-turbo-kaz-rus-v1", "whisper-turbo-kaz-rus-v1")],
           os.path.join(a.out, "FigureS3.png"))
