#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation.py -- Table 7: ablation of the spontaneity metric.

Rescores the whole corpus from the counters of the released manifest under ten
variants of the metric, and evaluates each against the validation sample with
the stratum weights that map the sample back to the corpus. The ground truth is
the majority of the listeners: a label is assigned only where at least two of
them concur, which is the rule of Section 5.2.

Variants: the baseline as released; each of the six live counters zeroed in
turn; equal weights on all six; the floor N0 on the denominator removed; and
strong fillers alone. The eleventh row of Table 7, the filter without the
reduplication stop-list, cannot be obtained by rescoring the manifest, because
the stop-list acts while the repetitions are counted; eval/ablation_stoplist.py
runs the released filter twice to produce it.

Rescoring reproduces the released label for 485,914 of the 485,950 rows. The 36
exceptions lie exactly on the strict threshold, where the floating-point sum of
the original run falls marginally below 5.0 while this recomputation reaches
it, so the rescored baseline selects 8,573 utterances where the released filter
selects 8,540. Table 7 reports the filter's own figure in the baseline row and
the rescored figures in the variant rows; the offset is immaterial beside the
differences between variants, and none of the 36 belongs to a released set.

    python3 ablation.py --release . [--out ablation.csv]
"""
import argparse, collections, csv, gzip, json, os

COUNTERS = ["strong_fillers", "weak_fillers", "discourse_markers", "repair_phrases",
            "repetitions", "russian_mixed", "non_speech", "cutoffs", "ellipsis"]
LABEL = {"strong_fillers": "without strong fillers",
         "weak_fillers": "without weak fillers",
         "discourse_markers": "without discourse markers",
         "repair_phrases": "without repair phrases",
         "repetitions": "without repetitions",
         "russian_mixed": "without Russian material"}


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def ground_truth(release):
    """majority of the listeners, a label only where at least two concur"""
    A = os.path.join(release, "annotation")
    g = lambda r: (r.get("ОЦЕНКА") or "").strip().upper()
    key = {r["id"]: r for r in read(os.path.join(A, "validation_key.csv"))}
    votes = collections.defaultdict(list)
    for n in (1, 2, 3):
        p = os.path.join(A, "validation_listener_%d.csv" % n)
        if not os.path.exists(p):
            continue
        for r in read(p):
            if g(r) in ("S", "P"):
                votes[r["id"]].append(g(r))
    truth = {}
    for i, v in votes.items():
        top, n = collections.Counter(v).most_common(1)[0]
        if n >= 2 and n > len(v) - n:
            truth[i] = top
    return key, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=".")
    ap.add_argument("--out", default="ablation.csv")
    ap.add_argument("--theta1", type=float, default=5.0)
    ap.add_argument("--min-tokens", type=int, default=6)
    a = ap.parse_args()

    lex = json.load(open(os.path.join(a.release, "filter/lexicons.json"), encoding="utf-8"))
    W = dict(lex["weights"])
    N0 = int(lex["normalization"]["min_denominator_tokens"])
    man = read(os.path.join(a.release, "manifest/manifest.csv.gz"))
    live = [c for c in COUNTERS if any(int(r.get(c, 0)) for r in man)]

    key, truth = ground_truth(a.release)
    byrec = {r["record_id"]: r for r in man}
    items = [(i, byrec[key[i]["record_id"]], t) for i, t in truth.items()
             if key[i]["record_id"] in byrec]
    Nh = collections.Counter((r["domain"].split("/")[-1], r["label"]) for r in man)
    nh = collections.Counter(key[i]["stratum"] for i, _, _ in items)
    weight = {}
    for i, _, _ in items:
        sub, lab = key[i]["stratum"].split("|")
        weight[i] = Nh[(sub, lab)] / nh[key[i]["stratum"]]
    print("live counters:", ", ".join(live))
    print("ground truth: majority of the listeners, %d of %d items settled"
          % (len(truth), len(key)))

    def score(r, weights, floor, only):
        # The terms are accumulated in this order, into a float starting at 0.0,
        # because the order is observable: two utterances of the corpus land on
        # 4.999999999999999 or on 5.0 depending on it, and the published Table 7
        # was produced with this one.
        den = max(int(r["tokens"]), floor) if floor else max(int(r["tokens"]), 1)
        s = 0.0
        for c in live:
            if only and c not in only:
                continue
            s += weights.get(c, 0.0) * 1000.0 * int(r.get(c, 0)) / den
        return s

    def evaluate(weights, floor, only):
        tp = fp = fn = 0.0
        for i, r, t in items:
            hit = int(r["tokens"]) >= a.min_tokens and score(r, weights, floor, only) >= a.theta1
            w = weight[i]
            if hit and t == "S": tp += w
            elif hit and t == "P": fp += w
            elif t == "S": fn += w
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F = 2 * P * R / (P + R) if P + R else float("nan")
        sel = sum(1 for r in man if int(r["tokens"]) >= a.min_tokens
                  and score(r, weights, floor, only) >= a.theta1)
        return P, R, F, sel

    variants = [("baseline, as released (rescored)", W, N0, None)]
    for c in live:
        w = dict(W); w[c] = 0.0
        variants.append((LABEL.get(c, "without " + c), w, N0, None))
    variants += [("equal weights", {c: 0.02 for c in live}, N0, None),
                 ("without the floor N0", W, None, None),
                 ("strong fillers only", W, N0, {"strong_fillers"})]

    print("\n%-36s %9s %9s %9s %10s"
          % ("variant", "precision", "recall", "F1", "selected"))
    rows = []
    for name, w, floor, only in variants:
        P, R, F, sel = evaluate(w, floor, only)
        print("%-36s %9.3f %9.3f %9.3f %10d" % (name, P, R, F, sel))
        rows.append([name, round(P, 3), round(R, 3), round(F, 3), sel])
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["variant", "precision_weighted", "recall_weighted", "f1_weighted",
                     "selected_corpus"])
        cw.writerows(rows)
    print("\nprecision, recall and F1 are corpus-weighted; selected = utterances of KSC2")
    print("reaching the strict threshold. Wrote", a.out)


if __name__ == "__main__":
    main()
