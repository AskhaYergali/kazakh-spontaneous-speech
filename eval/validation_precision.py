#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validation_precision.py -- Table 6 and the four ground truths of Section 5.2.

Every corpus-weighted figure is a Horvitz-Thompson estimate: an item of stratum
h carries the weight N_h / n_h, where N_h is the size of that stratum in the
released manifest and n_h is the number of items drawn from it that are USABLE
under the ground truth in force.  Items that are not usable -- undecidable by
ear, or outside the ground truth being tested -- are dropped and their stratum
is reweighted, so the frame always maps back to the whole of KSC2.

    python3 validation_precision.py --release . [--boot 10000]

Prints Table 6 and the four ground truths of Section 5.2.
"""
import argparse, collections, csv, gzip, math, os, random

THETA1, THETA2 = 5.0, 3.0


def read(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class Frame:
    """A ground truth plus the weights it implies."""

    def __init__(self, name, truth, key, N):
        self.name, self.truth, self.key = name, truth, key
        n = collections.Counter(key[i]["stratum"] for i in truth)
        self.w = {i: N[tuple(key[i]["stratum"].split("|"))] / n[key[i]["stratum"]]
                  for i in truth}
        self.by_stratum = collections.defaultdict(list)
        for i in truth:
            self.by_stratum[key[i]["stratum"]].append(i)

    def score(self, i):
        return float(self.key[i]["score"])

    def counts(self, ids, sel, positive, weighted):
        f = (lambda i: self.w[i]) if weighted else (lambda i: 1.0)
        tp = fp = fn = 0.0
        for i in ids:
            s, t = sel(self.score(i)), self.truth[i]
            if positive == "P":          # relabel truth; `sel` already selects the predicted-prepared set
                t = "S" if t == "P" else "P"
            if s and t == "S":
                tp += f(i)
            elif s:
                fp += f(i)
            elif t == "S":
                fn += f(i)
        return tp, fp, fn

    def prf(self, ids, sel, positive="S", weighted=True):
        tp, fp, fn = self.counts(ids, sel, positive, weighted)
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F = 2 * P * R / (P + R) if P + R else float("nan")
        return P, R, F

    def boot(self, sel, positive="S", B=10000, seed=20261014):
        rnd = random.Random(seed)
        acc = ([], [], [])
        for _ in range(B):
            ids = []
            for mem in self.by_stratum.values():
                ids.extend(rnd.choices(mem, k=len(mem)))
            for k, v in enumerate(self.prf(ids, sel, positive)):
                if v == v:
                    acc[k].append(v)
        q = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v))]) if v else (float("nan"),) * 2
        return [q(v) for v in acc]


def build(release, listener3=None):
    man = read(os.path.join(release, "manifest", "manifest.csv.gz"))
    N = collections.Counter((r["domain"].split("/")[-1], r["label"]) for r in man)
    A = os.path.join(release, "annotation")
    key = {r["id"]: r for r in read(os.path.join(A, "validation_key.csv"))}
    g = lambda r: (r.get("ОЦЕНКА") or "").strip().upper()
    l1 = {r["id"]: g(r) for r in read(os.path.join(A, "validation_listener_1.csv"))}
    l2 = {r["id"]: g(r) for r in read(os.path.join(A, "validation_listener_2.csv"))}
    adj = {r["id"]: r["СОГЛАСОВАННАЯ"].strip().upper()
           for r in read(os.path.join(A, "validation_disagreements.csv"))}
    l3 = {}
    if listener3:
        l3 = {r["id"]: g(r) for r in read(listener3)}
    ok = lambda v: v in ("S", "P")

    adjudicated, agreed, listener2, adversarial, majority = {}, {}, {}, {}, {}
    for i, r in key.items():
        a, b, c = l1.get(i, ""), l2.get(i, ""), l3.get(i, "")
        # primary: majority of the listeners who reached a decision, and only
        # where at least two of them concur, so that no label rests on one person
        votes = [v for v in (a, b, c) if ok(v)]
        if votes:
            top, n = collections.Counter(votes).most_common(1)[0]
            if n >= 2 and n > len(votes) - n:
                majority[i] = top
        if ok(a) and ok(b) and a == b:
            adjudicated[i] = agreed[i] = adversarial[i] = a
        elif ok(adj.get(i, "")):
            adjudicated[i] = adj[i]
            # adversarial: of the two listener labels, take the one that
            # contradicts the filter wherever they differ
            filt = "S" if float(r["score"]) >= THETA1 else "P"
            cand = [v for v in (a, b) if ok(v)]
            adversarial[i] = ("P" if filt == "S" else "S") if any(
                v != filt for v in cand) else adj[i]
        if ok(b):
            listener2[i] = b
    out = {}
    if l3:
        out["majority of three listeners"] = majority
    out.update({"first author adjudicates": adjudicated,
                "listener agreement only": agreed,
                "second listener alone": listener2,
                "every disagreement against the filter": adversarial})
    return man, N, key, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=".")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--listener3", help="validation_listener_3.csv, if a third listener settled "
                                        "the items the first two did not")
    a = ap.parse_args()
    man, N, key, truths = build(a.release, a.listener3)
    scored = sum(v for k, v in N.items() if k[1] != "too_short")
    print("corpus %d utterances, %d scored, %d below the six-token floor\n"
          % (len(man), scored, len(man) - scored))

    primary = next(iter(truths))
    F = Frame(primary, truths[primary], key, N)
    ids = sorted(F.truth)
    print("=== Table 6 (ground truth: %s, %d items) ===" % (primary, len(ids)))
    print("%-12s %-28s %-28s %-18s %s" % ("decision", "precision sample / weighted",
                                          "recall sample / weighted", "F1 weighted", "n"))
    for nm, sel, pos in (("strict", lambda s: s >= THETA1, "S"),
                         ("inclusive", lambda s: s >= THETA2, "S"),
                         ("prepared", lambda s: s < THETA2, "P")):
        tp, fp, fn = F.counts(ids, sel, pos, False)
        sP, sR = tp / (tp + fp) if tp + fp else float("nan"), tp / (tp + fn) if tp + fn else float("nan")
        cP, cR = wilson(tp, tp + fp), wilson(tp, tp + fn)
        P, R, Fs = F.prf(ids, sel, pos)
        (pl, ph), (rl, rh), (fl, fh) = F.boot(sel, pos, a.boot)
        print("%-12s %.2f [%.2f-%.2f] / %.2f [%.2f-%.2f]  %.2f [%.2f-%.2f] / %.2f [%.2f-%.2f]  "
              "%.2f [%.2f-%.2f]  %d"
              % (nm, sP, cP[0], cP[1], P, pl, ph, sR, cR[0], cR[1], R, rl, rh, Fs, fl, fh, int(tp + fp)))
        if nm == "inclusive":
            R_incl = R
    tot = sum(F.w[i] for i in ids if F.truth[i] == "S")
    base = sum(F.w[i] for i in ids)
    print("\nextrapolated spontaneous {:,.0f} of {:,.0f} scored = {:.1f}%".format(
        tot, base, 100 * tot / base))

    worst = []
    for drop in ids:
        kept = [i for i in ids if i != drop]
        G = Frame(primary, {i: F.truth[i] for i in kept}, key, N)
        _, R2, _ = G.prf(kept, lambda s: s >= THETA2)
        worst.append((abs(R2 - R_incl), drop, R2))
    worst.sort(reverse=True)
    print("\nleave-one-out influence on inclusive weighted recall (%.3f with every item):" % R_incl)
    for d, i, r2 in worst[:3]:
        print("  without %-5s %-42s recall %.3f  (%+.3f)"
              % (i, key[i]["stratum"], r2, r2 - R_incl))

    print("\n=== Section 5.2: weighted precision of the strict decision under four ground truths ===")
    for nm, t in truths.items():
        Fi = Frame(nm, t, key, N)
        P, _, _ = Fi.prf(sorted(t), lambda s: s >= THETA1)
        print("  %-40s %3d items   weighted precision %.3f" % (nm, len(t), P))


if __name__ == "__main__":
    main()
