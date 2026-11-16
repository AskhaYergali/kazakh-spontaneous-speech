#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harmony_fillers.py -- vowel harmony in filled pauses: reference vs recognizer output.

Reference implementation of the analysis reported in Section 6.3 of

    Yergaliyev, A.; Sharipbay, A.  A Curated Spontaneous-Speech Resource for
    Kazakh: Filter, Validation and Register Effects on ASR.  (submitted)

The code used for the first submission was not preserved.  This file
re-implements the analysis with every operational choice stated explicitly and
exposed as a command-line flag.  Run with its DEFAULT settings it reproduces
the figures printed in Section 6.3 and Figure 1.

----------------------------------------------------------------------------
Question
----------------------------------------------------------------------------
Kazakh has back/front vowel harmony.  A filled pause is transcribed in KSC2
either with a back vowel ("ы", "ыы", "ыыы", ...) or with a front vowel ("і",
"іі", ...).  If that distinction were a stable phonetic property conditioned
by context, it would follow the harmony of the neighbouring words.  Section
6.3 asks whether it does -- in the reference transcripts, and in recognizer
output.

----------------------------------------------------------------------------
Operationalization (defaults = Section 6.3)
----------------------------------------------------------------------------
Unit of analysis
    one filled-pause token occurrence.

Forms counted
    the back (ы-) and front (і-) series of filter/lexicons.json, single-letter
    spellings included:  ы ыы ыыы ыыыы ыыыыы ыыыыыы  і іі ііі іііі ііііі.
    Forms built on other vowels (аа, оо, ә, ...) and vowelless forms (мм, мхм,
    э...) are NOT counted: the analysis concerns the ы/і alternation only.
    Matching is whole-token after lower-casing and stripping punctuation, the
    same normalization the released filter applies.
    --forms all  counts every filler form that carries a classifiable vowel.

Outcome  Y = 1 if the filler is the front (і-) form.

Predictor  X = 1 if the context is front
    Context is the nearest NON-FILLER neighbour: the preceding word where
    there is one, the following word otherwise.  It is classified front or
    back by the MAJORITY of its harmony-bearing vowels; a word with no such
    vowel, or with equal numbers of each, yields no context and the filler is
    dropped.
        back  vowels: а о ұ ы
        front vowels: ә ө ү і е
        и, у, э and non-Kazakh vowels are not harmony-bearing.
    --context-rule and --tie change this.

Session
    record_id with the trailing "_<utterance number>" removed, i.e. the
    recording the utterances were cut from.  Subcorpus = `domain`.

Statistics
    * 2x2 table, odds ratio, Woolf (logit) 95 % CI, chi-square.
    * per-session share of the front form; Pearson r between reference and
      model over the sessions carrying at least --min-tokens fillers on BOTH
      sides (16 sessions at the default of 15); Fisher-z 95 % CI.
    * partial r with subcorpus means removed from both variables,
      df = n - 1 - (number of subcorpora).
    * mixed logistic regression of the REFERENCE form on context harmony and
      subcorpus with a session random intercept (BinomialBayesMixedGLM), and
      the same model on the model output.
    * deviance decomposition null -> +harmony -> +subcorpus, as a share of the
      null deviance.

----------------------------------------------------------------------------
Input
----------------------------------------------------------------------------
Per-utterance evaluation CSVs as written by eval/eval_zeroshot.py:

    record_id, domain, score, strong_fillers, russian_mixed, dur_sec,
    reference, hypothesis, wer_utt

NOTE.  The copies in the public release have `reference` and `hypothesis`
removed: that text is KSC2 material and may not be redistributed.  Regenerate
them with eval/eval_zeroshot.py against a local copy of KSC2.

----------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------
    python3 harmony_fillers.py \
        --runs results/abilmansplus__whisper-turbo-ksc2__spont_eval.csv \
        --lexicons filter/lexicons.json \
        --out out/harmony

Writes <out>_tokens.csv, <out>_sessions.csv, <out>_summary.json and a report
on stdout.  <out>_sessions.csv is the input of Figure 1.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

BACK_DEFAULT = "аоұы"
FRONT_DEFAULT = "әөүіе"
PUNCT_RE = re.compile(r"[^\w\sЀ-ӿ-]+", re.UNICODE)


def tokenize(text: str) -> list:
    if not text:
        return []
    return PUNCT_RE.sub(" ", text.lower()).split()


class Harmony:
    """Vowel-harmony classifier.  1 = back, 0 = front, None = undefined."""

    def __init__(self, back: str, front: str, tie: str) -> None:
        self.back, self.front, self.tie = set(back), set(front), tie
        if self.back & self.front:
            raise ValueError("vowel in both classes")

    def _directional(self, word: str, reverse: bool):
        for ch in (reversed(word) if reverse else word):
            if ch in self.back:
                return 1
            if ch in self.front:
                return 0
        return None

    def majority(self, word: str):
        nb = sum(1 for ch in word if ch in self.back)
        nf = sum(1 for ch in word if ch in self.front)
        if nb == 0 and nf == 0:
            return None
        if nb != nf:
            return 1 if nb > nf else 0
        if self.tie == "drop":
            return None
        if self.tie == "back":
            return 1
        if self.tie == "front":
            return 0
        if self.tie == "last":
            return self._directional(word, True)
        if self.tie == "first":
            return self._directional(word, False)
        raise ValueError(self.tie)

    def classify(self, word: str, rule: str):
        if rule == "majority":
            return self.majority(word)
        if rule == "last":
            return self._directional(word, True)
        if rule == "first":
            return self._directional(word, False)
        raise ValueError(rule)

    def has_vowel(self, word: str) -> bool:
        return any(ch in self.back or ch in self.front for ch in word)


def load_forms(path: str, har: Harmony, which: str):
    """which='yi' -> the ы- and і- series only; 'all' -> every voweled form."""
    with open(path, encoding="utf-8") as fh:
        lex = json.load(fh)
    every = set(lex.get("strong_fillers", [])) | set(lex.get("weak_fillers", []))
    if which == "all":
        forms = {f for f in every if har.has_vowel(f)}
    else:
        forms = {f for f in every if set(f) <= {"ы"} or set(f) <= {"і"}}
    if not forms:
        raise SystemExit("no usable filler forms")
    return forms, every


def session_of(record_id: str) -> str:
    return re.sub(r"(\.txt)+$", "", record_id).rsplit("_", 1)[0]


def is_front_form(form: str, har: Harmony):
    v = har._directional(form, False)
    return None if v is None else 1 - v


def extract(rows, field, forms, every_filler, har, args):
    """One record per counted filled-pause token that has a defined context."""
    out = []
    for r in rows:
        toks = tokenize(r[field])
        sess, dom = session_of(r["record_id"]), r.get("domain", "NA")
        for i, w in enumerate(toks):
            if w not in forms:
                continue
            y = is_front_form(w, har)
            if y is None:
                continue

            ctx = None
            if args.context_rule in ("nearest", "prev"):
                j = i - 1
                if args.skip_fillers:
                    while j >= 0 and toks[j] in every_filler:
                        j -= 1
                if j >= 0:
                    ctx = har.classify(toks[j], args.classify)
            if ctx is None and args.context_rule in ("nearest", "next"):
                k = i + 1
                if args.skip_fillers:
                    while k < len(toks) and toks[k] in every_filler:
                        k += 1
                if k < len(toks):
                    ctx = har.classify(toks[k], args.classify)
            if ctx is None:
                continue

            out.append({"record_id": r["record_id"], "session": sess,
                        "subcorpus": dom, "field": field, "form": w,
                        "front_form": y, "front_context": 1 - ctx})
    return out


# --------------------------------------------------------------------------
def two_by_two(recs):
    n = [[0, 0], [0, 0]]                      # [front context][front form]
    for d in recs:
        n[d["front_context"]][d["front_form"]] += 1
    nf, nb = n[1][0] + n[1][1], n[0][0] + n[0][1]
    a, b, c, e = n[1][1], n[1][0], n[0][1], n[0][0]
    corrected = min(a, b, c, e) == 0
    if corrected:
        a, b, c, e = a + .5, b + .5, c + .5, e + .5
    orv = (a * e) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
    tot = a + b + c + e
    chi2 = tot * (a * e - b * c) ** 2 / ((a + b) * (c + e) * (a + c) * (b + e))
    return {"n": len(recs), "table": n, "n_front_context": nf, "n_back_context": nb,
            "p_front_form_given_front_context": (n[1][1] / nf) if nf else float("nan"),
            "p_front_form_given_back_context": (n[0][1] / nb) if nb else float("nan"),
            "odds_ratio": orv,
            "ci95": [math.exp(math.log(orv) - 1.96 * se),
                     math.exp(math.log(orv) + 1.96 * se)],
            "chi2": chi2, "haldane_corrected": corrected}


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def fisher_ci(r, n, df_lost=1):
    eff = n - df_lost - 2
    if eff <= 0 or not -1 < r < 1:
        return [float("nan"), float("nan")]
    z, se = 0.5 * math.log((1 + r) / (1 - r)), 1 / math.sqrt(eff)
    return [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def session_table(recs, min_tokens):
    agg = defaultdict(lambda: [0, 0, None])
    for d in recs:
        a = agg[d["session"]]
        a[0] += d["front_form"]
        a[1] += 1
        a[2] = d["subcorpus"]
    return {s: {"front": v[0], "n": v[1], "rate": v[0] / v[1],
                "wilson": wilson(v[0], v[1]), "subcorpus": v[2]}
            for s, v in agg.items() if v[1] >= min_tokens}


def mixed_model(recs, seed, label):
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as exc:
        return {"error": "statsmodels/pandas unavailable: %s" % exc}

    df = pd.DataFrame(recs)
    if df["subcorpus"].nunique() < 2 or df["session"].nunique() < 2:
        return {"error": "need >=2 subcorpora and >=2 sessions"}
    np.random.seed(seed)
    base = "parliament" if "parliament" in set(df["subcorpus"]) else sorted(df["subcorpus"])[0]
    cats = [base] + [c for c in sorted(df["subcorpus"].unique()) if c != base]
    df["subcorpus"] = pd.Categorical(df["subcorpus"], categories=cats)

    out = {"label": label, "subcorpus_reference_level": base}
    try:
        res = BinomialBayesMixedGLM.from_formula(
            "front_form ~ front_context + C(subcorpus)",
            {"session": "0 + C(session)"}, df).fit_vb(verbose=False)
        out["fixed_effects_or"] = {
            nm: {"or": float(math.exp(res.fe_mean[k])),
                 "ci95": [float(math.exp(res.fe_mean[k] - 1.96 * res.fe_sd[k])),
                          float(math.exp(res.fe_mean[k] + 1.96 * res.fe_sd[k]))]}
            for k, nm in enumerate(res.model.exog_names)}
        out["session_random_intercept_sd"] = float(np.exp(res.vcp_mean[0]))
    except Exception as exc:
        out["mixed_error"] = repr(exc)
    try:
        dev = {}
        for nm, f in (("null", "front_form ~ 1"),
                      ("harmony", "front_form ~ front_context"),
                      ("harmony+subcorpus", "front_form ~ front_context + C(subcorpus)")):
            dev[nm] = smf.glm(f, data=df, family=sm.families.Binomial()).fit().deviance
        out["deviance"] = {k: float(v) for k, v in dev.items()}
        out["deviance_share"] = {
            "harmony": float((dev["null"] - dev["harmony"]) / dev["null"]),
            "subcorpus_given_harmony": float(
                (dev["harmony"] - dev["harmony+subcorpus"]) / dev["null"])}
    except Exception as exc:
        out["deviance_error"] = repr(exc)
    return out


# --------------------------------------------------------------------------
def analyse(path, forms, every, har, args):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("record_id", "reference", "hypothesis"):
        if col not in rows[0]:
            raise SystemExit(
                "%s has no '%s' column.  The public release ships these files without "
                "the reference/hypothesis text; regenerate them with "
                "eval/eval_zeroshot.py against a local copy of KSC2." % (path, col))

    ref = extract(rows, "reference", forms, every, har, args)
    hyp = extract(rows, "hypothesis", forms, every, har, args)

    res = {"run": os.path.basename(path), "n_utterances": len(rows),
           "settings": {k: getattr(args, k) for k in
                        ("forms", "context_rule", "classify", "tie", "skip_fillers",
                         "min_tokens", "back", "front", "seed")},
           "n_forms_counted": len(forms), "forms_counted": sorted(forms),
           "reference": two_by_two(ref), "model": two_by_two(hyp)}

    R, H = session_table(ref, args.min_tokens), session_table(hyp, args.min_tokens)
    shared = sorted(set(R) & set(H))
    s = {"n_reference": len(R), "n_model": len(H), "n_shared": len(shared)}
    if len(shared) >= 3:
        xs = [R[k]["rate"] for k in shared]
        ys = [H[k]["rate"] for k in shared]
        r = pearson(xs, ys)
        groups = defaultdict(list)
        for k in shared:
            groups[R[k]["subcorpus"]].append(k)
        cx, cy = [], []
        for _, mem in groups.items():
            mx = sum(R[k]["rate"] for k in mem) / len(mem)
            my = sum(H[k]["rate"] for k in mem) / len(mem)
            for k in mem:
                cx.append(R[k]["rate"] - mx)
                cy.append(H[k]["rate"] - my)
        pr = pearson(cx, cy)
        s.update({"reference_rate_range": [min(xs), max(xs)],
                  "model_rate_range": [min(ys), max(ys)],
                  "pearson_r": r, "pearson_ci95": fisher_ci(r, len(shared), 1),
                  "partial_r": pr,
                  "partial_ci95": fisher_ci(pr, len(shared), 1 + len(groups)),
                  "subcorpus_composition": {g: len(m) for g, m in sorted(groups.items())},
                  "within_subcorpus_r": {
                      g: {"n": len(m),
                          "r": (pearson([R[k]["rate"] for k in m],
                                        [H[k]["rate"] for k in m])
                                if len(m) >= 3 else None)}
                      for g, m in sorted(groups.items())}})
    res["sessions"] = s
    res["mixed_reference"] = mixed_model(ref, args.seed, "reference")
    res["mixed_model_output"] = mixed_model(hyp, args.seed, "model output")
    return res, R, H, shared, ref, hyp


def fmt_mixed(mm, indent="   "):
    L = []
    if "fixed_effects_or" in mm:
        for nm, d in mm["fixed_effects_or"].items():
            if nm == "Intercept":
                continue
            L.append("%s%-28s OR %6.2f  [%.2f, %.2f]"
                     % (indent, nm, d["or"], d["ci95"][0], d["ci95"][1]))
        L.append("%ssession random intercept SD  %.2f"
                 % (indent, mm["session_random_intercept_sd"]))
    if "deviance_share" in mm:
        L.append("%sdeviance share: harmony %.1f%%, subcorpus | harmony %.1f%%"
                 % (indent, 100 * mm["deviance_share"]["harmony"],
                    100 * mm["deviance_share"]["subcorpus_given_harmony"]))
    for k in ("error", "mixed_error", "deviance_error"):
        if k in mm:
            L.append("%s%s: %s" % (indent, k, mm[k]))
    return L


def report(res):
    L, w = [], None
    out = L.append
    out("=" * 76)
    out(res["run"] + "   (%d forms counted: %s)"
        % (res["n_forms_counted"], " ".join(res["forms_counted"])))
    out("=" * 76)
    for key, lab in (("reference", "REFERENCE"), ("model", "MODEL OUTPUT")):
        t = res[key]
        out("%-13s n=%-6d front context n=%-5d back context n=%-5d"
            % (lab, t["n"], t["n_front_context"], t["n_back_context"]))
        out("              P(front form | front context) = %.3f"
            % t["p_front_form_given_front_context"])
        out("              P(front form | back  context) = %.3f"
            % t["p_front_form_given_back_context"])
        out("              OR = %.2f  [%.2f, %.2f]   chi2 = %.1f%s"
            % (t["odds_ratio"], t["ci95"][0], t["ci95"][1], t["chi2"],
               "  (Haldane-corrected)" if t["haldane_corrected"] else ""))
    s = res["sessions"]
    out("-" * 76)
    out("sessions with >= %d fillers: %d reference, %d model, %d on both sides"
        % (res["settings"]["min_tokens"], s["n_reference"], s["n_model"], s["n_shared"]))
    if "pearson_r" in s:
        out("front-form share, reference  %.0f%%-%.0f%%"
            % (100 * s["reference_rate_range"][0], 100 * s["reference_rate_range"][1]))
        out("front-form share, model      %.0f%%-%.0f%%"
            % (100 * s["model_rate_range"][0], 100 * s["model_rate_range"][1]))
        out("r         = %+.3f  [%+.2f, %+.2f]"
            % (s["pearson_r"], s["pearson_ci95"][0], s["pearson_ci95"][1]))
        out("partial r = %+.3f  [%+.2f, %+.2f]   (subcorpus means removed)"
            % (s["partial_r"], s["partial_ci95"][0], s["partial_ci95"][1]))
        out("composition: %s" % ", ".join("%s %d" % kv
                                          for kv in s["subcorpus_composition"].items()))
        for g, d in s["within_subcorpus_r"].items():
            out("   within %-11s n=%-3d r=%s"
                % (g, d["n"], "n/a" if d["r"] is None else "%+.2f" % d["r"]))
    out("-" * 76)
    out("mixed logistic, REFERENCE form ~ harmony + subcorpus + (1|session)")
    L.extend(fmt_mixed(res["mixed_reference"]))
    out("mixed logistic, MODEL output   ~ harmony + subcorpus + (1|session)")
    L.extend(fmt_mixed(res["mixed_model_output"]))
    out("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Vowel harmony in filled pauses: reference vs recognizer output "
                    "(Section 6.3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="per-utterance evaluation CSV(s) with reference and hypothesis")
    ap.add_argument("--lexicons", required=True, help="filter/lexicons.json")
    ap.add_argument("--out", default="harmony", help="output prefix")
    ap.add_argument("--forms", default="yi", choices=["yi", "all"],
                    help="'yi' = the ы- and і- series only; 'all' = every voweled form")
    ap.add_argument("--context-rule", default="nearest",
                    choices=["nearest", "prev", "next"],
                    help="which neighbour supplies the context")
    ap.add_argument("--classify", default="majority", choices=["majority", "last", "first"],
                    help="how a context word is assigned a harmony class")
    ap.add_argument("--tie", default="drop", choices=["drop", "back", "front", "last", "first"],
                    help="what to do when a context word has equally many back and front vowels")
    ap.add_argument("--skip-fillers", dest="skip_fillers", action="store_true", default=True,
                    help="look past adjacent fillers for the context word")
    ap.add_argument("--no-skip-fillers", dest="skip_fillers", action="store_false",
                    help="take the immediate neighbour even if it is a filler")
    ap.add_argument("--min-tokens", type=int, default=15,
                    help="minimum fillers per session to enter the correlation")
    ap.add_argument("--back", default=BACK_DEFAULT, help="back vowel inventory")
    ap.add_argument("--front", default=FRONT_DEFAULT, help="front vowel inventory")
    ap.add_argument("--seed", type=int, default=20261014, help="seed for the variational fit")
    args = ap.parse_args(argv)

    har = Harmony(args.back, args.front, args.tie)
    forms, every = load_forms(args.lexicons, har, args.forms)
    sys.stderr.write("counting %d filler form(s): %s\n" % (len(forms), " ".join(sorted(forms))))

    paths = []
    for p in args.runs:
        paths.extend(sorted(glob.glob(p)) or [p])
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)

    summaries, tok_rows, sess_rows = [], [], []
    for p in paths:
        res, R, H, shared, ref, hyp = analyse(p, forms, every, har, args)
        summaries.append(res)
        print(report(res))
        for rec in ref + hyp:
            rec = dict(rec); rec["run"] = res["run"]; tok_rows.append(rec)
        for s in sorted(set(R) | set(H)):
            a, b = R.get(s), H.get(s)
            sess_rows.append({
                "run": res["run"], "session": s,
                "subcorpus": (a or b)["subcorpus"],
                "n_reference": a["n"] if a else 0, "n_model": b["n"] if b else 0,
                "front_reference": a["front"] if a else 0,
                "front_model": b["front"] if b else 0,
                "rate_reference": "%.6f" % a["rate"] if a else "",
                "rate_model": "%.6f" % b["rate"] if b else "",
                "wilson_lo_reference": "%.6f" % a["wilson"][0] if a else "",
                "wilson_hi_reference": "%.6f" % a["wilson"][1] if a else "",
                "wilson_lo_model": "%.6f" % b["wilson"][0] if b else "",
                "wilson_hi_model": "%.6f" % b["wilson"][1] if b else "",
                "in_correlation": int(s in shared)})

    with open(args.out + "_tokens.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["run", "record_id", "session", "subcorpus",
                                            "field", "form", "front_form", "front_context"])
        wr.writeheader(); wr.writerows(tok_rows)
    with open(args.out + "_sessions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sess_rows[0].keys()))
        wr.writeheader(); wr.writerows(sess_rows)
    with open(args.out + "_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summaries, fh, ensure_ascii=False, indent=2)
    sys.stderr.write("wrote %s_tokens.csv, %s_sessions.csv, %s_summary.json\n"
                     % (args.out, args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
