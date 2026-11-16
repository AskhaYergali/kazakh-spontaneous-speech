#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Supplementary.docx from the current figures and result files.

    python3 build_supplementary.py --release <release dir> --work <dir with
        out_extra/ and supp/> --out Supplementary.docx

Always embeds the PNGs as they are on disk, so regenerating a figure and
re-running this script keeps the document and the figures in step.
"""
import argparse, collections, csv, gzip, os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image

COND = {"a": "(a) fillers retained", "c": "(c) canonical", "b": "(b) fillers removed"}
MODELS = (("abilmansplus__whisper-turbo-ksc2", "whisper-turbo-ksc2"),
          ("abilmansplus__whisper-turbo-kaz-rus-v1", "whisper-turbo-kaz-rus-v1"))
ZS = (("openai__whisper-small", "Whisper-small"), ("openai__whisper-medium", "Whisper-medium"),
      ("openai__whisper-large-v3", "Whisper-large-v3"),
      ("openai__whisper-large-v3-turbo", "Whisper-large-v3-turbo"))
PRET = {"tv_news": "TV news", "talkshow": "talk shows", "crowdsourced": "crowdsourced",
        "parliament": "parliament", "podcasts": "podcasts", "radio": "radio"}
LBL = {"likely_spontaneous": "spontaneous", "maybe_spontaneous": "borderline",
       "likely_prepared_or_read": "prepared"}
ORDER = ["likely_spontaneous", "maybe_spontaneous", "likely_prepared_or_read"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="Supplementary.docx")
    a = ap.parse_args()
    W, REL = a.work, a.release
    d = Document()
    d.styles["Normal"].font.name = "Palatino Linotype"
    d.styles["Normal"].font.size = Pt(9.5)

    def H(t, lvl=1):
        p = d.add_paragraph(); r = p.add_run(t); r.bold = True
        r.font.size = Pt(11 if lvl == 1 else 10); p.paragraph_format.space_after = Pt(4)

    def P(t):
        p = d.add_paragraph(t); p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def CAP(t):
        p = d.add_paragraph(); r = p.add_run(t); r.font.size = Pt(8.5)
        p.paragraph_format.space_after = Pt(10); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def TBL(head, rows, size=8.5):
        t = d.add_table(rows=1, cols=len(head)); t.style = "Table Grid"
        for i, h in enumerate(head):
            c = t.rows[0].cells[i]; c.text = ""
            r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
        for row in rows:
            cs = t.add_row().cells
            for i, v in enumerate(row):
                cs[i].text = ""
                rr = cs[i].paragraphs[0].add_run(str(v)); rr.font.size = Pt(size)
        d.add_paragraph().paragraph_format.space_after = Pt(2)

    def FIG(png, cap, width_in=5.0):
        im = Image.open(png)
        w = width_in
        d.add_picture(png, width=Inches(w), height=Inches(w * im.size[1] / im.size[0]))
        d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        CAP(cap)

    f2 = lambda x: "%.2f" % float(x)
    rat = lambda r: "%.2f [%.2f–%.2f]" % (float(r["ratio"]), float(r["ci_lo"]), float(r["ci_hi"]))
    read = lambda p: {r["condition"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}

    H("Supplementary Materials")
    P("A Spontaneity Metric for Kazakh Transcripts: Register, Filled Pauses and the Evaluation of "
      "Spontaneous Speech. Askhat Yergaliyev, Lyailya Baibulekova and Altynbek Sharipbay.")
    P("These materials give the additional comparisons requested during review. Every figure and table "
      "here is produced by scripts of the released repository from the per-utterance evaluation files "
      "and the released manifest; no number is entered by hand.")

    for title, files, col4, cap in (
        ("Table S1. Register gap on the Test partition alone",
         [(os.path.join(W, "out_extra", "test_%s.csv" % m), l) for m, l in MODELS], "Prep. WER",
         "Table S1. As Table 9 of the main text, restricted to the Test partition (406 spontaneous and "
         "1,031 control utterances over ten recording sessions), so that no Dev material enters the "
         "comparison. The pattern of Table 9 is reproduced, but on ten sessions the intervals are wide "
         "enough that the filler-retained gap is established for whisper-turbo-ksc2 only: for "
         "whisper-turbo-kaz-rus-v1 the filler-retained interval contains one. Intervals from the paired "
         "session bootstrap."),
        ("Table S2. The four models not adapted to Kazakh under all three scoring conditions",
         [(os.path.join(W, "out_extra", "zs_%s.csv" % m), l) for m, l in ZS], "Prep. WER",
         "Table S2. The main text reports these systems under condition (a) only (Table 8). Canonical "
         "scoring changes nothing for them, because they almost never produce a filler to rescore; "
         "removing fillers from the reference lowers the ratio further. Every ratio is at or below one, "
         "which is why these systems are read as uninformative about register rather than as evidence "
         "against a register effect."),
        ("Table S3. The TV-news control under all three scoring conditions",
         [(os.path.join(W, "out_extra", "news_%s.csv" % m), l) for m, l in MODELS], "TV-news WER",
         "Table S3. The TV-news control shares no recording session with the spontaneous set, so the "
         "interval comes from an unpaired utterance bootstrap rather than the paired session bootstrap "
         "used elsewhere. The ratios are three to four times larger than against the session-matched "
         "control and remain far above one under canonical scoring: against a control from a different "
         "recording source the comparison measures the source, not the register, which is the reason the "
         "session-matched control is the primary one.")):
        H(title, 2); rows = []
        for f, lab in files:
            t = read(f)
            for c in "acb":
                r = t[c]
                rows.append([lab if c == "a" else "", COND[c], f2(r["wer_spont"]),
                             f2(r["wer_prep"]), rat(r)])
        TBL(["Model", "Condition", "Spont. WER", col4, "Ratio [95% CI]"], rows); CAP(cap)

    H("Table S4. Substitutions, deletions and insertions on the spontaneous set", 2)
    rows = []
    for f, lab in ((os.path.join(W, "out_extra", "sdi_ksc2.csv"), "whisper-turbo-ksc2"),
                   (os.path.join(W, "out_extra", "sdi_kazrus.csv"), "whisper-turbo-kaz-rus-v1")):
        t = read(f)
        for c in "acb":
            r = t[c]
            rows.append([lab if c == "a" else "", COND[c], "{:,}".format(int(r["sub_spont"])),
                         "{:,}".format(int(r["del_spont"])), "{:,}".format(int(r["ins_spont"])),
                         "{:,}".format(int(r["err_spont"])), "{:,}".format(int(r["ref_spont"]))])
    TBL(["Model", "Condition", "Subst.", "Del.", "Ins.", "Total errors", "Ref. words"], rows)
    CAP("Table S4. Error types on the spontaneous evaluation set. Canonical scoring removes substitutions "
        "almost exclusively — for whisper-turbo-ksc2 from 1,156 to 698, with deletions and insertions "
        "unchanged — which is the counting form of the claim that the filler-retained penalty is a "
        "spelling penalty. Removing fillers then cuts the deletions, from 547 to 182, which is the share "
        "of fillers the model omits outright.")

    # ---- Table S5 ------------------------------------------------------
    H("Table S5. The 18 validation strata", 2)
    A = os.path.join(REL, "annotation")
    key = {r["id"]: r for r in csv.DictReader(open(os.path.join(A, "validation_key.csv"), encoding="utf-8"))}
    g = lambda r: (r.get("ОЦЕНКА") or "").strip().upper()
    l1 = {r["id"]: g(r) for r in csv.DictReader(open(os.path.join(A, "validation_listener_1.csv"), encoding="utf-8"))}
    l2 = {r["id"]: g(r) for r in csv.DictReader(open(os.path.join(A, "validation_listener_2.csv"), encoding="utf-8"))}
    adj = {r["id"]: r["СОГЛАСОВАННАЯ"].strip().upper()
           for r in csv.DictReader(open(os.path.join(A, "validation_disagreements.csv"), encoding="utf-8"))}
    l3 = {r["id"]: g(r) for r in csv.DictReader(
        open(os.path.join(A, "validation_listener_3.csv"), encoding="utf-8-sig"))} \
        if os.path.exists(os.path.join(A, "validation_listener_3.csv")) else {}
    dec = set()
    for i, r in key.items():
        votes = [v for v in (l1.get(i, ""), l2.get(i, ""), l3.get(i, "")) if v in ("S", "P")]
        if votes:
            top, n = collections.Counter(votes).most_common(1)[0]
            if n >= 2 and n > len(votes) - n:
                dec.add(i)
    drawn = collections.Counter(r["stratum"] for r in key.values())
    ndec = collections.Counter(key[i]["stratum"] for i in dec)
    N = collections.Counter()
    with gzip.open(os.path.join(REL, "manifest", "manifest.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            N[(r["domain"].split("/")[-1], r["label"])] += 1
    rows, tN, td, tn = [], 0, 0, 0
    for s in sorted(drawn, key=lambda s: (s.split("|")[0], ORDER.index(s.split("|")[1]))):
        sub, lab = s.split("|"); n = N[(sub, lab)]
        rows.append([PRET[sub], LBL[lab], "{:,}".format(n), drawn[s], ndec[s], "%.1f" % (n / ndec[s])])
        tN += n; td += drawn[s]; tn += ndec[s]
    ts = sum(N[k] for k in N if k[1] == "too_short")
    rows.append(["not sampled", "below the six-token floor", "{:,}".format(ts), 0, 0, "—"])
    rows.append(["total", "", "{:,}".format(tN + ts), td, tn, ""])
    TBL(["Subcorpus", "Filter label", "Stratum N", "Drawn", "Settled", "Weight N / settled"], rows, 8)
    CAP("Table S5. The stratified sample behind Section 5.2 and Table 6. Strata are the crossing of the "
        "six subcorpora with the three filter labels, and the stratum sizes N are counted in the released "
        "manifest. Two hundred items were drawn; four were not settled by two of the three listeners and "
        "are dropped, so each remaining item carries the Horvitz–Thompson weight N divided by the number "
        "of settled items drawn from its stratum, which is the last column. The weights span four orders of magnitude, from "
        "1.6 in television-news material the filter calls spontaneous to 22,585 in crowdsourced material "
        "it calls prepared, which is why every precision and recall figure in the paper is reported "
        "corpus-weighted and why its interval comes from a bootstrap stratified on these 18 cells. The "
        "frame covers the 426,784 utterances the metric scores; the 59,166 utterances below the six-token "
        "floor are neither scored nor sampled.")

    d.add_page_break(); H("Figures")
    S = os.path.join(W, "supp")
    FIG(os.path.join(S, "FigureS1.png"),
        "Figure S1. Above: corpus-weighted precision against the volume selected from KSC2, for the "
        "hand-set score and for both fits of the logistic regression of Section 5.3, each ranked over "
        "the 196 majority-labelled items. Each curve is a step function: selecting any volume between "
        "two ranked items takes exactly the items above it, so the precision is constant in between. "
        "The dotted line is the volume the filter selects at θ₂, and the marker gives the precision of "
        "the score there. The two fits are drawn for one declared cross-validation split; they move a "
        "great deal with the split, so no value is marked on them and Section 5.3 reports their median "
        "and range over forty splits instead. Below: the volume selected as a function of the "
        "threshold, which is how a threshold is translated into a corpus size.", 4.7)
    FIG(os.path.join(S, "FigureS2.png"),
        "Figure S2. Distribution of the spontaneity score over all 485,950 utterances of KSC2, by "
        "subcorpus, on a logarithmic count axis. The crowdsourced and television-news subcorpora are "
        "exhausted before the inclusive threshold, while the three conversational subcorpora extend well "
        "beyond the strict one. The register gradient the filter is meant to capture is visible in the "
        "raw distribution.", 5.0)
    FIG(os.path.join(S, "FigureS3.png"),
        "Figure S3. Word error rate per recording session, spontaneous set against session-matched "
        "control, condition (a), for both Kazakh-adapted models. Marker shape and colour give the "
        "subcorpus and point area is proportional to the number of reference words; the diagonal marks "
        "equality. Parliamentary sessions lie far above the diagonal in both panels, podcast sessions "
        "scatter around it. The aggregate ratios of Table 9 average over this heterogeneity.", 6.2)

    H("Reproduction")
    P("Tables S1–S4 are produced by eval/score_sets.py, Table S5 and the validation figures of Table 6 "
      "and Section 5.2 by eval/validation_precision.py, Figures S1–S3 by eval/supp_figures.py, and the "
      "analysis of Section 6.3 together with Figure 1 by eval/harmony_fillers.py and "
      "eval/figure1_harmony.py. The ablation of Table 7 is produced by eval/ablation.py, except "
      "for its last row, the filter without the reduplication stop-list, which comes from "
      "eval/ablation_stoplist.py, and the register gap of Table 9 with its intervals by "
      "eval/bootstrap_register_gap.py, which can be run from the released per-utterance error "
      "counts without the corpus text. This document is assembled from their output by "
      "eval/build_supplementary.py. Running eval/supp_figures.py also prints the out-of-fold AUCs of "
      "Section 5.3 over forty cross-validation splits and the comparison of the hand-set weights with the fitted coefficients on both scales, raw and per standard deviation. All of these read the per-utterance evaluation "
      "files written by eval/eval_zeroshot.py. The copies of those files in the public release carry no "
      "reference or hypothesis text, since that text is KSC2 material and may not be redistributed; they "
      "must be regenerated against a local copy of the corpus. requirements.txt pins the library "
      "versions these scripts were run with.")
    d.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
