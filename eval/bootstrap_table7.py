#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Парный бутстреп по сессиям для Таблицы 7 статьи (Эксперимент A).

Для каждой модели и каждого условия шкалирования считает WER на спонтанном
наборе и на согласованном по сессиям подготовленном контроле, их отношение
и 95%-й процентильный интервал отношения. Пересэмплируются ЗАПИСНЫЕ СЕССИИ
(24 группы, одни и те же в обоих наборах), а не высказывания: высказывания
одной сессии не независимы. B = 10 000, зерно фиксировано.

Нормализация и инвентарь заполнителей — из eval_zeroshot.py без изменений.
WER считается на счётчиках ошибок (сумма ошибок / сумма слов эталона),
а не как среднее поутвердительных WER.

Два входа. Из полных прогонов с текстом (не публикуются по лицензии KSC2):
    python bootstrap_table7.py --runs results/asr_runs --out table7_ci.csv
Из публикуемого файла счётчиков ошибок (без текста) — воспроизводит ту же таблицу:
    python bootstrap_table7.py --counts results/summaries/wer_counts_table7.csv
"""
from __future__ import annotations
import argparse, csv, os, re
import numpy as np
import jiwer

FILLERS = {
    "ыы","ыыы","ыыыы","ыыыыы","ыыыыыы","іі","ііі","іііі","ііііі",
    "мм","ммм","мммм","аа","ааа","аааа","оо","ооо","ээ","эээ",
    "мхм","мһм","аха","ыһы","әһә","ы","і","м","э","эм","ә","о","а",
}
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SESSION_RE = re.compile(r"^(.*)_\d+$")
B = 10000
SEED = 20261014
CONDS = [("(a) fillers retained", False), ("(c) canonical", "canon"), ("(b) fillers removed", True)]
MODELS = ["abilmansplus__whisper-turbo-ksc2", "abilmansplus__whisper-turbo-kaz-rus-v1"]


def normalize(text, mode):
    toks = PUNCT_RE.sub(" ", text.lower()).split()
    if mode is True or mode == "stripped":
        toks = [x for x in toks if x not in FILLERS]
    elif mode == "canon":
        toks = ["<F>" if x in FILLERS else x for x in toks]
    return " ".join(toks)


def session(record_id, domain):
    base = os.path.basename(record_id)
    for suf in (".txt.txt", ".txt"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    m = SESSION_RE.match(base)
    return "{}/{}".format(domain, m.group(1) if (m and m.group(1)) else base)


def load(path, mode):
    """сессия -> (ошибки, слов эталона)"""
    agg = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rn = normalize(r["reference"], mode)
        hn = normalize(r["hypothesis"], mode)
        if not rn:                      # пустой эталон отбрасывается, как в eval_zeroshot
            continue
        o = jiwer.process_words([rn], [hn])
        e = o.substitutions + o.deletions + o.insertions
        n = o.substitutions + o.deletions + o.hits
        k = session(r["record_id"], r["domain"])
        a, b = agg.get(k, (0, 0))
        agg[k] = (a + e, b + n)
    return agg


def load_counts(path, model, setname, mode):
    """То же из файла счётчиков ошибок: record_id, session, err_a/ref_a, err_c/..., err_b/..."""
    agg = {}
    key = {False: "a", "canon": "c", True: "b"}[mode]
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["model"] != model or r["set"] != setname:
            continue
        if r["ref_" + key] == "":
            continue
        a, b = agg.get(r["session"], (0, 0))
        agg[r["session"]] = (a + int(r["err_" + key]), b + int(r["ref_" + key]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=None,
                    help="файл счётчиков ошибок вместо полных прогонов")
    ap.add_argument("--runs", default="results/asr_runs")
    ap.add_argument("--out", default="table7_ci.csv")
    a = ap.parse_args()
    out = []
    for tag in MODELS:
        sp_p = os.path.join(a.runs, tag + "__spont_eval.csv")
        pr_p = os.path.join(a.runs, tag + "__prep_eval_matched.csv")
        for cname, mode in CONDS:
            if a.counts:
                m = tag.split("__")[-1]
                sp = load_counts(a.counts, m, "spont_eval", mode)
                pr = load_counts(a.counts, m, "prep_eval_matched", mode)
            else:
                sp, pr = load(sp_p, mode), load(pr_p, mode)
            keys = sorted(set(sp) | set(pr))
            sE = np.array([sp.get(k, (0, 0))[0] for k in keys], float)
            sN = np.array([sp.get(k, (0, 0))[1] for k in keys], float)
            pE = np.array([pr.get(k, (0, 0))[0] for k in keys], float)
            pN = np.array([pr.get(k, (0, 0))[1] for k in keys], float)
            w_sp, w_pr = sE.sum() / sN.sum(), pE.sum() / pN.sum()
            rng = np.random.default_rng(SEED)
            idx = rng.integers(0, len(keys), size=(B, len(keys)))
            ws = sE[idx].sum(1) / sN[idx].sum(1)
            wp = pE[idx].sum(1) / pN[idx].sum(1)
            rs, gp = ws / wp, (ws - wp) * 100
            lo, hi = np.percentile(rs, [2.5, 97.5])
            glo, ghi = np.percentile(gp, [2.5, 97.5])
            out.append([tag.split("__")[-1], cname, round(w_sp * 100, 2), round(w_pr * 100, 2),
                        round(w_sp / w_pr, 2), round(lo, 2), round(hi, 2),
                        round((w_sp - w_pr) * 100, 2), round(glo, 2), round(ghi, 2), len(keys)])
            print("{:26s} {:22s} {:5.2f} {:5.2f}  {:4.2f} [{:.2f}-{:.2f}]  {:+5.2f} p.p. [{:+.2f};{:+.2f}]  сессий {}".format(*out[-1]))
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["model", "condition", "spont_wer", "prep_wer", "ratio", "ratio_ci_lo",
                    "ratio_ci_hi", "gap_pp", "gap_ci_lo", "gap_ci_hi", "sessions"])
        w.writerows(out)


if __name__ == "__main__":
    main()
