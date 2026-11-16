#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Считает Таблицу 7 для любой пары наборов: WER и CER в трёх условиях шкалирования,
отношение спонтанный/подготовленный и доверительный интервал парным бутстрэпом
по сессиям записи. Дополнительно — счётчики подстановок, удалений и вставок
(пункт «Error types» списка дополнительных сравнений).

Нормализация скопирована из eval_zeroshot.py дословно, иначе цифры не сойдутся
с опубликованными.

Запуск, пункт 8 (без прогона, подмножество уже посчитанного контроля):
    python score_sets.py \
        --spont results/asr_runs/MODEL__spont_eval.csv \
        --prep  results/asr_runs/MODEL__prep_eval_matched.csv \
        --prep-ids sets/prep_duration_matched.ids \
        --label "duration-matched control" --out table_matched.csv

Запуск, пункт 1 (после прогона наборов слушателей):
    python score_sets.py \
        --spont results/MODEL__listener_spont.csv \
        --prep  results/MODEL__listener_prep.csv \
        --label "listener-labelled" --out table_listener.csv
"""
import argparse, csv, math, random, re, sys

FILLERS = {
    "ыы", "ыыы", "ыыыы", "ыыыыы", "ыыыыыы",
    "іі", "ііі", "іііі", "ііііі",
    "мм", "ммм", "мммм", "аа", "ааа", "аааа", "оо", "ооо",
    "ээ", "эээ", "мхм", "мһм", "аха", "ыһы", "әһә",
    "ы", "і", "м", "э", "эм", "ә", "о", "а",
}
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
SESSION_RE = re.compile(r"^(.*)_\d+$")
SEED, B = 20261014, 10000

def normalize(text, mode):
    t = PUNCT_RE.sub(" ", text.lower())
    toks = t.split()
    if mode is True or mode == "stripped":
        toks = [x for x in toks if x not in FILLERS]
    elif mode == "canon":
        toks = ["<F>" if x in FILLERS else x for x in toks]
    return " ".join(toks)

def session_of(record_id):
    stem = re.sub(r"(\.txt)+$", "", record_id)
    m = SESSION_RE.match(stem)
    return m.group(1) if m else stem

def counts(ref, hyp):
    """ошибки и длина эталона на уровне слов, плюс S/D/I"""
    import jiwer
    o = jiwer.process_words(ref, hyp)
    err = o.substitutions + o.deletions + o.insertions
    n = o.substitutions + o.deletions + o.hits
    return err, n, o.substitutions, o.deletions, o.insertions

def char_counts(ref, hyp):
    import jiwer
    o = jiwer.process_characters(ref, hyp)
    err = o.substitutions + o.deletions + o.insertions
    n = o.substitutions + o.deletions + o.hits
    return err, n

def load(path, ids=None):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("пустой файл: %s" % path)
    for c in ("record_id", "reference", "hypothesis"):
        if c not in rows[0]:
            sys.exit("в %s нет колонки %s — нужен run-файл С ТЕКСТОМ" % (path, c))
    if ids is not None:
        keep = set(ids)
        rows = [r for r in rows if r["record_id"] in keep]
        miss = keep - {r["record_id"] for r in rows}
        if miss:
            print("предупреждение: %d записей из списка нет в run-файле" % len(miss))
    return rows

MODES = (("a", False), ("c", "canon"), ("b", True))

def utt_counts(rows, m):
    """[(ошибки, эталонных слов)] по каждому высказыванию — для непарного бутстрэпа"""
    mode = dict(MODES)[m]
    out = []
    for r in rows:
        ref, hyp = normalize(r["reference"], mode), normalize(r["hypothesis"], mode)
        if not ref:
            continue
        e, n, _, _, _ = counts(ref, hyp)
        out.append((e, n))
    return out


def tally(rows):
    """по сессиям: {session: {mode: [err, n, S, D, I, cerr, cn]}}"""
    out = {}
    for r in rows:
        s = session_of(r["record_id"])
        d = out.setdefault(s, {m: [0, 0, 0, 0, 0, 0, 0] for m, _ in MODES})
        for m, mode in MODES:
            ref, hyp = normalize(r["reference"], mode), normalize(r["hypothesis"], mode)
            if not ref:
                continue
            e, n, S, D, I = counts(ref, hyp)
            ce, cn = char_counts(ref, hyp)
            acc = d[m]
            acc[0] += e; acc[1] += n; acc[2] += S; acc[3] += D; acc[4] += I
            acc[5] += ce; acc[6] += cn
    return out

def rate(tal, sessions, m, char=False):
    e = sum(tal[s][m][5 if char else 0] for s in sessions if s in tal)
    n = sum(tal[s][m][6 if char else 1] for s in sessions if s in tal)
    return 100.0 * e / n if n else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spont", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--spont-ids", default=None)
    ap.add_argument("--prep-ids", default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="table.csv")
    ap.add_argument("--unpaired", action="store_true",
                    help="бутстрэп по высказываниям, а не по сессиям; нужен, когда\n                          наборы почти не делят сессии (набор слушателей)")
    args = ap.parse_args()

    rid = lambda p: [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
    sp = load(args.spont, rid(args.spont_ids) if args.spont_ids else None)
    pr = load(args.prep, rid(args.prep_ids) if args.prep_ids else None)
    tsp, tpr = tally(sp), tally(pr)
    shared = sorted(set(tsp) & set(tpr))
    print("набор: %s" % (args.label or "(без имени)"))
    print("высказываний: спонтанных %d, подготовленных %d" % (len(sp), len(pr)))
    print("сессий: спонтанных %d, подготовленных %d, общих %d"
          % (len(tsp), len(tpr), len(shared)))
    if len(shared) < 5 and not args.unpaired:
        print("ВНИМАНИЕ: общих сессий мало — добавьте --unpaired,\n"
              "          иначе интервалы по сессиям бессмысленны")
    if args.unpaired:
        print("интервалы: непарный бутстрэп по высказываниям")

    rows = []
    rnd = random.Random(SEED)
    for m, _ in MODES:
        ws = rate(tsp, tsp.keys(), m); wp = rate(tpr, tpr.keys(), m)
        cs = rate(tsp, tsp.keys(), m, True); cp = rate(tpr, tpr.keys(), m, True)
        ratio = ws / wp if wp else float("nan")
        boots = []
        if args.unpaired:
            A = [(r_[0], r_[1]) for r_ in utt_counts(sp, m)]
            P = [(r_[0], r_[1]) for r_ in utt_counts(pr, m)]
            for _ in range(B):
                ea = na = eb = nb = 0
                for _ in range(len(A)):
                    e, n = A[rnd.randrange(len(A))]; ea += e; na += n
                for _ in range(len(P)):
                    e, n = P[rnd.randrange(len(P))]; eb += e; nb += n
                if na and nb:
                    boots.append((ea / na) / (eb / nb))
            boots.sort()
        elif shared:
            for _ in range(B):
                pick = [shared[rnd.randrange(len(shared))] for _ in shared]
                a, b = rate(tsp, pick, m), rate(tpr, pick, m)
                if b and not math.isnan(a) and not math.isnan(b):
                    boots.append(a / b)
            boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else float("nan")
        hi = boots[int(0.975 * len(boots)) - 1] if boots else float("nan")
        S = sum(tsp[s][m][2] for s in tsp); D = sum(tsp[s][m][3] for s in tsp)
        I = sum(tsp[s][m][4] for s in tsp)
        rows.append({"condition": m, "wer_spont": round(ws, 2), "wer_prep": round(wp, 2),
                     "ratio": round(ratio, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "cer_spont": round(cs, 2), "cer_prep": round(cp, 2),
                     "sub_spont": S, "del_spont": D, "ins_spont": I,
                     "err_spont": sum(tsp[s][m][0] for s in tsp),
                     "ref_spont": sum(tsp[s][m][1] for s in tsp),
                     "err_prep": sum(tpr[s][m][0] for s in tpr),
                     "ref_prep": sum(tpr[s][m][1] for s in tpr)})
        print("  (%s)  WER %5.2f / %5.2f  отношение %.3f [%.2f–%.2f]   CER %5.2f / %5.2f"
              % (m, ws, wp, ratio, lo, hi, cs, cp))

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nтаблица: %s" % args.out)


if __name__ == "__main__":
    main()
