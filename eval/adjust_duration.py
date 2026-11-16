#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjust_duration.py -- Table 12: the register gap after standardizing the
prepared control for utterance duration, in two ways.

  pooled          контроль перевзвешивается по распределению длительностей
                  спонтанного набора, состав сессий не контролируется
  within-session  то же, но внутри каждой сессии записи, после чего сессии
                  агрегируются; одновременно устраняются и длина, и домен

Печатает состав доменов по интервалам длительности: без этого нельзя понять,
не измеряет ли «поправка на длину» на самом деле разницу доменов.

Запуск:
    python3 adjust_duration.py \
        --spont results/asr_runs/abilmansplus__whisper-turbo-ksc2__spont_eval.csv \
        --prep  results/asr_runs/abilmansplus__whisper-turbo-ksc2__prep_eval_matched.csv \
        --bins 3 --out table_adjusted_ksc2.csv
"""
import argparse, collections, csv, math, random, re, sys

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
MODES = (("a", False), ("c", "canon"), ("b", True))


def normalize(text, mode):
    toks = PUNCT_RE.sub(" ", text.lower()).split()
    if mode is True or mode == "stripped":
        toks = [x for x in toks if x not in FILLERS]
    elif mode == "canon":
        toks = ["<F>" if x in FILLERS else x for x in toks]
    return " ".join(toks)


def session_of(rid):
    stem = re.sub(r"(\.txt)+$", "", rid)
    m = SESSION_RE.match(stem)
    return m.group(1) if m else stem


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for c in ("record_id", "reference", "hypothesis", "dur_sec"):
        if not rows or c not in rows[0]:
            sys.exit("в %s нет колонки %s" % (path, c))
    return rows


def prepare(rows, edges):
    """-> [(session, bin, domain, {mode: (err, ref)})]"""
    import jiwer
    out = []
    for r in rows:
        d = float(r["dur_sec"])
        b = 0
        while b < len(edges) and d > edges[b]:
            b += 1
        per = {}
        for m, mode in MODES:
            ref, hyp = normalize(r["reference"], mode), normalize(r["hypothesis"], mode)
            if not ref:
                per[m] = (0, 0); continue
            o = jiwer.process_words(ref, hyp)
            per[m] = (o.substitutions + o.deletions + o.insertions,
                      o.substitutions + o.deletions + o.hits)
        out.append((session_of(r["record_id"]), b, r.get("domain", "?"), per))
    return out


def cells(items, m, sessions=None):
    """{(session, bin): [err, ref]}"""
    out = {}
    for s, b, _, per in items:
        if sessions is not None and s not in sessions:
            continue
        e, n = per[m]
        a = out.setdefault((s, b), [0, 0]); a[0] += e; a[1] += n
    return out


def plain(c):
    E = sum(v[0] for v in c.values()); N = sum(v[1] for v in c.values())
    return (100.0 * E / N if N else float("nan")), N


def standardize(sc, pc, within):
    """Возвращает (WER спонт. по общим ячейкам, стандартизованный WER контроля,
    доля удержанных эталонных слов спонтанного набора)."""
    if within:
        keys = [(s, b) for (s, b) in sc if (s, b) in pc and pc[(s, b)][1] > 0]
    else:                                  # объединяем сессии: ячейка = интервал
        sb = collections.defaultdict(lambda: [0, 0]); pb = collections.defaultdict(lambda: [0, 0])
        for (s, b), v in sc.items():
            sb[b][0] += v[0]; sb[b][1] += v[1]
        for (s, b), v in pc.items():
            pb[b][0] += v[0]; pb[b][1] += v[1]
        sc, pc = {(None, b): v for b, v in sb.items()}, {(None, b): v for b, v in pb.items()}
        keys = [(None, b) for b in sb if (None, b) in pc and pc[(None, b)][1] > 0]
    if not keys:
        return float("nan"), float("nan"), 0.0
    se = sum(sc[k][0] for k in keys); sn = sum(sc[k][1] for k in keys)
    num = sum(sc[k][1] * (100.0 * pc[k][0] / pc[k][1]) for k in keys)
    den = sn
    total = sum(v[1] for v in sc.values())
    return (100.0 * se / sn if sn else float("nan"),
            num / den if den else float("nan"),
            sn / total if total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spont", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default="table_adjusted.csv")
    ap.add_argument("--bins", type=int, default=3)
    args = ap.parse_args()

    sp_rows, pr_rows = load(args.spont), load(args.prep)
    durs = sorted(float(r["dur_sec"]) for r in sp_rows)
    edges = [durs[int(len(durs) * k / args.bins)] for k in range(1, args.bins)]
    sp, pr = prepare(sp_rows, edges), prepare(pr_rows, edges)
    shared = sorted({s for s, _, _, _ in sp} & {s for s, _, _, _ in pr})
    lab = []
    for b in range(len(edges) + 1):
        lo = "%.2f" % edges[b - 1] if b else "0"
        hi = "%.2f" % edges[b] if b < len(edges) else "inf"
        lab.append("%s-%s" % (lo, hi))

    print("границы интервалов, с: %s" % ", ".join("%.2f" % e for e in edges))
    print("высказываний: спонтанных %d, контроль %d; общих сессий %d\n"
          % (len(sp), len(pr), len(shared)))

    print("состав доменов по интервалам (доля высказываний)")
    for name, items in (("спонт.", sp), ("контр.", pr)):
        for b in range(len(edges) + 1):
            dom = collections.Counter(d for _, bb, d, _ in items if bb == b)
            tot = sum(dom.values()) or 1
            top = ", ".join("%s %.0f%%" % (k, 100.0 * v / tot)
                            for k, v in dom.most_common(4))
            print("  %s %-14s n=%4d  %s" % (name, lab[b], tot, top))
        print()

    rnd = random.Random(SEED)
    rows = []
    for m, _ in MODES:
        sc, pc = cells(sp, m), cells(pr, m)
        raw_s, _ = plain(sc); raw_p, _ = plain(pc)
        line = {"condition": m, "wer_spont": round(raw_s, 2),
                "wer_prep_raw": round(raw_p, 2)}
        for tag, within in (("pooled", False), ("within_session", True)):
            s_r, p_r, cov = standardize(sc, pc, within)
            ratio = s_r / p_r if p_r else float("nan")
            boots = []
            for _ in range(B):
                pick = {shared[rnd.randrange(len(shared))] for _ in shared}
                a, b_, _c = standardize(cells(sp, m, pick), cells(pr, m, pick), within)
                if b_ and not math.isnan(a) and not math.isnan(b_):
                    boots.append(a / b_)
            boots.sort()
            lo = boots[int(0.025 * len(boots))] if boots else float("nan")
            hi = boots[int(0.975 * len(boots)) - 1] if boots else float("nan")
            line.update({tag + "_wer_spont": round(s_r, 2),
                         tag + "_wer_prep": round(p_r, 2),
                         tag + "_ratio": round(ratio, 3),
                         tag + "_ci_lo": round(lo, 3), tag + "_ci_hi": round(hi, 3),
                         tag + "_coverage": round(cov, 3)})
            print("  (%s) %-14s  спонт. %5.2f | контроль %5.2f  отношение %.3f [%.2f–%.2f]"
                  "  охват %.0f%%" % (m, tag, s_r, p_r, ratio, lo, hi, 100 * cov))
        rows.append(line)
        print()

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("таблица: %s" % args.out)


if __name__ == "__main__":
    main()
