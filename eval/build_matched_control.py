#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_matched_control.py -- the one-to-one duration-matched control of
Section 6.5.

Для каждого спонтанного высказывания подбирает подготовленное из ТОЙ ЖЕ сессии
с ближайшей длительностью, без возвращения. Кандидаты берутся из уже
прогнанного контроля, поэтому новый прогон ASR не нужен — на выходе список
record_id, который score_sets.py применяет как фильтр к существующим run-файлам.

Запуск:
    python build_matched_control.py \
        --spont results/asr_runs/abilmansplus__whisper-turbo-ksc2__spont_eval.csv \
        --prep  results/asr_runs/abilmansplus__whisper-turbo-ksc2__prep_eval_matched.csv \
        --out   sets/prep_duration_matched.ids

Если пар не хватает, скрипт об этом скажет: тогда нужен добор кандидатов
из манифеста и отдельный прогон.
"""
import argparse, csv, re, sys

SESSION_RE = re.compile(r"^(.*)_\d+$")

def session_of(record_id):
    stem = re.sub(r"(\.txt)+$", "", record_id)
    m = SESSION_RE.match(stem)
    return m.group(1) if m else stem

def rd(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spont", required=True, help="run-файл спонтанного набора")
    ap.add_argument("--prep", required=True, help="run-файл подготовленного контроля")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="допуск в секундах для отчёта о качестве подбора")
    args = ap.parse_args()

    spont = rd(args.spont)
    prep = rd(args.prep)
    for name, rows in (("spont", spont), ("prep", prep)):
        if not rows or "dur_sec" not in rows[0]:
            sys.exit("в %s нет колонки dur_sec" % name)

    pool = {}
    for r in prep:
        pool.setdefault(session_of(r["record_id"]), []).append(
            (float(r["dur_sec"]), r["record_id"]))
    for s in pool:
        pool[s].sort()

    chosen, deltas, unmatched = [], [], []
    # длинные подбираем первыми: им труднее всего найти пару
    for r in sorted(spont, key=lambda x: -float(x["dur_sec"])):
        sess = session_of(r["record_id"])
        cand = pool.get(sess) or []
        if not cand:
            unmatched.append((r["record_id"], sess, float(r["dur_sec"]))); continue
        d = float(r["dur_sec"])
        best = min(range(len(cand)), key=lambda k: abs(cand[k][0] - d))
        dur, rid = cand.pop(best)
        chosen.append(rid); deltas.append(abs(dur - d))

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(chosen) + "\n")

    sd = sorted(float(r["dur_sec"]) for r in spont)
    cd = sorted(deltas)
    picked = {rid for rid in chosen}
    pd_ = sorted(float(r["dur_sec"]) for r in prep if r["record_id"] in picked)
    med = lambda v: v[len(v) // 2] if v else float("nan")
    mean = lambda v: sum(v) / len(v) if v else float("nan")

    print("подобрано пар: %d из %d" % (len(chosen), len(spont)))
    print("длительность, спонтанные:  среднее %.2f с, медиана %.2f с, сумма %.2f ч"
          % (mean(sd), med(sd), sum(sd) / 3600.0))
    print("длительность, подобранные: среднее %.2f с, медиана %.2f с, сумма %.2f ч"
          % (mean(pd_), med(pd_), sum(pd_) / 3600.0))
    print("|Δдлительности|: среднее %.2f с, медиана %.2f с, максимум %.2f с"
          % (mean(cd), med(cd), max(cd) if cd else float("nan")))
    within = sum(1 for x in cd if x <= args.tolerance)
    print("пар в пределах %.1f с: %d (%.1f%%)"
          % (args.tolerance, within, 100.0 * within / max(len(cd), 1)))
    if unmatched:
        print("\nбез пары осталось %d высказываний — в их сессиях кандидаты кончились:"
              % len(unmatched))
        for rid, sess, d in unmatched[:10]:
            print("   %s  сессия %s  %.1f с" % (rid, sess, d))
        print("если таких много или средняя |Δ| велика, нужен добор кандидатов "
              "из манифеста и отдельный прогон ASR по ним")
    print("\nсписок записей: %s" % args.out)


if __name__ == "__main__":
    main()
