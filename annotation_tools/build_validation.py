#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Формальная валидация фильтра (раздел 2.4) + аудиопроверка контрольной группы.

Собирает:
  validation_A.csv     — для вас (слепой файл, свой порядок записей)
  validation_B.csv     — для разметчика (те же записи, другой порядок)
  validation.KEY.csv   — метки фильтра и размеры страт. НЕ ОТКРЫВАТЬ до конца разметки,
                         сразу прислать для расчёта
  spotcheck_100.csv    — 100 записей контрольной группы для проверки загрязнения
  validation_package.zip — всё, что нужно разметчику: его файлы, аудио, программа, протокол

Выборка: 200 записей, стратифицированных по 6 подкорпусам и 3 меткам фильтра,
не пересекается с калибровочной выборкой и с прослушиванием заполнителей.

Запуск из корня корпуса:
    python3 code/build_validation.py manifest_v2.csv
"""
import collections
import csv
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import audio_path, flac_duration  # noqa: E402

TH1, TH2, MIN_TOK = 5.0, 3.0, 6
TARGET = {"likely_spontaneous": 70, "maybe_spontaneous": 60, "likely_prepared_or_read": 70}
DOMAINS = ["crowdsourced", "parliament", "podcasts", "radio", "talkshow", "tv_news"]
SEED = 20260923


def label(score):
    return ("likely_spontaneous" if score >= TH1 else
            "maybe_spontaneous" if score >= TH2 else "likely_prepared_or_read")


def excluded_ids():
    ids = set()
    for f, col in (("calibration_samples.csv", "record_id"), ("filler_audit.csv", "record_id")):
        if os.path.exists(f):
            ids |= {r[col] for r in csv.DictReader(open(f, encoding="utf-8"))}
    return ids


def allocate(target, avail):
    """Поровну по подкорпусам, недобор перераспределяется туда, где записи есть."""
    alloc = {d: 0 for d in avail}
    left = target
    while left > 0:
        open_d = [d for d in avail if alloc[d] < avail[d]]
        if not open_d:
            break
        for d in open_d:
            if left == 0:
                break
            alloc[d] += 1
            left -= 1
    return alloc


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "manifest_v2.csv"
    skip = excluded_ids()
    print("исключено (калибровка и прослушивание заполнителей): {}".format(len(skip)))

    strata = collections.defaultdict(list)
    for r in csv.DictReader(open(manifest, encoding="utf-8")):
        if int(r["tokens"]) < MIN_TOK or r["record_id"] in skip:
            continue
        dom = r["domain"].split("/")[-1]
        strata[(dom, label(float(r["score"])))].append(r)

    rnd = random.Random(SEED)
    chosen = []
    print("\n{:<26}{:>14}{:>10}".format("страта", "в корпусе", "в выборке"))
    for lab, tgt in TARGET.items():
        avail = {d: len(strata[(d, lab)]) for d in DOMAINS if strata[(d, lab)]}
        alloc = allocate(tgt, avail)
        for d in DOMAINS:
            pool = strata[(d, lab)][:]
            rnd.shuffle(pool)
            k, got = alloc.get(d, 0), 0
            for r in pool:
                if got == k:
                    break
                if flac_duration(audio_path(r["record_id"])) is None:
                    continue
                r["_stratum"] = "{}|{}".format(d, lab)
                r["_N"] = len(strata[(d, lab)])
                r["_n"] = k
                chosen.append(r)
                got += 1
            if k:
                print("{:<26}{:>14,}{:>10}".format("{} / {}".format(d, lab[:12]), len(strata[(d, lab)]), got))
    print("итого: {}".format(len(chosen)))

    for i, r in enumerate(sorted(chosen, key=lambda x: x["record_id"]), 1):
        r["_id"] = "V{:03d}".format(i)

    with open("validation.KEY.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "record_id", "domain", "label", "score", "stratum", "stratum_N", "stratum_n"])
        for r in chosen:
            w.writerow([r["_id"], r["record_id"], r["domain"], label(float(r["score"])),
                        r["score"], r["_stratum"], r["_N"], r["_n"]])

    for name, seed in (("validation_A.csv", 101), ("validation_B.csv", 202)):
        order = chosen[:]
        random.Random(seed).shuffle(order)
        with open(name, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "record_id", "preview", "ОЦЕНКА", "УВЕРЕННОСТЬ", "ЗАМЕТКА"])
            for r in order:
                w.writerow([r["_id"], r["record_id"], r["preview"], "", "", ""])

    ctrl = list(csv.DictReader(open("golden/prep_eval_matched.csv", encoding="utf-8")))
    rnd.shuffle(ctrl)
    ctrl = ctrl[:100]
    with open("spotcheck_100.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "record_id", "text", "ОТВЕТ", "ЗАМЕТКА"])
        for i, r in enumerate(ctrl, 1):
            w.writerow(["C{:03d}".format(i), r["record_id"], r["text"], "", ""])

    audio = {audio_path(r["record_id"]) for r in chosen} | {audio_path(r["record_id"]) for r in ctrl}
    here = os.path.dirname(os.path.abspath(__file__))
    with zipfile.ZipFile("validation_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(audio):
            z.write(p)
        z.write("validation_B.csv")
        z.write("spotcheck_100.csv")
        for extra in ("validate_listen.py", "protocol_razmetki.md"):
            src = os.path.join(here, extra)
            if os.path.exists(src):
                z.write(src, extra)
    print("\nфайлы: validation_A.csv, validation_B.csv, validation.KEY.csv, spotcheck_100.csv")
    print("пакет для разметчика: validation_package.zip ({:.0f} МБ, {} аудио)".format(
        os.path.getsize("validation_package.zip") / 1e6, len(audio)))


if __name__ == "__main__":
    main()
