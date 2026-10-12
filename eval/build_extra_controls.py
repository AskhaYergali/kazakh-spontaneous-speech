#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительные контроли к эксперименту B.

1) source_control.csv — ПОДГОТОВЛЕННЫЕ записи (S < theta_2) из тех же разговорных
   подкорпусов, что и Golden Dataset, согласованные с ним по числу записей и
   распределению длительности. Отличается от Golden только плотностью заполнителей,
   но не акустикой источника: разводит два фактора, слитых в random_train.

2) random_500 / random_1000 / random_2000 — подвыборки нецелевого набора для
   проверки зависимости эффекта от объёма данных.

Запуск из корня корпуса:
    python3 code/build_extra_controls.py manifest_v2.csv
"""
import collections
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import audio_path, flac_duration, write_set  # noqa: E402

TH2 = 3.0
SPONT_DOMAINS = ("podcasts", "radio", "talkshow", "parliament")
BIN = 1.0
SEED = 20260924


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "manifest_v2.csv"
    target = list(csv.DictReader(open("golden/spont_train.csv", encoding="utf-8")))
    need = collections.Counter(int(float(r["dur_sec"]) // BIN) for r in target)
    print("цель: {} записей, {:.2f} ч".format(
        len(target), sum(float(r["dur_sec"]) for r in target) / 3600))

    banned = set()
    for f in ("golden/spont_tune.csv", "golden/spont_eval.csv"):
        banned |= {r["speaker_group"] for r in csv.DictReader(open(f, encoding="utf-8"))}

    pool = [r for r in csv.DictReader(open(manifest, encoding="utf-8"))
            if r["domain"].startswith("Train/")
            and r["domain"].split("/")[1] in SPONT_DOMAINS
            and int(r["tokens"]) >= 6 and float(r["score"]) < TH2
            and r["speaker_group"] not in banned]
    print("пул подготовленных записей разговорных источников: {}".format(len(pool)))

    rnd = random.Random(SEED)
    rnd.shuffle(pool)
    chosen, got = [], collections.Counter()
    for r in pool:
        d = flac_duration(audio_path(r["record_id"]))
        if d is None:
            continue
        b = int(d // BIN)
        if got[b] < need[b]:
            got[b] += 1
            chosen.append(r)
            if len(chosen) == len(target):
                break
    short = sum(need[b] - got[b] for b in need if got[b] < need[b])
    if short:
        print("!! не хватило {} записей — добираю без учёта корзин".format(short))
        for r in pool:
            if len(chosen) == len(target):
                break
            if r in chosen:
                continue
            if flac_duration(audio_path(r["record_id"])) is not None:
                chosen.append(r)

    chosen.sort(key=lambda r: r["record_id"])
    st = write_set("golden/source_control.csv", chosen, ".")
    fill_t = sum(int(r["strong_fillers"]) for r in target)
    fill_c = sum(int(r["strong_fillers"]) for r in chosen)
    doms = collections.Counter(r["domain"].split("/")[1] for r in chosen)
    print("\nsource_control: {} записей, {:.2f} ч, групп дикторов {}".format(
        st["n"], st["hours"], st["speakers"]))
    print("по подкорпусам:", dict(doms.most_common()))
    print("сильных заполнителей: Golden {} / source_control {} / random_train {}".format(
        fill_t, fill_c,
        sum(int(r["strong_fillers"]) for r in
            csv.DictReader(open("golden/random_train.csv", encoding="utf-8")))))

    rows = list(csv.DictReader(open("golden/random_train.csv", encoding="utf-8")))
    rnd2 = random.Random(SEED + 1)
    rnd2.shuffle(rows)
    for n in (500, 1000, 2000):
        sub = sorted(rows[:n], key=lambda r: r["record_id"])
        path = "golden/random_{}.csv".format(n)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sub)
        h = sum(float(r["dur_sec"]) for r in sub) / 3600
        f = sum(int(r["strong_fillers"]) for r in sub)
        print("{}: {} записей, {:.2f} ч, заполнителей {}".format(path, n, h, f))


if __name__ == "__main__":
    main()
