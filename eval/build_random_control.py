#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контрольная выборка для эксперимента B: случайные записи KSC2 Train,
совпадающие с spont_train по ЧИСЛУ записей и РАСПРЕДЕЛЕНИЮ ДЛИТЕЛЬНОСТИ.

Выборка нецелевая: регистр не учитывается, записи берутся в естественной
пропорции корпуса (в основном начитка). Исключаются только too_short и
группы дикторов, отведённые под настройку и оценку, — иначе нарушится
дикторо-независимость.

Запуск из корня корпуса:
    python3 code/build_random_control.py manifest_v2.csv
"""
import csv
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import audio_path, flac_duration, write_set  # noqa: E402

BIN = 1.0          # ширина корзины длительности, секунды
SEED = 20260922


def bin_of(sec):
    return int(sec // BIN)


def main():
    manifest = sys.argv[1] if len(sys.argv) > 1 else "manifest_v2.csv"
    target = list(csv.DictReader(open("golden/spont_train.csv", encoding="utf-8")))
    need = collections.Counter(bin_of(float(r["dur_sec"])) for r in target)
    t_hours = sum(float(r["dur_sec"]) for r in target) / 3600
    print("цель: {} записей, {:.2f} ч, {} корзин по {} с".format(len(target), t_hours, len(need), BIN))

    banned = set()
    for f in ("golden/spont_tune.csv", "golden/spont_eval.csv"):
        banned |= {r["speaker_group"] for r in csv.DictReader(open(f, encoding="utf-8"))}

    pool = [r for r in csv.DictReader(open(manifest, encoding="utf-8"))
            if r["domain"].startswith("Train/") and int(r["tokens"]) >= 6
            and r["speaker_group"] not in banned]
    print("пул Train (без too_short и запрещённых групп): {}".format(len(pool)))

    rnd = random.Random(SEED)
    rnd.shuffle(pool)
    chosen, got = [], collections.Counter()
    for i, r in enumerate(pool):
        d = flac_duration(audio_path(r["record_id"]))
        if d is None:
            continue
        b = bin_of(d)
        if got[b] < need[b]:
            got[b] += 1
            chosen.append(r)
            if len(chosen) == len(target):
                break
        if i and i % 50000 == 0:
            print("  просмотрено {}, набрано {}".format(i, len(chosen)))

    short = {b: need[b] - got[b] for b in need if got[b] < need[b]}
    if short:
        print("!! не хватило записей в корзинах: {}".format(short))

    chosen.sort(key=lambda r: r["record_id"])
    st = write_set("golden/random_train.csv", chosen, ".")
    doms = collections.Counter(r["domain"].split("/")[1] for r in chosen)
    spont = sum(1 for r in chosen if float(r["score"]) >= 5.0)
    print("\nrandom_train: {} записей, {:.2f} ч (цель {:.2f} ч), групп дикторов {}".format(
        st["n"], st["hours"], t_hours, st["speakers"]))
    print("по подкорпусам:", dict(doms.most_common()))
    print("из них случайно попало спонтанных (S >= 5): {} ({:.1%})".format(spont, spont / len(chosen)))
    fill_t = sum(int(r["strong_fillers"]) for r in target)
    fill_r = sum(int(r["strong_fillers"]) for r in chosen)
    print("сильных заполнителей: Golden {} против random {} — вот разница, которую проверяет эксперимент".format(fill_t, fill_r))


if __name__ == "__main__":
    main()
