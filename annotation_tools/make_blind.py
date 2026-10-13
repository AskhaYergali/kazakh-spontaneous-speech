#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Готовит ослеплённую копию калибровочной выборки для второго слушателя.

Убирает всё, что может подсказать ответ: score, счётчики признаков, полосу,
домен, длительность. Оставляет только идентификатор, превью и пустую колонку
для оценки. Порядок перемешивается другим сидом, чтобы соседство записей
не совпадало с первой разметкой.

Запуск:
    python3 code/make_blind.py calibration_samples.csv -o annotator_2.csv

Сохраняет также карту соответствия blind_id -> record_id в отдельный файл,
который второму слушателю НЕ передаётся.
"""

import argparse
import csv
import os
import random
import sys

KEEP = ("record_id", "preview")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="исходный calibration_samples.csv")
    ap.add_argument("-o", "--output", default="annotator_2.csv")
    ap.add_argument("--seed", type=int, default=777, help="сид перемешивания")
    ap.add_argument("--keep-domain", action="store_true",
                    help="оставить колонку домена (по умолчанию скрыта)")
    args = ap.parse_args()

    with open(args.source, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("Пустой файл: " + args.source)

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)

    out, keymap = [], []
    for i, r in enumerate(rows, 1):
        bid = "K{:03d}".format(i)
        item = {"id": bid, "record_id": r["record_id"], "preview": r["preview"]}
        if args.keep_domain:
            item["domain"] = r["domain"]
        item["ОЦЕНКА"] = ""
        item["УВЕРЕННОСТЬ"] = ""
        item["ЗАМЕТКА"] = ""
        out.append(item)
        keymap.append({"id": bid, "record_id": r["record_id"],
                       "score": r["score"], "band": r["band"],
                       "domain": r["domain"]})

    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    keyfile = os.path.splitext(args.output)[0] + ".KEY.csv"
    with open(keyfile, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keymap[0].keys()))
        w.writeheader()
        w.writerows(keymap)

    print("Файл для второго слушателя: {} ({} записей)".format(args.output, len(out)))
    print("Скрыты: score, band, все счётчики признаков{}".format(
        "" if args.keep_domain else ", домен"))
    print("Карта соответствия (НЕ передавать слушателю): {}".format(keyfile))
    print()
    print("Инструкция слушателю: в колонке ОЦЕНКА ставить S или P,")
    print("в УВЕРЕННОСТЬ — 1/2/3 (высокая/средняя/низкая),")
    print("в ЗАМЕТКА — одно-два слова, что решило (ритм, паузы, чтение и т.п.).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
