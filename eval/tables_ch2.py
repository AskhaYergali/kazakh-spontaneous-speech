#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Досчитывает незаполненные ячейки таблиц 2.2 и 2.3 диссертации.

Метки пересчитываются из score по ТЕКУЩИМ порогам (theta_1 = 5.0, theta_2 = 3.0),
поэтому годится manifest_v2.csv, созданный ещё со старыми порогами.
Длительности читаются из заголовков FLAC.

Запуск из корня корпуса:
    python3 code/tables_ch2.py manifest_v2.csv
"""
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from build_dataset import audio_path, flac_duration  # noqa: E402

TH1, TH2, MIN_TOK = 5.0, 3.0, 6
ORDER = ["crowdsourced", "parliament", "podcasts", "radio", "talkshow", "tv_news"]
NAMES = {"crowdsourced": "Crowdsourced (read prompts)", "parliament": "Parliament",
         "podcasts": "Podcasts", "radio": "Radio", "talkshow": "Talk shows", "tv_news": "TV news"}


def label(r):
    if int(r["tokens"]) < MIN_TOK:
        return "too_short"
    s = float(r["score"])
    return "likely_spontaneous" if s >= TH1 else ("maybe_spontaneous" if s >= TH2 else "likely_prepared_or_read")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "manifest_v2.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print("записей: {}  (чтение длительностей займёт несколько минут)".format(len(rows)))

    by_dom = collections.defaultdict(lambda: collections.Counter())
    hours_dom = collections.Counter()
    part = collections.defaultdict(lambda: {"n": 0, "h": 0.0, "s": 0.0, "t": 0, "fs": 0.0})

    for i, r in enumerate(rows, 1):
        dom = r["domain"].split("/")[-1]
        lab = label(r)
        d = flac_duration(audio_path(r["record_id"])) or 0.0
        by_dom[dom][lab] += 1
        hours_dom[dom] += d
        p = part[lab]
        p["n"] += 1
        p["h"] += d
        p["s"] += float(r["score"])
        p["t"] += int(r["tokens"])
        p["fs"] += float(r["strong_filler_per_1000"])
        if i % 50000 == 0:
            print("  {} / {}".format(i, len(rows)))

    print("\nТАБЛИЦА 2.2 (проценты от всех файлов подкорпуса, включая too_short)")
    print("{:<30}{:>9}{:>9}{:>10}{:>10}{:>10}".format("Subcorpus", "Files", "Hours", "%likely", "%maybe", "%prep"))
    tot = collections.Counter()
    th = 0.0
    for dom in ORDER:
        c = by_dom[dom]
        n = sum(c.values())
        tot.update(c)
        th += hours_dom[dom]
        print("{:<30}{:>9,}{:>9.1f}{:>10.1f}{:>10.1f}{:>10.1f}".format(
            NAMES[dom], n, hours_dom[dom] / 3600,
            100 * c["likely_spontaneous"] / n, 100 * c["maybe_spontaneous"] / n,
            100 * c["likely_prepared_or_read"] / n))
    n = sum(tot.values())
    print("{:<30}{:>9,}{:>9.1f}{:>10.1f}{:>10.1f}{:>10.1f}".format(
        "Total, human speech", n, th / 3600,
        100 * tot["likely_spontaneous"] / n, 100 * tot["maybe_spontaneous"] / n,
        100 * tot["likely_prepared_or_read"] / n))

    print("\nТАБЛИЦА 2.3")
    print("{:<26}{:>11}{:>9}{:>10}{:>10}{:>10}".format("Partition", "Utterances", "Hours", "Mean S", "Mean tok", "Mean φs"))
    for lab in ("likely_spontaneous", "maybe_spontaneous", "likely_prepared_or_read", "too_short"):
        p = part[lab]
        k = max(p["n"], 1)
        print("{:<26}{:>11,}{:>9.1f}{:>10.2f}{:>10.1f}{:>10.1f}".format(
            lab, p["n"], p["h"] / 3600, p["s"] / k, p["t"] / k, p["fs"] / k))


if __name__ == "__main__":
    main()
