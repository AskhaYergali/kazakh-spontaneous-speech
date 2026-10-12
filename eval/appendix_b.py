#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Приложение B: результат фильтра по подкорпусам и разделам корпуса.

Метки пересчитываются по действующим порогам (theta_1 = 5.0, theta_2 = 3.0),
длительности читаются из заголовков FLAC. Вывод готов к переносу в текст.

Запуск из корня корпуса:
    python3 code/appendix_b.py manifest_v2.csv
"""
import collections, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import audio_path, flac_duration  # noqa: E402

TH1, TH2, MIN_TOK = 5.0, 3.0, 6
DOMS = ["crowdsourced", "parliament", "podcasts", "radio", "talkshow", "tv_news"]
LABS = ["likely_spontaneous", "maybe_spontaneous", "likely_prepared_or_read", "too_short"]


def label(r):
    if int(r["tokens"]) < MIN_TOK:
        return "too_short"
    s = float(r["score"])
    return LABS[0] if s >= TH1 else (LABS[1] if s >= TH2 else LABS[2])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "manifest_v2.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    n = collections.Counter(); h = collections.Counter(); sc = collections.defaultdict(list)
    for i, r in enumerate(rows, 1):
        if "/" not in r["domain"]:
            continue
        sp, dom = r["domain"].split("/", 1)
        if dom not in DOMS:
            continue
        lab = label(r)
        d = flac_duration(audio_path(r["record_id"])) or 0.0
        n[(sp, dom, lab)] += 1; h[(sp, dom, lab)] += d
        if lab != "too_short":
            sc[(dom, lab)].append(float(r["score"]))
        if i % 100000 == 0:
            print("  {} / {}".format(i, len(rows)), file=sys.stderr)

    print("\nB.1 — Записи и часы по разделу, подкорпусу и метке фильтра")
    print("{:<7}{:<14}".format("раздел", "подкорпус") + "".join("{:>26}".format(l[:24]) for l in LABS))
    for sp in ("Train", "Dev", "Test"):
        for dom in DOMS:
            if not any(n[(sp, dom, l)] for l in LABS):
                continue
            cells = "".join("{:>18,} / {:>5.1f}h".format(n[(sp, dom, l)], h[(sp, dom, l)] / 3600) for l in LABS)
            print("{:<7}{:<14}".format(sp, dom) + cells)

    print("\nB.2 — Распределение оценки S по подкорпусам (без too_short)")
    print("{:<14}{:>8}{:>9}{:>9}{:>9}{:>9}{:>9}".format("подкорпус", "n", "медиана", "p75", "p90", "p95", "макс"))
    for dom in DOMS:
        v = sorted(sc[(dom, LABS[0])] + sc[(dom, LABS[1])] + sc[(dom, LABS[2])])
        if not v:
            continue
        q = lambda p: v[min(len(v) - 1, int(len(v) * p))]
        print("{:<14}{:>8,}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}".format(
            dom, len(v), q(.5), q(.75), q(.9), q(.95), v[-1]))


if __name__ == "__main__":
    main()
