#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Калибровка порогов theta_1 / theta_2 по полному манифесту.

Что делает:
  1. Считает РЕАЛЬНУЮ длительность из заголовков FLAC (без внешних библиотек).
  2. Строит таблицу "порог -> записей / часов" по сплитам и доменам.
  3. Проверяет наличие аудио у отобранных записей.
  4. Оценивает качество прокси-группировки дикторов.
  5. Выгружает пограничные примеры для просмотра глазами (это и есть калибровка).

Запуск ИЗ КОРНЯ КОРПУСА:
    cd /path/to/ISSAI_KSC2
    python3 code/calibrate.py manifest_v2.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import struct
import sys

SPONT_DOMAINS = ("parliament", "podcasts", "radio", "talkshow")


def flac_duration(path: str):
    """Длительность FLAC в секундах из блока STREAMINFO. None, если не прочиталось."""
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"fLaC":
                return None
            header = fh.read(4)
            if len(header) < 4:
                return None
            block_len = int.from_bytes(header[1:4], "big")
            info = fh.read(min(block_len, 34))
            if len(info) < 18:
                return None
            # биты 80..123 = sample_rate(20) + channels(3) + bps(5) + total_samples(36)
            chunk = int.from_bytes(info[10:18], "big")
            sample_rate = (chunk >> 44) & 0xFFFFF
            total_samples = chunk & 0xFFFFFFFFF
            if not sample_rate or not total_samples:
                return None
            return total_samples / sample_rate
    except OSError:
        return None


def audio_path(record_id: str) -> str:
    base = record_id
    for suf in (".txt.txt", ".txt"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base + ".flac"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--root", default=".", help="корень корпуса (по умолчанию текущая папка)")
    ap.add_argument("--samples", default="calibration_samples.csv")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    rows = []
    with open(args.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["score"] = float(r["score"])
            r["tokens"] = int(r["tokens"])
            split, _, dom = r["domain"].partition("/")
            r["split"], r["dom"] = split, dom
            rows.append(r)
    print("записей в манифесте: {}".format(len(rows)))

    # ---------- 1. длительности (только для не-too_short) ----------
    print("\nЧитаю заголовки FLAC (это займёт 1-3 минуты)...")
    cache = {}
    missing = 0
    read = 0
    for r in rows:
        if r["label"] == "too_short":
            r["dur"] = 0.0
            continue
        p = os.path.join(args.root, audio_path(r["record_id"]))
        d = flac_duration(p)
        if d is None:
            missing += 1
            r["dur"] = None
        else:
            r["dur"] = d
            read += 1
    print("прочитано {} файлов, недоступно {}".format(read, missing))

    durs = [r["dur"] for r in rows if r.get("dur")]
    if durs:
        durs_sorted = sorted(durs)
        print("средняя длительность: {:.2f} с, медиана {:.2f} с".format(
            sum(durs) / len(durs), durs_sorted[len(durs_sorted) // 2]))
        print("сек на токен (медиана отношения): {:.3f}".format(
            sorted(r["dur"] / r["tokens"] for r in rows
                   if r.get("dur") and r["tokens"])[len(durs_sorted) // 2]))

    # ---------- 2. таблица порогов ----------
    train_sp = [r for r in rows
                if r["split"] == "Train" and r["dom"] in SPONT_DOMAINS
                and r["label"] != "too_short"]
    eval_sp = [r for r in rows
               if r["split"] in ("Dev", "Test") and r["dom"] in SPONT_DOMAINS
               and r["label"] != "too_short"]

    def hours(sel):
        return sum(r["dur"] for r in sel if r.get("dur")) / 3600.0

    print("\n" + "=" * 74)
    print("TRAIN, спонтанные подкорпуса: сколько наберётся при разных theta_1")
    print("=" * 74)
    print("{:>8}{:>10}{:>9}{:>10}{:>10}{:>10}{:>10}".format(
        "theta_1", "записей", "часов", "parliam", "podcast", "radio", "talkshow"))
    for th in (8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0):
        sel = [r for r in train_sp if r["score"] >= th]
        per = collections.Counter(r["dom"] for r in sel)
        print("{:>8.1f}{:>10}{:>9.1f}{:>10}{:>10}{:>10}{:>10}".format(
            th, len(sel), hours(sel),
            per["parliament"], per["podcasts"], per["radio"], per["talkshow"]))

    print("\nDEV+TEST, спонтанные подкорпуса (оценочный набор):")
    print("{:>8}{:>10}{:>9}".format("theta_1", "записей", "часов"))
    for th in (7.0, 5.0, 4.0, 3.5, 3.0):
        sel = [r for r in eval_sp if r["score"] >= th]
        print("{:>8.1f}{:>10}{:>9.2f}".format(th, len(sel), hours(sel)))

    # ---------- 3. загрязнение: сколько прошло из подготовленных ----------
    prep = [r for r in rows
            if r["dom"] in ("crowdsourced", "tv_news") and r["label"] != "too_short"]
    print("\nЛОЖНЫЕ СРАБАТЫВАНИЯ: записей из crowdsourced+tv_news выше порога")
    print("{:>8}{:>12}{:>14}".format("theta_1", "штук", "% от отбора"))
    for th in (7.0, 5.0, 4.0, 3.5, 3.0, 2.5):
        n_bad = sum(1 for r in prep if r["score"] >= th)
        n_good = sum(1 for r in train_sp + eval_sp if r["score"] >= th)
        tot = n_bad + n_good
        print("{:>8.1f}{:>12}{:>13.2f}%".format(
            th, n_bad, 100.0 * n_bad / tot if tot else 0.0))

    # ---------- 4. группировка дикторов ----------
    print("\nПРОКСИ ДИКТОРОВ по спонтанным подкорпусам:")
    print("{:<14}{:>10}{:>10}{:>12}{:>10}".format(
        "домен", "записей", "групп", "зап./группу", "макс"))
    for dom in SPONT_DOMAINS:
        sel = [r for r in rows if r["dom"] == dom]
        g = collections.Counter(r["speaker_group"] for r in sel)
        if not g:
            continue
        print("{:<14}{:>10}{:>10}{:>12.1f}{:>10}".format(
            dom, len(sel), len(g), len(sel) / len(g), max(g.values())))

    # ---------- 5. пограничные примеры ----------
    rnd = random.Random(args.seed)
    bands = [(8.0, 99.0), (6.0, 8.0), (5.0, 6.0), (4.0, 5.0),
             (3.5, 4.0), (3.0, 3.5), (2.5, 3.0), (2.0, 2.5),
             (1.0, 2.0), (0.01, 1.0)]
    out = []
    pool = [r for r in rows if r["label"] != "too_short"
            and r["dom"] in SPONT_DOMAINS]
    for lo, hi in bands:
        band = [r for r in pool if lo <= r["score"] < hi]
        rnd.shuffle(band)
        for r in band[:12]:
            out.append({
                "band": "{:.1f}-{:.1f}".format(lo, hi),
                "score": r["score"],
                "domain": r["domain"],
                "tokens": r["tokens"],
                "dur_sec": round(r["dur"], 1) if r.get("dur") else "",
                "f_s": r["strong_fillers"],
                "f_w": r["weak_fillers"],
                "d": r["discourse_markers"],
                "r_t": r["repetitions"],
                "m": r["russian_mixed"],
                "record_id": r["record_id"],
                "preview": r["preview"],
                "ВАША_ОЦЕНКА": "",
            })
    with open(args.samples, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print("\nКалибровочная выборка: {} ({} записей по 10 полосам оценки)".format(
        args.samples, len(out)))
    print("Прослушайте аудио, в колонке ВАША_ОЦЕНКА поставьте S (спонтанная)")
    print("или P (подготовленная). Порог ставим там, где S перестаёт")
    print("преобладать при движении вниз по шкале.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
