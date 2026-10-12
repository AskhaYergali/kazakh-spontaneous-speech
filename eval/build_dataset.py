#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка Golden Dataset из манифеста фильтра.

Формирует наборы для экспериментов B и C и считает цифры для таблицы 2.4:

  spont_train        Train, спонтанные подкорпуса, S >= theta_1   (обучение LoRA)
  spont_train_ext    то же, но S >= theta_2                        (абляция likely+maybe)
  spont_dev          Dev,  S >= theta_1                            (ранняя остановка)
  spont_test         Test, S >= theta_1                            (оценка, RQ2/RQ3)
  prep_dev/test      контрольная группа: crowdsourced, S < theta_2 (эксперимент C)

Проверяет дикторо-независимость между наборами и печатает готовую таблицу.

Запуск ИЗ КОРНЯ КОРПУСА:
    python3 code/build_dataset.py manifest_v2.csv --out golden/
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import sys

SPONT_DOMAINS = ("podcasts", "radio", "talkshow")   # parliament по умолчанию отдельно
PREP_DOMAIN = "crowdsourced"

FIELDS = ("record_id", "audio_path", "domain", "split", "speaker_group",
          "tokens", "dur_sec", "label", "score",
          "strong_fillers", "weak_fillers", "discourse_markers",
          "repetitions", "russian_mixed", "text")


TXT_TAIL_RE = re.compile(r"(?:\.txt)+$", re.IGNORECASE)


def audio_path(record_id: str) -> str:
    """Расширения зеркалятся: .txt -> .flac, .txt.txt -> .flac.flac.

    В Dev/Test подкорпуса tv_news файлы названы 412833.txt.txt / 412833.flac.flac,
    в остальных местах — одиночное расширение.
    """
    m = TXT_TAIL_RE.search(record_id)
    if not m:
        return record_id + ".flac"
    n = len(m.group(0)) // 4
    return record_id[: m.start()] + ".flac" * n


def flac_duration(path: str):
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"fLaC":
                return None
            hdr = fh.read(4)
            if len(hdr) < 4:
                return None
            info = fh.read(min(int.from_bytes(hdr[1:4], "big"), 34))
            if len(info) < 18:
                return None
            chunk = int.from_bytes(info[10:18], "big")
            sr = (chunk >> 44) & 0xFFFFF
            ts = chunk & 0xFFFFFFFFF
            return ts / sr if sr and ts else None
    except OSError:
        return None


def read_text(root: str, record_id: str) -> str:
    try:
        with open(os.path.join(root, record_id), encoding="utf-8", errors="replace") as fh:
            return " ".join(fh.read().split())
    except OSError:
        return ""


def write_set(path: str, rows, root: str) -> dict:
    total_dur, missing = 0.0, 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            ap = audio_path(r["record_id"])
            d = flac_duration(os.path.join(root, ap))
            if d is None:
                missing += 1
                d = 0.0
            total_dur += d
            split, _, dom = r["domain"].partition("/")
            w.writerow({
                "record_id": r["record_id"], "audio_path": ap,
                "domain": dom, "split": split,
                "speaker_group": r["speaker_group"],
                "tokens": r["tokens"], "dur_sec": round(d, 2),
                "label": r["label"], "score": r["score"],
                "strong_fillers": r["strong_fillers"],
                "weak_fillers": r["weak_fillers"],
                "discourse_markers": r["discourse_markers"],
                "repetitions": r["repetitions"],
                "russian_mixed": r["russian_mixed"],
                "text": read_text(root, r["record_id"]),
            })
    return {"n": len(rows), "hours": total_dur / 3600.0, "missing_audio": missing,
            "speakers": len({r["speaker_group"] for r in rows})}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="golden")
    ap.add_argument("--theta1", type=float, default=5.0)
    ap.add_argument("--theta2", type=float, default=3.0)
    ap.add_argument("--include-parliament", action="store_true",
                    help="включить parliament в спонтанные наборы (по умолчанию нет)")
    ap.add_argument("--control-size", type=int, default=2000,
                    help="размер контрольной группы на каждый из Dev и Test")
    ap.add_argument("--tune-frac", type=float, default=0.08,
                    help="доля групп дикторов Train, уходящих под раннюю остановку")
    ap.add_argument("--seed", type=int, default=20260918)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    with open(args.manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["score"] = float(r["score"])
            r["tokens"] = int(r["tokens"])
            r["split"], _, r["dom"] = r["domain"].partition("/")
            rows.append(r)
    print("манифест: {} записей".format(len(rows)))

    doms = list(SPONT_DOMAINS) + (["parliament"] if args.include_parliament else [])
    print("спонтанные подкорпуса: {}".format(", ".join(doms)))
    print("пороги: theta_1={} theta_2={}\n".format(args.theta1, args.theta2))

    def pick(splits, lo, domains, hi=None):
        out = []
        for r in rows:
            if r["split"] not in splits or r["dom"] not in domains:
                continue
            if r["label"] == "too_short":
                continue
            if lo is not None and r["score"] < lo:
                continue
            if hi is not None and r["score"] >= hi:
                continue
            out.append(r)
        return out

    # --- выделяем группы дикторов Train под настройку (ранняя остановка) ---
    train_pool = pick(("Train",), args.theta2, doms)
    train_groups = sorted({r["speaker_group"] for r in train_pool})
    rnd0 = random.Random(args.seed)
    rnd0.shuffle(train_groups)
    n_tune = max(5, int(len(train_groups) * args.tune_frac))
    tune_groups = set(train_groups[:n_tune])
    print("групп дикторов в Train: {}, из них под настройку: {}\n".format(
        len(train_groups), len(tune_groups)))

    def split_by_group(sel, inside):
        return [r for r in sel if (r["speaker_group"] in tune_groups) == inside]

    EVAL = ("Dev", "Test")
    # Согласованный контроль берётся ИЗ ТЕХ ЖЕ источников, что и спонтанный набор:
    # тот же канал записи и те же дикторы, отличается только регистр.
    BROADCAST = tuple(doms)
    NEWS = ("tv_news",)

    sets = {
        # обучение
        "spont_train":      split_by_group(pick(("Train",), args.theta1, doms), False),
        "spont_train_ext":  split_by_group(pick(("Train",), args.theta2, doms), False),
        # настройка / ранняя остановка (дикторы Train, не пересекаются с обучением)
        "spont_tune":       split_by_group(pick(("Train",), args.theta1, doms), True),
        # оценка: Dev+Test вместе ради мощности
        "spont_eval":       pick(EVAL, args.theta1, doms),
        # контроль с тем же каналом записи: те же источники + tv_news, но низкая оценка
        "prep_eval_matched": pick(EVAL, None, BROADCAST, hi=args.theta2),
        # вторичный контроль: краудсорсинг
        "prep_eval_crowd":  pick(EVAL, None, (PREP_DOMAIN,), hi=args.theta2),
        # третий контроль: подготовленная речь, трансляционный канал, ДРУГИЕ дикторы
        "prep_eval_news":   pick(EVAL, None, NEWS, hi=args.theta2),
    }

    rnd = random.Random(args.seed)
    for key in ("prep_eval_matched", "prep_eval_crowd", "prep_eval_news"):
        pool = sets[key]
        rnd.shuffle(pool)
        sets[key] = pool[: args.control_size]

    stats = {}
    for name, sel in sets.items():
        sel.sort(key=lambda r: r["record_id"])
        st = write_set(os.path.join(args.out, name + ".csv"), sel, args.root)
        stats[name] = st

    print("{:<20}{:>9}{:>9}{:>11}{:>12}".format(
        "набор", "записей", "часов", "дикторов", "без аудио"))
    for name in ("spont_train", "spont_train_ext", "spont_tune", "spont_eval",
                 "prep_eval_matched", "prep_eval_crowd", "prep_eval_news"):
        s = stats[name]
        print("{:<20}{:>9}{:>9.2f}{:>11}{:>12}".format(
            name, s["n"], s["hours"], s["speakers"], s["missing_audio"]))

    # дикторо-независимость
    print("\nПроверка дикторо-независимости:")
    g = {k: {r["speaker_group"] for r in v} for k, v in sets.items()}
    pairs = [("spont_train", "spont_tune"), ("spont_train", "spont_eval"),
             ("spont_train_ext", "spont_tune"), ("spont_train_ext", "spont_eval"),
             ("spont_tune", "spont_eval"), ("spont_train", "prep_eval_matched"),
             ("spont_train_ext", "prep_eval_matched")]
    ok = True
    for a, b in pairs:
        inter = g[a] & g[b]
        flag = "OK" if not inter else "ПЕРЕСЕЧЕНИЕ {}".format(len(inter))
        if inter:
            ok = False
        print("  {:<18} vs {:<16} {}".format(a, b, flag))
    print("  итог: {}".format("все наборы дикторо-независимы" if ok
                              else "ЕСТЬ ПЕРЕСЕЧЕНИЯ, разбиение править"))
    shared = g["spont_eval"] & g["prep_eval_matched"]
    print("\n  Совпадение дикторов между spont_eval и prep_eval_matched: "
          "{} из {} — ЭТО ЗАДУМАНО.".format(len(shared), len(g["spont_eval"])))
    print("  Регистровый разрыв измеряется внутри одних и тех же говорящих,")
    print("  поэтому канал записи и личность диктора контролируются.")

    meta = {"theta_1": args.theta1, "theta_2": args.theta2,
            "spont_domains": doms, "control_size": args.control_size,
            "tune_frac": args.tune_frac, "seed": args.seed, "stats": stats,
            "speaker_disjoint": ok}
    with open(os.path.join(args.out, "dataset_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print("\nНаборы и dataset_meta.json записаны в {}/".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
