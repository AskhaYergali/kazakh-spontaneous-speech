#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Прослушивание калибровочной выборки.

Проигрывает аудио, показывает превью транскрипта, ждёт одну клавишу.
Прогресс сохраняется после каждого ответа — можно прерваться и продолжить.

Клавиши:
    s - спонтанная речь
    p - подготовленная / читаемая
    r - повторить
    x - пропустить (неразборчиво / не понял)
    q - выйти, сохранив прогресс

Запуск ИЗ КОРНЯ КОРПУСА:
    python3 code/audit.py calibration_samples.csv

Прогресс пишется в calibration_samples.judged.csv
"""

import csv
import os
import subprocess
import sys
import termios
import tty

COL = "ВАША_ОЦЕНКА"


def getkey() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def audio_path(record_id: str) -> str:
    base = record_id
    for suf in (".txt.txt", ".txt"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    for cand in (base + ".flac", record_id[:-4] + ".flac", base + ".txt.flac"):
        if os.path.exists(cand):
            return cand
    return base + ".flac"


def play(path: str) -> bool:
    if not os.path.exists(path):
        return False
    for player in (["afplay", path], ["ffplay", "-nodisp", "-autoexit",
                                      "-loglevel", "quiet", path],
                   ["aplay", "-q", path]):
        try:
            subprocess.run(player, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "calibration_samples.csv"
    dst = os.path.splitext(src)[0] + ".judged.csv"

    with open(src, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())

    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as fh:
            done = {r["record_id"]: r.get(COL, "") for r in csv.DictReader(fh)}
        for r in rows:
            if done.get(r["record_id"]):
                r[COL] = done[r["record_id"]]

    def save():
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    todo = [r for r in rows if not r.get(COL)]
    print("Всего {}, осталось {}".format(len(rows), len(todo)))
    print("Клавиши: s=спонтанная  p=подготовленная  r=повторить  x=пропуск  q=выход\n")

    for i, r in enumerate(todo, 1):
        path = audio_path(r["record_id"])
        while True:
            print("=" * 70)
            print("[{}/{}]  score={}  {}  f_s={} f_w={} d={} r_t={} m={}".format(
                i, len(todo), r["score"], r["domain"],
                r["f_s"], r["f_w"], r["d"], r["r_t"], r["m"]))
            print(r["preview"][:200])
            print("-" * 70)
            if not play(path):
                print("!! аудио не найдено: {}".format(path))
            k = getkey()
            if k == "r":
                continue
            if k == "q":
                save()
                print("\nСохранено в {}. Запустите снова, чтобы продолжить.".format(dst))
                return 0
            if k in ("s", "p", "x"):
                r[COL] = {"s": "S", "p": "P", "x": "X"}[k]
                save()
                print("-> {}\n".format(r[COL]))
                break

    save()
    print("\nГотово. Результат: {}".format(dst))
    counts = {}
    for r in rows:
        counts[r.get(COL, "")] = counts.get(r.get(COL, ""), 0) + 1
    print("Итог:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
