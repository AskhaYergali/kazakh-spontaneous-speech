#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пакет для разметки на iPad: аудио переименовано по номерам строк таблицы.

На iPad: «Файлы» слева (слушать B001, B002 ...), Numbers справа (ставить ответы).
Никаких программ ставить не нужно.

Запуск из корня корпуса ПОСЛЕ build_validation.py:
    python3 code/make_ipad_package.py            # аудио как есть (FLAC)
    python3 code/make_ipad_package.py --m4a      # перекодировать в AAC, если FLAC не играет

Результат: ipad_package.zip — перекинуть на iPad через AirDrop и распаковать в «Файлах».
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile

OUT = "ipad_package"


def audio(rid):
    m = re.search(r'(?:\.txt)+$', rid)
    return rid + '.flac' if not m else rid[:m.start()] + '.flac' * (len(m.group(0)) // 4)


def put(src, dst_base, m4a):
    if m4a:
        dst = dst_base + ".m4a"
        try:
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", src, dst],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return os.path.basename(dst)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    dst = dst_base + ".flac"
    shutil.copyfile(src, dst)
    return os.path.basename(dst)


def build(src_csv, prefix, sub, cols_out, text_col, m4a):
    rows = list(csv.DictReader(open(src_csv, encoding="utf-8")))
    folder = os.path.join(OUT, sub)
    os.makedirs(folder, exist_ok=True)
    table = []
    for i, r in enumerate(rows, 1):
        num = "{}{:03d}".format(prefix, i)
        fname = put(audio(r["record_id"]), os.path.join(folder, num), m4a)
        table.append([num, fname, r[text_col][:240]] + [""] * len(cols_out) + [r["id"]])
    with open(os.path.join(OUT, sub + ".csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["№", "файл", "текст"] + cols_out + ["id"])
        w.writerows(table)
    return len(rows)


def main():
    m4a = "--m4a" in sys.argv
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    n1 = build("validation_B.csv", "B", "1_validation", ["ОЦЕНКА", "УВЕРЕННОСТЬ", "ЗАМЕТКА"], "preview", m4a)
    n2 = build("spotcheck_100.csv", "C", "2_spotcheck", ["ОТВЕТ", "ЗАМЕТКА"], "text", m4a)
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "protocol_razmetki.md")):
        shutil.copyfile(os.path.join(here, "protocol_razmetki.md"), os.path.join(OUT, "протокол.md"))
    with zipfile.ZipFile("ipad_package.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, OUT))
    print("валидация: {} записей, аудиопроверка: {} записей".format(n1, n2))
    print("ipad_package.zip: {:.0f} МБ".format(os.path.getsize("ipad_package.zip") / 1e6))


if __name__ == "__main__":
    main()
