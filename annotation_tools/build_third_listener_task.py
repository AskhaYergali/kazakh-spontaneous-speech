#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_third_listener_task.py -- blind listening package for a third annotator.

Builds, from the released validation key and a local copy of KSC2, a folder the
annotator works from on this Mac by double-clicking index.html.  Nothing in the
package reveals the answer: no filter label, no score, no subcorpus, no original
file name, no earlier judgement, and the order is randomized with a recorded seed.

    python3 build_third_listener_task.py \
        --ksc2 /path/to/ISSAI_KSC2 \
        --annotation /path/to/release/annotation \
        --out third_listener_package

By default the package holds the 53 items the two listeners did not settle: the
44 where they disagreed and the 9 neither could decide.  --subset all takes the
whole sample of 200.

Writes
    <out>/index.html           the task, opens in Safari or Chrome
    <out>/audio/001.m4a ...    converted for the browser, opaque names
    <out>/PROTOCOL.md          the same instructions as inside the page
    <out>_KEY_do_not_share.csv item -> id, record_id, label, score  (OUTSIDE <out>)

Hand over the <out> folder only.
"""

import argparse, csv, html, json, os, random, re, shutil, subprocess, sys

SEED = 20261115
HERE = os.path.dirname(os.path.abspath(__file__))

PROTOCOL_MD = """# Определение спонтанности речи на слух

## Задача

Для каждой записи нужно решить, **спонтанная** это речь или **подготовленная**, и
отметить, насколько вы уверены.

Порядок записей случайный. Ничего в этой папке не содержит подсказок: нумерация не
связана ни с источником записи, ни с правильным ответом. Вы не видите ни оценок
автоматического фильтра, ни ответов других слушателей — так и задумано: ваши ответы
сравнивают с их ответами, поэтому важно, чтобы вы не знали их заранее.

## Главное правило

Решает **функция** слова, а не само слово. Заполнитель паузы или дискурсивный маркер
говорит о спонтанном порождении тогда, когда он отмечает **перезапуск или
перепланирование** мысли, и не говорит — когда вводит пояснение к уже сказанному.

* «яғни» или «мысалы» перед уточнением — это пояснение, с подготовленной речью совместимо;
* «негізі» в точке, где говорящий перестраивает фразу, — признак спонтанности.

## В пользу спонтанной речи

* реакция на собеседника, ответ на реплику;
* наложение речи, перебивание;
* разговорные формы;
* повтор содержательного слова не подряд, а с возвратом к нему.

## В пользу подготовленной речи

* непрерывная линия мысли без перепланирования — **независимо от того, сколько в ней
  заполнителей паузы**;
* запинки, которые написаны в тексте и прочитаны ровной интонацией.

## Не является доказательством ни в ту, ни в другую сторону

* тема разговора и степень официальности;
* качество записи, акцент, темп речи;
* длина записи.

## Уверенность

`уверен` / `скорее уверен` / `не уверен`. Вариант «на слух решить невозможно» —
только для записей, где решение действительно нельзя принять: слишком коротко,
неразборчиво или обрывается. Это не вариант «затрудняюсь»: если склоняетесь к одному
из ответов, выберите его с пометкой «не уверен».

## Чего не делать

Не пытайтесь угадать, из какой передачи или корпуса взята запись. Не меняйте ответы
задним числом, пройдя всю подборку: первое решение по каждой записи и есть ответ.
"""


def md_to_html(md: str) -> str:
    out, in_ul = [], False
    for line in md.split("\n"):
        s = line.rstrip()
        if s.startswith("## "):
            if in_ul: out.append("</ul>"); in_ul = False
            out.append("<h3>%s</h3>" % inline(s[3:]))
        elif s.startswith("# "):
            continue
        elif s.startswith("* "):
            if not in_ul: out.append("<ul>"); in_ul = True
            out.append("<li>%s</li>" % inline(s[2:]))
        elif not s:
            if in_ul: out.append("</ul>"); in_ul = False
        else:
            if in_ul: out.append("</ul>"); in_ul = False
            out.append("<p>%s</p>" % inline(s))
    if in_ul: out.append("</ul>")
    return "\n".join(out)


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def audio_rel(record_id: str) -> str:
    """'Dev/tv_news/415515.txt.txt' -> 'Dev/tv_news/415515.flac.flac'."""
    m = re.search(r"(\.txt)*$", record_id)
    n = len(m.group(0)) // 4 if m.group(0) else 0
    return re.sub(r"(\.txt)+$", "", record_id) + ".flac" * max(n, 1)


def index_flacs(root: str):
    idx = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(".flac"):
                idx.setdefault(n, []).append(os.path.join(dirpath, n))
    return idx


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def convert(src: str, dst_noext: str):
    """Return the written filename. Safari plays m4a everywhere; flac only on
    recent versions, so convert when a converter is available."""
    if have("afconvert"):
        dst = dst_noext + ".m4a"
        r = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-q", "127",
                            "-s", "3", src, dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.getsize(dst) > 1024:
            return os.path.basename(dst)
        if os.path.exists(dst): os.remove(dst)
    if have("ffmpeg"):
        dst = dst_noext + ".m4a"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                            "-c:a", "aac", "-b:a", "128k", dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.getsize(dst) > 1024:
            return os.path.basename(dst)
        if os.path.exists(dst): os.remove(dst)
    dst = dst_noext + ".flac"
    shutil.copy2(src, dst)
    return os.path.basename(dst)


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ksc2", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["unsettled", "all"], default="unsettled")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dry-run", action="store_true", help="build the page without copying audio")
    a = ap.parse_args()

    A = a.annotation
    key = {r["id"]: r for r in read(os.path.join(A, "validation_key.csv"))}
    g = lambda r: (r.get("ОЦЕНКА") or "").strip().upper()
    l1 = {r["id"]: g(r) for r in read(os.path.join(A, "validation_listener_1.csv"))}
    l2 = {r["id"]: g(r) for r in read(os.path.join(A, "validation_listener_2.csv"))}

    if a.subset == "all":
        chosen = sorted(key)
        why = "the whole sample"
    else:
        dis = {i for i in key if l1.get(i) in ("S", "P") and l2.get(i) in ("S", "P")
               and l1[i] != l2[i]}
        und = {i for i in key if l1.get(i) not in ("S", "P") or l2.get(i) not in ("S", "P")}
        chosen = sorted(dis | und)
        why = "%d disagreements + %d undecided" % (len(dis), len(und))
    print("selected %d items (%s)" % (len(chosen), why))

    print("indexing .flac under %s ..." % a.ksc2)
    idx = {} if a.dry_run else index_flacs(a.ksc2)
    if not a.dry_run:
        print("  %d distinct basenames" % len(idx))

    resolved, missing = [], []
    for i in chosen:
        r = key[i]; rel = audio_rel(r["record_id"])
        if a.dry_run:
            resolved.append((r, rel)); continue
        direct = os.path.join(a.ksc2, rel)
        if os.path.isfile(direct):
            resolved.append((r, direct)); continue
        c = idx.get(os.path.basename(rel), [])
        if len(c) == 1:
            resolved.append((r, c[0]))
        else:
            want = os.path.basename(os.path.dirname(rel))
            pick = [x for x in c if os.path.basename(os.path.dirname(x)) == want]
            if len(pick) == 1:
                resolved.append((r, pick[0]))
            else:
                missing.append((i, rel, "%d candidates" % len(c) if c else "not found"))
    if missing:
        print("\n%d file(s) not resolved — package NOT written:" % len(missing))
        for i, rel, w in missing[:25]:
            print("  %-6s %-52s %s" % (i, rel, w))
        print("\nCheck --ksc2, or send me this list.")
        return 1

    rnd = random.Random(a.seed)
    rnd.shuffle(resolved)

    audio_dir = os.path.join(a.out, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    items, keyrows = [], []
    for n, (r, src) in enumerate(resolved, 1):
        tag = "%03d" % n
        fname = tag + ".m4a" if a.dry_run else convert(src, os.path.join(audio_dir, tag))
        items.append({"item": tag, "file": fname})
        keyrows.append({"item": tag, "id": r["id"], "record_id": r["record_id"],
                        "domain": r["domain"], "label": r["label"],
                        "score": r["score"], "stratum": r["stratum"],
                        "listener_1": l1.get(r["id"], ""), "listener_2": l2.get(r["id"], "")})
        if n % 10 == 0 or n == len(resolved):
            print("  prepared %d/%d" % (n, len(resolved)))

    tpl = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read()
    page = (tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
               .replace("__PROTOCOL__", json.dumps(md_to_html(PROTOCOL_MD), ensure_ascii=False)))
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(a.out, "PROTOCOL.md"), "w", encoding="utf-8") as fh:
        fh.write(PROTOCOL_MD)

    keyfile = os.path.abspath(a.out.rstrip("/\\")) + "_KEY_do_not_share.csv"
    with open(keyfile, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "id", "record_id", "domain", "label",
                                            "score", "stratum", "listener_1", "listener_2"])
        w.writeheader(); w.writerows(keyrows)

    if not a.dry_run:
        mb = sum(os.path.getsize(os.path.join(audio_dir, f))
                 for f in os.listdir(audio_dir)) / 1e6
        exts = {os.path.splitext(i["file"])[1] for i in items}
        print("\naudio: %d files, %.1f MB, format %s" % (len(items), mb, ", ".join(sorted(exts))))
    print("\npackage: %s" % os.path.abspath(a.out))
    print("  open index.html by double-clicking it")
    print("key (do NOT hand over): %s" % keyfile)
    print("randomization seed: %d" % a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
