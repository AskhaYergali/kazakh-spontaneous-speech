#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Шаг 1: инвентаризация корпуса KSC2.

Ничего не изменяет. Читает все .txt и отвечает на вопросы, без которых
нельзя написать лексиконы фильтра спонтанности:

  1. Сколько файлов, токенов, какое распределение длин по подкорпусам.
  2. Транскрибируются ли хезитации вообще и в каком написании.
  3. Есть ли неречевые пометки, обрывы, многоточия.
  4. Насколько присутствуют русские вкрапления.
  5. Какие токены вообще частотны (для ручного отбора маркеров).

Запуск:
    python3 explore_ksc2.py /путь/к/ISSAI_KSC2
    python3 explore_ksc2.py /путь/к/ISSAI_KSC2 --out отчет

Требует только стандартную библиотеку Python 3.8+.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------
# Алфавиты

KAZ_ONLY = set("әғқңөұүһі")          # буквы, которых нет в русском
RUS_ONLY = set("щъёцч")               # встречаются почти только в русизмах
CYRILLIC = re.compile(r"[а-яәғқңөұүһіё]", re.IGNORECASE)

# Токенизация: слово = последовательность букв/цифр, дефис внутри сохраняем
TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*-?", re.UNICODE)

# Кандидаты в хезитации: короткий токен из 1-2 уникальных гласных/сонорных,
# возможно с повторами букв. Ищем эмпирически, не по готовому списку.
HESITATION_CANDIDATE = re.compile(
    r"^(?:[аәеиоөуұүыіэ]|[мнң])[аәеиоөуұүыіэмнң]*$", re.UNICODE
)

# Русские служебные слова — грубый индикатор вкраплений (расширим позже)
RUS_FUNCTION = {
    "это", "как", "так", "что", "чтобы", "если", "потому", "который",
    "которые", "уже", "еще", "ещё", "очень", "тоже", "также", "просто",
    "например", "конечно", "вообще", "значит", "короче", "типа", "вот",
    "ну", "да", "нет", "или", "но", "для", "при", "над", "под", "без",
    "через", "между", "после", "перед", "около", "весь", "всё", "все",
    "мы", "вы", "они", "он", "она", "оно", "мой", "наш", "ваш", "их",
    "был", "была", "было", "были", "быть", "есть", "будет", "будут",
    "может", "можно", "нужно", "надо", "должен", "хорошо", "спасибо",
}

# Неречевые пометки и спецсимволы, которые могут встречаться в транскриптах
BRACKET_RE = re.compile(r"[\[\(<{]([^\]\)>}]{1,40})[\]\)>}]")
ELLIPSIS_RE = re.compile(r"\.{2,}|…")
TRAILING_HYPHEN_RE = re.compile(r"\b[^\W\d_]+-(?=\s|$)", re.UNICODE)


# ----------------------------------------------------------------------

def script_profile(token):
    """Вернуть 'kaz' / 'rus' / 'neutral' по буквенному составу токена."""
    letters = set(token.lower())
    if letters & KAZ_ONLY:
        return "kaz"
    if letters & RUS_ONLY:
        return "rus"
    return "neutral"


def scan(base_dir):
    stats = defaultdict(lambda: {
        "files": 0,
        "tokens": 0,
        "chars": 0,
        "lengths": [],
        "with_bracket": 0,
        "with_ellipsis": 0,
        "with_hyphen_cut": 0,
        "with_repetition": 0,
        "kaz_tokens": 0,
        "rus_tokens": 0,
        "neutral_tokens": 0,
        "rus_function_hits": 0,
        "no_audio": 0,
    })

    global_tokens = Counter()
    hesitation_candidates = Counter()
    bracket_contents = Counter()
    short_tokens = Counter()          # токены длиной 1-3 — среди них прячутся заполнители
    repeated_letter_tokens = Counter()  # токены с 3+ одинаковыми буквами подряд
    examples_repetition = []
    examples_hyphen = []
    examples_bracket = []

    triple_letter = re.compile(r"(.)\1{2,}", re.UNICODE)

    total_files = 0
    for root, _dirs, files in os.walk(base_dir):
        txts = [f for f in files if f.endswith(".txt")]
        if not txts:
            continue
        # подкорпус = имя папки, содержащей файлы
        domain = os.path.basename(root)
        for fname in sorted(txts):
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = fh.read().strip()
            except Exception as exc:
                print(f"[пропуск] {path}: {exc}", file=sys.stderr)
                continue

            total_files += 1
            s = stats[domain]
            s["files"] += 1
            s["chars"] += len(raw)

            audio = path[:-4] + ".flac"
            if not os.path.exists(audio):
                audio = path[:-4] + ".wav"
                if not os.path.exists(audio):
                    s["no_audio"] += 1

            low = raw.lower()
            tokens = TOKEN_RE.findall(low)
            n = len(tokens)
            s["tokens"] += n
            s["lengths"].append(n)

            global_tokens.update(tokens)

            for tok in tokens:
                prof = script_profile(tok)
                s[prof + "_tokens"] += 1
                if len(tok) <= 3:
                    short_tokens[tok] += 1
                if HESITATION_CANDIDATE.match(tok) and len(tok) <= 5:
                    hesitation_candidates[tok] += 1
                if triple_letter.search(tok):
                    repeated_letter_tokens[tok] += 1
                if tok in RUS_FUNCTION:
                    s["rus_function_hits"] += 1

            # непосредственные повторы токенов
            rep = any(tokens[i] == tokens[i + 1] for i in range(n - 1))
            if rep:
                s["with_repetition"] += 1
                if len(examples_repetition) < 15:
                    examples_repetition.append((domain, fname, raw[:160]))

            br = BRACKET_RE.findall(raw)
            if br:
                s["with_bracket"] += 1
                bracket_contents.update(x.strip().lower() for x in br)
                if len(examples_bracket) < 15:
                    examples_bracket.append((domain, fname, raw[:160]))

            if ELLIPSIS_RE.search(raw):
                s["with_ellipsis"] += 1

            if TRAILING_HYPHEN_RE.search(raw):
                s["with_hyphen_cut"] += 1
                if len(examples_hyphen) < 15:
                    examples_hyphen.append((domain, fname, raw[:160]))

    return {
        "stats": stats,
        "global_tokens": global_tokens,
        "hesitation_candidates": hesitation_candidates,
        "bracket_contents": bracket_contents,
        "short_tokens": short_tokens,
        "repeated_letter_tokens": repeated_letter_tokens,
        "examples_repetition": examples_repetition,
        "examples_hyphen": examples_hyphen,
        "examples_bracket": examples_bracket,
        "total_files": total_files,
    }


def pct(part, whole):
    return 0.0 if not whole else 100.0 * part / whole


def report(res, out_prefix):
    stats = res["stats"]
    lines = []
    add = lines.append

    add("=" * 78)
    add("ИНВЕНТАРИЗАЦИЯ КОРПУСА KSC2")
    add("=" * 78)
    add(f"Всего .txt файлов: {res['total_files']}")
    add("")

    add("-" * 78)
    add("1. ОБЪЁМ ПО ПОДКОРПУСАМ")
    add("-" * 78)
    hdr = f"{'подкорпус':<18}{'файлов':>9}{'токенов':>11}{'ср.длина':>10}{'медиана':>9}{'<5 ток.':>9}"
    add(hdr)
    for dom in sorted(stats):
        s = stats[dom]
        L = sorted(s["lengths"])
        mean = s["tokens"] / s["files"] if s["files"] else 0
        med = L[len(L) // 2] if L else 0
        tiny = sum(1 for x in L if x < 5)
        add(f"{dom:<18}{s['files']:>9}{s['tokens']:>11}{mean:>10.1f}{med:>9}{tiny:>9}")
    add("")

    add("-" * 78)
    add("2. ЕСТЬ ЛИ В ТРАНСКРИПТАХ СЛЕДЫ СПОНТАННОСТИ  (ключевой вопрос)")
    add("-" * 78)
    add(f"{'подкорпус':<18}{'повторы%':>10}{'обрывы%':>10}{'скобки%':>10}{'многоточ%':>11}")
    for dom in sorted(stats):
        s = stats[dom]
        f = s["files"]
        add(f"{dom:<18}{pct(s['with_repetition'], f):>10.1f}"
            f"{pct(s['with_hyphen_cut'], f):>10.1f}"
            f"{pct(s['with_bracket'], f):>10.1f}"
            f"{pct(s['with_ellipsis'], f):>11.1f}")
    add("")
    add("Если во ВСЕХ подкорпусах эти проценты близки к нулю — транскрипты")
    add("нормализованы, и счётчики c, e, n_s, r_t работать не будут.")
    add("Тогда фильтр строится на f_s, f_w, d, m, а в разделе 4.5 это")
    add("фиксируется как ограничение.")
    add("")

    add("-" * 78)
    add("3. КАНДИДАТЫ В ЗАПОЛНИТЕЛИ ПАУЗ (короткие вокалические токены)")
    add("-" * 78)
    add("Смотрите глазами: настоящие хезитации обычно с удлинением букв.")
    for tok, c in res["hesitation_candidates"].most_common(60):
        add(f"    {tok:<12} {c:>8}")
    add("")

    add("-" * 78)
    add("4. ТОКЕНЫ С УТРОЕННОЙ БУКВОЙ (самый надёжный признак хезитации)")
    add("-" * 78)
    if res["repeated_letter_tokens"]:
        for tok, c in res["repeated_letter_tokens"].most_common(60):
            add(f"    {tok:<16} {c:>8}")
    else:
        add("    НЕ НАЙДЕНО НИ ОДНОГО. Это плохая новость для счётчика f_s:")
        add("    удлинения в транскриптах не сохранены.")
    add("")

    add("-" * 78)
    add("5. СОДЕРЖИМОЕ СКОБОК (неречевые пометки n_s)")
    add("-" * 78)
    if res["bracket_contents"]:
        for tok, c in res["bracket_contents"].most_common(40):
            add(f"    {tok:<28} {c:>8}")
    else:
        add("    Скобочных пометок нет — счётчик n_s всегда будет нулевым.")
    add("")

    add("-" * 78)
    add("6. РУССКИЕ ВКРАПЛЕНИЯ (основа для счётчика m)")
    add("-" * 78)
    add(f"{'подкорпус':<18}{'каз.токен%':>12}{'рус.токен%':>12}{'рус.служебн/1000':>18}")
    for dom in sorted(stats):
        s = stats[dom]
        t = s["tokens"]
        add(f"{dom:<18}{pct(s['kaz_tokens'], t):>12.1f}"
            f"{pct(s['rus_tokens'], t):>12.1f}"
            f"{1000.0 * s['rus_function_hits'] / t if t else 0:>18.1f}")
    add("")

    add("-" * 78)
    add("7. ПРИМЕРЫ (проверьте глазами)")
    add("-" * 78)
    for title, key in [("Повторы токенов", "examples_repetition"),
                       ("Обрывы на дефис", "examples_hyphen"),
                       ("Скобочные пометки", "examples_bracket")]:
        add(f"  [{title}]")
        if not res[key]:
            add("     — не найдено")
        for dom, fn, txt in res[key][:8]:
            add(f"     ({dom}/{fn}) {txt}")
        add("")

    add("-" * 78)
    add("8. ПРОБЛЕМЫ ЦЕЛОСТНОСТИ")
    add("-" * 78)
    total_no_audio = sum(s["no_audio"] for s in stats.values())
    add(f"    .txt без парного аудио: {total_no_audio}")
    add("")

    text = "\n".join(lines)
    print(text)

    with open(out_prefix + "_report.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    # частотный список токенов — для ручного отбора маркеров
    with open(out_prefix + "_tokens.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "count", "script"])
        for tok, c in res["global_tokens"].most_common(5000):
            w.writerow([tok, c, script_profile(tok)])

    print(f"\nСохранено: {out_prefix}_report.txt")
    print(f"Сохранено: {out_prefix}_tokens.csv  (топ-5000 токенов)")


def main():
    ap = argparse.ArgumentParser(description="Инвентаризация корпуса KSC2")
    ap.add_argument("base_dir", help="корневая папка корпуса (сканируется рекурсивно)")
    ap.add_argument("--out", default="ksc2_inventory", help="префикс выходных файлов")
    args = ap.parse_args()

    if not os.path.isdir(args.base_dir):
        sys.exit(f"Папка не найдена: {args.base_dir}")

    res = scan(args.base_dir)
    if res["total_files"] == 0:
        sys.exit("Не найдено ни одного .txt. Проверьте путь и структуру папок.")
    report(res, args.out)


if __name__ == "__main__":
    main()
