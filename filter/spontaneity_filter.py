#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фильтр спонтанности казахской речи, версия 1.0.

Реализует метрику раздела 2.1 диссертации БЕЗ отступлений:

    phi_i(t) = 1000 * f_i(t) / max(N, N0)          (2.1, со сглаживанием)
    S(t)     = sum_i w_i * phi_i(t)                (2.2)
    label(t) = likely_spontaneous          если S >= theta_1
               maybe_spontaneous           если theta_2 <= S < theta_1
               likely_prepared_or_read     если S <  theta_2
               too_short                   если N <  min_tokens          (2.3)

Отличия от прототипа find_spntns.py:
  * строго линейная форма, никаких min()-потолков и абсолютных счётчиков;
  * сглаженная нормировка max(N, N0) вместо N — снимает взрыв на коротких
    записях, сохраняя линейность и разложимость S по признакам;
  * повторы считаются без наложения и со стоп-листом редупликаций;
  * счётчик m разделён на служебные слова и морфологические блендинги;
  * в манифест попадают ВСЕ записи, включая контрольную группу и too_short;
  * домен и ID диктора извлекаются и пишутся в манифест;
  * лексиконы вынесены в JSON с версией и SHA-256, хеш пишется в манифест.

Запуск:
    python3 spontaneity_filter.py /путь/к/ISSAI_KSC2 -o manifest.csv
    python3 spontaneity_filter.py /путь/к/корпусу --dump-repetitions rep.csv
    python3 spontaneity_filter.py --self-test

Только стандартная библиотека, Python 3.8+.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

FILTER_VERSION = "1.1.0"

TOKEN_RE = re.compile(r"[a-zа-яёәіңғүұқөһ]+", re.IGNORECASE)

NON_SPEECH_RE = re.compile(
    r"(\[(смех|пауза|шум|күлкі|үзіліс|laugh|pause|noise|breath)[^\]]*\]"
    r"|\((смех|пауза|шум|күлкі|үзіліс|laugh|pause|noise|breath)[^\)]*\)"
    r"|<(laugh|pause|noise|breath)[^>]*>)",
    re.IGNORECASE,
)
CUTOFF_RE = re.compile(r"(?<![-\w])[a-zа-яёәіңғүұқөһ]{2,}-(?=\s|$)", re.IGNORECASE)
ELLIPSIS_RE = re.compile(r"\.{3,}|…")

# record_id вида "senate_01_232", "09_04_120" -> группа "senate_01", "09_04"
SESSION_RE = re.compile(r"^(.*)_\d+$")

FIELDNAMES = [
    "record_id", "domain", "speaker_group", "tokens",
    "label", "score",
    "strong_fillers", "strong_filler_per_1000",
    "weak_fillers", "weak_filler_per_1000",
    "discourse_markers", "discourse_per_1000",
    "repair_phrases", "repair_per_1000",
    "repetitions", "repetition_per_1000",
    "russian_mixed", "russian_mixed_per_1000",
    "russian_function_hits", "morph_blend_hits",
    "non_speech", "cutoffs", "ellipsis",
    "filter_version", "lexicon_version", "lexicon_sha256",
    "preview",
]


# ----------------------------------------------------------------------
# Лексиконы


class Lexicons:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            raw_bytes = fh.buffer.read() if hasattr(fh, "buffer") else None
        with open(path, "rb") as fb:
            raw_bytes = fb.read()
        self.sha256 = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))

        self.version: str = data["lexicon_version"]
        self.strong = frozenset(data["strong_fillers"])
        self.weak = frozenset(data["weak_fillers"])
        self.discourse = frozenset(data["discourse_markers"])
        self.repair: List[str] = list(data["repair_phrases"])
        self.rus_func = frozenset(data["russian_function_words"])
        self.rus_infix: List[str] = list(data["russian_aspectual_infixes"])
        self.rus_stems: List[str] = list(data["russian_verb_stems_for_blends"])
        self.kaz_suffix: List[str] = list(data["kazakh_verbal_suffixes_for_blends"])
        self.redup_stop = frozenset(data["reduplication_stoplist"])

        self.w: Dict[str, float] = dict(data["weights"])
        norm = data["normalization"]
        self.per: int = int(norm["per"])
        self.n0: int = int(norm["min_denominator_tokens"])
        th = data["thresholds"]
        self.theta1: float = float(th["theta_1"])
        self.theta2: float = float(th["theta_2"])
        self.min_tokens: int = int(th["min_tokens"])

        # многословные русские маркеры обрабатываем как фразы
        self.rus_phrases = [p for p in self.rus_func if " " in p]
        self.rus_words = frozenset(p for p in self.rus_func if " " not in p)

        # фразы компилируем один раз — детерминизм и скорость
        self._repair_re = [
            (p, re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)")) for p in self.repair
        ]
        self._rus_phrase_re = [
            (p, re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)")) for p in self.rus_phrases
        ]


# ----------------------------------------------------------------------
# Подсчёт признаков


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def count_repetitions(tokens: List[str], stoplist: frozenset) -> int:
    """Непосредственные повторы БЕЗ наложения.

    Сканируем слева направо, на каждой позиции проверяем n-граммы
    от длинных к коротким; найденный повтор потребляет свои позиции.
    Одиночные повторы токенов из стоп-листа (казахские редупликации
    типа "бір бір", "әр әр") не считаются.
    """
    n_tok = len(tokens)
    i = 0
    total = 0
    while i < n_tok:
        matched = False
        for n in (3, 2, 1):
            if i + 2 * n > n_tok:
                continue
            if tokens[i:i + n] != tokens[i + n:i + 2 * n]:
                continue
            if n == 1 and tokens[i] in stoplist:
                continue
            # считаем длину серии повторов
            reps = 1
            j = i + n
            while j + n <= n_tok and tokens[j:j + n] == tokens[i:i + n]:
                reps += 1
                j += n
            total += reps - 1
            i = j
            matched = True
            break
        if not matched:
            i += 1
    return total


def count_morph_blends(tokens: List[str], lex: Lexicons) -> int:
    """Морфологические блендинги: русская основа/инфикс + казахский суффикс.

    Пример из раздела 2.1: "кіргізіватқан" — русский видовой инфикс -ыва-
    внутри казахской глагольной формы.
    """
    hits = 0
    for tok in tokens:
        if len(tok) < 7:
            continue
        has_kaz_suffix = any(tok.endswith(sfx) for sfx in lex.kaz_suffix)
        if not has_kaz_suffix:
            continue
        body = tok
        if any(inf in body for inf in lex.rus_infix):
            hits += 1
            continue
        if any(body.startswith(stem) for stem in lex.rus_stems):
            hits += 1
    return hits


def extract_speaker_group(record_id: str, domain: str) -> str:
    """Прокси ID диктора: сессия/эпизод из имени файла.

    parliament: senate_01_232  -> senate_01
    podcast:    09_04_120      -> 09_04
    Если шаблон не распознан (crowdsourced, tv_news), запись образует
    собственную одноэлементную группу — это консервативно и не нарушает
    дикторо-независимость разбиений.
    """
    base = os.path.basename(record_id)
    for suffix in (".txt.txt", ".txt"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    m = SESSION_RE.match(base)
    if m and m.group(1):
        return "{}/{}".format(domain, m.group(1))
    return "{}/{}".format(domain, base)


def analyze(record_id: str, domain: str, text: str, lex: Lexicons) -> Dict:
    text_lc = text.lower()
    tokens = tokenize(text)
    n = len(tokens)

    f_s = sum(1 for t in tokens if t in lex.strong)
    f_w = sum(1 for t in tokens if t in lex.weak)
    d = sum(1 for t in tokens if t in lex.discourse)
    r_p = sum(len(rx.findall(text_lc)) for _p, rx in lex._repair_re)
    r_t = count_repetitions(tokens, lex.redup_stop)

    rus_words = sum(1 for t in tokens if t in lex.rus_words)
    rus_phr = sum(len(rx.findall(text_lc)) for _p, rx in lex._rus_phrase_re)
    blends = count_morph_blends(tokens, lex)
    rus_func_hits = rus_words + rus_phr
    m_total = rus_func_hits + blends

    n_s = len(NON_SPEECH_RE.findall(text))
    c = len(CUTOFF_RE.findall(text))
    e = len(ELLIPSIS_RE.findall(text))

    denom = max(n, lex.n0)

    def phi(count: int) -> float:
        return 0.0 if denom == 0 else lex.per * count / denom

    phi_s, phi_w, phi_d = phi(f_s), phi(f_w), phi(d)
    phi_p, phi_t, phi_m = phi(r_p), phi(r_t), phi(m_total)
    phi_n, phi_c, phi_e = phi(n_s), phi(c), phi(e)

    w = lex.w
    score = (
        w["strong_fillers"] * phi_s
        + w["weak_fillers"] * phi_w
        + w["discourse_markers"] * phi_d
        + w["repair_phrases"] * phi_p
        + w["repetitions"] * phi_t
        + w["russian_mixed"] * phi_m
        + w["non_speech"] * phi_n
        + w["cutoffs"] * phi_c
        + w["ellipsis"] * phi_e
    )

    if n < lex.min_tokens:
        label = "too_short"
    elif score >= lex.theta1:
        label = "likely_spontaneous"
    elif score >= lex.theta2:
        label = "maybe_spontaneous"
    else:
        label = "likely_prepared_or_read"

    return {
        "record_id": record_id,
        "domain": domain,
        "speaker_group": extract_speaker_group(record_id, domain),
        "tokens": n,
        "label": label,
        "score": round(score, 4),
        "strong_fillers": f_s, "strong_filler_per_1000": round(phi_s, 2),
        "weak_fillers": f_w, "weak_filler_per_1000": round(phi_w, 2),
        "discourse_markers": d, "discourse_per_1000": round(phi_d, 2),
        "repair_phrases": r_p, "repair_per_1000": round(phi_p, 2),
        "repetitions": r_t, "repetition_per_1000": round(phi_t, 2),
        "russian_mixed": m_total, "russian_mixed_per_1000": round(phi_m, 2),
        "russian_function_hits": rus_func_hits,
        "morph_blend_hits": blends,
        "non_speech": n_s, "cutoffs": c, "ellipsis": e,
        "filter_version": FILTER_VERSION,
        "lexicon_version": lex.version,
        "lexicon_sha256": lex.sha256[:16],
        "preview": " ".join(text.split())[:220],
    }


# ----------------------------------------------------------------------
# Обход корпуса


SKIP_BASENAMES = {"all_text.txt"}
# Синтезированная речь: не человеческая, исключается из курирования целиком.
SKIP_DIRS = {"tts"}


def iter_transcripts(base: str):
    """Отдаёт (record_id, domain, text). Агрегатные файлы и TTS пропускаются."""
    base = os.path.abspath(base)
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for fname in sorted(files):
            if not fname.endswith(".txt"):
                continue
            if fname in SKIP_BASENAMES:
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, base)
            domain = os.path.basename(root) if root != base else "_root"
            parts = rel.split(os.sep)
            split = parts[0] if len(parts) > 2 else "_unknown"
            domain = "{}/{}".format(split, domain)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
            except OSError as exc:
                print("[пропуск] {}: {}".format(path, exc), file=sys.stderr)
                continue
            if not text:
                continue
            yield rel, domain, text


def dump_repetition_candidates(base: str, out_path: str, lex: Lexicons) -> None:
    """Ранжированный список повторяющихся токенов — для ручной курации стоп-листа."""
    counter: Counter = Counter()
    for _rid, _dom, text in iter_transcripts(base):
        toks = tokenize(text)
        for a, b in zip(toks, toks[1:]):
            if a == b:
                counter[a] += 1
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["token", "immediate_repeats", "already_in_stoplist"])
        for tok, cnt in counter.most_common():
            wr.writerow([tok, cnt, "yes" if tok in lex.redup_stop else "no"])
    print("Кандидаты в стоп-лист редупликаций: {} ({} уникальных токенов)".format(
        out_path, len(counter)))
    print("Просмотрите глазами верхние строки и допишите казахские редупликации")
    print("в lexicons.json -> reduplication_stoplist, затем перезапустите фильтр.")


# ----------------------------------------------------------------------
# Самопроверка


def self_test(lex: Lexicons) -> int:
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # 1. Пример из раздела 2.1 диссертации
    senate = ("ііі банк өздеріңіз білесіздер ол ііі сол себептен осы ііі "
              "бірыңғай оператордың ііі ішіне біз оны кіргізіватқан жоқпыз "
              "өйткені оның і жұмысы бөлек талқыланып бөлек ыыы істеу керек")
    r = analyze("senate_01_232.txt", "parliament", senate, lex)
    check("senate: N=28", r["tokens"] == 28)
    check("senate: f_s=5", r["strong_fillers"] == 5)
    check("senate: f_w>=1", r["weak_fillers"] >= 1)
    check("senate: блендинг кіргізіватқан пойман", r["morph_blend_hits"] >= 1)
    check("senate: likely_spontaneous", r["label"] == "likely_spontaneous")
    check("senate: speaker_group", r["speaker_group"] == "parliament/senate_01")

    # 2. Чистая подготовленная речь
    prep = ("құрметті депутаттар бүгінгі отырыстың күн тәртібінде тұрған "
            "мәселе бойынша баяндама жасауға рұқсат етіңіздер")
    r = analyze("x.txt", "parliament", prep, lex)
    check("prepared: S близка к нулю", r["score"] < lex.theta2)
    check("prepared: метка", r["label"] == "likely_prepared_or_read")

    # 3. Стоп-лист редупликаций
    r1 = analyze("a.txt", "crowdsourced", "бәрінің көңілінде бір бір алаң бар сонда", lex)
    check("redup: 'бір бір' не считается", r1["repetitions"] == 0)
    r2 = analyze("b.txt", "podcast", "аян ретінде аян ретінде берілген түстер деп", lex)
    check("redup: настоящий повтор биграммы считается", r2["repetitions"] == 1)
    r3 = analyze("c.txt", "podcast", "осындай осындай осындай мәселеге байланысты ы", lex)
    check("redup: серия из трёх даёт 2", r3["repetitions"] == 2)

    # 4. Отсутствие двойного счёта
    check("no double count: серия не считается и как слово, и как биграмма",
          r3["repetitions"] == 2)

    # 5. 'да' больше не русский маркер
    r = analyze("d.txt", "crowdsourced", "ол да келді біз да бардық сонда да жақсы болды", lex)
    check("'да' не считается русским вкраплением", r["russian_mixed"] == 0)

    # 6. Сглаживание на короткой записи
    short = "ммм кім екенін білсем бола ма"
    r = analyze("e.txt", "crowdsourced", short, lex)
    check("короткая запись: phi_s = 1000/20 = 50",
          abs(r["strong_filler_per_1000"] - 50.0) < 0.01)

    # 7. too_short достижима
    r = analyze("f.txt", "crowdsourced", "иә ммм жақсы", lex)
    check("too_short достижима", r["label"] == "too_short")

    # 8. Линейность: удвоение признака при той же длине удваивает S
    filler_words = ["сөз", "мәтін", "жұмыс", "уақыт", "адам", "қала", "жол",
                    "күн", "ай", "жыл", "үй", "кітап", "тіл", "ел", "жер",
                    "су", "ас", "ат", "қол"]
    t1 = " ".join(["ыыы"] + filler_words)
    t2 = " ".join(["ыыы", filler_words[0], "ыыы"] + filler_words[1:18])
    r1 = analyze("g.txt", "x", t1, lex)
    r2 = analyze("h.txt", "x", t2, lex)
    check("тест линейности: нет посторонних признаков",
          r1["repetitions"] == 0 and r2["repetitions"] == 0
          and r1["weak_fillers"] == 0 and r1["discourse_markers"] == 0)
    check("линейность S по f_s", abs(r2["score"] - 2 * r1["score"]) < 1e-9)
    s1, s2 = r1["score"], r2["score"]

    # 9. Монотонность
    check("монотонность", s2 > s1)

    # 10. Детерминизм
    a = analyze("i.txt", "x", senate, lex)
    b = analyze("i.txt", "x", senate, lex)
    check("детерминизм", a == b)

    if failures:
        print("ПРОВАЛЕНО {} проверок:".format(len(failures)))
        for f in failures:
            print("   - " + f)
        return 1
    print("Все самопроверки пройдены.")
    return 0


# ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Фильтр спонтанности казахской речи v" + FILTER_VERSION)
    ap.add_argument("input", nargs="?", help="папка корпуса (обходится рекурсивно)")
    ap.add_argument("-o", "--output", default="spontaneity_manifest.csv")
    ap.add_argument("-l", "--lexicons", default=None,
                    help="путь к lexicons.json (по умолчанию рядом со скриптом)")
    ap.add_argument("--dump-repetitions", metavar="CSV",
                    help="выгрузить кандидатов в стоп-лист редупликаций и выйти")
    ap.add_argument("--self-test", action="store_true", help="прогнать юнит-тесты и выйти")
    args = ap.parse_args()

    lex_path = args.lexicons or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lexicons.json")
    if not os.path.exists(lex_path):
        sys.exit("Не найден файл лексиконов: " + lex_path)
    lex = Lexicons(lex_path)

    if args.self_test:
        return self_test(lex)

    if not args.input:
        ap.error("укажите папку корпуса (или --self-test)")
    if not os.path.isdir(args.input):
        sys.exit("Папка не найдена: " + args.input)

    if args.dump_repetitions:
        dump_repetition_candidates(args.input, args.dump_repetitions, lex)
        return 0

    rows = []
    for rid, domain, text in iter_transcripts(args.input):
        rows.append(analyze(rid, domain, text, lex))

    if not rows:
        sys.exit("Не найдено ни одного .txt")

    # детерминированный порядок: по домену, затем по record_id
    rows.sort(key=lambda r: (r["domain"], r["record_id"]))

    with open(args.output, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        wr.writeheader()
        wr.writerows(rows)

    # сводка
    by_label: Counter = Counter(r["label"] for r in rows)
    by_dom: Dict[str, Counter] = {}
    for r in rows:
        by_dom.setdefault(r["domain"], Counter())[r["label"]] += 1

    print("Фильтр v{}, лексиконы v{} (sha256 {})".format(
        FILTER_VERSION, lex.version, lex.sha256[:16]))
    print("Нормировка: phi = {} * f / max(N, {})".format(lex.per, lex.n0))
    print("Пороги: theta_1 = {}, theta_2 = {}, min_tokens = {}".format(
        lex.theta1, lex.theta2, lex.min_tokens))
    print("Обработано записей: {}".format(len(rows)))
    print("Манифест: {}".format(args.output))
    print()
    order = ["likely_spontaneous", "maybe_spontaneous",
             "likely_prepared_or_read", "too_short"]
    hdr = "{:<16}".format("домен") + "".join("{:>26}".format(x) for x in order)
    print(hdr)
    for dom in sorted(by_dom):
        line = "{:<16}".format(dom[:15])
        for lab in order:
            line += "{:>26}".format(by_dom[dom][lab])
        print(line)
    print("{:<16}".format("ВСЕГО") + "".join("{:>26}".format(by_label[x]) for x in order))

    spk = len({r["speaker_group"] for r in rows})
    print("\nГрупп дикторов (прокси по имени файла): {}".format(spk))
    return 0


if __name__ == "__main__":
    sys.exit(main())