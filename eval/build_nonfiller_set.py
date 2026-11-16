#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пункт 1 рецензии: спонтанный набор, отобранный БЕЗ участия филлеров.

Берёт высказывания из тех же сессий записи, что и основной спонтанный набор,
в которых нет ни одного филлера — ни сильного, ни слабого, — но есть не менее
--min-other признаков спонтанности иного рода: дискурсивные маркеры, фразы
самоисправления, неслучайные повторы, русские вкрапления. Записи основного
набора и контроля исключаются.

Если разрыв сохраняется и на этом наборе, вывод «разрыв несёт регистр» не
может быть артефактом отбора по филлерам.

Запуск:
    python3 build_nonfiller_set.py \
        --manifest manifest/manifest.csv.gz \
        --spont-set manifest/sets/golden_spont_eval.csv \
        --prep-set  manifest/sets/golden_prep_eval_matched.csv \
        --corpus /path/to/ISSAI_KSC2 --out sets/
"""
import argparse, csv, gzip, os, re, sys


def rd(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--spont-set", required=True)
    ap.add_argument("--prep-set", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="sets")
    ap.add_argument("--min-other", type=int, default=2)
    ap.add_argument("--min-tokens", type=int, default=6)
    args = ap.parse_args()

    spont = rd(args.spont_set)
    sessions = {r["speaker_group"] for r in spont}
    used = {r["record_id"] for r in spont} | {r["record_id"] for r in rd(args.prep_set)}

    try:
        import soundfile as sf
    except ImportError:
        sys.exit("нужен soundfile: pip install soundfile")

    cols = ["record_id", "audio_path", "domain", "split", "speaker_group",
            "tokens", "dur_sec", "label", "score", "strong_fillers",
            "weak_fillers", "discourse_markers", "repetitions",
            "russian_mixed", "other_markers", "text"]
    rows, skipped = [], 0

    with gzip.open(args.manifest, "rt", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["speaker_group"] not in sessions or r["record_id"] in used:
                continue
            if int(r["strong_fillers"]) or int(r["weak_fillers"]):
                continue
            other = (int(r["discourse_markers"]) + int(r["repair_phrases"])
                     + int(r["repetitions"]) + int(r["russian_mixed"]))
            if other < args.min_other or int(r["tokens"]) < args.min_tokens:
                continue
            rid = r["record_id"]
            txt = os.path.join(args.corpus, rid)
            n_ext = len(re.search(r"(\.txt)*$", rid).group(0)) // 4
            wav = os.path.join(args.corpus,
                               re.sub(r"(\.txt)+$", "", rid) + ".flac" * max(n_ext, 1))
            if not (os.path.exists(txt) and os.path.exists(wav)):
                skipped += 1; continue
            try:
                info = sf.info(wav)
                dur = round(info.frames / float(info.samplerate), 2)
            except Exception:
                skipped += 1; continue
            with open(txt, encoding="utf-8") as f2:
                text = f2.read().strip()
            rows.append({"record_id": rid, "audio_path": wav,
                         "domain": r["domain"].split("/")[-1],
                         "split": rid.split("/")[0],
                         "speaker_group": r["speaker_group"], "tokens": r["tokens"],
                         "dur_sec": dur, "label": r["label"], "score": r["score"],
                         "strong_fillers": r["strong_fillers"],
                         "weak_fillers": r["weak_fillers"],
                         "discourse_markers": r["discourse_markers"],
                         "repetitions": r["repetitions"],
                         "russian_mixed": r["russian_mixed"],
                         "other_markers": other, "text": text})

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "nonfiller_spont.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    import collections
    print("отобрано %d высказываний, %.2f ч, сессий %d"
          % (len(rows), sum(r["dur_sec"] for r in rows) / 3600.0,
             len({r["speaker_group"] for r in rows})))
    print("домены:", dict(collections.Counter(r["domain"] for r in rows)))
    print("разделы:", dict(collections.Counter(r["split"] for r in rows)))
    print("средняя длительность %.2f с (у основного спонтанного набора 9.37 с)"
          % (sum(r["dur_sec"] for r in rows) / max(len(rows), 1)))
    if skipped:
        print("пропущено из-за отсутствия файлов: %d" % skipped)
    print("набор:", path)


if __name__ == "__main__":
    main()
