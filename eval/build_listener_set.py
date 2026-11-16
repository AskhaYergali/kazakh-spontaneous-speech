#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пункт 1 рецензии: набор, в котором метка спонтанности получена НЕ от фильтра,
а от слушателей (§5). Строит два набора в формате eval_zeroshot.py.

Запуск:
    python build_listener_set.py \
        --key   annotation/validation_key.csv \
        --l1    annotation/validation_listener_1.csv \
        --l2    annotation/validation_listener_2.csv \
        --manifest manifest/manifest.csv.gz \
        --corpus /path/to/ISSAI_KSC2 \
        --truth agree \
        --out sets/

Режимы --truth:
    agree       только случаи, где слушатели совпали независимо (основной)
    l2          разметка второго слушателя, не участвовавшего в разработке фильтра
    adjudicated как в статье: совпадения плюс согласованные решения
"""
import argparse, csv, gzip, os, re, sys

def rd(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--disagreements", default=None,
                    help="нужен только для --truth adjudicated")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--corpus", required=True, help="корень распакованного KSC2")
    ap.add_argument("--truth", default="agree", choices=["agree", "l2", "adjudicated"])
    ap.add_argument("--out", default="sets")
    args = ap.parse_args()

    key = {r["id"]: r for r in rd(args.key)}
    l1 = {r["id"]: r["ОЦЕНКА"].strip() for r in rd(args.l1)}
    l2 = {r["id"]: r["ОЦЕНКА"].strip() for r in rd(args.l2)}

    truth = {}
    if args.truth == "agree":
        for i in key:
            if l1.get(i) == l2.get(i) and l1.get(i) in ("S", "P"):
                truth[i] = l1[i]
    elif args.truth == "l2":
        truth = {i: v for i, v in l2.items() if v in ("S", "P")}
    else:
        if not args.disagreements:
            sys.exit("для --truth adjudicated нужен --disagreements")
        adj = {r["id"]: r["СОГЛАСОВАННАЯ"].strip() for r in rd(args.disagreements)}
        for i in key:
            v = l1.get(i) if l1.get(i) == l2.get(i) else adj.get(i)
            if v in ("S", "P"):
                truth[i] = v

    man = {r["record_id"]: r for r in rd(args.manifest)}

    try:
        import soundfile as sf
    except ImportError:
        sys.exit("нужен soundfile: pip install soundfile")

    os.makedirs(args.out, exist_ok=True)
    cols = ["record_id", "audio_path", "domain", "split", "speaker_group",
            "tokens", "dur_sec", "label", "score", "strong_fillers",
            "weak_fillers", "discourse_markers", "repetitions",
            "russian_mixed", "text", "listener_label"]

    buckets = {"S": [], "P": []}
    missing = []
    for vid, verdict in sorted(truth.items()):
        rid = key[vid]["record_id"]
        m = man.get(rid)
        if m is None:
            missing.append((rid, "нет в манифесте")); continue
        txt_path = os.path.join(args.corpus, rid)
        # в tv_news расширения удвоены: 415515.txt.txt / 415515.flac.flac
        n_ext = len(re.search(r"(\.txt)*$", rid).group(0)) // 4
        audio_rel = re.sub(r"(\.txt)+$", "", rid) + ".flac" * max(n_ext, 1)
        audio_path = os.path.join(args.corpus, audio_rel)
        if not os.path.exists(txt_path) or not os.path.exists(audio_path):
            missing.append((rid, "нет файла в корпусе")); continue
        with open(txt_path, encoding="utf-8") as fh:
            text = fh.read().strip()
        try:
            info = sf.info(audio_path)
            dur = round(info.frames / float(info.samplerate), 2)
        except Exception as exc:
            missing.append((rid, "аудио нечитаемо: %s" % exc)); continue
        buckets[verdict].append({
            "record_id": rid, "audio_path": audio_path,
            "domain": m["domain"].split("/")[-1], "split": rid.split("/")[0],
            "speaker_group": m["speaker_group"], "tokens": m["tokens"],
            "dur_sec": dur, "label": m["label"], "score": m["score"],
            "strong_fillers": m["strong_fillers"], "weak_fillers": m["weak_fillers"],
            "discourse_markers": m["discourse_markers"],
            "repetitions": m["repetitions"], "russian_mixed": m["russian_mixed"],
            "text": text, "listener_label": verdict,
        })

    for verdict, name in (("S", "listener_spont"), ("P", "listener_prep")):
        rows = buckets[verdict]
        path = os.path.join(args.out, name + ".csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        total = sum(float(r["dur_sec"]) for r in rows)
        sess = {r["speaker_group"] for r in rows}
        print("%-16s n=%3d  %5.2f ч  сессий %2d  ->  %s"
              % (name, len(rows), total / 3600.0, len(sess), path))

    both = {r["speaker_group"] for r in buckets["S"]} & {r["speaker_group"] for r in buckets["P"]}
    print("\nсессий, где есть и S, и P: %d  (парный бутстрэп опирается на них)" % len(both))
    if missing:
        print("\nпропущено %d записей:" % len(missing))
        for rid, why in missing[:10]:
            print("   %s — %s" % (rid, why))


if __name__ == "__main__":
    main()
