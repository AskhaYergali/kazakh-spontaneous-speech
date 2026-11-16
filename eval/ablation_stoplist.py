#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пункт 5 рецензии, последний вариант абляции: фильтр без стоп-листа редупликации.

Прогоняет ОПУБЛИКОВАННЫЙ фильтр дважды — с обычным лексиконом и с пустым
стоп-листом — и считает точность, полноту, F1 и объём отбора на тех же
191 размеченных элементах. Логика подсчёта повторов при этом не переписывается,
меняется только содержимое стоп-листа.

Запуск на Mac, из каталога корпуса:
    python3 ablation_stoplist.py \
        --filter  ~/kazakh-spontaneous-speech/filter/spontaneity_filter.py \
        --lexicons ~/kazakh-spontaneous-speech/filter/lexicons.json \
        --annotation ~/kazakh-spontaneous-speech/annotation \
        --corpus . --out ablation_stoplist.csv

Полный проход по 485 950 расшифровкам занимает несколько минут.
"""
import argparse, csv, json, math, os, subprocess, sys, tempfile

TH1, MIN_TOK = 5.0, 6

def rd(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))

def truth_map(ann):
    key = {r["id"]: r for r in rd(os.path.join(ann, "validation_key.csv"))}
    l1 = {r["id"]: r["ОЦЕНКА"].strip() for r in rd(os.path.join(ann, "validation_listener_1.csv"))}
    l2 = {r["id"]: r["ОЦЕНКА"].strip() for r in rd(os.path.join(ann, "validation_listener_2.csv"))}
    adj = {r["id"]: r["СОГЛАСОВАННАЯ"].strip()
           for r in rd(os.path.join(ann, "validation_disagreements.csv"))}
    out = {}
    for i, k in key.items():
        v = l1.get(i) if l1.get(i) == l2.get(i) else adj.get(i)
        if v in ("S", "P"):
            out[k["record_id"]] = (v, float(k["stratum_N"]) / float(k["stratum_n"]))
    return out

def run_filter(filt, lex, corpus, out):
    cmd = [sys.executable, filt, corpus, "-o", out, "-l", lex]
    print("   запускаю:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        sys.exit("фильтр завершился с ошибкой")
    return out

def evaluate(manifest, truth):
    tp = fp = fn = 0.0
    sel = 0
    with open(manifest, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            strict = r["label"] == "likely_spontaneous"
            if strict: sel += 1
            t = truth.get(r["record_id"])
            if not t: continue
            lab, w = t
            if strict and lab == "S": tp += w
            elif strict and lab == "P": fp += w
            elif (not strict) and lab == "S": fn += w
    P = tp/(tp+fp) if tp+fp else float("nan")
    R = tp/(tp+fn) if tp+fn else float("nan")
    return P, R, (2*P*R/(P+R) if P+R else float("nan")), sel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--lexicons", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--corpus", default=".")
    ap.add_argument("--out", default="ablation_stoplist.csv")
    ap.add_argument("--keep-manifests", action="store_true")
    args = ap.parse_args()

    truth = truth_map(args.annotation)
    print("размеченных элементов:", len(truth))
    L = json.load(open(args.lexicons, encoding="utf-8"))
    print("стоп-лист редупликации:", len(L.get("reduplication_stoplist", [])), "форм")

    tmp = tempfile.mkdtemp(prefix="ablation_")
    L2 = dict(L); L2["reduplication_stoplist"] = []
    L2["lexicon_version"] = str(L.get("lexicon_version", "?")) + "-nostoplist"
    lex2 = os.path.join(tmp, "lexicons_nostoplist.json")
    json.dump(L2, open(lex2, "w", encoding="utf-8"), ensure_ascii=False)

    rows = []
    for name, lex in (("со стоп-листом (базовая)", args.lexicons),
                      ("без стоп-листа редупликации", lex2)):
        man = os.path.join(tmp, name.split()[0] + ".csv")
        run_filter(args.filter, lex, args.corpus, man)
        P, R, F, sel = evaluate(man, truth)
        print("%-30s точн. %.3f  полн. %.3f  F1 %.3f  отобрано %d" % (name, P, R, F, sel))
        rows.append([name, round(P,3), round(R,3), round(F,3), sel])
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant","precision_weighted","recall_weighted","f1","selected_corpus"])
        w.writerows(rows)
    print("\nтаблица:", args.out)
    if args.keep_manifests:
        print("манифесты оставлены в", tmp)

if __name__ == "__main__":
    main()
