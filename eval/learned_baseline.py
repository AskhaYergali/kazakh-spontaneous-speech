#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучаемый baseline для метрики спонтанности (раздел 2.1, ответ рецензенту).

Сравнивает РУЧНУЮ формулу S(t) с логистической регрессией, обученной на тех же
признаках и на согласованных экспертных метках, при честной перекрёстной проверке.
Никаких зависимостей: регрессия реализована здесь же.

Нужны файлы (в корне корпуса):
    manifest_v2.csv            признаки по каждой записи
    validation.KEY.csv         id -> record_id, страты
    validation_A_judged.csv    разметка слушателя A
    <экспорт разметчика B>     CSV из Numbers (разделитель ; или ,)
    validation_disagreements.csv  согласованные метки по расхождениям

Запуск из корня корпуса:
    python3 code/learned_baseline.py --b 1_validation_3604.csv
"""
import argparse
import csv
import math
import random
import sys

FEATS = ["strong_filler_per_1000", "weak_filler_per_1000", "discourse_per_1000",
         "repair_per_1000", "repetition_per_1000", "russian_mixed_per_1000"]
HAND = {"strong_filler_per_1000": 0.060, "weak_filler_per_1000": 0.020,
        "discourse_per_1000": 0.015, "repair_per_1000": 0.040,
        "repetition_per_1000": 0.025, "russian_mixed_per_1000": 0.010}


def read_any(path):
    raw = open(path, encoding="utf-8-sig").read()
    delim = ";" if raw.split("\n")[0].count(";") > raw.split("\n")[0].count(",") else ","
    return list(csv.DictReader(raw.splitlines(), delimiter=delim))


def fit(X, y, sw=None, l2=1.0, iters=4000, lr=0.1):
    """Логистическая регрессия, полный градиентный спуск, L2 без штрафа на свободный член.
    sw — веса наблюдений (веса страт), позволяющие учиться под распределение корпуса."""
    n, m = len(X), len(X[0])
    if sw is None:
        sw = [1.0] * n
    tot = sum(sw)
    w = [0.0] * m
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * m
        gb = 0.0
        for xi, yi, si in zip(X, y, sw):
            z = b + sum(w[j] * xi[j] for j in range(m))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = (p - yi) * si
            gb += e
            for j in range(m):
                gw[j] += e * xi[j]
        for j in range(m):
            w[j] -= lr * (gw[j] / tot + l2 * w[j] / tot)
        b -= lr * gb / tot
    return w, b


def predict(w, b, x):
    z = b + sum(wi * xi for wi, xi in zip(w, x))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def auc(scores, labels, weights=None):
    """Взвешенный AUC по попарной формуле, с корректной обработкой совпадений."""
    if weights is None:
        weights = [1.0] * len(scores)
    pairs = sorted(zip(scores, labels, weights))
    pos = sum(w for _s, y, w in pairs if y == 1)
    neg = sum(w for _s, y, w in pairs if y == 0)
    if not pos or not neg:
        return float("nan")
    acc = 0.0
    neg_below = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        pw = sum(pairs[k][2] for k in range(i, j) if pairs[k][1] == 1)
        nw = sum(pairs[k][2] for k in range(i, j) if pairs[k][1] == 0)
        acc += pw * (neg_below + 0.5 * nw)
        neg_below += nw
        i = j
    return acc / (pos * neg)


def pr_at(scores, labels, weights, keep_mass):
    """Точность и полнота при отборе, дающем ту же взвешенную массу, что и ручной фильтр."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    total_pos = sum(weights[i] for i in range(len(labels)) if labels[i] == 1)
    sel_w = tp = 0.0
    for i in order:
        if sel_w >= keep_mass:
            break
        sel_w += weights[i]
        if labels[i] == 1:
            tp += weights[i]
    p = tp / sel_w if sel_w else 0.0
    r = tp / total_pos if total_pos else 0.0
    return p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_v2.csv")
    ap.add_argument("--key", default="validation.KEY.csv")
    ap.add_argument("--a", default="validation_A_judged.csv")
    ap.add_argument("--b", required=True, help="экспорт разметчика B")
    ap.add_argument("--adj", default="validation_disagreements.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()

    key = {r["id"]: r for r in read_any(args.key)}
    A = {r["id"]: r["ОЦЕНКА"].strip().upper() for r in read_any(args.a)}
    B = {r["id"].strip(): r["ОЦЕНКА"].strip().upper() for r in read_any(args.b) if r.get("id")}
    ADJ = {r["id"]: r["СОГЛАСОВАННАЯ"].strip().upper() for r in read_any(args.adj)}

    gold = {}
    for i in A:
        if i not in B or A[i] not in ("S", "P") or B[i] not in ("S", "P"):
            continue
        g = A[i] if A[i] == B[i] else ADJ.get(i, "")
        if g in ("S", "P"):
            gold[i] = 1 if g == "S" else 0
    print("эталонных меток: {} (S: {})".format(len(gold), sum(gold.values())))

    want = {key[i]["record_id"]: i for i in gold}
    feats = {}
    for r in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if r["record_id"] in want:
            feats[want[r["record_id"]]] = [float(r[f]) for f in FEATS]
    miss = [i for i in gold if i not in feats]
    if miss:
        print("!! нет признаков для {} записей, исключены".format(len(miss)))
        for i in miss:
            gold.pop(i)

    ids = sorted(gold)
    X = [feats[i] for i in ids]
    y = [gold[i] for i in ids]
    wt = [float(key[i]["stratum_N"]) / float(key[i]["stratum_n"]) for i in ids]
    hand = [sum(HAND[f] * v for f, v in zip(FEATS, x)) for x in X]

    # стандартизация по обучающей части считается внутри фолда
    rnd = random.Random(args.seed)
    idx = list(range(len(ids)))
    rnd.shuffle(idx)
    folds = [idx[k::args.folds] for k in range(args.folds)]
    oof = [0.0] * len(ids)
    oofw = [0.0] * len(ids)
    coefs, coefsw = [], []
    for k in range(args.folds):
        te = set(folds[k])
        tr = [i for i in idx if i not in te]
        mu = [sum(X[i][j] for i in tr) / len(tr) for j in range(len(FEATS))]
        sd = [max(1e-6, (sum((X[i][j] - mu[j]) ** 2 for i in tr) / len(tr)) ** .5) for j in range(len(FEATS))]
        Z = lambda x: [(x[j] - mu[j]) / sd[j] for j in range(len(FEATS))]
        Ztr = [Z(X[i]) for i in tr]
        ytr = [y[i] for i in tr]
        w, b = fit(Ztr, ytr)
        w2, b2 = fit(Ztr, ytr, sw=[wt[i] for i in tr])
        coefs.append([w[j] / sd[j] for j in range(len(FEATS))])
        coefsw.append([w2[j] / sd[j] for j in range(len(FEATS))])
        for i in te:
            oof[i] = predict(w, b, Z(X[i]))
            oofw[i] = predict(w2, b2, Z(X[i]))

    print("\nРАНЖИРОВАНИЕ (out-of-fold, {} фолдов)".format(args.folds))
    print("  ручная формула S:        AUC = {:.3f}   взвешенный AUC = {:.3f}".format(
        auc(hand, y), auc(hand, y, wt)))
    print("  регрессия, без весов:    AUC = {:.3f}   взвешенный AUC = {:.3f}".format(
        auc(oof, y), auc(oof, y, wt)))
    print("  регрессия, с весами страт: AUC = {:.3f} взвешенный AUC = {:.3f}".format(
        auc(oofw, y), auc(oofw, y, wt)))

    th2 = 3.0
    keep = sum(wt[i] for i in range(len(ids)) if hand[i] >= th2)
    ph, rh = pr_at(hand, y, wt, keep)
    pl, rl = pr_at(oof, y, wt, keep)
    plw, rlw = pr_at(oofw, y, wt, keep)
    print("\nПРИ ОДИНАКОВОМ ОБЪЁМЕ ОТБОРА (масса = отбор ручного фильтра при theta_2 = 3.0)")
    print("  ручная формула:          точность {:.3f}  полнота {:.3f}".format(ph, rh))
    print("  регрессия, без весов:    точность {:.3f}  полнота {:.3f}".format(pl, rl))
    print("  регрессия, с весами страт: точность {:.3f} полнота {:.3f}".format(plw, rlw))

    print("\nВЕСА: ручные против средних обученных (в исходных единицах признака)")
    avg = [sum(c[j] for c in coefsw) / len(coefsw) for j in range(len(FEATS))]
    sc = sum(HAND[f] for f in FEATS) / max(1e-9, sum(abs(a) for a in avg))
    print("  (обученные — модель с весами страт)")
    print("  {:<26}{:>10}{:>14}".format("признак", "ручной", "обученный*"))
    for j, f in enumerate(FEATS):
        print("  {:<26}{:>10.3f}{:>14.3f}".format(f, HAND[f], avg[j] * sc))
    mh = sum(HAND[f] for f in FEATS) / len(FEATS)
    ml = sum(avg[j] * sc for j in range(len(FEATS))) / len(FEATS)
    num = sum((HAND[f] - mh) * (avg[j] * sc - ml) for j, f in enumerate(FEATS))
    den = (sum((HAND[f] - mh) ** 2 for f in FEATS) *
           sum((avg[j] * sc - ml) ** 2 for j in range(len(FEATS)))) ** .5
    print("  * приведены к сумме ручных весов; корреляция Пирсона r = {:.2f}".format(num / den if den else float("nan")))


if __name__ == "__main__":
    main()
