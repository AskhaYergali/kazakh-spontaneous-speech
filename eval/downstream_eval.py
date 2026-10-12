#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент D: устойчивость последующей обработки текста к ошибкам распознавания.

Конвейер: аудио -> распознавание -> готовая NLP-модель -> результат.
Потолок — та же модель на ЭТАЛОННОМ транскрипте. Измеряется потеря при
переходе от эталона к гипотезе каждой системы распознавания.

Две задачи:
  NER        — извлечение именованных сущностей (модель, обученная на KazNERD).
               Метрика: precision/recall/F1 сущностей гипотезы относительно
               сущностей эталона (эталонное решение = потолок).
  Тональность — полярность отзыва (модель ISSAI на KazSAnDRA; доступ по
               принятию условий на Hugging Face). Метрика: доля совпавших решений.

Запуск на поде:
    python downstream_eval.py --files results/*spont_eval.csv results_B/*spont_eval.csv \
        --out results_D/downstream.csv
    # только NER, без тональности:
    python downstream_eval.py --files ... --no-sentiment
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

import torch
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          AutoTokenizer, PreTrainedTokenizerFast, pipeline)

NER_DEFAULT = "Abzalbek89/kk-bert-small-morph-bpe-kaznerd"
FILLERS = {"ыы","ыыы","ыыыы","ыыыыы","ыыыыыы","іі","ііі","іііі","ііііі","мм","ммм","мммм",
           "аа","ааа","аааа","оо","ооо","ээ","эээ","мхм","мһм","аха","ыһы","әһә",
           "ы","і","м","э","эм","ә","о","а"}
SENT_DEFAULT = "issai/rembert-sentiment-analysis-polarity-classification-kazakh"
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


FILLERS = {
    "ыы", "ыыы", "ыыыы", "ыыыыы", "ыыыыыы", "іі", "ііі", "іііі", "ііііі",
    "мм", "ммм", "мммм", "аа", "ааа", "аааа", "оо", "ооо", "ээ", "эээ",
    "мхм", "мһм", "аха", "ыһы", "әһә", "ы", "і", "м", "э", "эм", "ә", "о", "а",
}


def clean(t, strip=False):
    toks = PUNCT.sub(" ", t.lower()).split()
    if strip:
        toks = [w for w in toks if w not in FILLERS]
    return " ".join(toks)


def entities(pipe, texts, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        for res in pipe(texts[i:i + batch]):
            out.append(collections.Counter(
                (e["entity_group"], " ".join(e["word"].split())) for e in res))
    return out


def prf(gold, pred):
    """Сущности гипотезы против сущностей эталона, на уровне пар (тип, строка)."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        inter = sum((g & p).values())
        tp += inter
        fp += sum(p.values()) - inter
        fn += sum(g.values()) - inter
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="CSV с колонками reference и hypothesis (можно с масками)")
    ap.add_argument("--out", default="results_D/downstream.csv")
    ap.add_argument("--ner-model", default=NER_DEFAULT)
    ap.add_argument("--sentiment-model", default=SENT_DEFAULT)
    ap.add_argument("--no-sentiment", action="store_true")
    ap.add_argument("--strip-fillers", action="store_true",
                    help="удалить заполнители из эталона и гипотезы перед распознаванием сущностей")
    ap.add_argument("--per-utterance", default="results_D/per_utterance.csv",
                    help="разбивка по записям для расчёта доверительных интервалов")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    files = sorted({p for pat in args.files for p in glob.glob(pat)})
    if not files:
        sys.exit("файлы не найдены")
    print("систем к оценке: {}".format(len(files)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    dev = 0 if args.device.startswith("cuda") else -1
    print("загружаю NER: {}".format(args.ner_model))
    try:
        tk = AutoTokenizer.from_pretrained(args.ner_model)
    except Exception as exc:
        # некоторые репозитории объявляют нестандартный класс токенизатора —
        # собираем быстрый токенизатор прямо из tokenizer.json
        print("  AutoTokenizer не справился ({}), беру tokenizer.json".format(str(exc)[:80]))
        from huggingface_hub import hf_hub_download
        tk = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(args.ner_model, "tokenizer.json"),
            unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
            pad_token="[PAD]", mask_token="[MASK]", model_max_length=512)
    mdl_ner = AutoModelForTokenClassification.from_pretrained(args.ner_model)
    ner = pipeline("token-classification", model=mdl_ner, tokenizer=tk,
                   aggregation_strategy="simple", device=dev)
    print("  классов сущностей: {}".format(len(mdl_ner.config.id2label)))

    sent = None
    if not args.no_sentiment:
        try:
            print("загружаю классификатор тональности: {}".format(args.sentiment_model))
            tok = AutoTokenizer.from_pretrained(args.sentiment_model)
            mdl = AutoModelForSequenceClassification.from_pretrained(args.sentiment_model)
            sent = pipeline("text-classification", model=mdl, tokenizer=tok,
                            device=dev, truncation=True, max_length=256)
        except Exception as exc:
            print("  недоступна ({}), считаю только NER".format(str(exc)[:120]))

    rows = []
    per = []
    for f in files:
        data = [r for r in csv.DictReader(open(f, encoding="utf-8"))
                if r["reference"].strip() and r["hypothesis"].strip()]
        name = os.path.basename(f).replace("__final_adapter", "").replace(".csv", "")
        if args.strip_fillers:
            name += "__nofillers"
        refs = [clean(r["reference"], args.strip_fillers) for r in data]
        hyps = [clean(r["hypothesis"], args.strip_fillers) for r in data]
        print("\n=== {} : {} записей ===".format(name, len(data)))

        g, p = entities(ner, refs), entities(ner, hyps)
        for r, gg, pp in zip(data, g, p):
            inter = sum((gg & pp).values())
            per.append({"system": name, "record_id": r["record_id"], "tp": inter,
                        "fp": sum(pp.values()) - inter, "fn": sum(gg.values()) - inter})
        P, R, F, tp, fp, fn = prf(g, p)
        n_g, n_p = sum(sum(x.values()) for x in g), sum(sum(x.values()) for x in p)
        print("  сущности: эталон {}, гипотеза {} | P {:.3f} R {:.3f} F1 {:.3f}".format(n_g, n_p, P, R, F))
        row = {"system": name, "n": len(data), "ner_ref_entities": n_g,
               "ner_hyp_entities": n_p, "ner_precision": round(P, 4),
               "ner_recall": round(R, 4), "ner_f1": round(F, 4)}

        if sent:
            lab = lambda xs: [x["label"] for x in sent(xs, batch_size=16)]
            lg, lp = lab(refs), lab(hyps)
            agree = sum(1 for a, b in zip(lg, lp) if a == b) / len(lg)
            row["sentiment_agreement"] = round(agree, 4)
            print("  тональность: совпадение решений {:.1%}".format(agree))
        rows.append(row)

    if per:
        os.makedirs(os.path.dirname(args.per_utterance) or ".", exist_ok=True)
        with open(args.per_utterance, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per[0].keys()))
            w.writeheader(); w.writerows(per)
        print("\nразбивка по записям: {}".format(args.per_utterance))

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nсводка: {}".format(args.out))


if __name__ == "__main__":
    main()
