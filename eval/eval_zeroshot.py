#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент A: замер регистрового разрыва zero-shot.

Прогоняет модель по спонтанному и подготовленному разделам, считает WER и CER
в двух режимах шкалирования (с заполнителями и без) и сохраняет все гипотезы
для последующего разбора ошибок (H1-H3).

Запуск на поде:
    python eval_zeroshot.py --model openai/whisper-large-v3-turbo \
        --sets golden/spont_test.csv golden/prep_test.csv --out results/

Сравнение нескольких моделей: запускать по одной, результаты дописываются
в results/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

import torch
import jiwer
import soundfile as sf
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Заполнители из lexicons.json — режим "без заполнителей" удаляет их
# из ЭТАЛОНА И ГИПОТЕЗЫ одинаково.
FILLERS = {
    "ыы", "ыыы", "ыыыы", "ыыыыы", "ыыыыыы",
    "іі", "ііі", "іііі", "ііііі",
    "мм", "ммм", "мммм", "аа", "ааа", "аааа", "оо", "ооо",
    "ээ", "эээ", "мхм", "мһм", "аха", "ыһы", "әһә",
    "ы", "і", "м", "э", "эм", "ә", "о", "а",
}

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str, mode) -> str:
    """mode: False/'retained' — как есть; True/'stripped' — удалить заполнители;
    'canon' — любой заполнитель заменить на <F> (важен факт, не орфография).
    Каноническое шкалирование — ОСНОВНАЯ метрика: форма заполнителя в эталонах
    KSC2 записана непоследовательно (слабый сингармонизм, OR~1.6), и её
    расхождение с гипотезой не является ошибкой распознавания."""
    t = PUNCT_RE.sub(" ", text.lower())
    toks = t.split()
    if mode is True or mode == "stripped":
        toks = [x for x in toks if x not in FILLERS]
    elif mode == "canon":
        toks = ["<F>" if x in FILLERS else x for x in toks]
    return " ".join(toks)


def load_audio(path: str, sr: int = 16000):
    wav, orig = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig != sr:
        wav = librosa.resample(wav, orig_sr=orig, target_sr=sr)
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="полная модель или БАЗА, на которую накладывается адаптер")
    ap.add_argument("--adapter", default=None,
                    help="LoRA-адаптер PEFT поверх --model (например kaz-rus-v1)")
    ap.add_argument("--processor", default=None,
                    help="откуда брать препроцессор, если в репозитории модели его нет")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--language", default="kk",
                    help="код языка или 'none' для автоопределения")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = без ограничения")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name_for_tag = args.adapter or args.model
    lang = None if str(args.language).lower() == "none" else args.language
    tag = name_for_tag.replace("/", "__") + ("" if lang else "__autolang")

    proc_src = args.processor or args.model
    print("Препроцессор: {}".format(proc_src))
    processor = WhisperProcessor.from_pretrained(proc_src)

    print("Модель: {}".format(args.model))
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16)
    if args.adapter:
        from peft import PeftModel
        print("Адаптер: {} (сливаю в веса)".format(args.adapter))
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model = model.to(args.device).eval()

    forced = None
    if lang:
        try:
            forced = processor.get_decoder_prompt_ids(language=lang, task=args.task)
        except Exception as exc:
            print("  языковой токен не задан ({}), декодирую без него".format(exc))
    else:
        print("  язык не форсируется, модель определяет его сама")

    summary_path = os.path.join(args.out, "summary.csv")
    summary_rows = []

    for set_path in args.sets:
        name = os.path.splitext(os.path.basename(set_path))[0]
        with open(set_path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        print("\n=== {} : {} записей ===".format(name, len(rows)))

        hyps, refs, keep = [], [], []
        t0 = time.time()
        for i in range(0, len(rows), args.batch):
            chunk = rows[i: i + args.batch]
            audios, chunk_ok = [], []
            for r in chunk:
                try:
                    audios.append(load_audio(r["audio_path"]))
                    chunk_ok.append(r)
                except Exception:
                    continue
            if not audios:
                continue
            inputs = processor(audios, sampling_rate=16000,
                               return_tensors="pt", padding=True)
            feats = inputs.input_features.to(args.device, dtype=torch.float16)
            with torch.no_grad():
                gen = model.generate(
                    feats, num_beams=args.beam, max_new_tokens=220,
                    forced_decoder_ids=forced)
            out = processor.batch_decode(gen, skip_special_tokens=True)
            for r, h in zip(chunk_ok, out):
                hyps.append(h.strip())
                refs.append(r["text"])
                keep.append(r)
            done = i + len(chunk)
            if done % (args.batch * 10) < args.batch:
                el = time.time() - t0
                print("  {}/{}  {:.1f} зап/с".format(done, len(rows), done / max(el, 1e-9)))

        # метрики в двух режимах
        res = {}
        for mode, strip in (("retained", False), ("canon", "canon"), ("stripped", True)):
            R = [normalize(x, strip) for x in refs]
            H = [normalize(x, strip) for x in hyps]
            pairs = [(r, h) for r, h in zip(R, H) if r]
            if not pairs:
                continue
            R2 = [p[0] for p in pairs]
            H2 = [p[1] for p in pairs]
            res["wer_" + mode] = jiwer.wer(R2, H2)
            res["cer_" + mode] = jiwer.cer(R2, H2)

        print("  WER канонический (ОСНОВНОЙ) = {:.2%}".format(res.get("wer_canon", float("nan"))))
        print("  WER (с заполнителями) = {:.2%}".format(res.get("wer_retained", float("nan"))))
        print("  WER (без заполнителей) = {:.2%}".format(res.get("wer_stripped", float("nan"))))
        print("  CER (с заполнителями) = {:.2%}".format(res.get("cer_retained", float("nan"))))

        # подробные гипотезы для разбора ошибок
        det = os.path.join(args.out, "{}__{}.csv".format(tag, name))
        with open(det, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["record_id", "domain", "score", "strong_fillers",
                        "russian_mixed", "dur_sec", "reference", "hypothesis",
                        "wer_utt"])
            for r, ref, hyp in zip(keep, refs, hyps):
                rn, hn = normalize(ref, False), normalize(hyp, False)
                try:
                    u = jiwer.wer(rn, hn) if rn else ""
                except Exception:
                    u = ""
                w.writerow([r["record_id"], r["domain"], r["score"],
                            r["strong_fillers"], r["russian_mixed"],
                            r["dur_sec"], ref, hyp, u])

        summary_rows.append({
            "model": name_for_tag, "set": name, "n": len(keep),
            "beam": args.beam, "language": lang or "auto",
            **{k: round(v, 5) for k, v in res.items()},
        })

    new = not os.path.exists(summary_path)
    with open(summary_path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(summary_rows)

    print("\nСводка дописана в {}".format(summary_path))
    print("Подробные гипотезы: {}/{}__*.csv".format(args.out, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
