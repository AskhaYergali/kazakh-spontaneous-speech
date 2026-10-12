#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент B: LoRA-дообучение whisper-turbo-ksc2.

По умолчанию число эпох ФИКСИРОВАНО (3) и лучшая эпоха НЕ выбирается:
spont_tune взят из дикторов KSC2 Train, на котором обучалась сама базовая
whisper-turbo-ksc2, поэтому WER на нём отражает запоминание, а не обобщение.
Метрики на spont_tune печатаются только как диагностика.
Флаг --select включает прежний выбор эпохи — только для набора, не виденного базой.

Основной прогон и контроль отличаются только обучающим файлом:
    python train_lora.py --train golden/spont_train.csv  --out runs/main_r16
    python train_lora.py --train golden/random_train.csv --out runs/random_r16

Абляции:
    --rank 8 | 32         ранг адаптера
    --decoder-only        адаптеры только в декодере
    --train golden/spont_train_ext.csv    расширенный набор
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import random

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (EarlyStoppingCallback, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                          WhisperProcessor)

FILLERS = {
    "ыы", "ыыы", "ыыыы", "ыыыыы", "ыыыыыы", "іі", "ііі", "іііі", "ііііі",
    "мм", "ммм", "мммм", "аа", "ааа", "аааа", "оо", "ооо", "ээ", "эээ",
    "мхм", "мһм", "аха", "ыһы", "әһә", "ы", "і", "м", "э", "эм", "ә", "о", "а",
}
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(text, mode):
    t = PUNCT.sub(" ", text.lower()).split()
    if mode == "canon":
        t = ["<F>" if w in FILLERS else w for w in t]
    elif mode == "strip":
        t = [w for w in t if w not in FILLERS]
    return " ".join(t)


class SpeechSet(Dataset):
    def __init__(self, csv_path, processor, limit=0):
        rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
        self.rows = [r for r in rows if r["text"].strip() and os.path.exists(r["audio_path"])]
        if limit:
            self.rows = self.rows[:limit]
        self.p = processor
        print("{}: {} записей".format(csv_path, len(self.rows)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav, sr = sf.read(r["audio_path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        feats = self.p.feature_extractor(wav, sampling_rate=16000).input_features[0]
        labels = self.p.tokenizer(r["text"].strip()).input_ids
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, processor, start_id):
        self.p, self.start = processor, start_id

    def __call__(self, batch):
        feats = self.p.feature_extractor.pad(
            [{"input_features": b["input_features"]} for b in batch], return_tensors="pt")
        lab = self.p.tokenizer.pad([{"input_ids": b["labels"]} for b in batch], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.start).all().item():
            labels = labels[:, 1:]
        feats["labels"] = labels
        return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="abilmansplus/whisper-turbo-ksc2")
    ap.add_argument("--train", required=True)
    ap.add_argument("--tune", default="golden/spont_tune.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--decoder-only", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--select", action="store_true",
                    help="выбирать лучшую эпоху по --tune (только если база его не видела)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="для пробного запуска")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    processor = WhisperProcessor.from_pretrained(args.base, language="kazakh", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "kazakh"
    model.generation_config.task = "transcribe"

    targets = (r".*decoder.*\.(q_proj|v_proj)$" if args.decoder_only else ["q_proj", "v_proj"])
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=targets, bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train = SpeechSet(args.train, processor, args.limit)
    tune = SpeechSet(args.tune, processor, min(args.limit, 200) if args.limit else 0)
    collator = Collator(processor, model.config.decoder_start_token_id)
    tok = processor.tokenizer

    def metrics(pred):
        ids = pred.predictions
        lab = np.where(pred.label_ids != -100, pred.label_ids, tok.pad_token_id)
        hyp = tok.batch_decode(ids, skip_special_tokens=True)
        ref = tok.batch_decode(lab, skip_special_tokens=True)
        out = {}
        for mode in ("canon", "keep", "strip"):
            R = [norm(x, mode) for x in ref]
            H = [norm(x, mode) for x in hyp]
            keep = [(r, h) for r, h in zip(R, H) if r]
            out["wer_" + mode] = 100 * jiwer.wer([k[0] for k in keep], [k[1] for k in keep])
        return out

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        num_train_epochs=args.epochs, fp16=True, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=2, load_best_model_at_end=args.select,
        metric_for_best_model="wer_canon", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        logging_steps=25, report_to="none", remove_unused_columns=False,
        label_names=["labels"], dataloader_num_workers=4, seed=args.seed)

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train, eval_dataset=tune,
        data_collator=collator, compute_metrics=metrics,
        callbacks=([EarlyStoppingCallback(early_stopping_patience=args.patience)] if args.select else []))
    trainer.train()

    best = os.path.join(args.out, "best_adapter" if args.select else "final_adapter")
    model.save_pretrained(best)
    processor.save_pretrained(best)
    hist = [h for h in trainer.state.log_history if "eval_wer_canon" in h]
    json.dump({"args": vars(args), "selected": args.select,
               "best_checkpoint": trainer.state.best_model_checkpoint if args.select else None,
               "tune_history_diagnostic_only": hist},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2, ensure_ascii=False)
    if hist:
        print("\nдиагностика spont_tune (виден базой, для выбора НЕ используется):")
        for h in hist:
            print("  эпоха {:.0f}: canon {:.2f}  keep {:.2f}  strip {:.2f}".format(
                h["epoch"], h["eval_wer_canon"], h["eval_wer_keep"], h["eval_wer_strip"]))
    print("адаптер: " + best)


if __name__ == "__main__":
    main()
