# Kazakh Spontaneous Speech: Spontaneity Filter, Manifest and Annotations

[![DOI 10.5281/zenodo.23357132](https://zenodo.org/badge/1037572614.svg)](https://doi.org/10.5281/zenodo.23357132)

Archived release: <https://doi.org/10.5281/zenodo.23357132> — concept DOI, the latest version.

Supplementary material for the study of register-controlled evaluation and
parameter-efficient adaptation of Whisper on spontaneous Kazakh-Russian
conversational speech.

## What is here

| Path | Contents |
| --- | --- |
| `filter/spontaneity_filter.py` | the spontaneity filter: nine feature counters, length-normalised score S(t), deterministic single pass, standard library only |
| `filter/lexicons.json` | the nine lexicons and the nine feature weights, with version and SHA-256 |
| `filter/repetitions.csv` | word types with immediate repeats across the corpus, used to build the reduplication stop-list |
| `manifest/manifest.csv.gz` | the full manifest: one row per KSC2 utterance, with all feature counters, the score, and the spontaneity label |
| `manifest/sets/` | the experimental sets: Golden Dataset, matched source control, random controls of 500 / 1000 / 2000 utterances |
| `annotation/` | human annotation results: formal validation (two annotators), calibration pass, filler-spelling audit, adjudicated disagreements |
| `annotation_tools/` | scripts that build the blind annotation packages and score them |
| `eval/*.py` | evaluation scripts |
| `eval/results/asr_runs/` | per-utterance recognition results of every run: utterance id, domain, spontaneity score, duration, utterance WER |
| `eval/results/summaries/` | aggregated WER/CER tables and the downstream NER results, prefixed with the working folder they came from |
| `eval/results/PROVENANCE.csv` | every original path in the working tree, the released file it maps to, and its SHA-256 |

Runs were produced over several sessions in overlapping working folders. Files
that are byte-identical after the text columns are removed are published once;
`PROVENANCE.csv` lists all of their original paths, so the mapping is complete
in both directions.

## What is NOT here, and why

**No audio and no transcript text.** The underlying corpus is KSC2
(ISSAI, Nazarbayev University), released under its own licence. This
repository publishes only (a) utterance identifiers, (b) feature counts and
scores computed by our filter, (c) our own annotation labels, and (d) our
code. To reproduce any experiment, obtain KSC2 from ISSAI, place this
repository's `filter/` and `eval/` scripts at the corpus root, and re-run
them; the manifest here lets you verify that your run reproduces our scores
row by row.

Every table in this repository has had its text columns removed
(`text`, `preview`, `reference`, `hypothesis` and the Russian-named
equivalents in the annotation files). A regular-expression check over the
released files confirms that no corpus text remains; `SHA256SUMS.txt` fixes
the contents of the release.

## Thresholds

The labels in `manifest/manifest.csv.gz` were produced with

    theta_1 = 5.0      likely_spontaneous
    theta_2 = 3.0      maybe_spontaneous
    min_tokens = 6     below this, too_short

The `score` column is threshold-independent, so other thresholds can be
applied to it directly.

## Reproducing the manifest

    # with KSC2 unpacked at ./ISSAI_KSC2
    cd ISSAI_KSC2
    python3 /path/to/filter/spontaneity_filter.py \
        --lexicons /path/to/filter/lexicons.json \
        --out manifest_v2.csv

The script prints the lexicon SHA-256 it used. If it matches the
`lexicon_sha256` column of the published manifest, the scores are
bit-identical.

## Related artifacts

| Artifact | Location |
| --- | --- |
| LoRA adapter trained by `eval/train_lora.py` on `manifest/sets/golden_spont_train.csv` (13 MB, MIT) | [askhaYerga/whisper-turbo-ksc2-kk-spontaneous-lora](https://huggingface.co/askhaYerga/whisper-turbo-ksc2-kk-spontaneous-lora) — DOI [10.57967/hf/11281](https://doi.org/10.57967/hf/11281) |
| Base model it adapts (MIT) | [abilmansplus/whisper-turbo-ksc2](https://huggingface.co/abilmansplus/whisper-turbo-ksc2) |

## Licence

Code (`filter/`, `eval/`, `annotation_tools/`): MIT -- see `LICENSE`.

Data (`manifest/`, `annotation/`, `eval/results/`, `filter/lexicons.json`,
`filter/repetitions.csv`): CC BY 4.0 -- see `LICENSE-DATA`.

The KSC2 corpus itself is not redistributed here and remains under its
original licence.

## Citation

See `CITATION.cff`.
