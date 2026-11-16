# Kazakh Spontaneous Speech: Spontaneity Filter, Manifest and Annotations

Supplementary material for the study of register-controlled evaluation and
parameter-efficient adaptation of Whisper on spontaneous Kazakh-Russian
conversational speech.

## Version

This is **v1.1**. The ground truth of the formal validation changed in this
version: it is now the majority judgement of three listeners rather than
adjudication by the first author, and every corpus-weighted figure was
recomputed. `CHANGELOG.md` lists what moved and by how much.

## What is here

| Path | Contents |
| --- | --- |
| `filter/spontaneity_filter.py` | the spontaneity filter: nine feature counters, length-normalised score S(t), deterministic single pass, standard library only |
| `filter/lexicons.json` | the nine lexicons and the nine feature weights, with version and SHA-256 |
| `filter/repetitions.csv` | word types with immediate repeats across the corpus, used to build the reduplication stop-list |
| `manifest/manifest.csv.gz` | the full manifest: one row per KSC2 utterance, with all feature counters, the score, and the spontaneity label |
| `manifest/sets/` | the experimental sets: Golden Dataset, matched source control, random controls of 500 / 1000 / 2000 utterances, and `nonfiller_spont.csv`, the filler-free spontaneous set of Section 6.5 |
| `annotation/` | human annotation results: formal validation (three listeners, ground truth by majority), calibration pass, filler-spelling audit, test-retest |
| `annotation_tools/` | scripts that build the blind annotation packages and score them |
| `eval/*.py` | evaluation and analysis scripts, including the ones that produce the tables and figures of the paper |
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

One consequence is worth stating plainly. Any analysis that rescores the
recognition output -- the canonical and filler-removed conditions, and the
duration-standardised tables of Section 6.5 -- needs the reference text and
therefore cannot be run against the released `eval/results/asr_runs/` files
as they stand. Regenerate those files with `eval/eval_zeroshot.py` against
your own copy of KSC2 first. Everything that reads only the manifest, the
annotation files or the per-utterance WER column runs directly.

## Thresholds

The labels in `manifest/manifest.csv.gz` were produced with

    theta_1 = 5.0      likely_spontaneous
    theta_2 = 3.0      maybe_spontaneous
    min_tokens = 6     below this, too_short

The `score` column is threshold-independent, so other thresholds can be
applied to it directly. It is stored rounded to four decimal places, while
the `label` column records the comparison against the unrounded score. The
two can disagree exactly on a threshold: for one of the 584 utterances of the
spontaneous evaluation set the unrounded score falls below `theta_1` by a
floating-point residue of order 1e-15, so the row carries the borderline
label at a recorded score of 5.0. Rescoring the counters with exact rational
arithmetic reproduces the released label for 485,914 of the 485,950 rows; the
36 exceptions all lie on `theta_1` and belong to no released evaluation or
training set.

## Reproducing the manifest

    # with KSC2 unpacked at ./ISSAI_KSC2
    cd ISSAI_KSC2
    python3 /path/to/filter/spontaneity_filter.py \
        --lexicons /path/to/filter/lexicons.json \
        --out manifest_v2.csv

The script prints the lexicon SHA-256 it used. If it matches the
`lexicon_sha256` column of the published manifest, the scores are
bit-identical.

## Licence

Code (`filter/`, `eval/`, `annotation_tools/`): MIT -- see `LICENSE`.

Data (`manifest/`, `annotation/`, `eval/results/`, `filter/lexicons.json`,
`filter/repetitions.csv`): CC BY 4.0 -- see `LICENSE-DATA`.

The KSC2 corpus itself is not redistributed here and remains under its
original licence.

## Citation

See `CITATION.cff`.
