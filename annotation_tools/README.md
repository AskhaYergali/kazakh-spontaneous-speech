# Annotation tools

The scripts that produced the files in `annotation/`. They are published as
they were run, so their paths follow the working corpus tree, not this
repository's layout. Read this before running any of them.

## What each script does

| Script | Role |
| --- | --- |
| `build_validation.py` | draws the stratified validation sample of 200 utterances (70 likely_spontaneous, 60 maybe_spontaneous, 70 likely_prepared_or_read, spread evenly over the six subcorpora), writes the key and two independently shuffled blind copies, draws the 100-utterance spotcheck from the prepared matched control, and packs an archive for the second listener |
| `make_blind.py` | strips score, band, feature counts and subcorpus from the calibration sample and reshuffles it, so the second listener sees only an id and a preview |
| `make_ipad_package.py` | renames the audio B001-B200 and C001-C100 and builds a package that can be annotated on a tablet with no software installed |
| `validate_listen.py` | terminal listening tool for the validation sample and the spotcheck; plays audio, takes one keystroke, saves after every answer |
| `audit.py` | the same tool for the calibration pass |
| `listen_fillers.py` | blind listening for the 30 filler-spelling positions |
| `protocol_razmetki.md` | the instructions given to the listeners, in Russian |

## Keystrokes and the codes they write

Validation: `s` writes S, `p` writes P, `n` writes NS (no signal), `x`
writes X (unintelligible). Confidence is then 1, 2 or 3.
Spotcheck: `y` writes Д, `n` writes Н.
Filler audit: `a`, `b`, or `n` for cannot tell, then confidence 1 to 3.

## Running them

The scripts expect to sit in a `code/` directory at the root of an unpacked
KSC2 tree, with the manifest and the `golden/` directory beside it:

    ISSAI_KSC2/
      manifest_v2.csv
      golden/prep_eval_matched.csv
      code/build_validation.py, build_dataset.py, ...
      Train/ Dev/ Test/

Two consequences for this repository:

**`build_validation.py` will not run from `annotation_tools/` as published.**
It does `from build_dataset import audio_path, flac_duration` and puts its
own directory on the path, but `build_dataset.py` is published under
`eval/`. Copy both into one directory at the corpus root before running:

    cp annotation_tools/build_validation.py eval/build_dataset.py ISSAI_KSC2/code/
    cd ISSAI_KSC2 && python3 code/build_validation.py manifest_v2.csv

**The set files are published under different names.** The script reads
`golden/prep_eval_matched.csv`; in this repository that file is
`manifest/sets/golden_prep_eval_matched.csv`. Either restore the original
names under a `golden/` directory or edit the path in the script.

The listening tools need the audio, which this repository does not
redistribute, and a player: `afplay` on macOS, `ffplay` or `aplay` on
Linux. They write to `<input>.judged.csv` and save after every answer, so a
session can be interrupted and resumed.

## Seeds

`build_validation.py` uses seed 20260923 for the sample and 101 and 202 for
the two listeners' orderings; `make_blind.py` uses 777. With the same
manifest these reproduce the published samples exactly.
