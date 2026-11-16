# Changelog

All notable changes to this release are recorded here. Version numbers follow
the Zenodo versions of the record; the concept DOI 10.5281/zenodo.23357132
always resolves to the latest one.

## v1.1.0 -- 2026-11-16

### Changed: the ground truth of the formal validation

The validation sample was previously settled by adjudication: where the two
listeners disagreed, the first author re-listened and decided. Because that
author had also designed the filter, the 53 unsettled items were given to an
independent third listener, and the ground truth is now the **majority of the
three** -- a label is assigned only where at least two of them concur. Four
items remain undecided and are dropped, so the decidable set grows from 191 to
196.

The substitution does not move the headline figure: the corpus-weighted
precision of the strict decision is 0.983, the same value the earlier
adjudication gave. It does move the recall, because five newly settled items
change every stratum weight.

| Figure | v1.0 | v1.1 |
| --- | --- | --- |
| Decidable items | 191 | 196 |
| Weighted precision, strict | 0.983 | 0.983 |
| Weighted recall, strict | 0.087 | 0.072 |
| Weighted recall, inclusive | 0.27 | 0.22 |
| Weighted F1, strict / inclusive | 0.16 / 0.42 | 0.13 / 0.36 |
| Weighted precision, prepared label | 0.83 | 0.77 |
| Sample recall, strict | 0.415 | 0.403 |
| Extrapolated spontaneous utterances | ~96,000 (22%) | ~117,000 (27%) |
| Largest stratum weight | 24,843 | 22,585 |

The extrapolation is sensitive to one sampled utterance: a crowdsourced read
prompt with audible re-planning, undecided by the first listener and heard as
spontaneous by the other two, carries a weight of 22,585 and accounts for most
of the change. Dropping it returns the inclusive recall to 0.274 and the
extrapolated total to about 95,000. No other sampled item moves recall by more
than 0.012. `eval/validation_precision.py` prints this leave-one-out
diagnostic.

Anyone who used v1.0 and obtained different recall figures has not made an
error; this is the reason.

### Added

- `annotation/validation_listener_3.csv` -- the independent third listener's
  blind labels on the 53 items the first two left unsettled.
- `annotation_tools/build_third_listener_task.py` and `page_template.html` --
  the builder for that blind listening package.
- `manifest/sets/nonfiller_spont.csv` -- the filler-free spontaneous set of
  Section 6.5: 155 utterances from the sessions of the spontaneous evaluation
  set, carrying no strong or weak filler but at least two other spontaneity
  markers, disjoint from the evaluation set and from the matched control.
- `eval/results/asr_runs/*__nonfiller_spont.csv` and
  `eval/results/summaries/wer_cer_nonfiller.csv` -- results on that set.
- `eval/validation_precision.py`, `eval/supp_figures.py`,
  `eval/harmony_fillers.py`, `eval/figure1_harmony.py`, `eval/score_sets.py`,
  `eval/build_supplementary.py` -- the scripts that produce the tables and
  figures of the paper, previously not released.
- `eval/ablation.py` -- the ten rescored variants of Table 7. The eleventh row,
  the filter without the reduplication stop-list, comes from
  `eval/ablation_stoplist.py`, which runs the released filter twice.
- `requirements.txt` -- the libraries the scripts in `eval/` need, pinned to the
  versions the released results were produced with.
- `eval/adjust_duration.py`, `eval/build_matched_control.py`,
  `eval/build_nonfiller_set.py`, `eval/build_listener_set.py`,
  `eval/ablation_stoplist.py` -- the scripts behind the controls and the
  ablation.
- This file.

### Changed: `eval/build_listener_set.py`

Gained a `majority` mode, now the default, matching the ground truth of
Section 5.2; the `adjudicated` mode is kept for comparison and is no longer
described as the one the paper uses. Its docstring now also states why no
recognition result on these sets is reported: 42 of the 45 utterances of the
prepared side come from the Train partition and are recognised at 0.00% WER,
so the comparison would measure memorisation of the training material rather
than register.

### Documented

- How the rounded `score` column and the `label` column can disagree exactly
  on a threshold, and the 485,914 / 485,950 agreement of an exact-arithmetic
  rescoring (see `README.md`).
- Which analyses cannot be run from the released run files, because they need
  the reference text that KSC2's licence does not let us redistribute.
- Which two files carry corpus word forms and why: the audited filler spellings
  in `annotation/filler_spelling_audit_key.csv` and the repeated word types in
  `filter/repetitions.csv`. No utterance is recoverable from either.

### Unchanged

The filter, the lexicons, the manifest, the experimental sets of v1.0 and all
recognition results. No score and no label changed; only the human ground
truth they are evaluated against, and the figures that follow from it.

## v1.0.0 -- 2026-10-12

First public release: spontaneity filter, lexicons, full utterance-level
manifest for KSC2, experimental sets, human annotation results of two
listeners, and the evaluation scripts and per-utterance recognition results.
