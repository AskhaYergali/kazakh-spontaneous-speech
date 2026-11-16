# Annotation

Two separate annotation exercises on two different samples. They are not
two passes over the same material, and their agreement figures are not
interchangeable.

Since v1.1 the ground truth of the formal validation is the **majority of
three listeners**. A label is assigned only where at least two of the three
concur, so no label rests on a single person. The figures below were
recomputed under that rule; v1.0 used adjudication by the first author and
reported a corpus-weighted recall of 0.087 for the strict decision where this
version reports 0.072. See `../CHANGELOG.md`.

## Formal validation (`validation_*`, ids V001-V200)

A stratified sample of 200 utterances, 18 strata crossing six subcorpora
with the three filter labels; `validation_key.csv` carries each item's
stratum, the stratum population `stratum_N` and the sample size drawn
`stratum_n`, which is what the stratum weights are computed from.

Two listeners judged every item independently and blind to the filter's
label: `validation_listener_1.csv` and `validation_listener_2.csv`. Listener
2 worked from a blind package in which the audio was renamed B001-B200, and
that column is kept as `№`; join on `id` to recover the utterance from
`validation_key.csv`.
`ОЦЕНКА` is S (spontaneous), P (prepared or read) or NS. NS means the
recording carried no signal either way, usually because it was too short;
the protocol has a separate code X for audio that could not be made out at
all, and no item received it. The nine NS rows carry no confidence and are
excluded from every agreement and accuracy figure. `УВЕРЕННОСТЬ` is
1 (certain) to 3 (uncertain).

Agreement on the 191 items both listeners resolved to S or P: 77.0 per
cent, Cohen's kappa 0.501. On the 78 items where both were confident,
agreement was 92 per cent, kappa 0.80.

### The third listener (`validation_listener_3.csv`)

Because the first author was one of the two listeners and had also designed
the filter, the 53 items the first two did not settle -- the 44 where they
disagreed and the 9 neither could decide -- were given to an independent
third listener, under the same protocol, in randomised order and blind to
the filter label, to the score, to the subcorpus and to both earlier
judgements. The package was built by
`../annotation_tools/build_third_listener_task.py` with seed 20261115.
Columns are `id`, `record_id`, `ОЦЕНКА` and `УВЕРЕННОСТЬ`, as for the other
two; one item was returned as `-`, meaning no decision.

The third listener sided with listener 1 in 37 of the 44 disagreements and
reproduced the earlier joint adjudication in 41 of them (kappa 0.824). A
three-way agreement coefficient is not reported: the third listener heard
only the 53 unsettled items, and on a subset selected for disagreement any
such coefficient describes the selection rather than the listeners.

`validation_disagreements.csv` is kept for reference. It records the earlier
adjudication by the first author -- `A` is listener 1, `B` is listener 2,
`СОГЛАСОВАННАЯ` is the label after re-listening -- and is no longer the
ground truth.

### Ground truth and the figures it gives

Majority of the three listeners: 147 items where the first two agreed
independently, plus 49 settled by the third. Four items remain undecided
and are dropped, leaving **196** of the 200.

Against those labels the filter's `likely_spontaneous` class has precision
0.824 and recall 0.403 on the sample, and precision 0.983 with recall 0.072
once the strata are weighted back to the corpus. The weighted pair is the
one that describes the corpus; the unweighted pair describes the sample
only. Precision is uneven across subcorpora: 12/12 on podcasts, 12/12 on
radio, 11/11 on talkshow, 10/12 on parliament, 10/11 on tv_news and 1/10 on
crowdsourced. The filter is not reliable on crowdsourced speech, which is
why that subcorpus carries no interpretive weight in the study. The six
counts sum to 56 correct out of 68 positives, which is the 0.824 reported
above.

Removing the first author from the adjudication does not move the headline
figure: the weighted precision of the strict decision is 0.983 under the
majority rule and was 0.983 under the earlier adjudication. What moves it is
the choice of which listener to follow -- 0.969 on the 147 independent
agreements, 0.793 on the labels of listener 2 alone. All of these are
printed by `../eval/validation_precision.py`.

## Calibration pass (`calibration_*`, ids K001-K120)

An earlier sample of 120 utterances drawn evenly from ten score bands, used
to place the thresholds. `calibration_key.csv` gives each item's score and
band. Two listeners judged it: `calibration_listener_1.csv` and
`calibration_listener_2.csv`.

Agreement 70.0 per cent, Cohen's kappa 0.353. The disagreements are
one-sided: of 36, thirty-three are listener 1 saying P where listener 2
says S, and listener 2 returned S for 97 of 120 items against listener 1's
67. The second listener is systematically more willing to call speech
spontaneous. `calibration_disagreements.csv` holds the adjudication, which
went the other way from the validation sample: 25 of 36 in favour of
listener 2, 10 in favour of listener 1, and one left at `none`, meaning no
decision was reached. `calibration_listener_2.csv` has an empty
`УВЕРЕННОСТЬ` column because that listener did not record confidence.

## Test-retest (`spotcheck_*`)

One listener was asked, for each utterance, whether the audio contains a
hesitation, repetition, filled pause or cut-off word that the reference
transcript does not record (`ОТВЕТ`: Д yes, Н no). The 100 utterances are
drawn from the prepared matched control set, not at random from the corpus,
which is why 84 of the 100 answers are Н: that set is read speech and its
transcripts omit little. The first 33 were then re-judged blind. Four of 33
decisions changed, kappa 0.639. That figure bounds how high agreement
between two different listeners on this material can be.

## Filler spelling audit (`filler_spelling_audit*`)

Thirty positions where the reference transcript and the model wrote a
filled pause differently. The listener heard the audio and chose variant A
or B without knowing which was which; the key records what A was and the
two spellings. A third option, "cannot tell them apart", was available and
was never used: the listener always heard a vowel. The reference spelling
was chosen 15 times out of 30, two-sided binomial p = 1.000. Neither
spelling matches what people hear better than the other.
