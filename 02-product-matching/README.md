# 02 · Product matching with one counterpart per offer

Post: TODO-devto-2

## The problem

Two shops sell the same TV. One calls it "Sony KDL-40V512 LCD TV - black", the other "SONY Lcd Tv kdl40v512 (black)"
at a slightly different price, next to its neighbour model kdl40v513. Is it the same product?

Asked pair by pair, a matcher (an LLM or a classifier) does well — and happily says "yes" to two different offers for
the same product, because each question is asked alone. We also want to know which answers can go out without a person
looking.

## Why it is solved this way

1. **Comparison is code's job.** Normalising model codes, comparing sizes, colours and prices is deterministic,
   testable and cheap.
2. **A head decides** over those facts: a closed-form ridge head fitted on labelled pairs, with a probability per answer.
3. **A promise needs a signal that separates right from wrong.** The head's probability over good facts does, so a
   threshold calibrated on held-out pairs says which answers go out alone.
4. **A rule across items belongs after the items' decisions.** "Each offer has at most one counterpart" is enforced on
   all answers together, exactly.

## What solvi does here

- `Catalog` turns plain functions into typed facts, each traced (`@cat.fn`, `@cat.check`, `@cat.extract` with a
  quote of where a model number was read).
- `System.fit("match", examples, select=False)` fits the head over every fact.
- `System.guarantee("match", calib, max_risk=0.01)`: P(answered alone *and* wrong) ≤ 1% of all pairs, for pairs like
  the calibration ones; below the threshold the answer has `status != "ok"` — a person should look.
- `solvi.core.sets.decide_set(items, [AtMostOne(...)])` keeps the most probable combination of answers in which every
  offer matches at most one candidate (`exact: True` = solved exactly, not greedily).

The comparison code itself (`pairfacts.py`, about 200 lines) is ordinary Python written for these two shops; solvi did
not write it.

## Results

### Toy: `matching_toy.py` (synthetic offers with near-miss models, no data, seconds)

From [`expected/matching_toy.out.txt`](expected/matching_toy.out.txt):

```
head over 7 facts, fitted on 866 pairs
guarantee on 878 pairs: threshold 0.6053, answered alone 96.5%
  P(answered alone and wrong) ≤ 0.01 — a share of all inputs — for inputs like the calibration examples
test, 874 pairs: F1 head 0.964 (offers with two counterparts: 13) -> with one counterpart per offer 0.987 (0); 13 answers changed, exact: True
answered alone 96.3%, wrong among them 9 (1.03% of all pairs; promise 1%)
```

Note the last line: 9 of 874 pairs (1.03%) were answered alone and wrong against a promise of 1%. That is not a bug:
`max_risk` (conformal risk control) bounds the share *on average over calibration sets*, and one calibration set and
one test set of 874 pairs fluctuate around it. If you need "at most 1% with high probability", use `max_error=` (learn
then test, holds with probability ≥ 1 − delta), which lets fewer pairs through.

### Real data: `abtbuy_matching.py` (Abt-Buy, 1,081 + 1,092 offers)

Fitted on the 5,743 labelled train pairs, calibrated on the 1,916 valid pairs, scored on the 1,916 test pairs (206
matches). No model reads the text, no LLM. From [`expected/abtbuy_matching.out.txt`](expected/abtbuy_matching.out.txt):

```
head over 17 facts, fitted on 5743 train pairs
guarantee on 1916 valid pairs: threshold 0.8133, answered alone 96.6%
  P(answered alone and wrong) ≤ 0.01 — a share of all inputs — for inputs like the calibration examples

test, 1916 pairs
  head alone:                precision 0.989, recall 0.879, F1 0.931; offers matched to more than one counterpart: 1
  head + one counterpart:    precision 0.995, recall 0.879, F1 0.933; offers matched to more than one counterpart: 0
  one counterpart changed 1 answers, feasible True, exact True
  answered alone 96.9%; answered alone and wrong 11 = 0.57% of all pairs (promise 1%); wrong among the matches answered alone 0 of 164
```

For comparison, on solvi's [task stand](https://github.com/solvi-ai/solvi/tree/main/benchmarks/tasks) (same splits,
measured with solvi 0.8.0) the same LLM-free recipe against `openai/gpt-oss-120b` asked about each pair:
F1 0.933 vs 0.872, and 0 offers matched to two counterparts vs 19; the "one counterpart" rule applied to the LLM's
answers alone raised its F1 to 0.909. On a strong head the rule changes little (1 answer here); on a weak matcher it
helps most.

## Where it does not help

- **It is supervised.** The head needs labelled pairs (5,743 here). A zero-shot LLM is a different trade-off, not a
  weaker version of this one. Without labels, use an LLM as the per-pair decider (`solvi.models.llm(...)`) and keep the
  set rule after it; with a reasoning model, do not force a JSON schema on the reply and give it `max_tokens` room.
- **It does not read the offers for you.** `pairfacts.py` is the domain work.
- **The promise covers pairs like the calibration ones.** A new product category, a shop that writes codes a new way:
  calibrate again. And "1% of all pairs" can hold while the *matches* given alone are wrong more often when most pairs
  are easy non-matches; `groups="answer"` puts the promise inside each answer.
- **Refits move the threshold.** Recalibrate after every refit; a fixed ridge penalty keeps it steadier.
- **No growth over time.** Facts added from a person's answers to escalations gave no measurable gain in solvi's tests
  on matching.

## Run it

```bash
uv venv && uv pip install -r requirements.txt
uv run python matching_toy.py                  # offline, seconds
uv run python fetch_data.py                    # Abt-Buy, about 0.5 MB, downloaded at run time only
uv run python abtbuy_matching.py               # seconds
uv run pytest                                  # the toy always; Abt-Buy when downloaded
```

## Data

- Toy: synthetic, generated by the script.
- **Abt-Buy** (the copy [`matchbench/Abt-Buy`](https://huggingface.co/datasets/matchbench/Abt-Buy) on Hugging Face;
  the benchmark goes back to Köpcke, Thor, Rahm, VLDB 2010, Leipzig benchmark datasets). **No licence is stated.** It
  is therefore only ever downloaded by `fetch_data.py` and never committed here; the script prints counts and scores,
  not offers. Do not add `data/` to the repository (it is in `.gitignore`).
