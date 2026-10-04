# 01 · Routing support tickets with a promise on the errors

Post: TODO-devto-1

## The problem

A support router reads a ticket and picks a team. Most routers are a classifier plus a threshold someone picked by eye:
nobody can say how often it is wrong, and nobody notices when a new kind of ticket starts arriving — the router puts it
on a known team, because a known team is all it can answer.

What we want instead: a router that **states its error up front** ("at most 5% wrong among the answers I give alone"),
keeps it, hands what it is unsure of to a slower reader and then to a person, records every decision — and has a
defence for tickets of a kind nobody labelled.

## Why it is solved this way

- **System 1**, the fast router, is a small head over facts that plain functions compute from the ticket. Cheap,
  deterministic, testable.
- **The slow path** reads more (in production an LLM; here a stand-in function that also reads the subject line). It
  gets only the slice of tickets where it measurably beats System 1.
- **A person** gets whatever neither can answer within the promise.
- **The promise** is calibrated on labelled tickets the head did not see. It holds for tickets *like those*; for a new
  kind of ticket, the open-set gate (`novel="auto"`) sizes the threshold for a share of tickets nobody labelled and
  raises a flag when that share jumps.

## What solvi does here

`solvi.build(question, examples, catalog=..., max_error=0.05, slow=..., novel=...)` does the wiring: fits System 1,
calibrates its threshold, probes whether the slow path may answer a slice, builds the gate, and stores every decision
in a hash-chained file. `s.explain()` prints every choice and the numbers behind it; `s.replay_all()` re-checks every
stored decision without the slow path.

In the real-data script the router is **not** solvi — it is TF-IDF + logistic regression with an "act" head (the
probability that its answer is right). solvi wraps it as a decision part and adds the promise, the open-set gate
(`OpenSetGate`, `leave_out`), the store and replay.

## Results

### Toy: `routing_toy.py` (synthetic tickets, no data, no model, seconds)

Four teams, 2,000 labelled tickets; then a stream of 1,200 where, from ticket 600 on, a third are fraud reports — a team
that does not exist in the labels. From [`expected/routing_toy.out.txt`](expected/routing_toy.out.txt):

```
Its promise: error among the answers given alone ≤ 0.05 with probability ≥ 0.9, for inputs like the calibration examples. Calibrated on 250 examples: threshold 0.5895, answered alone 82.0%, error among them 0.00%, P(alone and wrong) 0.00%; AUROC of the signal 0.95.
…
  slice 'guarantee' (39 calibration examples): the slow path's answer when its confidence ≥ 1 (right on this slice: System 1 56%, slow path 95%, slow path agreeing with System 1 95%)
…
novel=False
  before the shift: by {'s1': 452, 's2': 148, 'human': 0}, wrong among answered alone 0.5% of 600
  after  the shift: by {'s1': 395, 's2': 205, 'human': 0}, wrong among answered alone 37.2% of 600
  replay failures: 0
novel='auto'
  before the shift: by {'s1': 165, 's2': 0, 'human': 435}, wrong among answered alone 0.0% of 165
  after  the shift: by {'s1': 92, 's2': 0, 'human': 508}, wrong among answered alone 0.0% of 92
  replay failures: 0
  open-set gate flag: 680 | signals below 0.59: CUSUM 10.6 ≥ 10.3 (tuned to a share of 0.1; since decision 177)
```

The plain promise is excellent until the shift and then quietly wrong on 37.2% of what it answers alone. The gate keeps
0.0% wrong before and after and flags the change 80 tickets in — and pays for it every day: before any shift it answers
165 of 600 alone instead of 600. The toy is built to make the effect visible; it is not a benchmark.

### Real data: `banking77_stream.py` (Banking77, 77 bank-customer intents)

57 intents to fit and calibrate on; a stream of 2,000 requests where, from request 1,000 on, the rest of the test set
mixes in 20 intents the router never saw. Promise: at most 5% wrong among the answers given alone. From
[`expected/banking77_stream.out.txt`](expected/banking77_stream.out.txt) (about 1.5 minutes on 4 CPU threads):

```
plain promise
  before the shift: answered alone  80.8%, wrong among them  2.5% (20 of 808)
  after  the shift: answered alone  58.5%, wrong among them 16.4% (96 of 585)
    of which known intents:  answered alone  82.9%, wrong among them  1.8% (9 of 498)
    of which unseen intents: answered alone  21.8%, wrong among them 100.0% (87 of 87)
  stored decisions: chain verifies True; 100 replayed, 0 failed

open-set gate
  before the shift: answered alone  57.6%, wrong among them  0.7% (4 of 576)
  after  the shift: answered alone  13.7%, wrong among them  0.7% (1 of 137)
    of which known intents:  answered alone  22.6%, wrong among them  0.0% (0 of 136)
    of which unseen intents: answered alone   0.3%, wrong among them 100.0% (1 of 1)
  flag: 68 requests after the shift
```

Same shape as the toy, on real requests: the plain promise breaks without a sound (16.4% against 5%; every request about
an unseen intent that it answered alone was wrong), the gate keeps it (0.7%) by answering far less. The split and the
numbers are the same as Banking77 on solvi's task stand
([benchmarks/tasks](https://github.com/solvi-ai/solvi/tree/main/benchmarks/tasks)).

## Where it does not help

- **It does not make the router more accurate.** The router is as good as its facts and its slow path.
- **The promise does not hold between an abrupt shift and its detection.** The gate shrinks that window; it does not
  close it. When the flag goes up (`s.gate.flag_at`), stop answering alone until people have looked and the thresholds
  are calibrated again.
- **The gate costs answers in calm times** (57.6% answered alone instead of 80.8% on Banking77 before any shift). Use it
  only where new kinds of input are expected; if they come gradually, solvi's `track=(200,)` costs less; if nothing new
  can appear, the plain promise is the right one.
- **It does not learn the new kind.** Labelling it, adding the option and recalibrating are your job.
- **No growth over time.** On streams of one kind of decision, facts added from a person's answers to escalations gave
  no measurable gain in solvi's own tests; the store gives accountability (sources, retraction), not growth.

## Run it

```bash
uv venv && uv pip install -r requirements.txt
uv run python routing_toy.py                                  # offline, seconds
uv run python fetch_data.py                                   # Banking77, about 1 MB
OMP_NUM_THREADS=4 uv run python banking77_stream.py           # about 1.5 minutes
uv run pytest                                                 # the toy always; Banking77 when downloaded
```

With a real LLM as the slow path, replace `slow=reader` in `routing_toy.py` by
`slow=solvi.models.llm(URL, MODEL, api_key=KEY)` (any OpenAI-compatible server) and add `price=` for the dollars; the
rest does not change.

## Data

- Toy: synthetic, generated by the script.
- [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), **CC BY 4.0**: Casanueva, Temčinas, Gerz,
  Henderson, Vulić, "Efficient Intent Detection with Dual Sentence Encoders", 2020. Downloaded by `fetch_data.py`
  from the authors' repository; not stored here.
