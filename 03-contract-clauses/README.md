# 03 · Contract clauses with quotes you can check

Post: TODO-devto-3

## The problem

Contract review is where "the model said so" is least acceptable. A reviewer wants the passage, not a paraphrase of it;
"this contract has no termination-for-convenience clause" is a real answer, different from "I don't know"; and someone
has to decide which answers a lawyer must look at.

## Why it is solved this way

- **The answer type says what a valid answer is.** `Maybe[Span[str]]`: an exact piece of the text with its offsets, or
  `Unknown` — "not stated". A span is checked against the text; an answer that is not literally there is rejected,
  never passed on.
- **The model proposes, it does not certify.** Its confidence is one input to a decision about who looks, not that
  decision.
- **Long contracts are retrieved, not stuffed.** `long="retrieve"` splits a contract into sections, BM25 picks the few
  that bear on the question by the clause's own vocabulary (`retrieve_query`), and the model reads about 3,000 tokens;
  the quote still points into the whole document.

## What solvi does here

- `solvi.models.llm(URL, MODEL, ...)` talks to any OpenAI-compatible server; `.decision(name, question, "contract",
  Maybe[Span[str]])` asks one typed question. An invalid or cut-off reply never becomes a value: the decision escalates.
  With reasoning on, solvi does not force a JSON schema on the reply (on solvi's task stand, forcing it on a reasoning
  model cost accuracy).
- `find_quote` matches a model's quote on a normalized view (Unicode NFKC, dashes, quotes, whitespace) and stores the
  contract's own text at offsets into the original. A paraphrase is rejected. `Catalog(quotes="literal")` keeps the
  strict rule.

## Results

### `contract_clauses.py`: a small contract, two traps

A seven-section services agreement where Section 4 is titled "Exclusivity" but is a non-compete, and Section 7 lets a
party terminate for *breach*, which is not termination for convenience. Part 1 asked `openai/gpt-oss-120b` (3 requests,
about $0.0001); part 2 runs offline. From [`expected/contract_clauses.out.txt`](expected/contract_clauses.out.txt):

```
Part 1 — an LLM proposes, solvi keeps only passages that are in the contract
  governing_law                contract[775:930] = '6. Governing Law. This Agreement is governed by the laws of the Republ'… literal: True, confidence 1.00
  non_compete                  contract[450:637] = '4. Exclusivity. During the Term and for twelve (12) months after it, P'… literal: True, confidence 0.99
  termination_for_convenience  not stated (confidence 0.95)

Part 2 — what happens to a quote a model writes (offline)
  'governed by the laws of the Republic of Ireland'       -> found at [811:858], stored as 'governed by the laws of the Republic of Ireland'
  'governed by the laws of the Republic\xa0of\xa0Ireland' -> found at [811:858], stored as 'governed by the laws of the Republic of Ireland'
  'governed by Irish law'                                 -> not in the text: rejected
```

The model saw through both traps, and every passage is `contract[start:end]`: a reviewer's UI can highlight it and an
auditor can check it without trusting anyone.

### `cuad_contracts.py`: three real contracts from CUAD

Three contracts of CUAD's held-out test set (a web-hosting agreement, an IP agreement, a promotion agreement; 15,000 to
31,000 characters), three kinds of clause each, answered with `long="retrieve"` and compared with the lawyers'
annotations. 9 requests to `openai/gpt-oss-120b`, $0.0013. From
[`expected/cuad_contracts.out.txt`](expected/cuad_contracts.out.txt) (quotes cut at 60 characters):

```
CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT (15,176 characters)
  Governing Law                lawyers: present | contract[14093:14380] = 'This Agreement was entered into in the State of Florida, and'… literal: True, confidence 0.99 -> right
  Non-Compete                  lawyers: absent  | not stated (confidence 0.95) -> right
  Termination For Convenience  lawyers: present | contract[10880:10996] = 'Either party may terminate this Agreement without cause at a'… literal: True, confidence 0.99 -> right
…
9 of 9 right against the annotation; 7 of 7 quotes literally in the contract
```

Read that as a demonstration, not a score: the three contracts were picked (before the run) for a mix of present and
absent clauses, and nine questions say little. The measured numbers come from solvi's task stand on 1,025 questions
(25 held-out contracts × 41 kinds of clause; solvi 0.8.0, the same model), where the full method — this retrieval plus
a yes/no check of each quoted passage and a calibrated trust score under `System.guarantee(signal="trust",
max_risk=0.03)` — gave:

| | plain LLM call | solvi solution |
|---|---|---|
| accuracy | 0.882 | 0.899 |
| quotes literally in the contract | 260 of 398 | 263 of 263 |
| false claims where the clause is absent (of 721) | 60 | 27 |
| clauses found (of 304) | 243 | 227 |
| answered without a person; wrong among them | 100%; 11.8% | 67.6%; 4.0% |

The script of that run, with the check and the trust score, is
[benchmarks/tasks/cuad/solution.py](https://github.com/solvi-ai/solvi/blob/main/benchmarks/tasks/cuad/solution.py).

## Where it does not help

- **It finds fewer clauses.** On the stand, 227 against 243 of 304 present: the check and the strict quote rule cost
  recall. If a missed clause is worse for you than a false claim, that trade goes the wrong way.
- **It does not make the model read better.** Accuracy moved from 0.882 to 0.899; most of the gain is in *which*
  answers can be trusted.
- **A promise needs labelled questions** of your own contracts and covers contracts like those. A new contract type or
  language: calibrate again; retrieval matches words, so put the document's own words into `retrieve_query`.
- **A quote in the text is not proof that it answers the question.** The check question helps; it is still a model.

## Run it

```bash
uv venv && uv pip install -r requirements.txt
uv run python contract_clauses.py --offline                                   # part 2 only, no model
export LLM_URL=https://openrouter.ai/api/v1 LLM_KEY=...                       # or a local Ollama / vLLM URL
uv run python contract_clauses.py                                             # 3 requests
uv run python fetch_data.py && uv run python cuad_contracts.py                # CUAD test set (7 MB kept), 9 requests
uv run pytest            # offline part always; the LLM invariants (every quote literal) when LLM_URL is set
```

`LLM_MODEL` picks another model (default `openai/gpt-oss-120b`). With another model or on another day the answers may
differ from `expected/`; the test then checks only the invariant that every quote is literally in the contract.

## Data

- The services agreement in `contract_clauses.py` is made up for this recipe.
- [CUAD v1](https://github.com/TheAtticusProject/cuad) (The Atticus Project), **CC BY 4.0**: Hendrycks, Burns, Chen,
  Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021. Downloaded by
  `fetch_data.py`; not stored here. `expected/cuad_contracts.out.txt` quotes short passages of three CUAD contracts,
  with this attribution.
