# solvi recipes

Real tasks solved with [solvi](https://github.com/solvi-ai/solvi), one folder per task. Each recipe shows how the task
is solved, why it is solved this way, what solvi does in it — and where it does not help. Every number in a recipe's
README is the output of a script in that folder, kept verbatim in its `expected/` folder.

The recipes live apart from the library on purpose: a recipe may use any data, model or service, and its data has its
own licence. The library stays small; the recipes show it in context.

## Recipes

| # | Recipe | One line | Needs | Post |
|---|---|---|---|---|
| 01 | [support-routing](01-support-routing/) | Route tickets under a stated error (`solvi.build`): a fast head, a slow path, a person; then a stream where a new kind of ticket appears. Real data: Banking77 with 20 intents the router never saw. | nothing (toy); Banking77 download, scikit-learn | TODO-devto-1 |
| 02 | [product-matching](02-product-matching/) | Code reads the offers, a fitted head decides, a 1% risk promise says what goes out alone, and "one counterpart per offer" is enforced across all answers. Real data: Abt-Buy. | nothing (toy); Abt-Buy download | TODO-devto-2 |
| 03 | [contract-clauses](03-contract-clauses/) | An LLM proposes a clause as an exact span or "not stated"; solvi keeps only quotes that are in the contract. Real data: three CUAD contracts. | an OpenAI-compatible LLM (part of it runs offline); CUAD download | TODO-devto-3 |
| 04 | [agent-tool-guard](04-agent-tool-guard/) | `solvi.Guard` + `solvi.Knowledge` in front of a support agent's tools: provenance of arguments, the customer's yes, a written gate, a backend refusal learned, calls that must not repeat. | nothing (scripted agent) | TODO-devto-4 |
| 05 | [environment-agent](05-environment-agent/) | `solvi.Agent` on a toy crafting world: the same world met again needs a fraction of the slow decisions; protection by default, justified risk as an option. | nothing | TODO-devto-5 |
| 06 | [pokemon-world-map](06-pokemon-world-map/) | System 1 and System 2 on the world map of Pokémon Red (a recording; no ROM, no graphics). A video of the run will be added here. | — | — |

The overview of solvi 1.0 these posts belong to: TODO-devto-overview.

## How a recipe is organised

```
NN-name/
  README.md          the problem, why this way, what solvi does here, results (real outputs), limits, how to run, data + licence
  *.py               the runnable scripts; a toy version with no data and no model comes first where possible
  fetch_data.py      downloads the real data from its source (data is never stored in this repository)
  requirements.txt   solvi pinned to the release the outputs were made with (solvi==1.0.0)
  expected/          the scripts' output, verbatim, from the run the README quotes
  test_*.py          runs the scripts and compares with expected/ — offline where possible; a test that needs
                     downloaded data or an LLM is skipped without them
```

## Run one

With [uv](https://docs.astral.sh/uv/):

```bash
cd 01-support-routing
uv venv && uv pip install -r requirements.txt
uv run python routing_toy.py                       # offline, seconds
uv run python fetch_data.py && uv run python banking77_stream.py
uv run pytest
```

With pip:

```bash
cd 01-support-routing
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python routing_toy.py
pytest
```

Python 3.11 or newer. The tests of every recipe run in CI on Python 3.11 and 3.14 (offline: no data, no model).

**LLM steps.** Only recipe 03 calls a model. It talks to any OpenAI-compatible server through two environment
variables: `LLM_URL` (for example `https://openrouter.ai/api/v1`, or a local Ollama / vLLM URL) and `LLM_KEY`;
`LLM_MODEL` picks the model (default `openai/gpt-oss-120b`). No key is stored anywhere in this repository. Without
`LLM_URL` the LLM part is skipped and the offline part still runs.

## What solvi does here, and what it does not

solvi is a library for decision systems you can check: plain Python functions compute facts, a model proposes where
judgement is needed, and solvi's checks decide. Across these recipes it supplies:

- **promises on errors** calibrated on your own labelled examples (`max_error`, `max_risk`) and an open-set gate for
  inputs nobody labelled;
- **typed answers that are checked**: a quote must be literally in the source, a rule across many answers holds
  exactly, a tool call's arguments must come from the user;
- **a record of every decision**, hash-chained, that replays without the model;
- **knowledge with sources** that an agent acts on and that can be taken back.

It does not:

- **make a model or a classifier more accurate.** The answers are as good as the facts and models you give it; most of
  what solvi adds is *which* answers can go out without a person, and a record of why.
- **keep a promise for inputs unlike the calibration examples.** A new kind of input breaks the plain promise; the
  open-set gate shrinks the window, it does not close it (recipe 01).
- **get better over time by itself on streams of one kind of decision** (routing, matching) or for a support agent
  with tools. Growth from accumulated knowledge was shown only where an agent meets the same world again (recipes 05
  and 06). Elsewhere the knowledge store gives accountability (sources, retraction), not growth.
- **write the domain code for you.** In recipe 02 the comparison code is about 200 lines of ordinary Python.

Each recipe's README has a "Where it does not help" section with the limits measured on that task.

## Versions

Outputs in `expected/` were produced with `solvi==1.0.0` from PyPI on Python 3.11 (and checked on 3.14). When a recipe
moves to a newer solvi, its outputs are re-run and the README's quotes updated in the same commit.

## Licence

Code: Apache-2.0 (see [LICENSE](LICENSE)), like solvi. Data is not part of this repository: each `fetch_data.py`
downloads it from its source, under its own licence, named in the recipe's README.
