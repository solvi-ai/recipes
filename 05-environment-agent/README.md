# 05 · An agent that learns the world it acts in

Post: TODO-devto-5

## The problem

Most LLM agents start every task from zero: they explore, find the way, and the next day explore again. Some keep a
memory, but it is usually text the model wrote about itself, which nobody can check or take back. We want an agent that
keeps **verified** knowledge about the world it acts in, acts fast where that knowledge predicts the action works,
deliberates where it does not, and keeps written rules as hard checks in both.

## Why it is solved this way

Each step is one decision with two possible answerers:

- **System 1** takes an action the knowledge predicts will work and that advances an open goal: a goal's *skill* (the
  action after which its done check turned true before) or the next step of a confirmed route.
- **System 2** is a search over the offered actions when System 1 has nothing it is sure of: an open goal's action
  first, then what was never tried here, then the way to the nearest place with untried actions.
- **Hard checks hold in both**: the agenda's gates, a hard refusal of the action model, the failure memory.

After every step the knowledge is written back from the outcome — never from the agent's own guess. Rules (what an
operator needs) carry across worlds; map facts belong to the map they were seen on.

## What solvi does here

`solvi.Knowledge(vocabulary=...)` (an action model that learns from outcomes over the conditions you name, skills, map
facts, goals with done checks in code, agenda gates), `solvi.Agent(env, knowledge=km, key=...)`, `RiskBudget` for
justified risk, and `agent.replay()`: every decision is stored with what it was given and re-checked without the
environment. The world is any object with `reset(seed)`, `actions(state)` and `step(action) → Outcome`;
`solvi.testing.conformance.check_environment` checks one of your own.

## Results

`environment_agent.py` is a toy crafting world: 8 places joined by exits, resources (tree, stone, iron, water), six
operators with the environment's own checks, six goals. The agent knows none of the rules. No model, a few seconds.
It is the same file as [examples/25_environment_agent.py](https://github.com/solvi-ai/solvi/blob/main/examples/25_environment_agent.py)
in the solvi repository. From [`expected/environment_agent.out.txt`](expected/environment_agent.out.txt):

```
Part 1 — the same world met again needs fewer slow decisions
  world 7, run 1: 61 steps, 6 of 6 goals, System 1 0, System 2 61, fallback 0, refused 23
  world 7, run 2: 12 steps, 6 of 6 goals, System 1 11, System 2 1, fallback 0, refused 0
  world 8 (new): 17 steps, 6 of 6 goals, System 1 4, System 2 13, fallback 0, refused 3
  every decision replays: 90 of 90; the knowledge journal verifies: True
  learned: collect_stone needs ["'stone'"] here and a pickaxe ['True']

Part 2 — protection vs justified risk (30 episodes on a world with a breaking bridge to the iron)
  protect : iron in 2 of 30 episodes, fell 1 times, 0 risky crossings, 5.07 goals per episode
  risk    : iron in 24 of 30 episodes, fell 6 times, 27 risky crossings, 5.80 goals per episode
  the gate 'no bridge without the stone pickaxe' held in both: True
```

Run 1 explores (61 steps, all System 2, 23 actions refused by the world). Run 2 in the same world: 12 steps, 11 by
System 1, nothing refused. In a new world the agent keeps the rules and re-learns the map: 17 steps.

Part 2: the iron lies only across a rope bridge that breaks one time in three. With protection (the default) the agent
fell once and never crossed again — iron in 2 of 30 episodes. With `RiskBudget` it crossed when the expected gain
outweighed the estimated risk: it fell 6 times and got the iron in 24 episodes. The written gate held in both; gates,
hard predictions and the failure memory are never traded.

The same idea on a real game map is recipe [06](../06-pokemon-world-map/).

## Where it does not help

- **Growth was shown only in environments met again** — this toy and the Pokémon map. Not on streams of one kind of
  decision (recipes 01, 02) or for a support agent with tools (recipe 04); there the knowledge gives accountability,
  not growth.
- **It does not shorten the first exploration.** Run 1 is a search; knowledge pays off on the second visit.
- **The action model learns only what the environment checks**, over your vocabulary. A rule the world does not
  enforce must be written as a gate.
- **Rules learned in one world can be over-cautious in another.** A condition that held at every success by accident
  does not transfer; the model then says "unknown", not "wrong".
- **Justified risk lowers the cost of protection; it does not promise to do as well as an agent with no knowledge.**
  In a harder dungeon game it won back part of what protection cost, not all. Measure it on your own metric first.
- **The knowledge store is measured to 10,000 items**, not beyond.

## Run it

```bash
uv venv && uv pip install -r requirements.txt
uv run python environment_agent.py
uv run pytest
```

## Data

None: the world is generated by the script from a seed.
