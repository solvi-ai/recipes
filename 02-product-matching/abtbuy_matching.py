"""Abt-Buy: is a pair of offers from two shops (Abt, Buy) the same product? Real data, no model, no LLM.

1. Code reads the offers into typed facts (pairfacts.py: brand, model number, other codes, sizes, colours, price, and
   how A's and B's compare).
2. `System.fit(..., select=False)` fits a closed-form ridge head over every fact on the labelled train pairs.
3. `System.guarantee(max_risk=0.01)` calibrates a threshold on the valid pairs: a pair is answered alone only when
   P(answered alone and wrong) <= 1% of all pairs, for pairs like these; the rest is handed to a person.
4. `decide_set` with `AtMostOne` on both sides: an offer has at most one counterpart in the other shop, so the most
   probable combination of answers with at most one match per offer is kept (exactly, not greedily).

Fitted and calibrated on train / valid; test is only scored. Prints counts and scores, never the offers (the data has no
stated licence: it is downloaded by fetch_data.py and not redistributed).

    python fetch_data.py && python abtbuy_matching.py
"""
from collections import Counter

from pairfacts import Idf, build, load, pairs, state
from solvi import System
from solvi.core.sets import AtMostOne, Item, decide_set

MAX_RISK = 0.01


def score(rows, key="match"):
    tp = sum(r[key] and r["gold"] for r in rows)
    fp = sum(r[key] and not r["gold"] for r in rows)
    fn = sum(r["gold"] and not r[key] for r in rows)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    a = Counter(r["abt"] for r in rows if r[key])
    b = Counter(r["buy"] for r in rows if r[key])
    conflicts = sum(v > 1 for v in a.values()) + sum(v > 1 for v in b.values())
    return (f"precision {pr:.3f}, recall {rc:.3f}, F1 {2 * pr * rc / max(pr + rc, 1e-9):.3f}; "
            f"offers matched to more than one counterpart: {conflicts}")


if __name__ == "__main__":
    abt, buy = load()
    cat, question = build(Idf(list(abt.values()) + list(buy.values())))
    system = System(cat, [question])
    examples = lambda ps: [(state(abt[p["abt"]], buy[p["buy"]]), "yes" if p["match"] else "no") for p in ps]  # noqa: E731
    train, valid, test = pairs("train"), pairs("valid"), pairs("test")
    print(f"Abt-Buy: {len(abt)} + {len(buy)} offers; labelled pairs: train {len(train)}, valid {len(valid)}, "
          f"test {len(test)} ({sum(p['match'] for p in test)} matches)")

    head = system.fit("match", examples(train), select=False)
    print(f"head over {len(head.features)} facts, fitted on {len(train)} train pairs")
    rep = system.guarantee("match", examples(valid), max_risk=MAX_RISK)
    print(f"guarantee on {rep['n']} valid pairs: threshold {rep['threshold']:.4f}, answered alone {rep['answered']:.1%}")
    print(" ", rep["promise"])

    rows = []
    for p in test:
        r = system.ask(state(abt[p["abt"]], buy[p["buy"]]), store=False)["match"]
        py = float(r.probs["yes"])
        rows.append({"abt": p["abt"], "buy": p["buy"], "gold": p["match"], "p": py, "match": py >= 0.5,
                     "alone": r.status == "ok"})

    items = [Item((r["abt"], r["buy"]), {"yes": r["p"], "no": 1 - r["p"]}, keys={"abt": r["abt"], "buy": r["buy"]})
             for r in rows]
    out = decide_set(items, [AtMostOne("abt"), AtMostOne("buy")])
    for r in rows:
        r["one"] = out[(r["abt"], r["buy"])].answer == "yes"
    print(f"\ntest, {len(rows)} pairs")
    print(f"  head alone:                {score(rows)}")
    print(f"  head + one counterpart:    {score(rows, 'one')}")
    print(f"  one counterpart changed {len(out.changed)} answers, feasible {out.feasible}, exact {out.exact}")
    alone = [r for r in rows if r["alone"]]
    wrong = sum(r["match"] != r["gold"] for r in alone)
    yes = [r for r in alone if r["match"]]
    print(f"  answered alone {len(alone) / len(rows):.1%}; answered alone and wrong {wrong} = {wrong / len(rows):.2%} "
          f"of all pairs (promise {MAX_RISK:.0%}); wrong among the matches answered alone "
          f"{sum(not r['gold'] for r in yes)} of {len(yes)}")
