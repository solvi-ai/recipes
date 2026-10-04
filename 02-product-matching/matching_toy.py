"""Product matching — code reads the offers, a head decides, a guarantee says what goes out alone, and
"one counterpart per offer" is enforced across the answers (solvi 1.0.0 from PyPI, synthetic offers, no model)."""
import random
import re

from solvi import Answer, Catalog, Question, System
from solvi.core.sets import AtMostOne, Item, decide_set

BRANDS = ["Sony", "Samsung", "Panasonic", "Canon", "Bose", "Garmin"]
KINDS = ["LCD TV", "camera", "headphones", "GPS", "soundbar"]
COLOURS = ["black", "silver", "white"]


def product(rng, n):
    letters = "".join(rng.choice("ABCDKLMSVX") for _ in range(rng.choice([2, 3])))
    return {"id": n, "brand": rng.choice(BRANDS), "kind": rng.choice(KINDS), "colour": rng.choice(COLOURS),
            "code": f"{letters}-{rng.randint(10, 99)}{rng.choice('ABVX')}{rng.randint(100, 999)}",
            "price": round(rng.uniform(40, 1500), 2)}


def offer_a(p):                                 # shop A: "Sony KDL-40V512 LCD TV - black"
    return f"{p['brand']} {p['code']} {p['kind']} - {p['colour']}", p["price"]


def offer_b(p, rng):                            # shop B writes codes and names its own way, prices drift
    code = p["code"].replace("-", rng.choice(["", " ", "-"])).lower() if rng.random() < 0.85 else ""
    name = f"{p['brand'].upper()} {p['kind'].title()} {code} ({p['colour']})".replace("  ", " ")
    return name, round(p["price"] * rng.uniform(0.85, 1.15), 2)


def sibling(p, rng):                            # a near miss: same brand and kind, a neighbouring model
    q = dict(p)
    q["code"] = p["code"][:-1] + str((int(p["code"][-1]) + rng.randint(1, 8)) % 10)
    q["colour"] = rng.choice(COLOURS)
    q["price"] = round(p["price"] * rng.uniform(0.7, 1.3), 2)
    return q


cat = Catalog()


def codes(text):
    return {re.sub(r"[^a-z0-9]", "", m.lower()) for m in re.findall(r"[A-Za-z]{2,3}[- ]?\d{2}[A-Za-z]\d{3}", text)}


@cat.fn
def code_equal(a_name, b_name) -> bool:          # the model codes, normalised, are the same
    return bool(codes(a_name) & codes(b_name))


@cat.fn
def code_missing(b_name) -> bool:                # shop B left the code out
    return not codes(b_name)


@cat.fn
def same_brand(a_name, b_name) -> bool:
    return a_name.split()[0].lower() == b_name.split()[0].lower()


@cat.fn
def same_colour(a_name, b_name) -> bool:
    return a_name.rsplit(" ", 1)[-1].lower() == b_name.rsplit("(", 1)[-1].strip(")").lower()


@cat.fn
def price_gap(a_price, b_price) -> float:        # relative price difference
    return abs(a_price - b_price) / max(a_price, b_price)


def pairs(rng, products):
    """Each A offer against its true counterpart in B (when B has it) and two near misses B also sells."""
    out = []
    for p in products:
        a_name, a_price = offer_a(p)
        cands = [(sibling(p, rng), "no"), (sibling(p, rng), "no")]
        if rng.random() < 0.9:
            cands.append((p, "yes"))
        for k, (q, label) in enumerate(cands):
            b_name, b_price = offer_b(q, rng)
            out.append(({"a_name": a_name, "a_price": a_price, "b_name": b_name, "b_price": b_price},
                        label, (p["id"], f"{p['id']}-{k}")))
    return out


rng = random.Random(3)
train, valid, test = (pairs(rng, [product(rng, i + 1000 * s) for i in range(300)]) for s in range(3))
system = System(cat, [Question("match", "Are these two offers the same product?", Answer.yes_no())])
head = system.fit("match", [(x, y) for x, y, _ in train], select=False)
rep = system.guarantee("match", [(x, y) for x, y, _ in valid], max_risk=0.01)
print(f"head over {len(head.features)} facts, fitted on {len(train)} pairs")
print(f"guarantee on {rep['n']} pairs: threshold {rep['threshold']:.4f}, answered alone {rep['answered']:.1%}")
print(" ", rep["promise"])

rows = []
for x, y, (a_id, b_id) in test:
    r = system.ask(x, store=False)["match"]
    rows.append({"a": a_id, "b": b_id, "p": float(r.probs["yes"]), "alone": r.status == "ok", "label": y})


def f1(pred):
    tp = sum(m and r["label"] == "yes" for m, r in zip(pred, rows))
    fp = sum(m and r["label"] == "no" for m, r in zip(pred, rows))
    fn = sum((not m) and r["label"] == "yes" for m, r in zip(pred, rows))
    return 2 * tp / (2 * tp + fp + fn)


head_pred = [r["p"] >= 0.5 for r in rows]
items = [Item((r["a"], r["b"]), {"yes": r["p"], "no": 1 - r["p"]}, keys={"a": r["a"]}) for r in rows]
out = decide_set(items, [AtMostOne("a")])
set_pred = [out[(r["a"], r["b"])].answer == "yes" for r in rows]
alone = [r for r in rows if r["alone"]]
wrong_alone = sum((r["p"] >= 0.5) != (r["label"] == "yes") for r in alone)


def two_counterparts(pred):
    hits = {}
    for m, r in zip(pred, rows):
        hits[r["a"]] = hits.get(r["a"], 0) + m
    return sum(v > 1 for v in hits.values())


print(f"test, {len(rows)} pairs: F1 head {f1(head_pred):.3f} (offers with two counterparts: {two_counterparts(head_pred)})"
      f" -> with one counterpart per offer {f1(set_pred):.3f} ({two_counterparts(set_pred)}); "
      f"{len(out.changed)} answers changed, exact: {out.exact}")
print(f"answered alone {len(alone) / len(rows):.1%}, wrong among them {wrong_alone} "
      f"({wrong_alone / len(rows):.2%} of all pairs; promise 1%)")
