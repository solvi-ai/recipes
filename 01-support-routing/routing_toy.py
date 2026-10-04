"""Route support tickets with a promise on the errors, a slow path and a person (solvi 1.0.0 from PyPI).

Synthetic tickets (no data download): four teams, keyword facts computed by the catalog from the body, a head fitted
by solvi.build, a stand-in for the slow path (an LLM in practice) that also
reads the subject line. Then a stream where, from ticket 600 on, a third of
the tickets are a kind nobody labelled (fraud reports)."""
import random
import tempfile
from pathlib import Path

import solvi
from solvi import Answer, Catalog, Question

TEAMS = ["billing", "shipping", "technical", "account"]
WORDS = {"billing": ["charged", "invoice", "refund", "payment", "price"],
         "shipping": ["parcel", "delivery", "courier", "tracking", "arrived"],
         "technical": ["error", "crash", "login", "app", "bug"],
         "account": ["password", "email", "profile", "delete", "username"],
         "fraud": ["stolen", "unauthorized", "scam", "hacked", "suspicious"]}
FILLER = ["please", "help", "today", "order", "again", "thanks", "urgent", "still", "my", "the"]

cat = Catalog()


def words_of(team):                                  # one computed fact per team: how many of its words the text has
    def hits(text):
        low = text.lower()
        return sum(w in low for w in WORDS[team])
    hits.__name__ = f"{team}_words"
    return hits


for team in TEAMS:
    cat.fn(words_of(team))


def ticket(rng, kind):
    own = rng.sample(WORDS[kind], rng.choice([1, 1, 2]))
    noise = rng.sample(WORDS[rng.choice(TEAMS)], 1) if rng.random() < 0.35 else []
    words = own + noise + rng.sample(FILLER, 4)
    rng.shuffle(words)
    hint = kind if rng.random() < 0.97 else rng.choice(TEAMS)          # the subject line, which System 1 does not read
    return {"text": " ".join(words), "subject": f"Re: {rng.choice(WORDS[hint])} (ticket {rng.randint(10000, 99999)})"}


def reader(state):                                   # the slow path: a stand-in that also reads the subject line
    subject = state["subject"].lower()
    for t in TEAMS:
        if any(w in subject for w in WORDS[t]):
            return t
    low = state["text"].lower()                      # a subject no team's words match: the body decides
    return max(TEAMS, key=lambda t: sum(w in low for w in WORDS[t]))


rng = random.Random(1)
examples = []
for _ in range(2000):
    kind = rng.choice(TEAMS)
    examples.append((ticket(rng, kind), kind))      # the team a person routed it to



def run(novel):
    store = Path(tempfile.mkdtemp()) / "decisions.jsonl"
    s = solvi.build(Question("team", "Which team handles this ticket?", Answer.choice(TEAMS)), examples,
                    catalog=cat, max_error=0.05, slow=reader, storage=store, novel=novel)
    stream = random.Random(2)                      # the same stream for both setups
    tally = {"before": [], "after": []}
    for n in range(1200):
        kind = stream.choice(TEAMS) if n < 600 or stream.random() > 1 / 3 else "fraud"
        res = s.ask(ticket(stream, kind))
        tally["before" if n < 600 else "after"].append((res.by, res.answer == kind))
    return s, tally


for novel in (False, "auto"):
    s, tally = run(novel)
    if novel is False:
        print(s.explain())
        print()
    print(f"novel={novel!r}")
    for part, rows in tally.items():
        alone = [ok for by, ok in rows if by != "human"]
        by = {k: sum(b == k for b, _ in rows) for k in ("s1", "s2", "human")}
        print(f"  {part:6s} the shift: by {by}, wrong among answered alone "
              f"{(1 - sum(alone) / len(alone)) if alone else 0:.1%} of {len(alone)}")
    print("  replay failures:", len(s.replay_all()))
    if s.gate is not None:
        print("  open-set gate flag:", s.gate.flag_at, "|", (s.gate.why or "")[:110])
