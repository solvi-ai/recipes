"""Banking77 as a stream of bank-customer requests, routed under a promise: at most 5% wrong among the answers given
alone. From request 1,000 on, a share of the requests is about 20 intents the router was never shown.

Real data (Banking77, CC BY 4.0, downloaded by fetch_data.py; nothing is redistributed here), no LLM, no network at
run time. Two setups on the same stream:

  plain  `System.guarantee(max_error=0.05)`: the plain promise, which knows nothing of new intents;
  gate   `OpenSetGate`: the promise sized for a share of requests the router has no answer for, simulated on the
         calibration set by refitting the router three times without a third of its intents; the gate follows the
         share as the stream goes and raises a CUSUM flag when it jumps.

The router itself is not solvi: TF-IDF + logistic regression with nearest-neighbour signals and an "act" head (the
probability that its answer is right), wrapped as a solvi decision part. solvi adds the promise, the gate, the stored
hash-chained decisions and their replay.

    python fetch_data.py && python banking77_stream.py          # a few minutes on 4 CPU threads
"""
import csv
import hashlib
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from solvi import Catalog, System
from solvi.core.guarantees.openset import OpenSetGate, leave_out
from solvi.core.store import JSONLStorage

DATA = Path(__file__).resolve().parent / "data" / "banking77"
SPLIT_SEED = 20261002          # the same split as solvi's task stand (benchmarks/tasks/prepare.py)
LEAVE_OUT_SEED = 100           # which intents are left out together in the simulation
MAX_ERROR = 0.05
TASK = "What is the customer's request about?"


def label(intent):
    """The option as the decider names it: the intent's name in words."""
    return intent.replace("_", " ")


def split():
    """fit / calib rows of 57 known intents, and a stream of 2,000: 1,000 known-intent requests, then 1,000 drawn from
    the rest of the test set (20 of the 77 intents appear only there)."""
    if not (DATA / "train.csv").exists():
        sys.exit("no data: run `python fetch_data.py` first")
    train = list(csv.DictReader(open(DATA / "train.csv", encoding="utf-8")))
    test = list(csv.DictReader(open(DATA / "test.csv", encoding="utf-8")))
    cats = sorted({r["category"] for r in train})
    rng = random.Random(SPLIT_SEED)
    unseen = set(rng.sample(cats, 20))
    known = [c for c in cats if c not in unseen]
    row = lambda r, i, pre: {"id": f"{pre}{i}", "text": r["text"], "intent": r["category"]}          # noqa: E731
    tr = [row(r, i, "tr") for i, r in enumerate(train) if r["category"] in known]
    rng.shuffle(tr)
    cut = len(tr) * 4 // 5
    te = [row(r, i, "te") for i, r in enumerate(test)]
    rng.shuffle(te)
    before = [r for r in te if r["intent"] in known][:1000]
    used = {r["id"] for r in before}
    after = rng.sample([r for r in te if r["id"] not in used], 1000)
    stream = [{**r, "phase": "before", "known": True} for r in before] + \
             [{**r, "phase": "after", "known": r["intent"] in known} for r in after]
    return tr[:cut], tr[cut:], [{"n": i, **r} for i, r in enumerate(stream)], known


# ------------------------------------------------------------------------- the router (not solvi): classifier + act head
class Clf:
    """The baseline's classifier (the same TF-IDF and LogisticRegression settings) plus nearest-neighbour signals."""

    def __init__(self, rows, C=20):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts, y = [r["text"] for r in rows], [r["intent"] for r in rows]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
        self.X, self.CX = self.vec.fit_transform(texts), self.cvec.fit_transform(texts)
        self.lr = LogisticRegression(C=C, max_iter=2000).fit(self.X, y)
        self.classes = list(self.lr.classes_)
        self.y = np.array([self.classes.index(v) for v in y])
        self.unigrams = TfidfVectorizer(ngram_range=(1, 1)).fit(texts)

    def signals(self, texts, k=10):
        """→ (log-probabilities [n, K], features [n, 13]): the classifier's margins and entropy, how close the nearest
        training texts are (word and character n-grams) and whether they share the answer, length, known words."""
        X, CX = self.vec.transform(texts), self.cvec.transform(texts)
        raw = self.lr.decision_function(X)
        z = raw - raw.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        p = np.exp(logp)
        top = np.argsort(-p, 1)[:, :2]
        p1, p2 = p[np.arange(len(p)), top[:, 0]], p[np.arange(len(p)), top[:, 1]]
        rs = np.sort(raw, 1)
        ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
        feats = [np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1)), p1 - p2, ent, rs[:, -1], rs[:, -1] - rs[:, -2]]
        for M, Q in ((self.X, X), (self.CX, CX)):
            S = (Q @ M.T).toarray()
            idx = np.argsort(-S, 1)[:, :k]
            pred = top[:, 0]
            feats += [S[np.arange(len(S)), idx[:, 0]], np.where(self.y[None, :] == pred[:, None], S, -1).max(1),
                      (self.y[idx] == pred[:, None]).mean(1)]
        an, voc = self.unigrams.build_analyzer(), self.unigrams.vocabulary_
        toks = [an(t) for t in texts]
        feats += [np.log1p([len(t) for t in toks]), np.array([np.mean([w in voc for w in t]) if t else 0.0 for t in toks])]
        return logp, np.stack(feats, 1)


def act_head(rows, intents, folds=3, seed=0):
    """A logistic regression P(the answer is right) from the signals. Its training rows come from `rows` alone: `folds`
    times a classifier is fitted without one group of intents and one part of the examples; the held-out examples of
    its own intents give right / wrong, the left-out intents give "wrong" (the classifier cannot name them)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    its = sorted(intents)
    random.Random(seed).shuffle(its)
    groups = [set(its[i::folds]) for i in range(folds)]
    rng = random.Random(seed + 1)
    part = {id(r): rng.randrange(folds) for r in rows}
    F, ok = [], []
    for j, g in enumerate(groups):
        c = Clf([r for r in rows if r["intent"] not in g and part[id(r)] != j])
        for rs in ([r for r in rows if r["intent"] not in g and part[id(r)] == j], [r for r in rows if r["intent"] in g]):
            logp, f = c.signals([r["text"] for r in rs])
            F.append(f)
            ok += [int(c.classes[i] == r["intent"]) for i, r in zip(logp.argmax(1), rs)]
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.concatenate(F), np.array(ok))


class Scorer:
    """The classifier and its act head behind solvi's scorer protocol: per item the log-probabilities of its options
    and the act logit."""

    def __init__(self, clf, head):
        self.clf, self.head = clf, head
        self.index = {label(c): i for i, c in enumerate(clf.classes)}
        self.model_id = "banking77/tfidf-lr+act"
        self._fp = hashlib.sha256(clf.lr.coef_.tobytes()).hexdigest()[:16]

    def fingerprint(self):
        return self._fp

    def logits(self, items):
        logp, F = self.clf.signals([it.text for it in items])
        p = np.clip(self.head.predict_proba(F)[:, 1], 1e-6, 1 - 1e-6)
        act = np.log(p / (1 - p))
        return [{"logits": logp[i, [self.index[o] for o in it.options]], "act": float(act[i])} for i, it in enumerate(items)]


def decider(rows, intents):
    """A solvi decision part over `intents`, fitted on `rows`: it never escalates by itself (min_act=0), so whatever
    gates it — a guarantee, an open-set gate — decides alone."""
    from solvi.core.deciders import DecideModel
    rows = [r for r in rows if r["intent"] in set(intents)]
    model = DecideModel(Scorer(Clf(rows), act_head(rows, intents)), meta={"format": "stand-in", "temperature": 1.0, "act": {}},
                        cache_size=10 ** 6)
    return model.decision("intent", TASK, "text", [label(i) for i in intents], min_act=0.0)


# --------------------------------------------------------------------------------------------- solvi: promise and gate
def run(fit, calib, stream, known, plain):
    part = decider(fit, known)
    examples = [(r["text"], label(r["intent"])) for r in calib]
    store = Path(tempfile.mkdtemp()) / "decisions.jsonl"
    cat = Catalog()
    system = System(cat, [part.question(cat)], storage=JSONLStorage(store))
    gate = None
    if plain:
        system.guarantee("intent", [({"text": x}, y) for x, y in examples], max_error=MAX_ERROR, signal="act")
    else:
        ds = part.decide([x for x, _ in examples])                  # the deployed router's signals on calib
        signals = [d.extra["act"] for d in ds]
        right = [d.value == y for d, (_, y) in zip(ds, examples)]
        sim = leave_out(examples, lambda kept: decider(fit, [i for i in known if label(i) in set(kept)]),
                        seed=LEAVE_OUT_SEED)
        gate = OpenSetGate.calibrate(signals, right, sim["novel"], max_error=MAX_ERROR)
        system.guarantee("intent", promise=gate, signal="act")

    back = {label(i): i for i in known}
    rows, flag_n = [], None
    for x in stream:                                                # the stream is fed as text; labels are not read
        r = system.ask({"text": x["text"]})["intent"]
        rows.append({**x, "said": back.get(r.answer) if r.status == "ok" else None})
        if gate is not None and gate.flag_at is not None and flag_n is None:
            flag_n = x["n"]
    ids = [rec.id for rec in system.storage.query()]
    random.Random(0).shuffle(ids)
    failed = sum(not system.storage.get(i).trace.replay(system)["ok"] for i in ids[:100])
    return rows, flag_n, gate, system.storage.verify()["ok"], failed


def tally(rows):
    alone = [r for r in rows if r["said"]]
    wrong = sum(r["said"] != r["intent"] for r in alone)
    return f"answered alone {len(alone) / len(rows):6.1%}, wrong among them {wrong / max(len(alone), 1):5.1%} ({wrong} of {len(alone)})"


if __name__ == "__main__":
    fit, calib, stream, known = split()
    digest = hashlib.sha256("".join(r["id"] for r in stream).encode()).hexdigest()[:12]
    print(f"Banking77: {len(fit)} requests to fit, {len(calib)} to calibrate, {len(known)} known intents; "
          f"a stream of {len(stream)} (ids {digest}), shift at request 1000")
    print(f"promise: at most {MAX_ERROR:.0%} wrong among the answers given alone")
    for name, plain in (("plain promise", True), ("open-set gate", False)):
        rows, flag_n, gate, chain_ok, failed = run(fit, calib, stream, known, plain)
        before = [r for r in rows if r["phase"] == "before"]
        after = [r for r in rows if r["phase"] == "after"]
        print(f"\n{name}")
        print(f"  before the shift: {tally(before)}")
        print(f"  after  the shift: {tally(after)}")
        print(f"    of which known intents:  {tally([r for r in after if r['known']])}")
        print(f"    of which unseen intents: {tally([r for r in after if not r['known']])}")
        if gate is not None:
            where = "never" if flag_n is None else ("before the shift (false alarm)" if flag_n < 1000
                                                    else f"{flag_n - 1000} requests after the shift")
            print(f"  flag: {where}")
        print(f"  stored decisions: chain verifies {chain_ok}; 100 replayed, {failed} failed")
