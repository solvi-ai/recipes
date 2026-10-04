"""Three real contracts from CUAD's held-out set, three kinds of clause each: an LLM proposes a passage or "not stated",
solvi keeps a passage only when it is literally in the contract, and the answer is compared with the lawyers'
annotation.

Long contracts: `long="retrieve"` splits a contract into sections, BM25 picks the few that bear on the question (by the
words in `retrieve_query`: the clause's own vocabulary), and the model reads about 3,000 tokens of them, not the whole
contract; a quote still points into the whole document.

Data: CUAD v1 (CC BY 4.0), downloaded by fetch_data.py. Model: any OpenAI-compatible server (9 requests of about
3,000 tokens; with openai/gpt-oss-120b on OpenRouter well under a cent).

    python fetch_data.py
    LLM_URL=https://openrouter.ai/api/v1 LLM_KEY=... python cuad_contracts.py
"""
import json
import os
import re
import sys
from pathlib import Path

from solvi import Maybe, Span, Unknown
from solvi.models import llm

DATA = Path(__file__).resolve().parent / "data" / "cuad" / "test.json"
CONTRACTS = ("CENTRACKINTERNATIONALINC", "HERTZGLOBALHOLDINGS", "CYBERIANOUTPOSTINC")   # title prefixes
QUERY = {   # the words such a clause is usually written with (from solvi's task stand, benchmarks/tasks/cuad/queries.py)
    "Governing Law": "governed by construed in accordance with laws of the state governing law jurisdiction without "
                     "regard conflict",
    "Non-Compete": "compete competing competitive competition shall not directly or indirectly engage business "
                   "territory restrict",
    "Termination For Convenience": "terminate at any time without cause for convenience upon days prior written notice "
                                   "for any reason or no reason",
}


def words(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", t.lower()))


def overlaps(quote, gold):
    """The stand's rule: the quote and a gold passage share at least half their words, or one contains the other."""
    a, b = words(quote), words(gold)
    return bool(a and b) and (len(a & b) / len(a | b) >= 0.5 or norm(quote) in norm(gold) or norm(gold) in norm(quote))


def slug(category):
    return re.sub(r"\W+", "_", category).strip("_").lower()


if __name__ == "__main__":
    if not DATA.exists():
        sys.exit("no data: run `python fetch_data.py` first")
    url = os.environ.get("LLM_URL")
    if not url:
        sys.exit("set LLM_URL (and LLM_KEY) to an OpenAI-compatible server")
    model = llm(url, os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"), api_key=os.environ.get("LLM_KEY", "none"),
                max_tokens=2000, max_len=3000, timeout=240, retries=2, extra_body={"reasoning": {"effort": "low"}})
    data = json.loads(DATA.read_text(encoding="utf-8"))["data"]
    tally = {"right": 0, "questions": 0, "quotes": 0, "literal": 0}
    for prefix in CONTRACTS:
        doc = next(c for c in data if c["title"].startswith(prefix))
        text, qas = doc["paragraphs"][0]["context"], doc["paragraphs"][0]["qas"]
        print(f"\n{doc['title'][:80]} ({len(text):,} characters)")
        for category, query in QUERY.items():
            qa = next(q for q in qas if q["id"].endswith("__" + category))
            details = " ".join(qa["question"].split("Details:")[-1].split())
            part = model.decision(slug(category), f'Which passage of the contract is the "{category}" clause a lawyer '
                                  f"should review? ({details})", "contract", Maybe[Span[str]],
                                  long="retrieve", retrieve_query=query)
            d = part(contract=text)
            v, gold = d.value, [a["text"] for a in qa["answers"]]
            stated = "present" if gold else "absent"
            if v is Unknown:
                said, ok = f"not stated (confidence {float(d.conf):.2f})", not gold
            elif v is None or d.escalate:
                said, ok = f"escalated to a person: {str(d.escalate)[:80]}", None
            else:
                literal = text[v.start:v.end] == str(v.value)
                tally["quotes"] += 1
                tally["literal"] += literal
                ok = any(overlaps(str(v.value), g) for g in gold)
                said = (f"contract[{v.start}:{v.end}] = {' '.join(str(v.value).split())[:60]!r}… literal: {literal}, "
                        f"confidence {float(d.conf):.2f}")
            tally["questions"] += 1
            tally["right"] += bool(ok)
            verdict = "a person decides" if ok is None else ("right" if ok else "WRONG")
            print(f"  {category:28s} lawyers: {stated:7s} | {said} -> {verdict}")
    print(f"\n{tally['right']} of {tally['questions']} right against the annotation; "
          f"{tally['literal']} of {tally['quotes']} quotes literally in the contract")
