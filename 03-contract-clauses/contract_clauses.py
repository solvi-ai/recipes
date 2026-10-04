"""Contract clauses with quotes you can check (solvi 1.0.0 from PyPI).

Part 1 asks an LLM (any OpenAI-compatible server; gpt-oss-120b by default, LLM_MODEL to change it) for three kinds
of clause as `Maybe[Span[str]]`: a passage of the contract with its offsets, or "not stated". Part 2, offline, shows
what solvi does with a quote a model writes: a typographic variant is found on the normalized view and stored as the
contract's own text; a paraphrase is not found and is rejected.

    LLM_URL=https://openrouter.ai/api/v1 LLM_KEY=... python contract_clauses.py     # any OpenAI-compatible server
    python contract_clauses.py --offline                                             # part 2 only, no model
"""
import os
import sys

from solvi import Maybe, Span, Unknown
from solvi.core import find_quote
from solvi.models import llm

CONTRACT = """MASTER SERVICES AGREEMENT

1. Services. Northwind Analytics Ltd ("Provider") shall deliver the data services described in Exhibit A to
Bluefin Retail GmbH ("Client").

2. Fees. Client shall pay the fees in Exhibit B within thirty (30) days of each invoice. Late amounts bear interest
at 1% per month.

3. Term. This Agreement starts on 1 November 2026 and continues for twenty-four (24) months unless terminated for
material breach under Section 7.

4. Exclusivity. During the Term and for twelve (12) months after it, Provider shall not provide substantially similar
services to any competitor of Client operating in the European Union.

5. Confidentiality. Each party shall keep the other party's Confidential Information secret and use it only to
perform this Agreement.

6. Governing Law. This Agreement is governed by the laws of the Republic of Ireland, and the courts of Dublin have
exclusive jurisdiction over any dispute.

7. Breach. Either party may terminate this Agreement if the other party commits a material breach and fails to cure
it within thirty (30) days of written notice.
"""

KINDS = {
    "governing_law": "Which passage says which state's or country's law governs the contract?",
    "non_compete": "Which passage restricts a party from competing or serving competitors?",
    "termination_for_convenience": "Which passage lets a party terminate without cause (for convenience), "
                                   "regardless of breach?",
}

url = os.environ.get("LLM_URL")
if "--offline" in sys.argv or not url:
    print("Part 1 skipped: set LLM_URL (and LLM_KEY) to an OpenAI-compatible server to run it")
else:
    model = llm(url, os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"), api_key=os.environ.get("LLM_KEY", "none"),
                extra_body={"reasoning": {"effort": "low"}})
    print("Part 1 — an LLM proposes, solvi keeps only passages that are in the contract")
    for name, question in KINDS.items():
        part = model.decision(name, question, "contract", Maybe[Span[str]])
        d = part(contract=CONTRACT)
        v = d.value
        if v is Unknown:
            print(f"  {name:28s} not stated (confidence {float(d.conf):.2f})")
        elif v is None or d.escalate:
            print(f"  {name:28s} escalated: {d.escalate}")
        else:
            same = CONTRACT[v.start:v.end] == str(v.value)
            print(f"  {name:28s} contract[{v.start}:{v.end}] = {str(v.value)[:70]!r}… literal: {same}, "
                  f"confidence {float(d.conf):.2f}")

print()
print("Part 2 — what happens to a quote a model writes (offline)")
for written in ["governed by the laws of the Republic of Ireland",          # as in the text
                "governed by the laws of the Republic\u00a0of\u00a0Ireland",  # no-break spaces, as models type them
                "governed by Irish law"]:                                    # a paraphrase
    q = find_quote(written, {"contract": CONTRACT})
    print(f"  {written!r:55s} -> " + (f"found at [{q.start}:{q.end}], stored as {q.value!r}" if q else "not in the text: rejected"))
