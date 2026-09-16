"""
CareBot -- a RAG-based patient-support chatbot.

Answers questions about clinic policy (scheduling, billing, prescription
refills) by retrieving the most relevant passages from data/index.json
and grounding the model's answer in them. Explicitly refuses to give
medical advice, diagnosis, or dosing guidance -- that boundary is
enforced in the system prompt below, not left to the model's judgment.

Usage:
    python src/chatbot.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from retrieval import retrieve

CHAT_MODEL = "gpt-4o-mini"
MIN_RELEVANCE_SCORE = 0.25  # below this, treat retrieval as "no good match"

SYSTEM_PROMPT = """You are CareBot, a patient-support assistant for a clinic.

Rules:
- Answer ONLY using the "Context" passages provided below. Do not use outside knowledge.
- If the context does not contain the answer, say so plainly and suggest the patient
  contact the front desk, rather than guessing.
- Never provide medical advice, diagnosis, treatment recommendations, or guidance on
  medication dosing, even if asked directly. Redirect those questions to a licensed
  provider.
- Keep answers short, plain-language, and friendly. Avoid clinical jargon.
"""


def build_prompt(query: str, passages: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {p['source']}]\n{p['text']}" for p in passages
    )
    return f"Context:\n{context}\n\nPatient question: {query}"


def answer(query: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")
    client = OpenAI(api_key=api_key)

    passages = retrieve(query, top_k=3)
    if not passages or passages[0]["score"] < MIN_RELEVANCE_SCORE:
        return (
            "I don't have information on that in the clinic knowledge base. "
            "Please contact the front desk directly for help with this."
        )

    prompt = build_prompt(query, passages)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    reply = response.choices[0].message.content
    sources = ", ".join(sorted({p["source"] for p in passages}))
    return f"{reply}\n\n(Sources: {sources})"


def main() -> None:
    print("CareBot -- patient support assistant. Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        print(f"\nCareBot: {answer(query)}\n")


if __name__ == "__main__":
    main()
