# CareBot — RAG-Based Patient Support Chatbot

CareBot answers patient questions about clinic policy — scheduling, billing,
and prescription refills — by retrieving the most relevant passages from a
knowledge base and generating a grounded answer from them, instead of relying
on the model's general knowledge. It explicitly refuses to give medical
advice, diagnosis, or dosing guidance, and redirects those questions to a
licensed provider.

This repository is a rebuilt, open-source version of a CareBot concept
originally prototyped as a custom GPT. The GPT builder doesn't expose source
code, so this version reimplements the same retrieval-augmented-generation
architecture as a standalone Python project: a real ingestion pipeline, a
real retrieval layer, and a real generation step, all inspectable and
runnable.

**Knowledge base note:** the documents in `data/` are sample clinic-policy
text I wrote for this demo (scheduling, billing, prescriptions) — not real
patient or clinical data.

All project code lives in the [`carebot/`](carebot/) folder.

## Architecture

```
data/*.txt  --ingest.py-->  data/index.json (chunks + embeddings)
                                    |
user query --embed + cosine similarity--> top-k relevant chunks
                                    |
                         chatbot.py builds a grounded prompt
                                    |
                              OpenAI chat model --> answer + cited sources
```

- **`src/ingest.py`** — loads the `.txt` knowledge-base files, splits them
  into overlapping ~120-word chunks (so retrieval returns focused passages,
  not whole documents), embeds each chunk with OpenAI's embedding model, and
  writes the result to `data/index.json`.
- **`src/retrieval.py`** — embeds the incoming question and ranks every
  indexed chunk by cosine similarity, returning the top-k matches with their
  similarity scores.
- **`src/chatbot.py`** — the CLI entry point. Builds a prompt from the
  retrieved passages, enforces a system prompt that restricts the model to
  the provided context and blocks medical-advice questions, and prints the
  answer along with which source file(s) it came from. If nothing in the
  knowledge base scores above a relevance threshold, it says so instead of
  guessing.

## Setup

```bash
git clone https://github.com/redietg0105/CareBot-.git
cd CareBot-/carebot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Usage

```bash
python src/ingest.py     # builds data/index.json from the knowledge base
python src/chatbot.py    # starts the interactive chat
```

Example session:

```
You: Can I get a same-day appointment?
CareBot: Same-day appointments depend on provider availability and aren't
guaranteed, so it's best to call ahead and ask what's open. (Sources:
kb_appointments.txt)

You: What dose of ibuprofen should I take?
CareBot: I can't give medical advice on medication dosing — that question
needs a licensed provider. I can help with scheduling, billing, or refill
questions instead.
```

## Design decisions worth noting

- **Chunking with overlap** rather than embedding whole documents, so a
  question that lands in the middle of a policy section still retrieves
  cleanly.
- **A relevance-score floor** before answering, so the bot says "I don't
  know" instead of generating a plausible-sounding but ungrounded answer
  when the knowledge base has nothing relevant.
- **The refusal rule lives in the system prompt, not a keyword filter** —
  medical-advice questions are blocked by instructing the model on its role
  and boundaries, which generalizes better than pattern-matching on words
  like "dose" or "symptom."

## Possible extensions

- Swap the in-memory cosine-similarity index for a real vector database
  (e.g. Chroma, pgvector) once the knowledge base grows past a few dozen
  documents.
- Add conversation memory so follow-up questions ("what about telehealth
  ones?") resolve against the prior turn.
- Add a small evaluation set of Q&A pairs to track retrieval accuracy as the
  knowledge base changes.

## Tech stack

Python, OpenAI API (`text-embedding-3-small`, `gpt-4o-mini`), NumPy, python-dotenv

---

**Author:** Rediet Gelaw · [LinkedIn](https://www.linkedin.com/in/redietgelaw) · [GitHub](https://github.com/redietg0105)
