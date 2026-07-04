# Jira Scout — Smart Ticket Discovery

A semantic search engine over Jira-style tickets. Given an incoming problem
statement, it surfaces the most relevant existing tickets — the same
duplicate-detection / "smart ticket discovery" problem described in the
`AI Engineering Initiatives` section of my resume.

> "Designed an LLM-powered semantic search engine over Jira that analyses
> incoming problem statements and surfaces the most relevant existing tickets
> using embedding models and vector similarity."

This repo is a standalone, runnable that demonstrates the approach end
to end on a sample dataset — no Jira instance or API keys required.

## Why this matters

Duplicate ticket creation is a real cost: engineers re-investigate issues
that were already root-caused, and historical fixes get buried under
near-identical new tickets. Keyword search (Jira's default) misses
duplicates that are worded differently ("login throws 500" vs. "users
can't sign in, server error"). Semantic search catches these because it
compares *meaning*, not exact words.

## Architecture

```
                ┌─────────────────────┐
   New ticket   │                     │
   title +   ─▶ │   FastAPI backend   │
   description  │   (app/main.py)     │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │  TicketSearchEngine │   builds a vector index over
                │ (search_engine.py)  │   the ticket corpus at startup
                └─────────┬───────────┘
                          │ delegates vectorization to
                          ▼
                ┌─────────────────────┐
                │  EmbeddingBackend    │  pluggable:
                │  (embeddings.py)     │  • TF-IDF (default, offline)
                │                      │  • sentence-transformers (optional)
                └──────────────────────┘
                          │
                          ▼
                cosine similarity ranking
                          │
                          ▼
                Top-K most similar tickets
                returned to the UI
```

**Design decision worth calling out in an interview:** the embedding layer
is behind an abstract `EmbeddingBackend` interface. This ships with a
TF-IDF backend so it runs instantly with no downloads or API keys, but
swapping in a transformer-based sentence embedding model (or an LLM
embeddings API) is a one-line config change (`EMBEDDING_BACKEND=sentence-transformer`)
with zero changes to the search engine or API layer. This is the same
pattern you'd use in production to start cheap and upgrade quality later
without a rewrite.

## Known limitation of the default backend (and how it's mitigated)

TF-IDF matches on literal word overlap, not meaning — it is **not** true
semantic search, despite the name of this project. Two failure modes this
surfaces in practice, and what's done about each:

1. **Vocabulary mismatch.** If a query uses a word that never appears in
   the ticket corpus (e.g. searching "login **issue**" when every ticket
   says "**error**"), that word contributes nothing to the score, and you
   get a different, weaker result than a synonymous query would. Fixed
   here with a small domain synonym map (`app/embeddings.py`) applied to
   both the indexed corpus and incoming queries, so "issue" / "problem" /
   "bug" / "error" and "login" / "signin" / "sign in" all normalize to the
   same token before vectorizing.
2. **Typos and unseen word forms.** "sigin" (missing a letter) shares no
   tokens with "sign", so it scores zero by default. Fixed with a fuzzy
   correction pass — but restricted to a small, curated list of canonical
   domain terms rather than the full corpus vocabulary. Matching against
   the *entire* vocabulary is unsafe: a typo can end up spelling-distance
   closer to an unrelated incidental word (e.g. "signing", from some other
   ticket that happens to mention "signing up") than to the word you
   actually meant. Spelling similarity isn't the same as topical relevance.

Neither patch makes TF-IDF a real semantic model — they close the two most
visible gaps on a small, informal ticket corpus. For genuine semantic
understanding (synonyms the map doesn't anticipate, larger typos, true
meaning-level similarity), switch to the `SentenceTransformerBackend` — see
"Upgrading the embedding backend" below. That's the actual fix; the
synonym/typo patches are a pragmatic stand-in that needs no model download.

## Project structure

```
jira-scout/
├── app/
│   ├── main.py          # FastAPI app + REST endpoints
│   ├── search_engine.py # index building + cosine similarity search
│   ├── embeddings.py     # pluggable embedding backends
│   └── models.py         # Ticket data model
├── data/
│   └── tickets.json      # sample ticket corpus (15 tickets, incl. 6 near-duplicate pairs)
├── static/
│   └── index.html         # single-page search UI
├── tests/
│   └── test_search.py
├── requirements.txt
└── requirements-optional.txt
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Then open **http://localhost:8000** — you'll see a search box. Try:

- "users can't sign in, server error"
- "export is cutting off large reports"
- "app crashes when tapping a notification on android"

Each of these has a genuine near-duplicate already in the sample dataset,
so you should see a high-scoring match with its original resolution
surfaced — exactly the "don't re-investigate what's already fixed" workflow
this is meant to demonstrate.

## Running the tests

```bash
pip install pytest
pytest tests/
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/search` | POST | `{"query": "...", "top_k": 5}` → ranked similar tickets |
| `/api/duplicate-check` | POST | `{"title": "...", "description": "...", "top_k": 3}` → duplicate candidates for a new ticket |
| `/api/tickets` | GET | List the full sample corpus |
| `/api/health` | GET | Health check + index size |

## Upgrading to real Jira data

Swap `data/tickets.json` for a Jira export (via the Jira REST API
`/rest/api/3/search` endpoint) mapped to the `Ticket` model in
`app/models.py`. No other code changes are required.

## Upgrading the embedding backend

```bash
pip install -r requirements.txt -r requirements-optional.txt
EMBEDDING_BACKEND=sentence-transformer uvicorn app.main:app --reload
```

## License

MIT — see [LICENSE](LICENSE).
