"""
Jira Scout — Smart Ticket Discovery
FastAPI backend serving semantic ticket search over a sample Jira-style
ticket corpus.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .search_engine import TicketSearchEngine

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "tickets.json"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Jira Scout",
    description="LLM-style semantic search over Jira tickets — surfaces the most relevant existing tickets for an incoming problem statement.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build the search index once at startup.
engine = TicketSearchEngine.from_json_file(DATA_PATH)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class DuplicateCheckRequest(BaseModel):
    title: str
    description: str
    top_k: int = 3


class TicketResult(BaseModel):
    id: str
    title: str
    description: str
    status: str
    resolution: str
    created: str
    labels: List[str]
    score: float


@app.get("/api/tickets")
def list_tickets():
    return [t.to_dict() for t in engine.tickets]


@app.post("/api/search", response_model=List[TicketResult])
def search(req: SearchRequest):
    results = engine.search(req.query, top_k=req.top_k)
    return [{**t.to_dict(), "score": round(score, 4)} for t, score in results]


@app.post("/api/duplicate-check", response_model=List[TicketResult])
def duplicate_check(req: DuplicateCheckRequest):
    results = engine.duplicate_candidates(req.title, req.description, top_k=req.top_k)
    return [{**t.to_dict(), "score": round(score, 4)} for t, score in results]


@app.get("/api/health")
def health():
    return {"status": "ok", "tickets_indexed": len(engine.tickets)}


# Serve the simple frontend
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
