import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.search_engine import TicketSearchEngine  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.json"


def test_engine_loads_all_tickets():
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    assert len(engine.tickets) > 0


def test_search_finds_relevant_duplicate():
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    results = engine.search("users can't log in, server error 500", top_k=3)
    assert len(results) > 0
    top_ticket, top_score = results[0]
    # The known near-duplicate pair in the sample data is PROJ-101 / PROJ-114
    assert top_ticket.id in {"PROJ-101", "PROJ-114"}
    assert top_score > 0.1


def test_search_respects_top_k():
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    results = engine.search("dashboard widgets not loading", top_k=2)
    assert len(results) <= 2


def test_duplicate_candidates_helper():
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    results = engine.duplicate_candidates(
        "Export missing rows",
        "Exporting a large report drops rows after a certain point with no error.",
    )
    assert len(results) > 0
    assert results[0][0].id in {"PROJ-067", "PROJ-142"}


def test_login_error_and_login_issue_return_consistent_top_result():
    """Regression test: 'login error' and 'login issue' should surface the
    same top duplicate, since 'issue' and 'error' are domain synonyms here
    — this was the original bug report (inconsistent results depending on
    which synonym the user happened to type)."""
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    error_results = engine.search("login error", top_k=3)
    issue_results = engine.search("login issue", top_k=3)
    assert error_results[0][0].id == issue_results[0][0].id == "PROJ-114"


def test_typo_signin_still_finds_the_login_ticket():
    """Regression test: a typo'd/no-space variant of 'sign in' should still
    surface the real login ticket, not an unrelated ticket that happens to
    share a similarly-spelled but unrelated word."""
    engine = TicketSearchEngine.from_json_file(DATA_PATH)
    results = engine.search("sigin issue", top_k=3)
    assert len(results) > 0
    assert results[0][0].id == "PROJ-114"
    # The unrelated dashboard-widgets ticket (which merely contains the
    # word "signing" in an unrelated sentence) must not outrank it.
    result_ids = [t.id for t, _ in results]
    if "PROJ-133" in result_ids:
        assert result_ids.index("PROJ-133") > result_ids.index("PROJ-114")
