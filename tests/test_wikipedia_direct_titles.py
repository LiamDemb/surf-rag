"""Tests for Wikipedia direct title existence checks (no redirect resolution)."""

from __future__ import annotations

import pytest

from surf_rag.core.wikipedia_client import WikipediaClient


def test_titles_are_direct_mainspace_pages_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    wiki = WikipediaClient()

    def fake_get(params: dict) -> dict:
        assert params.get("prop") == "info"
        return {
            "query": {
                "pages": [
                    {"title": "Foo", "ns": 0, "pageid": 1},
                    {"title": "Bar", "ns": 0, "pageid": 2},
                ]
            }
        }

    monkeypatch.setattr(wiki, "_get", fake_get)
    ok, msg = wiki.titles_are_direct_mainspace_pages(["Foo", "Bar"])
    assert ok is True
    assert msg == "ok"


def test_titles_are_direct_mainspace_pages_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wiki = WikipediaClient()

    def fake_get(params: dict) -> dict:
        return {
            "query": {
                "pages": [
                    {"title": "Nope", "ns": 0, "missing": True},
                ]
            }
        }

    monkeypatch.setattr(wiki, "_get", fake_get)
    ok, msg = wiki.titles_are_direct_mainspace_pages(["Nope"])
    assert ok is False
    assert msg.startswith("missing:")


def test_titles_are_direct_mainspace_pages_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wiki = WikipediaClient()

    def fake_get(params: dict) -> dict:
        return {
            "query": {
                "pages": [
                    {"title": "Rd", "ns": 0, "pageid": 3, "redirect": True},
                ]
            }
        }

    monkeypatch.setattr(wiki, "_get", fake_get)
    ok, msg = wiki.titles_are_direct_mainspace_pages(["Rd"])
    assert ok is False
    assert msg.startswith("redirect:")
