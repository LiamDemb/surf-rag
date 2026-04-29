from __future__ import annotations

import os
from typing import Any
import requests

_DEFAULT_TIMEOUT_S = 15
_USER_AGENT = "surf-rag/2wiki-dataset-download"


def nq_document_title(row: dict[str, Any]) -> str:
    """Compatibility wrapper for NQ title extraction."""
    try:
        from surf_rag.benchmark import nq_document_title as _nq_document_title

        return _nq_document_title(row)
    except Exception:
        document = row.get("document")
        if isinstance(document, dict):
            t = document.get("title")
            if t is not None and str(t).strip():
                return str(t).strip()
        for key in ("document_title", "title"):
            v = row.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""


def _load_app_env() -> None:
    try:
        from surf_rag.config.env import load_app_env as _load

        _load()
        return
    except Exception:
        pass
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        pass


def twowiki_supporting_titles(row: dict[str, Any]) -> list[str]:
    """Extract unique supporting-fact titles from a 2Wiki row."""
    supporting = row.get("supporting_facts")
    if isinstance(supporting, dict):
        titles = supporting.get("title")
        if isinstance(titles, list):
            return list(dict.fromkeys(str(t).strip() for t in titles if str(t).strip()))

    if isinstance(supporting, list):
        out: list[str] = []
        for item in supporting:
            if isinstance(item, (list, tuple)) and item:
                title = str(item[0]).strip()
            elif isinstance(item, dict):
                title = str(item.get("title", "")).strip()
            else:
                title = ""
            if title:
                out.append(title)
        return list(dict.fromkeys(out))
    return []


def _wiki_api_url(language: str) -> str:
    lang = (language or "en").strip().lower() or "en"
    return f"https://{lang}.wikipedia.org/w/api.php"


def titles_direct_mainspace_status(
    titles: list[str], *, language: str = "en"
) -> tuple[bool, str]:
    """Return `(ok, reason)` for title validation against MediaWiki `prop=info`."""
    uniq = list(dict.fromkeys(str(t).strip() for t in titles if str(t).strip()))
    if not uniq:
        return False, "empty_titles"
    _load_app_env()

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "info",
        "titles": "|".join(uniq),
    }
    headers = {"User-Agent": _USER_AGENT}
    wiki_token = (os.environ.get("WIKIMEDIA_OAUTH2_ACCESS_TOKEN") or "").strip()
    if wiki_token:
        headers["Authorization"] = f"Bearer {wiki_token}"
    resp = requests.get(
        _wiki_api_url(language),
        params=params,
        headers=headers,
        timeout=_DEFAULT_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages")
    if not isinstance(pages, list):
        return False, "bad_api_response"
    if len(pages) < len(uniq):
        return False, "incomplete_response"

    for page in pages:
        title = str(page.get("title", "")).strip()
        if page.get("invalid"):
            return False, f"invalid:{title}"
        if page.get("missing"):
            return False, f"missing:{title}"
        if page.get("ns") != 0:
            return False, f"non_main:{title}"
        if page.get("redirect"):
            return False, f"redirect:{title}"
    return True, "ok"


def titles_all_exist_head(titles: list[str], *, language: str = "en") -> bool:
    """
    Return True if all titles are direct existing mainspace pages.

    The function keeps the legacy name used by dataset scripts, but it validates via
    MediaWiki API `prop=info` instead of HTTP HEAD to avoid redirect/locale ambiguity.
    """
    ok, _ = titles_direct_mainspace_status(titles, language=language)
    return ok


def wikipedia_find_page_head(title: str, *, language: str = "en") -> bool:
    """Compatibility helper for scripts that validate one title at a time."""
    t = str(title or "").strip()
    if not t:
        return False
    return titles_all_exist_head([t], language=language)
