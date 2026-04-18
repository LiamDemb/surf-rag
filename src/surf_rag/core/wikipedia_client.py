from __future__ import annotations

import email.utils
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RAQR/0.1 (research corpus build)"

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 8
_DEFAULT_MAX_BACKOFF_S = 90.0


@dataclass(frozen=True)
class WikiPage:
    title: str
    page_id: Optional[str]
    revision_id: Optional[str]
    url: Optional[str]
    html: Optional[str]
    outgoing_titles: List[str]


class WikipediaClient:
    def __init__(
        self,
        throttle_s: float = 0.1,
        *,
        oauth2_access_token: Optional[str] = None,
        max_retries: Optional[int] = None,
        max_backoff_s: Optional[float] = None,
    ) -> None:
        self.throttle_s = throttle_s
        self._last_call = 0.0
        self._max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(os.environ.get("WIKIPEDIA_MAX_RETRIES", str(_DEFAULT_MAX_RETRIES)))
        )
        self._max_backoff_s = (
            float(max_backoff_s)
            if max_backoff_s is not None
            else float(
                os.environ.get("WIKIPEDIA_MAX_BACKOFF_S", str(_DEFAULT_MAX_BACKOFF_S))
            )
        )

        token = oauth2_access_token
        if token is None:
            token = os.environ.get("WIKIMEDIA_OAUTH2_ACCESS_TOKEN", "").strip() or None
        elif not str(token).strip():
            token = None

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip",
            }
        )
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    @property
    def oauth2_authenticated(self) -> bool:
        return "Authorization" in self._session.headers

    def _sleep_if_needed(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.throttle_s:
            time.sleep(self.throttle_s - delta)
        self._last_call = time.time()

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> float | None:
        raw = resp.headers.get("Retry-After")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            dt = email.utils.parsedate_to_datetime(raw)
            if dt is not None:
                return max(0.0, dt.timestamp() - time.time())
        except (TypeError, OSError, ValueError):
            pass
        return None

    def _cap_backoff(self, s: float) -> float:
        return min(max(0.0, s), self._max_backoff_s)

    def _get(self, params: Dict[str, str]) -> dict:
        self._sleep_if_needed()
        last_resp: requests.Response | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(WIKI_API, params=params, timeout=30)
            except requests.RequestException:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(
                    self._cap_backoff(2.0**attempt + random.random()),
                )
                continue

            last_resp = resp
            if resp.status_code == 429:
                ra = self._retry_after_seconds(resp)
                delay = (
                    self._cap_backoff(ra)
                    if ra is not None and ra > 0
                    else self._cap_backoff(2.0**attempt + random.random())
                )
                logger.warning(
                    "Wikipedia API 429; retry in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)
                continue

            resp.raise_for_status()
            return resp.json()

        if last_resp is not None:
            last_resp.raise_for_status()
        raise RuntimeError("Wikipedia API request failed after retries")

    def resolve_title(self, title: str) -> Optional[str]:
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "redirects": "1",
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" not in page:
                return page.get("title")
        return None

    def fetch_html(self, title: str) -> WikiPage:
        data = self._get(
            {
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text|links|revid",
                "redirects": "1",
                "formatversion": "2",
            }
        )
        parse = data.get("parse", {})
        html = parse.get("text")
        revid = parse.get("revid")
        page_id = parse.get("pageid")
        resolved_title = parse.get("title", title)
        outgoing = [
            link.get("title")
            for link in parse.get("links", [])
            if link.get("ns") == 0 and link.get("title")
        ]
        url = f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}"
        return WikiPage(
            title=resolved_title,
            page_id=str(page_id) if page_id is not None else None,
            revision_id=str(revid) if revid is not None else None,
            url=url,
            html=html,
            outgoing_titles=outgoing,
        )

    @staticmethod
    def parse_redirects_response(data: dict) -> Dict[str, List[str]]:
        redirect_map: Dict[str, List[str]] = {}
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title")
            if not title:
                continue
            redirects = [r.get("title") for r in page.get("redirects", []) if r.get("title")]
            if redirects:
                redirect_map[title] = redirects
        return redirect_map

    def fetch_redirects(self, titles: List[str]) -> Dict[str, List[str]]:
        if not titles:
            return {}
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "prop": "redirects",
                "rdnamespace": "0",
                "rdlimit": "max",
                "redirects": "1",
                "titles": "|".join(titles),
            }
        )
        return self.parse_redirects_response(data)

    def search_titles(self, query: str, limit: int = 5) -> List[str]:
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
            }
        )
        return [item.get("title") for item in data.get("query", {}).get("search", [])]
