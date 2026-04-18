from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "RAQR/0.1 (research corpus build)"


class WikidataClient:
    def __init__(
        self,
        throttle_s: float = 0.1,
        max_retries: int = 5,
        backoff_s: float = 1.0,
    ) -> None:
        self.throttle_s = throttle_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._last_call = 0.0

    def _sleep_if_needed(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.throttle_s:
            time.sleep(self.throttle_s - delta)
        self._last_call = time.time()

    def _get(self, params: Dict[str, str]) -> dict:
        attempts = 0
        while True:
            self._sleep_if_needed()
            resp = requests.get(
                WIKIDATA_API,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()

            attempts += 1
            if attempts > self.max_retries:
                resp.raise_for_status()

            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait_s = max(float(retry_after), self.backoff_s)
            else:
                wait_s = self.backoff_s * (2 ** (attempts - 1))
            time.sleep(wait_s)

    def get_wikipedia_title(self, qid: str) -> Optional[str]:
        data = self._get(
            {
                "action": "wbgetentities",
                "format": "json",
                "props": "sitelinks",
                "ids": qid,
                "sitefilter": "enwiki",
            }
        )
        entity = data.get("entities", {}).get(qid, {})
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki")
        if not enwiki:
            return None
        return enwiki.get("title")

    def get_claim_qids(self, qid: str, props: List[str], limit: int = 20) -> List[str]:
        data = self._get(
            {
                "action": "wbgetentities",
                "format": "json",
                "props": "claims",
                "ids": qid,
            }
        )
        entity = data.get("entities", {}).get(qid, {})
        claims = entity.get("claims", {})
        found: List[str] = []
        for prop in props:
            for claim in claims.get(prop, []):
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value", {})
                if isinstance(value, dict) and value.get("id"):
                    found.append(value["id"])
                    if len(found) >= limit:
                        return found
        return found
