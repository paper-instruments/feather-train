"""Exa transport for Feather's provider-neutral web tools."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .fetch import WebFetchResult, WebFetchUnavailable
from .search import WebSearchResult

logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 256 * 1024
_MAX_HIGHLIGHT_CHARS = 2_000
_MAX_CONTENT_CHARS = 10_000
_LIVECRAWL_TIMEOUT_MILLISECONDS = 20_000
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.5
_REQUEST_TIMEOUT_SECONDS = 30.0
_TIMEOUT = httpx.Timeout(_REQUEST_TIMEOUT_SECONDS, connect=5.0)
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})
_SAFE_TAG = re.compile(r"[A-Z0-9_]{1,64}\Z")
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
_CRAWL_HTTP_STATUS = re.compile(r"CRAWL_HTTP_[1-5][0-9]{2}")
_DOMAIN = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,62}")


class ExaConfigurationError(ValueError):
    """Invalid credentials or request configuration; retrying cannot repair it."""


class ExaResponseError(RuntimeError):
    """A retryable invalid or oversized Exa response."""


class _ExaFetchUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class _ExaCrawlUnavailable(ExaResponseError):
    pass


_retry_exa = retry(
    retry=retry_if_exception_type((ExaResponseError, httpx.TransportError, TimeoutError)),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_random_exponential(multiplier=_BACKOFF_BASE_SECONDS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@dataclass(frozen=True, slots=True)
class ExaWebProvider:
    """Direct, bounded adapter for Exa Search and Contents."""

    api_key: str = field(repr=False)
    transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = field(
        default=None,
        repr=False,
    )
    blocked_domains: tuple[str, ...] = ()
    blocked_content_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= len(self.api_key) <= 512 or any(
            not 0x21 <= ord(character) <= 0x7E for character in self.api_key
        ):
            raise ExaConfigurationError("EXA_API_KEY is required when a web tool is enabled")
        object.__setattr__(
            self, "blocked_domains", tuple(dict.fromkeys(_normalize_domain(d) for d in self.blocked_domains))
        )
        object.__setattr__(
            self,
            "blocked_content_markers",
            tuple(dict.fromkeys(_normalize_marker(m) for m in self.blocked_content_markers)),
        )

    @_retry_exa
    async def search(self, query: str, *, limit: int) -> list[WebSearchResult]:
        payload = {
            "query": query,
            "type": "auto",
            "numResults": limit,
            "contents": {
                "highlights": {"maxCharacters": _MAX_HIGHLIGHT_CHARS},
                "livecrawlTimeout": _LIVECRAWL_TIMEOUT_MILLISECONDS,
            },
        }
        if self.blocked_domains:
            payload["excludeDomains"] = list(self.blocked_domains)
        body = await self._request(
            "search",
            payload=payload,
        )
        return _parse_search_results(
            body, limit=limit,
            blocked_domains=self.blocked_domains,
            blocked_content_markers=self.blocked_content_markers,
        )

    async def fetch(
        self,
        url: str,
        *,
        max_characters: int,
    ) -> WebFetchResult | WebFetchUnavailable:
        if not 1 <= max_characters <= _MAX_CONTENT_CHARS:
            raise ValueError(f"Exa content limit must be between 1 and {_MAX_CONTENT_CHARS}")
        if _is_blocked_source(
            {"url": url}, self.blocked_domains, self.blocked_content_markers
        ):
            return WebFetchUnavailable(reason="not_available")

        try:
            return await self._fetch(url, max_characters=max_characters)
        except _ExaCrawlUnavailable:
            return WebFetchUnavailable(reason="not_available")

    @_retry_exa
    async def _fetch(
        self, url: str, *, max_characters: int
    ) -> WebFetchResult | WebFetchUnavailable:
        try:
            body = await self._request(
                "contents",
                payload={
                    "urls": [url],
                    "text": {"maxCharacters": max_characters},
                    "livecrawlTimeout": _LIVECRAWL_TIMEOUT_MILLISECONDS,
                },
            )
        except _ExaFetchUnavailable as exc:
            return WebFetchUnavailable(reason=exc.reason)
        return _parse_fetch_result(
            body,
            requested_url=url,
            max_characters=max_characters,
            blocked_domains=self.blocked_domains,
            blocked_content_markers=self.blocked_content_markers,
        )

    async def _request(
        self,
        operation: str,
        *,
        payload: dict[str, object],
    ) -> bytes:
        async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                transport=(
                    self.transport_factory() if self.transport_factory is not None else None
                ),
                timeout=_TIMEOUT,
                follow_redirects=False,
            ) as client:
                async with client.stream(
                    "POST",
                    f"https://api.exa.ai/{operation}",
                    headers={
                        "x-api-key": self.api_key,
                        "content-type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if not 200 <= response.status_code < 300:
                        await _raise_for_status(response, operation=operation)
                    return await _read_bounded_body(response)


async def _read_bounded_body(response: httpx.Response) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > _MAX_RESPONSE_BYTES:
            raise ExaResponseError(f"Exa response exceeded {_MAX_RESPONSE_BYTES} bytes")

    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
            raise ExaResponseError(f"Exa response exceeded {_MAX_RESPONSE_BYTES} bytes")
        body.extend(chunk)
    return bytes(body)


async def _raise_for_status(
    response: httpx.Response,
    *,
    operation: str,
) -> None:
    error_payload = await _read_error_metadata(response)
    tag = error_payload.get("tag")
    request_id = _safe_value(
        response.headers.get("x-request-id") or response.headers.get("request-id"),
        pattern=_SAFE_REQUEST_ID,
    ) or error_payload.get("requestId")
    message = f"Exa {operation} failed with HTTP {response.status_code}"
    if tag:
        message += f" ({tag})"
    if request_id:
        message += f" (request ID {request_id})"
    if operation == "contents":
        if response.status_code == 403 and tag == "ROBOTS_FILTER_FAILED":
            raise _ExaFetchUnavailable("not_available")
        if response.status_code == 400 and tag == "INVALID_URLS":
            raise _ExaFetchUnavailable("unsupported")
        if response.status_code == 400 and tag == "NO_CONTENT_FOUND":
            raise _ExaFetchUnavailable("not_available")
        if response.status_code == 422 and tag == "FETCH_DOCUMENT_ERROR":
            raise _ExaFetchUnavailable("not_available")
    if (
        response.status_code in _TRANSIENT_STATUS_CODES
        or response.status_code >= 500
        or response.status_code == 422
        or (response.status_code == 403 and tag == "CONTENT_FILTER_ERROR")
    ):
        raise ExaResponseError(message)
    raise ExaConfigurationError(message)


async def _read_error_metadata(response: httpx.Response) -> dict[str, str]:
    try:
        body = await _read_bounded_body(response)
    except (ExaResponseError, httpx.HTTPError):
        return {}
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, str] = {}
    tag = _safe_value(payload.get("tag"), pattern=_SAFE_TAG)
    request_id = _safe_value(payload.get("requestId"), pattern=_SAFE_REQUEST_ID)
    if tag:
        metadata["tag"] = tag
    if request_id:
        metadata["requestId"] = request_id
    return metadata


def _safe_value(value: object, *, pattern: re.Pattern[str]) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value):
        return value
    return None


def _parse_search_results(
    body: bytes, *, limit: int,
    blocked_domains: tuple[str, ...] = (),
    blocked_content_markers: tuple[str, ...] = (),
) -> list[WebSearchResult]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExaResponseError("Exa returned an invalid JSON response") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ExaResponseError("Exa response did not contain a results list")

    raw_results = payload["results"]
    if len(raw_results) > limit:
        raise ExaResponseError(f"Exa returned {len(raw_results)} results for a limit of {limit}")
    return [
        _parse_search_result(raw_result, index=index)
        for index, raw_result in enumerate(raw_results)
        if not _is_blocked_source(raw_result, blocked_domains, blocked_content_markers)
    ]


def _parse_search_result(raw_result: object, *, index: int) -> WebSearchResult:
    if not isinstance(raw_result, dict):
        raise ExaResponseError(f"Exa result {index} was not an object")

    url = raw_result.get("url")
    title = raw_result.get("title")
    highlights = raw_result.get("highlights")
    published_date = raw_result.get("publishedDate")

    if not isinstance(url, str) or not url:
        raise ExaResponseError(f"Exa result {index} did not contain a URL")
    if title is not None and not isinstance(title, str):
        raise ExaResponseError(f"Exa result {index} contained an invalid title")
    if highlights is None:
        highlights = []
    if not isinstance(highlights, list) or not all(
        isinstance(highlight, str) for highlight in highlights
    ):
        raise ExaResponseError(f"Exa result {index} contained invalid highlights")
    if published_date is not None and not isinstance(published_date, str):
        raise ExaResponseError(f"Exa result {index} contained an invalid publication date")

    try:
        return WebSearchResult(
            title=title.strip() if title and title.strip() else url,
            url=url,
            snippet="\n".join(highlight.strip() for highlight in highlights if highlight.strip()),
            published_date=published_date,
        )
    except ValueError as exc:
        raise ExaResponseError(f"Exa result {index} was invalid: {exc}") from exc


def _parse_fetch_result(
    body: bytes,
    *,
    requested_url: str,
    max_characters: int,
    blocked_domains: tuple[str, ...] = (),
    blocked_content_markers: tuple[str, ...] = (),
) -> WebFetchResult | WebFetchUnavailable:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExaResponseError("Exa returned an invalid JSON response") from exc
    if not isinstance(payload, dict):
        raise ExaResponseError("Exa contents response was not an object")

    raw_results = payload.get("results")
    raw_statuses = payload.get("statuses")
    if not isinstance(raw_results, list):
        raise ExaResponseError("Exa contents response did not contain a results list")
    if not isinstance(raw_statuses, list) or len(raw_statuses) != 1:
        raise ExaResponseError("Exa contents response did not contain exactly one status")

    status = raw_statuses[0]
    if not isinstance(status, dict) or status.get("id") != requested_url:
        raise ExaResponseError("Exa contents response contained an invalid status")
    if status.get("status") == "error":
        if raw_results:
            raise ExaResponseError("Exa contents response contained results for a failed fetch")
        return _parse_fetch_unavailable(status)
    if status.get("status") != "success":
        raise ExaResponseError("Exa contents response contained an invalid status")
    if status.get("error") is not None:
        raise ExaResponseError("Exa contents success contained unexpected error metadata")
    if len(raw_results) != 1:
        raise ExaResponseError("Exa contents response did not contain exactly one result")

    raw_result = raw_results[0]
    if not isinstance(raw_result, dict):
        raise ExaResponseError("Exa contents result was not an object")
    if raw_result.get("id") != requested_url:
        raise ExaResponseError("Exa contents result contained an invalid ID")
    if _is_blocked_source(raw_result, blocked_domains, blocked_content_markers):
        return WebFetchUnavailable(reason="not_available")
    resolved_url = raw_result.get("url")
    title = raw_result.get("title")
    content = raw_result.get("text")
    if not isinstance(resolved_url, str) or not resolved_url:
        raise ExaResponseError("Exa contents result did not contain a URL")
    if title is not None and not isinstance(title, str):
        raise ExaResponseError("Exa contents result contained an invalid title")
    if not isinstance(content, str):
        raise ExaResponseError("Exa contents result did not contain text")
    if len(content) > max_characters:
        raise ExaResponseError(f"Exa contents result exceeded {max_characters} characters")
    try:
        return WebFetchResult(
            url=resolved_url,
            title=title.strip() if title and title.strip() else resolved_url,
            content=content,
        )
    except ValueError as exc:
        raise ExaResponseError(f"Exa contents result was invalid: {exc}") from exc


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    while True:
        decoded = html.unescape(unquote(normalized))
        if decoded == normalized:
            return decoded
        normalized = unicodedata.normalize("NFKC", decoded).casefold()


def _normalize_domain(domain: str) -> str:
    if not isinstance(domain, str):
        raise ExaConfigurationError("blocked_domains must contain DNS names")
    try:
        normalized = _normalize_text(domain).strip().rstrip(".").encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ExaConfigurationError("blocked_domains must contain DNS names") from exc
    if not _DOMAIN.fullmatch(normalized):
        raise ExaConfigurationError("blocked_domains must contain DNS names")
    return normalized


def _normalize_marker(marker: str) -> str:
    if not isinstance(marker, str) or not (normalized := _normalize_text(marker).strip()):
        raise ExaConfigurationError("blocked_content_markers must contain nonempty strings")
    return normalized


def _is_blocked_source(
    source: object, domains: tuple[str, ...], markers: tuple[str, ...]
) -> bool:
    if not isinstance(source, dict) or not (domains or markers):
        return False
    url = source.get("url")
    if domains and isinstance(url, str):
        try:
            host = urlsplit(_normalize_text(url)).hostname or ""
            host = host.rstrip(".").encode("idna").decode("ascii")
        except (ValueError, UnicodeError):
            host = ""
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return True
    # Inspect the complete provider result before snippets or extracted text are bounded.
    if markers:
        for text in _source_strings(source):
            normalized = _normalize_text(text)
            if any(marker in normalized for marker in markers):
                return True
    return False


def _source_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _source_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _source_strings(child)


def _parse_fetch_unavailable(status: dict[str, object]) -> WebFetchUnavailable:
    error = status.get("error")
    if not isinstance(error, dict):
        raise ExaResponseError("Exa contents failure did not contain error metadata")
    tag = _safe_value(error.get("tag"), pattern=_SAFE_TAG)
    reasons = {
        "CRAWL_HTTP_400": "not_available",
        "CRAWL_NOT_FOUND": "not_found",
        "CRAWL_UNEXPECTED_CONTENT_TYPE": "unsupported",
        "SOURCE_NOT_AVAILABLE": "not_available",
        "UNSUPPORTED_URL": "unsupported",
    }
    if tag in reasons:
        return WebFetchUnavailable(reason=reasons[tag])
    # The source site refused or failed Exa's crawler (403 paywalls, 5xx): an ordinary
    # unavailable source the model can work around, never a reason to abort the run.
    if tag and _CRAWL_HTTP_STATUS.fullmatch(tag):
        return WebFetchUnavailable(reason="not_found" if tag == "CRAWL_HTTP_404" else "not_available")
    if tag in {
        "CRAWL_TIMEOUT",
        "CRAWL_LIVECRAWL_TIMEOUT",
        "CRAWL_UNKNOWN_ERROR",
    }:
        raise _ExaCrawlUnavailable(f"Exa contents failed ({tag})")
    raise ExaResponseError(f"Exa contents failed ({tag or 'UNKNOWN_ERROR'})")
