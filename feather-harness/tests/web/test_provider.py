import asyncio
import json

import httpx
import pytest

from feather_harness.tools.web.exa import ExaConfigurationError, ExaResponseError, ExaWebProvider
from feather_harness.tools.web.fetch import WebFetchResult, WebFetchUnavailable
from feather_harness.tools.web.search import WebSearchResult

pytestmark = pytest.mark.asyncio
URL = "https://example.com/article"


def provider_for(handler):
    return ExaWebProvider("secret-key", transport_factory=lambda: httpx.MockTransport(handler))


def fetched_page(**changes):
    return {
        "results": [{"id": URL, "url": URL, "title": "Article", "text": "Evidence", **changes}],
        "statuses": [{"id": URL, "status": "success"}],
    }


async def test_search_preserves_request_limits_and_ranked_excerpts():
    def respond(request):
        assert request.url == "https://api.exa.ai/search"
        assert request.headers["x-api-key"] == "secret-key"
        assert json.loads(request.content) == {
            "query": "latest filing",
            "type": "auto",
            "numResults": 2,
            "contents": {"highlights": {"maxCharacters": 2000}, "livecrawlTimeout": 20000},
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": URL,
                        "title": " Filing ",
                        "highlights": [" First ", "", "Second"],
                        "publishedDate": "2026-07-25",
                    },
                    {"url": "https://example.org/source", "title": None, "highlights": None},
                ]
            },
        )

    provider = provider_for(respond)
    results = await provider.search("latest filing", limit=2)
    assert results == [
        WebSearchResult(
            title="Filing", url=URL, snippet="First\nSecond", published_date="2026-07-25"
        ),
        WebSearchResult(
            title="https://example.org/source", url="https://example.org/source", snippet=""
        ),
    ]
    assert "secret-key" not in repr(provider)


async def test_fetch_preserves_request_limit_and_resolved_url():
    def respond(request):
        assert request.url == "https://api.exa.ai/contents"
        assert json.loads(request.content) == {
            "urls": [URL],
            "text": {"maxCharacters": 100},
            "livecrawlTimeout": 20000,
        }
        return httpx.Response(200, json=fetched_page(url="https://example.com/canonical"))

    result = await provider_for(respond).fetch(URL, max_characters=100)
    assert result == WebFetchResult(
        url="https://example.com/canonical", title="Article", content="Evidence"
    )


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/admin", "https://example.com/" + "a" * 2_000, "https://example.com/a b"],
)
async def test_search_skips_a_source_url_the_tools_cannot_render(url):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"results": [{"url": url, "title": "Bad"}, {"url": URL, "title": "Good"}]}
        )

    results = await provider_for(respond).search("filing", limit=2)
    assert results == [WebSearchResult(title="Good", url=URL)]
    assert len(requests) == 1


async def test_fetch_resolved_to_a_non_public_url_is_unavailable_without_retry():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=fetched_page(url="http://10.0.0.5/article"))

    result = await provider_for(respond).fetch(URL, max_characters=100)
    assert result == WebFetchUnavailable(reason="not_available")
    assert len(requests) == 1


@pytest.mark.parametrize("failure", [503, 429, "transport", "malformed", "oversized"])
async def test_search_retries_transient_failures_and_recovers(failure):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 3:
            return httpx.Response(200, json={"results": []})
        if failure == "transport":
            raise httpx.ConnectError("connection interrupted")
        if failure == "malformed":
            return httpx.Response(200, json={"results": [{"url": URL, "highlights": 17}]})
        if failure == "oversized":
            return httpx.Response(200, content=b" " * (256 * 1024 + 1))
        return httpx.Response(failure)

    assert await provider_for(respond).search("query", limit=1) == []
    assert len(requests) == 3
    assert all(request.content == requests[0].content for request in requests)


@pytest.mark.parametrize("operation", ["search", "fetch"])
async def test_auth_failure_is_non_retryable_and_does_not_echo_private_input(operation, caplog):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            401,
            json={
                "tag": "INVALID_API_KEY",
                "requestId": "req-1",
                "message": "secret-key private-query",
                "query": "private-query",
            },
        )

    provider = provider_for(respond)
    with pytest.raises(ExaConfigurationError) as error:
        if operation == "search":
            await provider.search("private-query", limit=1)
        else:
            await provider.fetch(URL, max_characters=100)
    assert len(requests) == 1
    assert (
        str(error.value)
        == f"Exa {'search' if operation == 'search' else 'contents'} failed with HTTP 401 (INVALID_API_KEY) (request ID req-1)"
    )
    assert "secret-key" not in str(error.value) + caplog.text
    assert "private-query" not in str(error.value) + caplog.text


@pytest.mark.parametrize(
    ("status", "tag", "reason"),
    [
        (200, "CRAWL_NOT_FOUND", "not_found"),
        # A site refusing Exa's crawler once aborted whole agent runs (FrontierFinance, 2026-10-09).
        (200, "CRAWL_HTTP_403", "not_available"),
        (200, "CRAWL_HTTP_404", "not_found"),
        (200, "CRAWL_HTTP_503", "not_available"),
        (200, "UNSUPPORTED_URL", "unsupported"),
        (403, "ROBOTS_FILTER_FAILED", "not_available"),
        (400, "INVALID_URLS", "unsupported"),
        (422, "FETCH_DOCUMENT_ERROR", "not_available"),
    ],
)
async def test_source_unavailability_is_returned_without_retry(status, tag, reason):
    requests = []

    def respond(request):
        requests.append(request)
        body = (
            {"tag": tag}
            if status != 200
            else {
                "results": [],
                "statuses": [{"id": URL, "status": "error", "error": {"tag": tag}}],
            }
        )
        return httpx.Response(status, json=body)

    assert await provider_for(respond).fetch(URL, max_characters=100) == WebFetchUnavailable(
        reason=reason
    )
    assert len(requests) == 1


@pytest.mark.parametrize("tag", ["CRAWL_TIMEOUT", "NEW_PROVIDER_ERROR"])
async def test_exhausted_crawl_timeout_is_unavailable_but_unknown_errors_propagate(tag):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [],
                "statuses": [{"id": URL, "status": "error", "error": {"tag": tag}}],
            },
        )

    provider = provider_for(respond)
    if tag == "CRAWL_TIMEOUT":
        assert await provider.fetch(URL, max_characters=100) == WebFetchUnavailable(
            reason="not_available"
        )
    else:
        with pytest.raises(ExaResponseError, match="NEW_PROVIDER_ERROR"):
            await provider.fetch(URL, max_characters=100)
    assert len(requests) == 3


@pytest.mark.parametrize("changes", [{"id": "https://example.com/other"}, {"text": "x" * 101}])
async def test_fetch_rejects_wrong_page_or_overlong_content(changes):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=fetched_page(**changes))

    with pytest.raises(ExaResponseError):
        await provider_for(respond).fetch(URL, max_characters=100)
    assert len(requests) == 3


async def test_cancellation_propagates_without_retry():
    requests = []

    def respond(request):
        requests.append(request)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await provider_for(respond).search("query", limit=1)
    assert len(requests) == 1


@pytest.mark.parametrize("key", ["", "key\nInjected: header"])
async def test_invalid_credentials_fail_before_request(key):
    with pytest.raises(ExaConfigurationError):
        ExaWebProvider(key)
