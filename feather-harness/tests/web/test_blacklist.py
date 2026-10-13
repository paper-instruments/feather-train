"""Research policies apply before any provider source reaches the agent."""

import json

import httpx
import pytest

from feather_harness.tools.web.exa import ExaConfigurationError, ExaWebProvider
from feather_harness.tools.web.fetch import WebFetchResult, WebFetchUnavailable

pytestmark = pytest.mark.asyncio
URL = "https://www.sec.gov/filing.html"


def guarded_provider(handler, **policy):
    return ExaWebProvider(
        "test-key", transport_factory=lambda: httpx.MockTransport(handler),
        blocked_domains=policy.get("blocked_domains", ("github.com", "bigfinancebench.com")),
        blocked_content_markers=policy.get("blocked_content_markers", ("bigfinancebench", "rogoai")),
    )


@pytest.mark.parametrize("url", [
    "https://GITHUB.COM./repo",
    "https://raw.githubusercontent.com/file",
    "https://answer.BIGFINANCEBENCH.com/answer",
    "https://%62igfinancebench.com/answer",
    "https://%2562igfinancebench.com/answer",
    "https://ｂｉｇｆｉｎａｎｃｅｂｅｎｃｈ.com/answer",
])
async def test_direct_blocked_fetch_never_sends_an_http_request(url):
    def reject_request(request):
        pytest.fail("blocked source reached the provider")

    provider = guarded_provider(
        reject_request,
        blocked_domains=("github.com", "githubusercontent.com", "bigfinancebench.com"),
    )
    assert await provider.fetch(url, max_characters=100) == WebFetchUnavailable("not_available")


@pytest.mark.parametrize("changes", [
    {"url": "https://answers.bigfinancebench.com/q"},
    {"url": "https://%2567ithub.com/repo"},
    {"title": "Answer by RoGoAI"},
    {"text": "x" * 2000 + " %2562igfinancebench answer"},
    {"text": "Answer by R&#79;GOAI"},
    {"text": "Answer by ＲＯＧＯＡＩ"},
])
async def test_search_filters_complete_raw_sources_before_excerpt_bounding(changes):
    def respond(request):
        payload = json.loads(request.content)
        assert payload["excludeDomains"] == ["github.com", "bigfinancebench.com"]
        blocked = {"url": "https://example.org/mirror", "title": "Mirror", "highlights": []}
        if "text" in changes:
            blocked["highlights"] = [changes["text"]]
        else:
            blocked.update(changes)
        return httpx.Response(200, json={"results": [
            blocked,
            {"url": URL, "title": "Primary filing", "highlights": ["Revenue was $100."]},
        ]})

    results = await guarded_provider(respond).search("revenue", limit=2)
    assert len(results) == 1
    assert results[0].url == URL
    assert results[0].snippet == "Revenue was $100."


@pytest.mark.parametrize("changes", [
    {"url": "https://answers.bigfinancebench.com/q"},
    {"url": "https://%2567ithub.com/repo"},
    {"title": "Answer by RoGoAI"},
    {"text": "x" * 2000 + " %2562igfinancebench answer"},
    {"text": "Answer by R&#79;GOAI"},
    {"text": "Answer by ＲＯＧＯＡＩ"},
])
async def test_fetch_rejects_redirects_and_full_raw_content_without_retry(changes):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "results": [{"id": URL, "url": URL, "title": "Filing", "text": "Revenue", **changes}],
            "statuses": [{"id": URL, "status": "success"}],
        })

    assert await guarded_provider(respond).fetch(URL, max_characters=100) == WebFetchUnavailable("not_available")
    assert len(requests) == 1


async def test_domain_boundaries_allow_unrelated_primary_source_and_fetch_content():
    url = "https://notgithub.com/article"

    def respond(request):
        return httpx.Response(200, json={
            "results": [{"id": url, "url": url, "title": "Article", "text": "Revenue"}],
            "statuses": [{"id": url, "status": "success"}],
        })

    assert await guarded_provider(respond).fetch(url, max_characters=100) == WebFetchResult(url, "Article", "Revenue")


@pytest.mark.parametrize("policy", [
    {"blocked_domains": ("https://github.com",)},
    {"blocked_domains": ("*.github.com",)},
    {"blocked_domains": ("1.2.3.4",)},
    {"blocked_content_markers": ("  ",)},
    {"blocked_content_markers": (None,)},
])
async def test_invalid_policy_fails_before_provider_use(policy):
    with pytest.raises(ExaConfigurationError):
        guarded_provider(None, **policy)


async def test_policy_normalizes_and_deduplicates_domains_and_markers():
    provider = guarded_provider(
        None, blocked_domains=("GitHub.COM.", "github.com"),
        blocked_content_markers=(" ROGOAI ", "rogoai"),
    )
    assert provider.blocked_domains == ("github.com",)
    assert provider.blocked_content_markers == ("rogoai",)
