import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from feather_harness import ChatCompletionsModel, run_local
from feather_harness.core.compaction import DEFAULT_COMPACTION_PROMPT, CompactionConfig
from feather_harness.local.filesystem import LocalFileRuntime
from feather_harness.production import (
    PRODUCTION_COMPACTION_PROMPT,
    PRODUCTION_PROMPT,
    PRODUCTION_TOOL_DESCRIPTIONS,
    VISION_DISABLED_FACT,
    build_production_instructions,
    production_compaction,
)
from feather_harness.prompt import RESOURCE_ROOT, build_instructions
from feather_harness.scripts.install_skills import (
    BUNDLE_ROOT,
    install_production_skill_tree,
    install_production_skills,
    install_skills,
)
from feather_harness.tools.files import FileTools
from feather_harness.tools.web.fetch import WebFetchUnavailable
from feather_harness.tools.web.search import WebSearchResult
from tests.core.test_images import IMAGE, PNG
from tests.local.test_cli import response

FORMATS = ("docx", "pdf", "pptx", "xlsx")


@pytest.mark.parametrize(
    "options, production, image_read",
    [
        ({}, True, True),
        ({"profile": "train"}, False, False),
        ({"profile": "prod", "enable_image_tools": False}, True, False),
        ({"profile": "train", "enable_image_tools": True}, False, True),
    ],
)
async def test_startup_profiles_and_image_overrides_reach_model_requests(
    tmp_path: Path, options: dict, production: bool, image_read: bool
) -> None:
    (tmp_path / "page.png").write_bytes(PNG)
    task = "Inspect page.png without changing the supplied task.\n"
    async with _model(
        response(tools=[("read", {"file_path": "page.png"})]), response("Finished.")
    ) as (model, requests):
        result = await run_local(task, model=model, workspace=tmp_path, **options)

    first = requests[0]
    instructions = first["messages"][0]["content"]
    assert first["messages"][1] == {"role": "user", "content": task}
    if production:
        assert (RESOURCE_ROOT / "production_prompt.md").read_text().strip() in instructions
        assert "## Host capabilities" not in instructions
        assert "headless" not in instructions
    else:
        assert instructions == build_instructions()
    tools = {tool["function"]["name"]: tool["function"] for tool in first["tools"]}
    assert "view_image" not in tools
    assert tools["read"]["description"].startswith(
        "Read a text file or an image." if image_read else "Read a text file with line numbers."
    )
    assert (
        "text file or image" in tools["read"]["parameters"]["properties"]["file_path"]["description"]
    ) is image_read
    observation = next(message for message in requests[1]["messages"] if message["role"] == "tool")
    if image_read:
        assert all(part["type"] == "text" for part in observation["content"])
        assert requests[1]["messages"][-1]["role"] == "user"
        assert requests[1]["messages"][-1]["content"][-1] == IMAGE
    else:
        assert observation["content"] == "Vision is not available in this environment."
        assert requests[1]["messages"][-1] == observation
    assert result.final_answer == "Finished."


class _WebProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int) -> list[WebSearchResult]:
        self.queries.append(query)
        return [WebSearchResult(title="Filing", url="https://example.com/filing")]

    async def fetch(self, url: str, *, max_characters: int) -> WebFetchUnavailable:
        return WebFetchUnavailable(reason="not_found")


@pytest.mark.parametrize("profile", ["train", "prod"])
async def test_web_tools_are_offered_only_with_a_provider(tmp_path: Path, profile: str) -> None:
    async with _model(response("Finished.")) as (model, requests):
        await run_local("Do the task", model=model, workspace=tmp_path, profile=profile)
    assert not {"web_search", "web_fetch"} & {
        tool["function"]["name"] for tool in requests[0]["tools"]
    }

    provider = _WebProvider()
    async with _model(
        response(tools=[("web_search", {"query": "annual filing"})]), response("Finished.")
    ) as (model, requests):
        await run_local(
            "Do the task", model=model, workspace=tmp_path, profile=profile,
            web_provider=provider,
        )
    assert {"web_search", "web_fetch"} <= {
        tool["function"]["name"] for tool in requests[0]["tools"]
    }
    assert provider.queries == ["annual filing"]
    observation = next(message for message in requests[1]["messages"] if message["role"] == "tool")
    assert "https://example.com/filing" in observation["content"]


async def test_profiles_change_prose_not_argument_contracts_or_text_invocation(
    tmp_path: Path,
) -> None:
    recorded = {}
    for profile in ("train", "prod"):
        workspace = tmp_path / profile
        workspace.mkdir()
        (workspace / "notes.txt").write_text("before")
        async with _model(
            response(
                tools=[
                    ("read", {"file_path": "notes.txt"}),
                    ("read", {"file_path": "notes.txt", "offset": 0}),
                ]
            ),
            response(
                tools=[
                    ("edit", {"file_path": "notes.txt", "old_string": "before", "new_string": "after"})
                ]
            ),
            response("Updated."),
        ) as (model, requests):
            await run_local(
                "Update notes.txt", model=model, workspace=workspace, profile=profile,
                enable_image_tools=True,
            )
        recorded[profile] = requests
        assert (workspace / "notes.txt").read_text() == "after"

    assert recorded["prod"][0]["tools"] != recorded["train"][0]["tools"]
    assert _without_production_descriptions(recorded["prod"][0]["tools"]) == (
        _without_production_descriptions(recorded["train"][0]["tools"])
    )
    for production, training in zip(recorded["prod"][1:], recorded["train"][1:], strict=True):
        assert production["messages"][1:] == training["messages"][1:]


async def test_reusing_a_production_workspace_restores_the_training_manuals(
    tmp_path: Path,
) -> None:
    async with _model(response("Done.")) as (model, _):
        await run_local("Do the task", model=model, workspace=tmp_path)
    assert "## Professional guidance" in (tmp_path / ".agents/docx/SKILL.md").read_text()
    assert (tmp_path / ".agents/design.md").is_file()

    async with _model(response("Done.")) as (model, requests):
        await run_local("Do the task", model=model, workspace=tmp_path, profile="train")

    assert requests[0]["messages"][0]["content"] == build_instructions()
    for name in FORMATS:
        installed = (tmp_path / ".agents" / name / "SKILL.md").read_bytes()
        assert installed == (BUNDLE_ROOT / "skills" / name / "SKILL.md").read_bytes()


async def test_production_keeps_the_trained_catalog_and_appends_guidance(tmp_path: Path) -> None:
    async with _model(response("Done.")) as (model, requests):
        await run_local("Do the task", model=model, workspace=tmp_path)

    instructions = requests[0]["messages"][0]["content"]
    training = build_instructions()
    assert instructions.endswith(training[training.index("## Skills") :])
    assert "# Tool capability instructions" in instructions
    assert "# Sandbox capability instructions" not in instructions
    for name in FORMATS:
        manual = (BUNDLE_ROOT / "skills" / name / "SKILL.md").read_bytes()
        guide = (BUNDLE_ROOT / "production_skills" / f"{name}.md").read_bytes()
        installed = (tmp_path / ".agents" / name / "SKILL.md").read_bytes()
        assert installed == manual.rstrip(b"\n") + b"\n\n" + guide
    icons = BUNDLE_ROOT / "production_skills/icons"
    indexed = {line.split("\t")[0] for line in (icons / "index.tsv").read_text().splitlines()}
    assert indexed == {path.stem for path in icons.glob("*.png")}
    for name in ("index.tsv", "LICENSE", "recolor.py", "factory.png"):
        assert (tmp_path / ".agents/icons" / name).read_bytes() == (icons / name).read_bytes()
    guidance = [PRODUCTION_PROMPT]
    guidance += [path.read_text() for path in (BUNDLE_ROOT / "production_skills").glob("*.md")]
    for path in {path for text in guidance for path in re.findall(r"`(\.agents/[^`]+)`", text)}:
        assert (tmp_path / path).is_file(), path


def test_production_instructions_are_utf8_with_a_legacy_locale_encoding(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "résumé"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: résumé\ndescription: Tailor a résumé — keep it brief.\n---\n",
        encoding="utf-8",
    )
    # Run in a separate process: the prompts are read at import time with the locale encoding.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from feather_harness.production import build_production_instructions; "
            "sys.stdout.write(build_production_instructions(skills_root=Path(sys.argv[1])))",
            str(tmp_path / "skills"),
        ],
        capture_output=True,
        env={
            **os.environ,
            "LC_ALL": "C",
            "PYTHONCOERCECLOCALE": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "0",
        },
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    instructions = result.stdout.decode("utf-8")
    assert PRODUCTION_PROMPT in instructions
    assert "- résumé: Tailor a résumé — keep it brief." in instructions


def test_a_host_owned_skill_tree_serves_every_production_reference(tmp_path: Path) -> None:
    # Desktop app data paths contain spaces, such as macOS Application Support.
    root = tmp_path / "Application Support" / "skills"
    install_production_skill_tree(root)
    instructions = build_production_instructions(
        skills_root=root,
        enable_image_tools=False,
        skills_path=root.as_posix(),
        vision_fact="The view_image tool describes images.",
    )

    guidance = [instructions, *(path.read_text() for path in root.rglob("*.md"))]
    assert not any(".agents/" in text for text in guidance)
    pattern = re.escape(root.as_posix()) + r"/[\w./-]+"
    referenced = {path.rstrip(".") for text in guidance for path in re.findall(pattern, text)}
    assert f"{root.as_posix()}/icons/recolor.py" in referenced
    for path in referenced:
        assert Path(path).is_file(), path
    assert f"(file: {root.as_posix()}/docx/SKILL.md)" in instructions
    assert "The view_image tool describes images." in instructions
    assert VISION_DISABLED_FACT not in instructions
    for name in FORMATS:
        installed = root / name / "SKILL.md"
        manual = (BUNDLE_ROOT / "skills" / name / "SKILL.md").read_text()
        assert installed.read_text().startswith(manual.rstrip("\n"))
        for link in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", installed.read_text()):
            if not link.startswith(("https://", "http://")):
                assert (installed.parent / link).is_file(), (installed, link)


def test_icon_recolor_keeps_the_glyph_and_changes_only_its_color(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    from feather_harness.production_skills.icons.recolor import recolor

    recolor("factory", "#FFFFFF", tmp_path / "white.png")
    source = Image.open(BUNDLE_ROOT / "production_skills/icons/factory.png").convert("RGBA")
    white = Image.open(tmp_path / "white.png")
    assert white.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    assert all(white.getchannel(band).getextrema() == (255, 255) for band in "RGB")
    assert white.info["dpi"] == source.info["dpi"]


def test_every_icon_is_cropped_to_its_artwork_at_one_resolution() -> None:
    Image = pytest.importorskip("PIL.Image")
    icons = sorted((BUNDLE_ROOT / "production_skills/icons").glob("*.png"))
    padded = []
    for path in icons:
        image = Image.open(path)
        if image.convert("RGBA").getbbox() != (0, 0, *image.size) or tuple(map(round, image.info["dpi"])) != (320, 320):
            padded.append(path.stem)
    assert icons and padded == []


def test_production_guidance_uses_banned_punctuation_only_to_name_it() -> None:
    guidance = {
        path.name: path.read_text() for path in (BUNDLE_ROOT / "production_skills").glob("*.md")
    }
    guidance["production_prompt.md"] = PRODUCTION_PROMPT
    guidance["production_compaction_prompt.md"] = PRODUCTION_COMPACTION_PROMPT
    for name, text in guidance.items():
        for line in text.splitlines():
            if "—" in line or "·" in line:
                assert re.search(r"em dash|middle dot|Confidential · Draft", line), (name, line)


@pytest.mark.parametrize(
    "profile, prompt, expected",
    [
        ("prod", DEFAULT_COMPACTION_PROMPT, PRODUCTION_COMPACTION_PROMPT),
        ("train", DEFAULT_COMPACTION_PROMPT, DEFAULT_COMPACTION_PROMPT),
        ("prod", "Custom continuation prompt.", "Custom continuation prompt."),
    ],
)
async def test_production_compaction_prompt_replaces_only_the_default(
    tmp_path: Path, profile: str, prompt: str, expected: str
) -> None:
    # A minimal context window compacts before the first turn, so the first request is the summary.
    compaction = CompactionConfig(enabled=True, context_window_tokens=30001, prompt=prompt)
    async with _model(response("Summary."), response("Done.")) as (model, requests):
        await run_local(
            "Do the task", model=model, workspace=tmp_path, profile=profile, compaction=compaction
        )
    assert requests[0]["messages"][-1]["content"].startswith(expected)


@pytest.mark.parametrize(
    "profile, settings, compactions",
    [("prod", {}, 1), ("prod", {"enabled": False}, 0), ("train", {}, 0)],
)
async def test_only_production_compacts_by_default_and_explicit_disable_wins(
    tmp_path: Path, profile: str, settings: dict, compactions: int
) -> None:
    # Harbor and the CLI pass compaction as JSON, so the profile default must survive it.
    compaction = CompactionConfig.model_validate_json(
        CompactionConfig(context_window_tokens=30001, **settings).model_dump_json()
    )
    async with _model(response("Summary."), response("Done.")) as (model, _):
        result = await run_local(
            "Do the task", model=model, workspace=tmp_path, profile=profile, compaction=compaction
        )
    assert result.compactions == compactions


def test_production_compaction_keeps_the_default_trigger() -> None:
    config = production_compaction(CompactionConfig())
    assert config.enabled
    assert config.threshold_tokens == 101_072


@pytest.mark.parametrize("profile, requests_sent", [("prod", 2), ("train", 1)])
async def test_only_production_recovers_an_empty_response(
    tmp_path: Path, profile: str, requests_sent: int
) -> None:
    async with _model(response(), response("Done.")) as (model, requests):
        result = await run_local("Do the task", model=model, workspace=tmp_path, profile=profile)
    assert len(requests) == requests_sent
    assert result.recoveries == requests_sent - 1


async def test_invalid_profile_selection_fails_before_workspace_or_model_effects(
    tmp_path: Path,
) -> None:
    async with _model() as (model, requests):
        with pytest.raises(ValueError, match="Unknown Feather profile"):
            await run_local("Do the task", model=model, workspace=tmp_path, profile="unknown")
    assert requests == []
    assert list(tmp_path.iterdir()) == []


async def test_each_start_replaces_stale_bundled_skills(tmp_path: Path) -> None:
    install_skills(tmp_path)
    install_production_skills(tmp_path)
    (tmp_path / ".agents/xlsx/SKILL.md").write_text("Stale guidance from an older harness")
    (tmp_path / ".agents/design.md").write_text("Stale house style")

    async with _model(response("Done.")) as (model, _):
        await run_local("Do the task", model=model, workspace=tmp_path)
    assert "## Professional guidance" in (tmp_path / ".agents/xlsx/SKILL.md").read_text()
    assert (tmp_path / ".agents/design.md").read_bytes() == (
        BUNDLE_ROOT / "production_skills/design.md"
    ).read_bytes()

    (tmp_path / ".agents/xlsx/SKILL.md").write_text("Stale guidance from an older harness")
    async with _model(response("Done.")) as (model, _):
        await run_local("Do the task", model=model, workspace=tmp_path, profile="train")
    assert (tmp_path / ".agents/xlsx/SKILL.md").read_bytes() == (
        BUNDLE_ROOT / "skills/xlsx/SKILL.md"
    ).read_bytes()


@pytest.mark.parametrize(
    "profile, target", [("train", ".agents"), ("prod", ".agents/design.md")]
)
async def test_skill_installation_refuses_symlinks_before_writing(
    tmp_path: Path, profile: str, target: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_text("Private material")
    if target == ".agents":
        (workspace / target).symlink_to(outside, target_is_directory=True)
    else:
        install_skills(workspace)
        (workspace / target).symlink_to(outside / "sentinel.txt")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    async with _model() as (model, requests):
        with pytest.raises(FileExistsError, match="symlink"):
            await run_local("Do the task", model=model, workspace=workspace, profile=profile)

    assert requests == []
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert (workspace / target).is_symlink()


@pytest.mark.parametrize("replacement", ["hard_link", "directory"])
async def test_skill_replacement_refuses_linked_or_irregular_paths(
    tmp_path: Path, replacement: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    install_skills(workspace)
    skill = workspace / ".agents/xlsx/SKILL.md"
    skill.unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("File shared with another checkout")
    if replacement == "hard_link":
        skill.hardlink_to(outside)
    else:
        skill.mkdir()
    (workspace / ".agents/docx/SKILL.md").write_text("Stale manual")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    async with _model() as (model, requests):
        with pytest.raises(FileExistsError, match="hard link|not a regular file"):
            await run_local("Do the task", model=model, workspace=workspace, profile="train")

    assert requests == []
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


async def test_production_skills_and_their_local_links_fit_default_reads(tmp_path: Path) -> None:
    install_skills(tmp_path)
    install_production_skills(tmp_path)
    read = FileTools(LocalFileRuntime(tmp_path)).read_tool()
    for path in [tmp_path / ".agents/design.md"] + [
        tmp_path / ".agents" / name / "SKILL.md" for name in FORMATS
    ]:
        result = await read.invoke(json.dumps({"file_path": str(path)}))
        assert result.error is None
        assert "[Output truncated:" not in result.content, path
        assert "Call read again" not in result.content, path
        for link in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", path.read_text()):
            if not link.startswith(("https://", "http://")):
                assert (path.parent / link).is_file(), (path, link)


async def test_grep_description_claims_match_search_behavior(tmp_path: Path) -> None:
    # The production grep description tells the model neither search below returns a match.
    (tmp_path / "wrapped.md").write_text("The covenant is tested\nquarterly on a rolling basis.\n")
    with zipfile.ZipFile(tmp_path / "credit.docx", "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("word/document.xml", "<w:t>" + "Change of control clause. " * 40 + "</w:t>")
    grep = FileTools(LocalFileRuntime(tmp_path)).grep_tool()

    wrapped = await grep.invoke(json.dumps({"pattern": "tested quarterly"}))
    office = await grep.invoke(json.dumps({"pattern": "Change of control"}))

    assert "No matches found" in wrapped.content
    assert "No matches found" in office.content
    description = PRODUCTION_TOOL_DESCRIPTIONS["grep"]
    assert "single line" in description and "Office documents and PDFs" in description


@asynccontextmanager
async def _model(*replies: dict) -> AsyncIterator[tuple[ChatCompletionsModel, list[dict]]]:
    requests = []
    responses = iter(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(responses))

    async with AsyncOpenAI(
        api_key="test", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        yield ChatCompletionsModel(client=client, model="test"), requests


def _without_production_descriptions(tools: list[dict]) -> list[dict]:
    normalized = deepcopy(tools)
    for tool in normalized:
        function = tool["function"]
        function.pop("description")
        if function["name"] == "read":
            function["parameters"]["properties"]["file_path"].pop("description")
    return normalized
