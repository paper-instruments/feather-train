import hashlib
import json

import pytest

from feather_harness.production import build_production_instructions
from feather_harness.prompt import build_instructions
from tests.local.test_cli import cli, model_server, response


@pytest.mark.parametrize("profile", ["train", "prod"])
@pytest.mark.parametrize("failed", [False, True])
def test_native_cli_composes_once_and_persists_effective_prompt_even_on_failure(
    tmp_path, profile, failed
):
    task = "Literal task café\n\n"
    additional = "Work autonomously inside {{ workspace }}."
    metadata_path = tmp_path / "logs/eval-prompt.json"
    replies = (
        [(500, {"error": {"message": "provider failed", "type": "server_error"}})] * 3
        if failed else [(200, response("Done."))]
    )
    with model_server(replies) as (url, requests):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "model": "test", "base_url": url, "profile": profile,
            "additional_instructions": additional, "eval_prompt_output": str(metadata_path),
            "model_retry": {"max_attempts": 3, "base_backoff_seconds": 0},
        }))
        result = cli(tmp_path, "--config", str(config), "-", input=task)
    assert result.returncode == (1 if failed else 0), result.stderr
    sent = requests[0][1]["messages"]
    assert sent[1] == {"role": "user", "content": task}
    system = sent[0]["content"]
    assert system.count("Work autonomously inside") == 1
    assert f"Work autonomously inside {tmp_path}." in system
    assert system.count("# Additional task instructions") == 1
    assert system.endswith(f"# Additional task instructions\n\nWork autonomously inside {tmp_path}.")
    base = (
        build_production_instructions(skills_root=tmp_path / ".agents")
        if profile == "prod" else build_instructions()
    )
    assert system.startswith(base)
    metadata = json.loads(metadata_path.read_text())
    assert metadata["profile"] == profile
    assert metadata["system_prompt_sha256"] == hashlib.sha256(system.encode()).hexdigest()
