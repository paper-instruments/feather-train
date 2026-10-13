# Feather Train

Feather Train manages rollouts and grading on snapshotted task worlds in E2B.
Paper Instruments worlds combine filesystem state and MCP services, can be
represented as Harbor tasks, and are prepared offline as E2B snapshots for fast
rollout startup. The package takes task datasets and those snapshots as inputs.

- **Remote execution:** run Feather Harness unchanged, with the agent loop outside
  the sandbox and tools executing inside it. The runtime minimizes E2B round trips
  and recovers tool results after connection drops.
- **Isolated worlds and grading:** restore Docker Compose environments with separate
  agent, MCP, and grader containers, and run Paper Instruments' task graders remotely.
- **Rollout observability:** Braintrust records agent execution, grading, rewards,
  and rollout metadata for inspecting production runs.
- **Training bookkeeping:** Verifiers and Renderers record inference tokens,
  logprobs, and attribution used to construct training samples.

A reference Tinker recipe connects these rollouts to asynchronous training. Small
changes can adapt the integration to other frameworks, such as SkyRL, Baseten Loops,
or Fireworks. See [Architecture](architecture.md) for the design and adaptation
points.

## Installation

From the workspace root, with uv installed:

```bash
uv sync --package feather-train --extra training
```

Choose extras according to what you need:

| Installation | Includes |
| --- | --- |
| `feather-train` | E2B runtime and Harness integration |
| `feather-train[verifiers]` | Verifiers environment, grading, rendering, and Braintrust |
| `feather-train[tinker]` | Verifiers integration and Tinker sampling client |
| `feather-train[training]` | Tinker client plus the Cookbook training recipe |

The `training` extra installs the versions used by the reference recipe. Dependency
pins are in [pyproject.toml](pyproject.toml).

### Agent image dependencies

Bake `feather-harness[office]`, Bash, ripgrep, and Poppler into the agent container
before snapshotting. Install optional LibreOffice/OCR tools there when tasks need
them. Installing packages on the coordinator does not expose them to sandbox tools.
See the Harness [system dependency commands](../feather-harness/README.md#recommended-packages).

## Reference Tinker training recipe

Start with [examples/tinker.json](examples/tinker.json). Set your dataset and
immutable revision, choose a model available to your Tinker account, and configure
the credentials below.

```bash
uv run --package feather-train --extra training python -m feather_train.recipes.tinker.train \
  feather-train/examples/tinker.json
```

The example selects up to 32 tasks, with eight rollouts per task and two batches of
16 groups: up to 256 rollouts.

The JSON config controls the dataset, model, harness, rewards, and training schedule.
See [recipe configuration](recipe.md) for the main settings.

## Credentials

Set these in the process launching the recipe. The entrypoint does not load a `.env`
file itself. Credentials stay outside the JSON configuration.

| Variable | When needed | Purpose |
| --- | --- | --- |
| `TINKER_API_KEY` | Required | Tinker sampling and training. |
| `E2B_API_KEY` | Required | Create/resume the baked E2B sandboxes. |
| `HF_TOKEN` | Private/gated datasets or model tokenizers | Hugging Face access. |
| `BRAINTRUST_API_KEY` | Braintrust enabled (including the example config) | Rollout and grading traces. Disable with `harness.enable_braintrust=false` when unused. |
| `EXA_API_KEY` | Tasks with web tools | Host-side web search/fetch. |
| `OPENAI_VERIFIER_API_KEY` | Tasks whose grader uses an LLM judge | Judge provider credentials; separate from the policy model. |
| `OPENAI_VERIFIER_BASE_URL` | Judge requires a provider endpoint | OpenAI-compatible judge endpoint, unless `harness.grading.judge_base_url` is set. |
| `REDUCTO_API_KEY` | Task grader uses Reducto | Document processing during grading. |
| `WANDB_API_KEY` | W&B logging requested via `wandb_project` | Cookbook training metrics. |

Judge credentials are read from the launching process and forwarded to the verifier
container's grading command. Set `harness.grading.judge_model` or
`harness.grading.judge_base_url` to override the judge model or endpoint.
Braintrust project and experiment names use `harness.bt_project` and
`harness.bt_experiment_name`.

## Observability

Braintrust receives rollout/grading traces, `rollout_group_id`, the group's
`sampled_policy_version`, and final shaped rewards with penalty metrics. Policy
versions are run-local checkpoint steps. After each successful optimizer step,
`training_batch_id` links the consumed groups to that update. W&B training metrics
are optional. See [grading and rewards](grading.md) for reward calculation.

## Evaluation

For API-only evaluation on baked tasks, use `uv run --package feather-train --extra verifiers python -m feather_train.eval`
with its [EvalConfig](src/feather_train/eval.py). That path uses individual grading
and does not apply the group length penalty.
Set `sandbox_timeout_minutes` to match the training run's sandbox lifetime; it
defaults to 180 minutes and is separate from the per-request model timeout.

The standalone evaluation JSON accepts `profile: "prod"`; the training recipe
accepts `harness.profile: "prod"`.
This selects the standalone Feather production startup condition:
professional prompt and tool guidance, production compaction continuation,
response recovery, and the production skills and design assets in each task
workspace. The default remains `train` for training and historical evals.
`enable_image_tools` is independent and defaults to false; leave it false for a
text-only comparison. Braintrust rollout metadata records the selected profile.

`EvalConfig` also accepts `additional_instructions` and `compaction`. Direct
`load_environment` / `FeatherHarnessConfig` integrations additionally accept
`prompt_token_budget`, `enable_image_reads`, `model_request_retries`,
`blocked_egress_domains`, and `blocked_web_content_markers`. The default SDK retry
count is two. The existing optional `model_retry` outage gate remains available to
direct harness integrations; leaving it unset preserves the ordinary retry policy.

For local evidence capture on benchmark tasks, set `capture_deliverables: true`
and explicit `capture_paths`, for example `["/filesystem/deliverables"]`, in the
evaluation JSON. Capture runs before grading removes the agent and saves an
archive, final response, and manifest under the evaluation output directory.
It rejects runtime/private roots, does not follow symlinks, and fails instead of
silently truncating oversized outputs. Capture errors are recorded without
changing the grading result. Local capture needs no Modal dependency. Direct
harness integrations may opt into `capture_volume_name`; that mode requires the
caller to install Modal and mount the corresponding output volume.

Task worlds may declare Harbor shared or separate verifier modes, an absolute
working directory, collection hooks, artifact sources, and a verifier deadline.
They must already be baked with the corresponding images and tests. This package
resumes prepared snapshots; it does not provide a snapshot-baking command.
HTTP/SSE MCP descriptors preserve headers, response deadlines, and error policy
when their endpoint is rewritten for the sandbox. See [grading](grading.md).

## Development checks

From the workspace root:

```bash
uv run --package feather-train --extra training --extra dev pytest -c feather-train/pyproject.toml feather-train/tests -m "not docker" -q
uv run --package feather-train --extra dev ruff check feather-train/src feather-train/tests
```

For tool execution tests in Docker:

```bash
docker build -t feather-e2b-test -f feather-train/tests/Dockerfile feather-train/tests
uv run --package feather-train --extra training --extra dev pytest -c feather-train/pyproject.toml feather-train/tests -m docker -q
```

Omit `-m` to run both suites. Local and Docker tests need no cloud credentials.

## License

[MIT](LICENSE).

Standalone evaluation JSON accepts `enable_image_tools: true` for native workspace image inspection and MCP image results. This requires a vision-capable endpoint; it is disabled by default and managed training renderer clients reject image prompts before sampling.
