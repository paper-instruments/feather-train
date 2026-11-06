# Ultramarine evaluation reference

Use the [Harbor configuration](../../feather-eval/examples/ultramarine.yaml) as
an explicit starting condition for knowledge-work evaluations. It selects the
production prompts, image tools, max reasoning, sampling, and context compaction
used in our S30 evaluations. It starts with the bundled file-tool smoke task,
one attempt, and one concurrent trial. The smoke task is an integration check,
not a scored benchmark or a test of long-context compaction.

## Run on Harbor

From the repository root, with Docker running, `PAPERINSTRUMENTS_API_KEY` set to your
Paper API key, and `EXA_API_KEY` set for web search and fetch:

```bash
uv sync --package feather-eval
uv build --package feather-harness --wheel --out-dir dist
uv run --package feather-eval harbor run --config feather-eval/examples/ultramarine.yaml
```

The YAML maps `PAPERINSTRUMENTS_API_KEY`, matching the
[API quickstart](https://docs.paperinstruments.com/docs/ultramarine), to the
`OPENAI_API_KEY` environment variable consumed by the adapter inside the sandbox.
It also forwards `EXA_API_KEY` for the enabled web tools; the task sandbox needs
egress to `api.exa.ai`.

Copy the YAML and replace `tasks` with your Harbor task paths, or use `datasets`
for a dataset directory or registered dataset. Its paths
are relative to the directory where you launch Harbor. Rebuild the wheel after
changing the harness, and retain the configuration, wheel, and its recorded hash
with each result. Increase concurrency only after checking your API limits and
available credit. The example disables whole-trial retries; model transport and
compaction-summary retries still follow the [harness retry policy](usage.md#configuration).

## Paper packages belong in the task image

The Harbor adapter installs the harness into a separate Python 3.12 environment,
plus shell utilities and LibreOffice. It installs the bundled Paper **skills**,
but does not install Paper Python packages into the task's interpreter. Selecting
`profile: prod` does not change the task's dependencies.

The bundled smoke task already has `paper-docx==0.2.0`, `paper-pptx==0.2.0`, and
`paper-xlsx==0.2.1`. For your own tasks, apply the
[task-image installation recipe](../../feather-eval/README.md#prepare-the-task-image)
after the task's existing dependencies. It replaces upstream `python-docx`,
`python-pptx`, and `openpyxl` with Paper, adds the document and script dependencies,
and runs the three Paper doctor commands. Keep the task's intended `python` on
`PATH`; installing packages only in the harness's isolated environment does not
make them available to task scripts. Task graders sharing that interpreter will
also use Paper, so record this environment change when comparing harnesses.

Before a full run, execute the doctor commands in the final task image, as the
task user with its normal `PATH`:

```bash
python --version
paper-docx-doctor
paper-pptx-doctor
paper-xlsx-doctor
```

See [recommended packages](../README.md#recommended-packages) for additional
rendering, OCR, font, and Python dependencies. The model needs an executable
Python environment and these packages in addition to the skills' instructions.

## Sampling and context

| Setting | Reference configuration |
| --- | --- |
| Profile | `prod` |
| Temperature / top-p | `1.0` / `0.95` |
| Reasoning effort | `max` |
| Image reads / web tools | Enabled / enabled |
| Harness context budget | 131,072 tokens |
| Compaction trigger | 101,072 estimated tokens |
| Recent turns retained | 2 |
| Execution / compaction output ceilings | 131,072 / 131,072 tokens |
| Turn limit | 500 |

`profile: prod` enables compaction by default in Harbor, the CLI, and `run_local`,
and selects the production compaction prompt; set `enabled: false` to turn it off.
`profile: train` compacts only when `enabled: true` is set. The example sets it
explicitly either way. Without an explicit `trigger_tokens`, the trigger is
`min(85% of context_window_tokens, context_window_tokens - 30000)`. The example
pins the resulting 101,072-token trigger. This is checked between model calls;
it is not a hard upper bound on each request's input.

`prompt_token_budget` uses the pinned GLM processor to count each outgoing prompt
and clamp requested output to the remaining context room. The adapter installs
the required `token-count` extra automatically. The first use needs network
access to download that processor. This 131K harness budget deliberately compacts
earlier than the public deployment's 262,144-token total window. Changing the
compaction window does not change the deployment's actual limit.

All output limits include reasoning and answer/tool tokens together. Execution
and summary limits are separate settings. Max reasoning can increase latency and can
still consume the full output allowance without a useful answer. Treat a change
to effort, compaction, or output limits as a separate evaluation condition.

## Relationship to historical evaluations

The S30 APEX showcase and Terminal-Bench 4.0 evaluations used the common condition
of `prod`, image tools, temperature `1`, top-p `0.95`, max reasoning, a 131,072-token
context budget, enabled compaction, and two retained turns. They configured
131,072-token execution and compaction output ceilings, with output clamped to
the remaining context room. Those were dedicated serving deployments and the
runs had different tasks, concurrency, and turn limits; there was no single
identical job configuration for every evaluation.

The public API example keeps the shared base's 131,072-token execution and
compaction ceilings and 500-turn limit. It starts at concurrency one and omits
the historical provider-specific `top_k: 0` extension. The historical
Terminal-Bench 4.0 run overrode the turn limit to unlimited; the shared base and
SpreadsheetBench configuration used 500. The example uses the public API and a
Docker smoke task rather than the historical dedicated endpoint and benchmark
sandboxes, so it is not an exact reproduction of a historical score. Configured
output allowances are also subject to the serving pool's enforced output ceiling.

For a controlled comparison, keep task IDs, sandbox contents, package versions,
prompts, sampling, compaction, output budgets, attempt counts, and scoring fixed.
Report model calls, transport retries, compactions, and failed trials separately.
Inspect `run.json`, `harness-condition.json`, `eval-prompt.json`, and the ATIF
segments; a successful smoke task alone does not establish benchmark quality.
