# Train and production profiles

Feather has one agent loop. `run_local` and the CLI select a bundle at startup;
there is no mid-run switching, completion critic, or artifact gate. The only
corrective continuation is `prod`'s single recovery from a cut-off or empty response.

| Setting | `train` | `prod` (default) |
| --- | --- | --- |
| System prose | Original minimal prompt | Professional knowledge-work prompt |
| Skills catalog | Existing `.agents` catalog | Same catalog entries |
| Format skills | Original technical manuals | Same manuals with professional guidance appended |
| Design assets | Not referenced (an earlier `prod` run may leave the files) | `.agents/design.md` and the `.agents/icons/` Lucide set, referenced by the prompt and format skills |
| Built-in tool descriptions | Original descriptions and schemas | Usage guidance for eight built-ins |
| Image tools, when no override is supplied | Disabled | Enabled |
| Compaction prompt, when none is supplied | Original prompt | Structured continuation prompt |
| Cut-off or empty model responses | Original handling | One recovery per run; interrupted tool calls fail, incomplete summaries retry |
| Compaction mechanics and execution | Shared loop and retry budget | Shared loop and retry budget |

## Selecting the bundle

```python
result = await run_local(task, model=model, workspace=workspace, profile="train")
result = await run_local(task, model=model, workspace=workspace)  # prod
```

CLI: `feather --profile train ...` or `feather --profile prod ...`.
JSON also supports `profile` and `enable_image_tools`:

```json
{
  "model": "YOUR_MODEL",
  "profile": "prod",
  "enable_image_tools": false
}
```

The example disables image tools for a text-only endpoint. An omitted or null
`enable_image_tools` follows the profile; an explicit boolean wins. No model-name
heuristic establishes image support. `train` with an explicit image override
enables image reads and MCP images, as in `prod`.

## Production skills

`prod` keeps one catalog and one skill per format. At install, each bundled
`docx`, `pptx`, `xlsx`, and `pdf` skill receives its `production_skills/<name>.md`
guidance appended after the trained technical manual. The skill's name, description,
and manual text are unchanged, so the catalog entries match training and the model
reads the guidance when it opens the skill it was trained to use. The shared house
style is one file, `.agents/design.md`; the prompt and each format skill point to it
for new or restyled deliverables. The pptx guidance also points to `.agents/icons/`, the
Lucide line set as 128 px PNGs with a tag index and a recolor helper. Neither is a catalog
entry, so neither appears in a training prompt. Each installed file fits the ordinary default `read` budget.

`run_local` reinstalls bundled skills at every start, replacing earlier copies:
`train` restores the original manuals and `prod` rewrites the combined skills and
design assets. A workspace reused across profiles or harness versions therefore always
carries the current run's files, and user edits to these harness-owned files are not
preserved. Installation still refuses symlinked skill paths. Task skills from
`skills_dir` are copied afterwards and still refuse to replace installed content.
Hosts that assemble Feather without `run_local` must install the matching profile's
skills in each task workspace at startup, or keep them in a directory they own.

A host that keeps skills outside the task workspace, such as a desktop app that must not
write into a user's project, calls `install_production_skill_tree(root)` with an absolute
directory. It installs the complete tree there: the bundled manuals with their references,
the production guidance, `design.md`, and the icons. The guidance names skill files by path,
so the installer rewrites every `.agents/` reference to `root`. The host then passes the
same path as `skills_path` to `build_production_instructions`, which rewrites the prompt's
references and catalog paths to match. A host whose image support matches neither
standard case, for example a text-only `read` beside its own image tool, supplies its own
`vision_fact` in place of the enabled or disabled sentence; `host_facts` still describes
the rest of its environment.

Native Feather Train E2B evals accept `profile: prod` in `env_args`; their
training-compatible default remains `train`. The E2B runtime archive carries the
same production guidance, installs it in the task workspace before rollout, and
uses the production prompt and tool behavior on the coordinator. Optional host capability statements describe differences in the execution environment.

Production changes tool descriptions without changing invocation contracts. `read`
uses distinct descriptions and parameter help for text-only and image-enabled runs;
image input is handled by `read` rather than a separate `view_image` tool.

## What the prose is intended to change

The [Ultramarine publication](https://www.paperinstruments.com/blog/ultramarine)
describes the deliberately minimal training posture. Production conditions that
policy with guidance for consulting, finance, accounting, law, and operations:
useful investigation, source-grounded calculations, scoped edits, and professional
deliverables without process narration, redundant inventories, or irrelevant additions.

The reviewed trace evidence motivated separating arithmetic checks from checking the
interpretation against the request, preserving unrelated workbook conventions, and
keeping investigation depth distinct from final-deliverable length. These are prompt
judgments, not runtime validators or measured guarantees of improved quality.

Format guidance includes an exact typography system, neutral colors, and proportional
checks. Client templates take precedence. Rendering is conditional on layout needs and
available tools; it is not a mandatory workflow. Reviewer rationale and qualification
notes are excluded from the model-facing resources. The ordinary image loader serves
both source evidence and rendered outputs, with the existing path, MIME, and size checks.

## Response recovery

`prod` enables `FeatherAgent(response_recovery=True)`; direct callers, including training,
keep it off. The adapter records each response's `finish_reason`. Before any tool runs, a
response is treated as failed when the provider cut it off (`length`, `error`, or
`content_filter`) or when it has no tool call and no visible text. A failed response with
tool calls stops the run without executing any of them, and a filtered response stops without
feedback. Otherwise the run appends one fixed feedback message and continues, once per run
across all compactions; a second failure stops with `ModelBehaviorError`, and no feedback is
added when no turn remains. Refusals keep their existing handling. A compaction response that
is empty retries up to four attempts; with response validation enabled, summaries cut off
by `length` or `error` also retry. Filtering and refusal fail immediately. Every failed
attempt remains in the trace and cannot replace the execution context. `RunResult.recoveries`
and the Harbor metadata record whether a recovery happened.

## Task-runner guidance

Production prose follows the Train PR 636 stack's task-runner adaptation. It omits
instructions that assume a live interaction channel, including clarification questions,
live status updates, and approval workflows. The default system prompt has no host
capability section. Hosts may explicitly supply `host_facts` and `vision_fact` (or
`enable_image_tools` to select the standard image sentence) when their capabilities
need to be described. Host-owned skill paths and opt-in subagent guidance remain supported.

Both profiles append `additional_instructions` in one final system section, expanding
`{{ workspace }}` only in that generic guidance. The task request itself stays literal.
`eval_prompt_output` saves the selected profile and effective parent prompt hash before
model execution, including subagent guidance when enabled.

## Compatibility boundary

`build_instructions()`, bare `FileTools`, the technical manuals, tool schemas, model
serialization, and the core runner keep their minimal training defaults. Native training
currently assembles those lower-level components directly; this change does not migrate
its prompt. In-repository editable workspace dependencies use the checked-out source:
unchanged version numbers do not isolate new builds from these changes. A caller upgrading `run_local` or the CLI
without selecting a profile intentionally receives `prod`, not historical behavior.
Select `train` explicitly when reproducing the training condition.

Harbor explicitly defaults to `prod`, with images still disabled unless requested.
It accepts `train` for reproduction and records profile, effective image setting, and
installed wheel SHA-256 in `harness-condition.json` and agent context metadata. Existing deployed immutable wheels are not changed by a source merge.

Compaction retains the original task, recent history, and todo handling. Optional
`trigger_tokens` selects an explicit compaction threshold, and
`exclude_unreplayed_reasoning` subtracts reasoning that the model will not see again
from the context estimate.
Its `max_tokens` summary ceiling defaults to 65,536 and can be configured independently
of the agent ceiling. Summaries inherit the agent's other sampling and provider settings;
a configured GLM token budget clamps output to remaining context room.
No approval/UI framework, sandbox provisioning,
external vision reviewer, application event framework, or tool-mutation retry is added.
Local commands retain the host's permissions; a profile is not a security boundary.
