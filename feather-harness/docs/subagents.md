# Parallel subagents

Delegation is optional. Existing library calls and CLI configurations do not expose
agent tools or change instructions unless enabled. The local runner supports it in
`prod`; enabling it in `train` is rejected before starting tools.

```json
{
  "profile": "prod",
  "subagents": {"max_concurrent": 3, "max_agents": 8, "max_turns": 50}
}
```

These fields limit simultaneous active assignments, total children created while the
manager is open, and each child's cumulative model turns across follow-ups. They are not
a dollar budget: context length, compaction calls, and provider retries also cost
tokens. The caller still owns inference timeouts and retry policy.

## Model-facing behavior

- `spawn_agent(prompt)` returns an ID immediately. The assignment must contain the
  necessary question, constraints, and source paths. Parent conversation history is
  not automatically copied.
- `wait_agent(agent_ids, timeout)` returns selected statuses and their latest answers.
  It returns as soon as any selected child finishes, waiting up to 120 seconds by default
  and 600 at most, so a long child costs the parent few turns. A timeout does not cancel
  children, and cancelling the parent run still interrupts a wait.
  A completion already returned by a wait does not interrupt the next wait for remaining
  children. New follow-up results still return immediately. Zero timeout inspects immediately,
  and a selection with no remaining work returns immediately even if its reports were seen.
- `followup_agent(agent_id, prompt)` continues a successfully completed child's own
  context. It does not reset the lifetime turn limit. Running, failed, cancelled,
  exhausted, or explicitly stopped sessions cannot be continued.
- `stop_agent(agent_id)` stops that child and waits for resource cleanup.
- `list_agents()` lists this parent's children. IDs from another parent are not valid.

The parent remains responsible for incorporating results. Children cannot delegate.
A failed child is reported to the parent without stopping its siblings. Normal parent
completion, failure, and cancellation all close child sessions; unfinished assignments
are cancelled, not silently detached to continue spending or editing.

Failed or turn-exhausted activations return a labelled partial execution excerpt, not a
successful answer. It contains up to four recent assistant/tool text messages, capped at
1,000 characters each, and the local saved-result path. Tool-only turns can therefore
surface useful work. Compaction summaries, reasoning fields, and image data are excluded.
No extra model call or tool execution is made, and the complete saved trace is unchanged.

The existing loop still executes ordinary tools sequentially. Parallelism comes from
independent child runs, not from concurrently executing every tool in an assistant turn.

## Context and workspace ownership

Children have separate histories, todo lists, file-read revision state, shell workers,
MCP connections, and compaction transcripts. Completed children keep these resources
until closed, allowing follow-ups to use their own earlier work and background tasks.
Follow-up uses the active, possibly compacted context, not the expanded archival trace.

The filesystem is shared. Only one agent should write a given file at a time, and each
preserves other agents' edits. This is coordination guidance, not an atomic write lock or an OS sandbox.
For a shared workbook or document, parallel research or separate draft fragments are
usually easier to coordinate than concurrent writes to the same final file.

Local child traces live under `.feather/agents/<agent_id>/`, with a separate result for
each completed or failed activation, a compaction transcript when needed, and shell
logs. These records may contain source data and model output; treat them like the
parent trace. In-memory child sessions are not a crash-resumable job service.

## Embedding in an application

`Subagents` accepts a `run_child(agent_id, serve)` coroutine. The host creates a fresh
`FeatherAgent` and awaits `serve(agent)` inside that child's resource scopes. Serve
keeps the session alive between assignments. Keep MCP connection entry and exit in
their owning task, using `run_with_mcp` as the local runner does.

The host supplies its authenticated model adapter and sandboxed tools. Do not replace
an application's sandbox with `LocalShellRuntime`, reuse a parent's mutable tool
instances, or pass delegation tools to a child. A stateless API client may be shared;
streaming adapters must attribute each child's events to that child.

Register `subagent_tools(manager)` on the parent, append `PARENT_GUIDANCE` only when
those tools are present, and include `manager.compaction_state()` alongside existing
tool state. Close the manager before closing parent-owned clients. `save_result` can
persist each activation's `RunResult`; `results(agent_id)` exposes those same records.
The sink may return a readable result path to include with partial diagnostics, or `None`.

## Children that outlive the parent's reply

The local runner closes children when the parent run ends. An application can instead
keep one `Subagents` manager open across parent runs, so the parent may reply while
children keep working and continue when their reports arrive. The manager's lifetime is
the host's `async with` block. Nothing else changes about how children run.

Each finished activation is reported to the parent once. `wait_agent` reports what it
returns. `take_unreported()` returns every other finished activation and marks it
reported, `has_unreported()` checks without taking, and `wait_for_unreported()` blocks
until one exists. A stop the parent asked for
is not reported back. The host delivers unreported results as the next parent input with
`render_subagent_reports(statuses)`, which labels them as subagent reports rather than a
message from the person.

A host that does this should:

- append `BACKGROUND_PARENT_GUIDANCE` instead of `PARENT_GUIDANCE`, so the parent knows
  it may reply before children finish and that their reports arrive in a later message;
- take unreported results at the start of every parent run, including runs a new user
  message starts, and never run two parent runs at once;
- keep the children's model clients and sandbox processes alive between parent runs,
  and close the manager when the conversation ends.

This package does not implement desktop UI, approval dialogs, credential storage,
cross-process resume, peer messaging, nested teams, or automatic conflict resolution.

Children inherit the run's additional system guidance and optional web provider. The
parent and children keep separate file revision state and transcripts, while sharing
the configured provider. Prompt metadata records the parent's effective system prompt,
including delegation guidance and limits.
