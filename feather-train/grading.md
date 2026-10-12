# Grading and rewards

Feather Train runs Paper Instruments' task graders after agent execution and before
world teardown. Factory-style worlds use a separate peer verifier. Official
Harbor worlds declare their verifier mode in the task template.

- With no `verifier.environment_mode`, the baked peer verifier grades after the
  solver container is removed and receives grading credentials at that point.
- `shared` copies baked tests into the agent container and grades there, using
  the declared working directory. This mode intentionally shares the solver's
  filesystem and process environment.
- `separate` runs declared collect hooks, copies artifacts from the agent or
  `world` service, removes the solver, and grades in a fresh baked tests-image
  verifier. It requires explicit artifact paths and a prepared tests image.

Factory tasks reject explicit shared or separate modes at task validation, before
execution. Their peer-verifier isolation and grading state contract are required.

Templates can set `verifier.timeout_sec` to override the grading deadline.
Declared trajectory artifacts are staged before collection when
`grading.stage_agent_trace` is enabled.

## Grader contract

- **Inputs:** task instructions and tests, plus output artifacts or the agent's
  execution trace as required by the task. Judge credentials are supplied at grading
  time.
- **Reward:** `/logs/verifier/reward.json` with a `reward` value takes precedence;
  otherwise `/logs/verifier/reward.txt` contains the numeric score. Invalid rewards
  fail without retry, and stale reward files are cleared before each attempt.
- **Details:** optional `reward_detail.json` enriches criterion reports and Braintrust
  traces. Ordinary read or report-conversion failures preserve the score and record
  the enrichment failure; permanent provider errors still propagate.

## Reward shaping

For rollout $i$, let $g_i$ be the grader score, $n_i$ the number of tool calls,
and $b_i$ the number of schema-invalid or unknown tool calls.

### Individual adjustments

The invalid-tool deduction is capped:

$$
p_i^{\mathrm{invalid}} = \min(\alpha b_i,\; c)
$$

The individually adjusted reward is:

$$
r_i =
\begin{cases}
0 & \text{if zero\_tool\_calls is enabled and } n_i = 0, \\
g_i - p_i^{\mathrm{invalid}} & \text{otherwise.}
\end{cases}
$$

| Symbol | Configuration field | Default |
| --- | --- | --- |
| $\alpha$ | `invalid_tool_call_penalty` | `0.0` |
| $c$ | `max_invalid_tool_call_penalty` | `1.0` |
| Zero-tool override | `zero_tool_calls` | `false` |

Ordinary tool execution errors are not included in $b_i$. API-only evaluation
ends at this individual reward.

### Group length penalties

For training, let $S$ contain the siblings with no rollout error. This includes
completed rollouts with a zero or negative reward. The shared scale is their mean
**after individual adjustments, before length penalties**, floored at zero:

$$
\mu = \max\left(0,\; \frac{1}{|S|}\sum_{j\in S} r_j\right)
$$

For each $i\in S$:

$$
p_i^{\mathrm{model}} = \mu\, w_m\, \frac{\max(M_i-A,\;0)}{L_m}
$$

$$
p_i^{\mathrm{tool}} = \mu\, w_t\, \frac{T_i}{L_t}
$$

$$
R_i = r_i - p_i^{\mathrm{model}} - p_i^{\mathrm{tool}}
$$

| Symbol | Meaning / configuration field |
| --- | --- |
| $M_i$ | Generated model tokens, including reasoning, tool calls, and compaction summaries. |
| $T_i$ | Rendered tool-result body tokens. Repeated context is counted once; final results left unsent at the turn limit are included. |
| $A$ | `output_token_free_allowance`, applied once per rollout; defaults to `0`. |
| $w_m$, $L_m$ | `num_output_tokens_weight`, `output_token_reference_length`. |
| $w_t$, $L_t$ | `num_returned_tool_tokens_weight`, `returned_tool_token_reference_length`. Tool weight defaults to `0`; its reference length is required when enabled. |

- Reference lengths scale the deduction; they are not token limits. At one reference
  length of charged tokens, the deduction is $\mu$ times the corresponding weight.
- A zero weight disables that component. If the group mean is nonpositive, both
  length deductions are zero. Length penalties are uncapped, and $R_i$ is not
  clamped to zero.
- Failed rollouts receive no length deduction and are excluded from training. If
  none remain, the reference recipe stops. The zero-tool override applies before
  length shaping, so it does not guarantee a final reward of zero during training.

**Example:** with $r_i=0.8$, $\mu=0.5$, $M_i=40{,}000$, $A=0$, $T_i=20{,}000$,
$L_m=L_t=40{,}000$, $w_m=0.075$, and $w_t=0.01$:

$$
p_i^{\mathrm{model}}=0.0375,\qquad
p_i^{\mathrm{tool}}=0.0025,\qquad
R_i=0.76
$$

## Retries

Connection recovery retrieves the same grader execution. A new grading attempt
starts only after the previous attempt has stopped and its reward files are cleared,
preventing overlapping writes. Exit code 2 and invalid rewards fail without a grading
retry; other nonzero exits and unavailable rewards follow the configured retry policy.
A retry cannot undo external judge requests already sent.

Implementation: [grader](src/feather_train/grading/grader.py),
[individual rewards](src/feather_train/grading/rewards.py),
[group length penalty](src/feather_train/grading/length_penalty.py).
