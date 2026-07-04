# Runtime Enforcement Contract

This document is the canonical contract for strict NeatapticTS orchestration
enforcement at runtime.

It defines what the hooks can prove today, which actions require repo-owned
runtime proof, how consecutive gate failures escalate, and where the
`manual-only` / `bridge-required` boundary still applies.

## Scope

The current strict runtime proof applies to **write** and **execute** actions:

- write: `apply_patch`, `edit`, `create`, `create_file`, `createFile`,
  `editFiles`, `replace_string_in_file`, `writeFile`, `vscode_renameSymbol`
- execute: `powershell`, `task`

Read/search actions still pass through the existing workflow/Cortex preflight,
but they do not currently require a per-action runtime proof carrier.

## Repo-owned runtime proof carrier

Because the CLI hook payload does not expose the full flow trigger and
delegation chain directly, the repo now uses a **repo-owned runtime enforcement
context carrier**.

Prepare it before the next strict write/execute action:

```bash
node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs \
  --prepare \
  --flow-id=<named-flow-id> \
  --agent=<current-agent> \
  --delegator-chain=<json-or-csv> \
  --required-skills=<json-or-csv> \
  --required-specialists=<json-or-csv> \
  --plan=<active-plan-path> \
  --tool-name=<expected-tool> \
  --action-class=<write|execute>
```

The carrier must provide:

- `flowId`
- `currentAgent`
- `delegatorChain`
- `requiredSkills`
- `requiredSpecialists`
- `planPath`
- `allowedActionClass`
- `expectedToolName`
- `actionId`
- `expiresAt`

The hook now returns an explicit `recoveryHint` for blocked strict actions. The
stored carrier `sessionId` is diagnostic; the live hook session is the source of
truth for matching the active action.

## Hook enforcement behavior

### Session start

`session-start-cortex-mcp-preflight.mjs` initializes the baseline carrier for
the current session in `data/hook-context-<session>.json`.

### PreToolUse

`pretool-workflow-cortex-preflight.mjs` now:

1. validates the runtime proof carrier for strict write/execute actions,
2. records `runtime-action-prepass` on success,
3. records `runtime-proof-mismatch` plus a gate exception on failure,
4. still runs workflow MCP self-check and Cortex-first-search checks.

When a strict action is blocked, the stderr output includes the recovery hint
and operators can preflight the carrier with:

```bash
node scripts/agent-customization/enforcement/runtime-enforcement-context.mjs --diagnose --tool-name=<tool> --plan=<active-plan> --session-id=<id>
```

### PostToolUse

`refresh-cortex-after-write.mjs` now:

1. re-validates the runtime proof carrier for strict write/execute actions,
2. records `runtime-action-postpass` on success,
3. clears the prepared one-shot action context,
4. still runs workflow sync hook-checks and Cortex refresh on the existing
   bounded surfaces.

## Learning-log event contract

The runtime enforcement path now adds structured events to
`.github/ai-learning/learning-log.jsonl`:

- `runtime-action-prepass`
- `runtime-action-postpass`
- `runtime-proof-mismatch`
- `gate-exception`
- `gate-escalation`

Those events are correlated by `sessionId` and `actionId` where applicable.

## Retry behavior (no consecutive-failure threshold)

`gate-exception-counter.mjs` supports durable counting from the learning
log for visibility and audit purposes only. Gate exceptions are recorded but
**do not trigger an automatic escalation** after a fixed number of failures.
Continue retrying the failing gate, workflow step, or validation until the
issue is fully resolved or a true technical limit is reached. The only valid
stopping condition is a documented theoretical or practical absolute, not an
arbitrary retry count. No concessions.

## Workflow-gap audit coverage

`workflow-gap-audit.mjs` now reasons about runtime enforcement evidence in
addition to gate failures and escalations. It reports:

- pre-action vs post-action pass coverage,
- runtime proof mismatches,
- missing post-action pairs,
- post-action passes without matching pre-action passes.

## Boundaries that remain intact

This runtime proof contract does **not** widen the archived architecture
boundary:

- no undocumented UI inspection,
- no direct inference of selected active agent or model state,
- no promotion of `manual-only` or `bridge-required` facts into fake direct-MCP
  claims,
- no shell-generalized MCP execution.

The carrier proves **repo-owned orchestration intent**, not unsupported client
state.
