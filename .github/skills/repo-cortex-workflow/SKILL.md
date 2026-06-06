---
name: repo-cortex-workflow
description: 'Use when maintaining Repo Cortex health: checking semantic-index freshness, choosing incremental versus forced rebuilds, regenerating the browser snapshot, running Cortex lifecycle gates, recovering from workflow or corpus MCP drift, or packaging durable Cortex evidence for hooks and CI. Use this workflow instead of ad hoc rebuild guesses.'
argument-hint: 'Describe the Cortex symptom, failing command or gate, known stale paths or plan binding, whether snapshot or dense search is in scope, and the desired green proof.'
user-invocable: false
disable-model-invocation: false
---

# Repo Cortex Workflow

Use this skill for Repo Cortex lifecycle maintenance only: semantic index
freshness, browser snapshot currency, dense-search readiness, unified Cortex
gates, and workflow or corpus MCP binding checks. It does not own
corpus-changing source edits — use it after those edits land, or when
maintenance surfaces drift red on their own.

## When to Use

- `validate-index.mjs` reports `stale_paths`, `missing_paths`, or
  `over_age_paths`.
- Only part of the corpus appears stale after localized changes and you need the
  narrowest safe repair.
- `docs/assets/semantic-snapshot.json` is older than the index or browser
  consumers are serving stale search state.
- `cortex-index.gate.mjs` fails and you need to isolate whether the index,
  snapshot, corpus MCP, or workflow MCP is red.
- `build-index.mjs --json-health` is unavailable, returns `status: "error"`, or
  disagrees with follow-up validation.
- Workflow MCP calls fail because the active plan binding or session override is
  wrong.
- Session-start or post-write automation needs a predictable Cortex maintenance
  packet with machine-readable evidence.

## Task Packet

Pass a compact packet with the failing surface, any known scope, and the proof
you want at the end.

```text
Use repo-cortex-workflow for localized semantic-index drift.
Symptom: validate-index reports stale_paths after edits under .github/skills/.
Failing command: node scripts/semantic-index/validate-index.mjs --json
Known scope: only skill docs changed; browser snapshot should stay current.
Desired end state: validate-index passes, snapshot is current, and cortex-index gate returns {pass: true}.
```

## Required Workflow

1. Start with the narrowest classifier:
   `node scripts/semantic-index/validate-index.mjs --json`
   Treat `stale_paths`, `missing_paths`, and `over_age_paths` as the decision
   surface.
2. Choose the smallest safe repair:
   - **Over-age only**: run `npm run index:session-start` to touch still-fresh
     rows and rebuild only changed or new corpus files.
   - **Stale or missing corpus rows**: run
     `node scripts/semantic-index/build-index.mjs --json`. The builder is
     already incremental; it reindexes changed or new files and skips unchanged
     rows.
   - **Suspected freshness-proof corruption or chunker drift**: rerun
     `node scripts/semantic-index/build-index.mjs --force --json` only after a
     normal incremental build fails to clear the same paths.
   - **Snapshot only**: run `npm run index:build-snapshot`. Use `npm run docs`
     only when broader docs outputs also need regeneration.
   - **Dense-search readiness after corpus change**: run `npm run index:prewarm`
     after the index is healthy.
3. Prefer a post-build health summary when available:
   `node scripts/semantic-index/build-index.mjs --json-health`
   If that returns `status: "error"` or is unavailable on the checkout, fall
   back to the builder's `--json` summary plus a fresh
   `validate-index.mjs --json` pass.
4. Run the unified lifecycle gate for final proof:
   `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
   Read its evidence as four separate surfaces: `index_fresh`,
   `corpus_mcp_alive`, `workflow_mcp_alive`, and snapshot currency.
5. For workflow MCP binding drift, prefer the least sticky override first:
   - Use a per-call `plan_path` override when the calling tool supports it.
   - If the whole session is pointed at the wrong plan, run
     `node scripts/agent-customization/plan-session-redirect.mjs --plan=<plan> --json`,
     confirm the returned `plan_path`, then rerun the failing self-check or
     gate.
6. After each repair, rerun the same narrow check before escalating. Do not
   chain multiple fixes on guesswork.

## Command Sequence

- Freshness status: `node scripts/semantic-index/validate-index.mjs --json`
- Incremental rebuild: `node scripts/semantic-index/build-index.mjs --json`
- Forced rebuild when evidence says normal freshness proofs are untrustworthy:
  `node scripts/semantic-index/build-index.mjs --force --json`
- Post-build health summary when available:
  `node scripts/semantic-index/build-index.mjs --json-health`
- Session-start partial refresh: `npm run index:session-start`
- Snapshot regeneration: `npm run index:build-snapshot`
- Dense-search prewarm: `npm run index:prewarm`
- Unified gate when available:
  `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
- Corpus MCP fallback:
  `node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json`
- Workflow MCP self-check fallback:
  `node scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs --plan=<plan> --self-check --json`
- Session redirect when available:
  `node scripts/agent-customization/plan-session-redirect.mjs --plan=<plan> --json`

## Failure Recovery and Edge Cases

- **Database missing or empty**: treat that as a build prerequisite failure. Run
  `node scripts/semantic-index/build-index.mjs --json` before any snapshot or
  MCP checks.
- **Builder exits non-zero**: keep stdout or stderr as evidence, stop the
  sequence, and do not run snapshot regeneration or gates as if the index were
  healthy.
- **`--json-health` says `status: "error"`**: trust the failure; rerun
  `build-index.mjs --json` or `validate-index.mjs --json` to capture more
  detail.
- **Ambiguous gate result**: if the unified gate fails but `validate-index`
  passes, isolate the failing surface with:
  - `node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json`
  - `node scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs --plan=<plan> --self-check --json`
  - `npm run index:build-snapshot` when snapshot currency is the only red
    surface.
- **Gate script unavailable on the checkout**: fall back to the three
  underlying checks above and report that the unified gate surface is missing,
  not merely failing.
- **Snapshot regeneration fails or produces unreadable JSON**: report it as a
  snapshot-specific failure; do not collapse it into an index-freshness failure
  if `validate-index` is green.
- **Health summary surface unavailable**: prefer machine-readable `--json`
  outputs and keep the fallback explicit in the final evidence.

## Partial Rebuild Decision Rules

- Do not invent manual per-path SQLite surgery for a small stale subset.
  `build-index.mjs` is already subset-aware through freshness proofs and is the
  correct targeted rebuild tool.
- Use `npm run index:session-start` when rows are merely over-age and the
  on-disk corpus still matches the stored proof.
- Reserve `--force` for suspected freshness-proof corruption, schema drift, or
  parser changes that make "unchanged" rows untrustworthy.
- Rebuild the browser snapshot only after the index is green, because the
  snapshot is downstream of the SQLite corpus.

## Persistent-Issue Escalation and Feedback

When the same Cortex issue survives one full classify → repair → revalidate
loop, surface a compact incident summary instead of repeating the same command
sequence.

Include:

- failing command or gate and exit code,
- latest `fixHint`,
- unresolved `stale_paths`, `missing_paths`, or gate evidence fields,
- active plan binding source (per-call `plan_path`, session override, or
  default startup plan),
- whether snapshot and dense-search follow-up steps were intentionally skipped.

Escalate to `00-helping` when:

- a required script or gate is missing from the checkout,
- workflow or corpus MCP self-checks keep failing after the binding is
  corrected,
- the same gate fails three consecutive times in the same session,
- manual host intervention is required to restart an MCP server or repair local
  toolchain state.

## Audit and Rollback Evidence

Capture durable before/after evidence for every non-trivial Cortex repair:

- `validate-index.mjs --json` output before and after the repair,
- builder `--json` or `--json-health` summary,
- `cortex-index.gate.mjs --json` result when available,
- any session redirect payload showing the written `plan_path`,
- snapshot regeneration summary when `docs/assets/semantic-snapshot.json`
  changed.

When a binding change is temporary, also record the rollback command
(`plan-session-redirect.mjs --clear` or the previous `--plan=...`) so the
session can be restored without guessing.

## Automation Integration

Prefer existing automation surfaces over bespoke shell glue:

- **Session-start maintenance**: `npm run index:session-start`, or the repo hook
  `scripts/agent-customization/hooks/session-start-cortex-mcp-preflight.mjs`
  when the environment supports SessionStart hooks.
- **Post-write refresh**:
  `scripts/agent-customization/hooks/refresh-cortex-after-write.mjs` already
  chains build-index, build-browser-snapshot, dense prewarm, and the Cortex gate
  after file writes.
- **CI or maintenance gates**: run
  `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` after
  corpus-changing source, plan, agent, or skill edits when the pipeline needs a
  single pass or fail Cortex verdict.
- **Dense-search consumers**: pair corpus rebuilds with `npm run index:prewarm`
  only in environments that actually need warm dense search.

## Guardrails

- Do not hand-edit `data/semantic-index.sqlite` or
  `docs/assets/semantic-snapshot.json`.
- Do not use SQLite file modification time as a freshness proxy when index
  content timestamps are available.
- Do not treat workflow MCP binding failures as proof that the corpus needs a
  rebuild.
- Keep plan-path overrides inside the repository `plans/` tree and prefer the
  least sticky override that solves the problem.
- Do not skip classification and jump straight to `--force`; normal incremental
  rebuilds are the default.
- Do not regenerate the snapshot from a known-stale index.
- Do not report a Cortex issue as fixed until the same check that was failing
  has been rerun successfully.

## Expected Final Output

A strong Cortex maintenance pass should report:

- the initial classifier output and what it proved,
- the exact repair command chosen and why it was the smallest safe action,
- any fallback used because `--json-health`, the unified gate, or an MCP surface
  was unavailable,
- the final gate or self-check evidence,
- any remaining manual step or escalation owner.
