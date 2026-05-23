---
name: repo-cortex-workflow
description: 'Use when: checking Repo Cortex freshness, sequencing semantic-index rebuilds, regenerating the semantic snapshot, invoking Cortex lifecycle validation, or resolving neataptic-workflow-mcp binding and override questions.'
argument-hint: 'Describe the Cortex health issue, rebuild trigger, or MCP binding question.'
user-invocable: false
disable-model-invocation: false
---

# Repo Cortex Workflow

Use this skill when work needs a durable Repo Cortex maintenance sequence rather
than ad hoc rebuild guesses.

## Workflow

1. Start with the smallest freshness check that can explain the symptom.
   Prefer `node scripts/semantic-index/validate-index.mjs --json` to identify
   stale, missing, or over-age paths before deciding on a rebuild.
2. Rebuild only when freshness evidence says the corpus is stale or incomplete.
   Use `node scripts/semantic-index/build-index.mjs`, then prefer
   `node scripts/semantic-index/build-index.mjs --json-health` when that health
   summary surface is available.
3. Treat `docs/assets/semantic-snapshot.json` as generated output. When Repo
   Cortex data changed and snapshot consumers need fresh search state, run
   `npm run docs` from the source-of-truth pipeline instead of editing the
   published snapshot directly.
4. Prefer a single gate result over manual multi-command interpretation when the
   unified Cortex lifecycle gate exists. Run
   `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` to
   confirm index freshness, MCP reachability, and snapshot currency together.
5. For workflow MCP binding issues, prefer a per-call `plan_path` override when
   that surface is available. If the session needs a temporary plan redirect,
   use the dedicated redirect helper, confirm the written override, and retry the
   failing MCP call against the intended plan.
6. Rerun the same narrow health check after each corrective action so the next
   step is evidence-based rather than cumulative guesswork.

## Command Sequence

- Freshness status: `node scripts/semantic-index/validate-index.mjs --json`
- Rebuild: `node scripts/semantic-index/build-index.mjs`
- Post-build health summary when available:
  `node scripts/semantic-index/build-index.mjs --json-health`
- Snapshot regeneration: `npm run docs`
- Unified gate when available:
  `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`
- Session redirect when available:
  `node scripts/agent-customization/plan-session-redirect.mjs --plan=<plan> --json`

## Guardrails

- Do not hand-edit generated snapshot artifacts.
- Do not use SQLite file modification time as a freshness proxy when index
  content timestamps are available.
- Do not treat MCP binding failures as proof that the corpus needs a rebuild.
- Keep plan-path overrides inside the repository `plans/` tree.