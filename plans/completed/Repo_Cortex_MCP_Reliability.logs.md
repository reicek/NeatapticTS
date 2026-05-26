# Repo Cortex MCP Reliability Log

**Status:** [DONE]

## Audit scope

- Objective: harden Repo Cortex Layer 4 before embeddings by shipping the scout agents,
  the `repo-cortex-workflow` skill, the unified lifecycle gate, session redirect support,
  docs TSConfig precheck coverage, semantic-index `fixHint` and `--json-health` diagnostics,
  and the workflow MCP per-call `plan_path` override.
- Bounded implementation surface: `.github/agents/**`, `.github/skills/repo-cortex-workflow/**`,
  `scripts/agent-customization/**`, `scripts/semantic-index/**`, tracker files, and the learning log.
- Closure surface: `plans/README.md`, `plans/Roadmap.md`,
  `plans/completed/Repo_Cortex_MCP_Reliability.plans.md`,
  `plans/completed/Repo_Cortex_MCP_Reliability.logs.md`, and
  `.github/ai-learning/learning-log.jsonl`.

## Files changed

- `.github/agents/repo-cortex-scout.agent.md`
- `.github/agents/cortex-embeddings-scout.agent.md`
- `.github/skills/repo-cortex-workflow/SKILL.md`
- `.github/copilot-instructions.md`
- `.gitignore`
- `CLAUDE.md`
- `jest.config.mjs`
- `scripts/agent-customization/gates/cortex-index.gate.mjs`
- `scripts/agent-customization/gates/cortex-index.gate.test.ts`
- `scripts/agent-customization/mcp/mcp-utils.mjs`
- `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs`
- `scripts/agent-customization/plan-session-redirect.mjs`
- `scripts/agent-customization/validate-tsconfig-docs.mjs`
- `scripts/semantic-index/build-index.health.test.ts`
- `scripts/semantic-index/build-index.mjs`
- `scripts/semantic-index/validate-index.fixhint.test.ts`
- `scripts/semantic-index/validate-index.mjs`
- `data/semantic-index.sqlite`
- `plans/README.md`
- `plans/Roadmap.md`
- `plans/Repo_Cortex_MCP_Reliability.plans.md` -> archived as `plans/completed/Repo_Cortex_MCP_Reliability.plans.md`
- `plans/completed/Repo_Cortex_MCP_Reliability.logs.md`
- `.github/ai-learning/learning-log.jsonl`

## Gates passed

- `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` -> PASS with
  `pass: true`, `schema_version: 1`, `index_documents: 849`, `index_fresh: true`,
  `corpus_mcp_alive: true`, `workflow_mcp_alive: true`, and `fixHint: null`.
- `node scripts/agent-customization/gates/plan-sync.gate.mjs --json` -> PASS before archival.
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json` -> PASS with
  the expected standalone descriptor for 07-logging closure.

## Validation evidence summary

- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` -> PASS.
- `node scripts/agent-customization/validate-agent-graph.mjs --json` -> PASS.
- `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` -> PASS.
- `node scripts/agent-customization/inventory-customizations.mjs --json` -> PASS with the new
  agents and skill visible.
- `node scripts/agent-customization/plan-session-redirect.mjs --plan=plans/Repo_Cortex_MCP_Reliability.plans.md --json`
  plus two `--clear` runs -> PASS.
- `node scripts/agent-customization/validate-tsconfig-docs.mjs --json` -> PASS.
- `node scripts/semantic-index/validate-index.mjs --json` failed after the archive move and
  again after the final archived tracker text update until the semantic index was rebuilt;
  the final pass returned `documents: 849`, `stale_paths: []`, and `fixHint: null`.
- `node scripts/semantic-index/build-index.mjs --json-health` -> PASS with `status: ok`,
  `total_documents: 849`, `removed_documents: 0`, and a valid `elapsed_ms` field on the
  final Step 07 freshness rebuild.
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md`
  -> PASS before archival.
- `Test-Path plans/completed/Repo_Cortex_MCP_Reliability.plans.md` -> `True`.
- `Test-Path plans/completed/Repo_Cortex_MCP_Reliability.logs.md` -> `True`.
- `node scripts/agent-customization/validate-plan-sync.mjs --json` now exits 1 because the
  validator requires an explicit `--plan`; Step 07 therefore used
  `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md`
  -> PASS, and direct README or Roadmap checks confirmed the completed Layer 4 archive path.
- Focused Jest slices all passed for `scripts/semantic-index/validate-index`,
  `scripts/semantic-index/build-index`, and `scripts/agent-customization/gates/cortex-index`.

## Learning events

- Added one durable `agent-system-gap` event to `.github/ai-learning/learning-log.jsonl`.
- Recorded lesson 1: the fresh-process workflow MCP per-call `plan_path` override path lacked
  a focused red contract, which let the missing `requireString` import escape until Step 05.
- Recorded lesson 2: workflow plan parsing depended on the exact `## Validation gates`
  sentinel heading, so a temporary renamed heading blocked `[WIP]` phase detection.

## Residual risks

- The already-running in-editor workflow MCP runtime still needs a VS Code reload or MCP
  server restart to pick up the repaired `neataptic-workflow-mcp.mjs` code.
- Fresh-process workflow MCP probes are green, so the remaining risk is runtime pickup timing,
  not repo-side implementation correctness.
