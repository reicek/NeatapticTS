# Repo Cortex MCP Reliability (Repo Cortex - Layer 4)

**Status:** [DONE]

## Purpose

Layer 4 hardened the Repo Cortex MCP infrastructure before the Embeddings layer by adding
two Cortex-focused scout agents, the durable `repo-cortex-workflow` skill, a unified
`cortex-index.gate.mjs`, a session redirect CLI plus per-call workflow MCP `plan_path`
override support, a docs TSConfig precheck, and richer semantic-index diagnostics through
`validate-index.mjs` `fixHint` output and `build-index.mjs --json-health` summaries.
This archive is the durable reopen baseline between
[Semantic_Knowledge_Browser_Snapshot.plans.md](Semantic_Knowledge_Browser_Snapshot.plans.md)
and [Semantic_Knowledge_Embeddings.plans.md](Semantic_Knowledge_Embeddings.plans.md).

## Artifact summary

### New artifacts

| Artifact                      | Path                                                      | Notes                                                         |
| ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| Repo Cortex Scout agent       | `.github/agents/repo-cortex-scout.agent.md`               | Hidden Tier 3 specialist; `user-invocable: false`             |
| Cortex Embeddings Scout agent | `.github/agents/cortex-embeddings-scout.agent.md`         | Hidden Tier 3 specialist; `user-invocable: false`             |
| Repo Cortex workflow skill    | `.github/skills/repo-cortex-workflow/SKILL.md`            | Durable Cortex workflow: freshness, rebuild, MCP validation   |
| Cortex lifecycle gate         | `scripts/agent-customization/gates/cortex-index.gate.mjs` | Gate contract `{ pass, evidence, fixHint, owner }`            |
| Plan session redirect         | `scripts/agent-customization/plan-session-redirect.mjs`   | Per-session `plan_path` override for `neataptic-workflow-mcp` |
| Docs TSConfig precheck        | `scripts/agent-customization/validate-tsconfig-docs.mjs`  | Validates `tsconfig.docs.json` before docs pipeline runs      |

### Enhanced artifacts

| Artifact            | Path                                                         | Enhancement                                                    |
| ------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| Index validator     | `scripts/semantic-index/validate-index.mjs`                  | Add structured `fixHint` for stale, missing, over-age failures |
| Index builder       | `scripts/semantic-index/build-index.mjs`                     | Add `--json-health` flag: emit post-build summary JSON         |
| Workflow MCP server | `scripts/agent-customization/mcp/neataptic-workflow-mcp.mjs` | Add per-call `plan_path` override parameter                    |

### Supporting evals / inventory / gate registration

| Artifact            | Path                                                                     | Notes                                         |
| ------------------- | ------------------------------------------------------------------------ | --------------------------------------------- |
| Skill trigger evals | `.github/skills/repo-cortex-workflow/` (or evals folder)                 | should-trigger and should-not-trigger queries |
| Agent graph recheck | `validate-agent-graph.mjs` run post-add                                  | Confirm new agents pass tier graph            |
| Gate registration   | `scripts/agent-customization/gates/` catalog or `workflow-gap-audit.mjs` | `cortex-index.gate.mjs` visible in audit      |
| Inventory check     | `scripts/agent-customization/inventory-customizations.mjs --json`        | New skill and agents appear                   |

## Final state

- Repo Cortex Scout and Cortex Embeddings Scout shipped as hidden Tier 3 specialists.
- `repo-cortex-workflow` shipped as the durable Layer 4 workflow owner.
- `cortex-index.gate.mjs`, `plan-session-redirect.mjs`, and `validate-tsconfig-docs.mjs`
  now cover the lifecycle gate, runtime plan redirect, and docs precheck seams.
- `validate-index.mjs` now emits structured `fixHint` output, `build-index.mjs` now emits
  `--json-health` summaries, and `neataptic-workflow-mcp.mjs` now accepts validated
  per-call `plan_path` overrides.
- The fresh-process workflow MCP probe is green for both the default active binding and the
  per-call `plan_path` override path.
- The already-running in-editor workflow MCP runtime still needs a VS Code reload or MCP
  server restart to pick up the repaired server code; the repo-side implementation is complete.

## Final validation evidence

- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json` -> PASS with
  `ok: true`, `errors: 0`, `warnings: 0` after agent tool alias normalization.
- `node scripts/agent-customization/validate-agent-graph.mjs --json` -> PASS with `issues: []`.
- `node scripts/agent-customization/validate-skill-frontmatter.mjs --json` -> PASS with `issues: []`.
- `node scripts/agent-customization/inventory-customizations.mjs --json` -> PASS with
  `agents: 57`, `userInvocableAgents: 8`, and `skills: 51`; both scout agents and the
  `repo-cortex-workflow` skill are present.
- `node scripts/agent-customization/gates/cortex-index.gate.mjs --json` -> PASS with
  `schema_version: 1`, `index_documents: 849`, `index_fresh: true`,
  `corpus_mcp_alive: true`, `workflow_mcp_alive: true`, and `fixHint: null`.
- `node scripts/agent-customization/plan-session-redirect.mjs --plan=plans/Repo_Cortex_MCP_Reliability.plans.md --json`
  plus two `--clear` runs -> PASS with write, clear, and idempotent second-clear behavior.
- `node scripts/agent-customization/validate-tsconfig-docs.mjs --json` -> PASS with
  `missing_paths: []`.
- `node scripts/semantic-index/validate-index.mjs --json` failed after the archive move and
  again after the final archived tracker text update until the semantic index was rebuilt;
  the final pass returned `documents: 849`, `stale_paths: []`, and `fixHint: null`.
- `node scripts/semantic-index/build-index.mjs --json-health` -> PASS with `status: ok`,
  `total_documents: 849`, `removed_documents: 0`, and a valid `elapsed_ms` field on the
  final Step 07 freshness rebuild.
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Repo_Cortex_MCP_Reliability.plans.md`
  -> PASS before archival.
- `node scripts/agent-customization/gates/plan-sync.gate.mjs --json` -> PASS before archival.
- Focused Jest slices all passed for `scripts/semantic-index/validate-index`,
  `scripts/semantic-index/build-index`, and `scripts/agent-customization/gates/cortex-index`.
- `Test-Path plans/completed/Repo_Cortex_MCP_Reliability.plans.md` -> `True`.
- `Test-Path plans/completed/Repo_Cortex_MCP_Reliability.logs.md` -> `True`.
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json` -> PASS with
  the expected `log-completion-marker` standalone descriptor.
- `node scripts/agent-customization/validate-plan-sync.mjs --json` now aborts with
  `--plan=<path> is required`; Step 07 therefore used
  `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/mcp-active-binding.plans.md`
  -> PASS, plus direct README and Roadmap checks confirming the completed Layer 4 path.

## Reopen conditions

- Workflow MCP or corpus MCP health regresses.
- Per-call `plan_path` override behavior or workflow plan parsing regresses.
- Layer 5 or Layer 6 Repo Cortex work needs to amend the Layer 4 baseline rather than build on it.

## Audit log

- Durable completion notes now live in [Repo_Cortex_MCP_Reliability.logs.md](Repo_Cortex_MCP_Reliability.logs.md).
