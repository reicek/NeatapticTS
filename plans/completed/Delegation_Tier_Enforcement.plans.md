# Delegation Tier Enforcement (Agentic Workflow Enforcement Prerequisite)

**Status:** [DONE]

## Scope

- Define and enforce the 5-layer agent delegation tier graph across the repo's custom agent surface.
- Keep the closed baseline limited to agent frontmatter, customization validation scripts, gate MCP tooling, operator docs, and the tracker alignment needed to unblock Repo Cortex Layer 4.
- Preserve this archive as the reopen point for future changes to tier rules, tier reporting, or tier-aware MCP discovery.

## Final state

- All 55 `.github/agents/*.agent.md` files now carry YAML `tier:` frontmatter with the enforced distribution Tier 1 = 8, Tier 2 = 10, Tier 3 = 33, Tier 4 = 4.
- `scripts/agent-customization/tier-graph-utils.mjs`, `tier-inventory.mjs`, `validate-agent-graph.mjs`, `gates/tier-enforcement-gate.mjs`, `tier-audit-report.mjs`, and `mcp/cortex-tier-tool.mjs` now define and expose the tier policy; `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs` serves `query_tier_graph` on the existing gate MCP surface.
- `.github/copilot-instructions.md` and `scripts/agent-customization/README.md` now document the tier graph, operator commands, and enforcement boundaries.
- Repo Cortex Layer 4 is no longer blocked on tier-graph formalization; the archived contract is ready for future hidden-agent additions to validate against.

## Audit summary

- Step 02 [DONE] completed the 55-agent inventory with zero preexisting `user-invocable` rule violations and identified `validate-agent-graph.mjs` as the extension point.
- Step 03 [DONE] added four red contracts for non-Tier-1 `user-invocable`, illegal upward delegation, and the tier-enforcement gate behavior in the `agent-customization-scripts` Jest project.
- Step 04 [DONE] implemented the tier inventory, validator, gate, report, MCP tool wiring, and YAML `tier:` frontmatter on all 55 agent files, including removal of the `implementation-pattern-coordinator -> solid-split` same-tier delegation edge.
- Step 05 [DONE] and Step 06 [DONE] preserved clean repo-local validation and documented the final policy surface in `.github/copilot-instructions.md` plus `scripts/agent-customization/README.md`.
- Step 07 [DONE] closed the workstream by adding the compressed same-boundary log, aligning `plans/README.md`, `plans/Roadmap.md`, `plans/completed/README.md`, and `plans/Repo_Cortex_MCP_Reliability.plans.md`, then archiving this plan under `plans/completed/`.
- Repo-local validation is complete. The only remaining external operator caveat is that live VS Code discovery of `query_tier_graph` may still require restarting `neataptic-gate-mcp` or reloading VS Code.

## Reopen conditions

- The tier graph changes, including any new tier, delegation rule, or `user-invocable` policy change.
- A future agent-customization lane needs to expand `validate-agent-graph.mjs`, tier inventory reporting, or the gate MCP `query_tier_graph` surface.
- VS Code or MCP client behavior changes enough that the current external discovery caveat needs a repo-owned fix rather than an operator restart or reload.

## Audit log

- See [Delegation_Tier_Enforcement.logs.md](Delegation_Tier_Enforcement.logs.md).
