# Agentic Workflow Architecture

**Status:** [DONE]

## Scope

- Refactor NeatapticTS agent customizations into a phase-driven, skill-first
  workflow with exactly seven numbered user-invocable agents.
- Keep specialist agents hidden behind explicit allow-lists, validate the
  customization graph mechanically, and preserve durable tracker-first working
  rules.
- Close the bounded MCP runtime-visibility lane without inferring unsupported
  client state or widening into unrelated runtime behavior.

## Final state

- The repo now exposes exactly seven numbered user-invocable phase agents,
  hidden specialists, durable workflow skills, qualified model-routing rules,
  plan-driven phase packets, and repo-side validators for frontmatter,
  delegation, and plan sync.
- The approved MCP ownership model is locked: `neataptic-workflow-mcp` owns
  `repo-static` workflow facts, `neataptic-validation-mcp` owns active-step
  allow-listed `direct-MCP` validation commands, `neataptic-vscode-bridge`
  remains the provisional `bridge-required` owner for model snapshots or hook
  observations only, and selected active agent, the full live agent list,
  tool-picker state, and the current selected model UI state remain `manual
  until a documented API exists`.
- Internal MCP workflow documentation is aligned at
  `.github/skills/mcp-local-server-workflow/SKILL.md`, and no public or
  generated documentation surface required closure-time widening.
- Phase 11 closed the workstream because no concrete repo-owned MCP runtime-
  visibility follow-up remained beyond future documented APIs, bridge-required
  observations, or manual-only client-state boundaries.

## Audit summary

- Phases 1-5 delivered the agent, skill, validator, model-routing, and
  delegation baseline that now governs repo customization work.
- Phases 6-10 classified, implemented, validated, and documented the bounded
  MCP runtime-visibility surface without introducing shell-generalized MCP
  execution or unsupported client-state inference.
- Phase 11 used the Step 02 closure-ready decision and the Step 05 active-plan
  validation evidence to archive the lane, add the same-boundary log, align
  `plans/README.md` and `plans/Roadmap.md`, and remove the stale active-plan
  `Handoff query` instead of leaving a false `[WIP]` frontier behind.
- Step 05 remains the final active-plan packet gate preserved for this archive:
  `node scripts/agent-customization/validate-plan-phase-packets.mjs --plan=plans/Agentic_Workflow_Architecture.plans.md`
  and `node scripts/agent-customization/validate-plan-sync.mjs` both passed
  with `0 errors, 0 warnings`, and `git diff --check --
  plans/Agentic_Workflow_Architecture.plans.md` produced no output while the
  active tracker still lived in `plans/` as an untracked file.

## Reopen conditions

- Microsoft exposes a documented inspection or bridge API that changes the
  current `manual until a documented API exists` boundary.
- A future customization request needs new repo-owned workflow architecture,
  model-routing, validation, or MCP runtime-visibility behavior beyond this
  archived baseline.
- The approved ownership labels or allow-list rules drift away from the
  archived `repo-static`, `direct-MCP`, `bridge-required`, and manual-only
  contract.

## Audit log

- See [Agentic_Workflow_Architecture.logs.md](Agentic_Workflow_Architecture.logs.md).