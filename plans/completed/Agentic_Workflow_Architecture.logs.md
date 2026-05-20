# Agentic Workflow Architecture Log

**Status:** [DONE]

## Audit scope

- Objective: close the standalone meta-workflow lane after delivering the
  seven-phase user-invocable agent architecture, hidden specialist delegation,
  skill-first workflow knowledge, validation scripts, and the bounded MCP
  runtime-visibility ownership model.
- Coverage included `.github/agents/`, `.github/skills/`,
  `scripts/agent-customization/`, and tracker alignment across `plans/README.md`,
  `plans/Roadmap.md`, the archive index, and the final archived plan/log pair.

## Durable milestones

### [DONE] Workflow architecture foundation

- Registered the workstream, locked model and frontmatter contracts, built the
  customization inventory and validator scripts, and moved durable workflow
  policy into skills.
- Shipped exactly seven numbered user-invocable phase agents while keeping
  specialist agents hidden behind explicit parent allow-lists.

### [DONE] Bounded MCP runtime-visibility surface

- Classified runtime facts into `repo-static`, `direct-MCP`,
  `bridge-required`, and `manual until a documented API exists`.
- Shipped `neataptic-workflow-mcp` for repo-static workflow facts and
  `neataptic-validation-mcp` for active-step allow-listed shell-free
  validation commands only, while keeping bridge-required and manual-only
  facts outside the shipped surface.

### [DONE] Validation, documentation, and closure

- Validated the MCP surface with the inherited smoke suite and plan-sync
  checks, documented the shipped runtime-boundary story in
  `.github/skills/mcp-local-server-workflow/SKILL.md`, and confirmed in Phase
  11 Step 02 that no concrete repo-owned follow-up remained.
- Archived the compressed plan/log pair under `plans/completed/` and aligned
  `plans/README.md`, `plans/Roadmap.md`, and `plans/completed/README.md` to
  the terminal `[DONE]` state.

## Controls and evidence

- Active-workstream validation covered
  `validate-agent-frontmatter.mjs --strict`,
  `validate-skill-frontmatter.mjs --strict`,
  `validate-agent-graph.mjs`,
  `validate-plan-phase-packets.mjs`, and
  `validate-plan-sync.mjs`.
- Preserved final active-plan packet evidence from Phase 11 Step 05 before the
  archive move: `node scripts/agent-customization/validate-plan-phase-packets.mjs --plan=plans/Agentic_Workflow_Architecture.plans.md`
  and `node scripts/agent-customization/validate-plan-sync.mjs` both passed
  with `0 errors, 0 warnings`, and `git diff --check --
  plans/Agentic_Workflow_Architecture.plans.md` produced no output while the
  active plan path remained untracked.
- Final archive validation for Step 07 ran
  `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/completed/Agentic_Workflow_Architecture.plans.md`
  plus `git diff --check -- plans/README.md plans/Roadmap.md
  plans/completed/README.md
  plans/completed/Agentic_Workflow_Architecture.plans.md
  plans/completed/Agentic_Workflow_Architecture.logs.md`.

## Reopen triggers

- A future documented VS Code or Copilot API changes the current bridge or
  manual-only ownership boundaries.
- A new customization lane needs workflow-architecture changes that are not
  covered by the archived seven-phase agent, validator, and MCP ownership
  baseline.