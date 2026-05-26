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

### [DONE] Post-closure Tier-0 structured-v1 maintenance

- Tightened the eight numbered Tier-0 agent prompt templates so field
  position is explicit in-context, including the requirement that
  `FILES_CHANGED` appear immediately before `KEY_FINDINGS` even when one or
  both values are `NONE`.
- Removed the output-validator length gate that had suppressed exact-order
  diagnostics when a required Tier-0 field was missing, so missing-field and
  shifted-order failures now report together.
- Restored `.github/agents/docs-example-writer.agent.md` to the Tier 4
  auxiliary contract after a structured-v1 follow-up drifted the agent to the
  Tier 3 scout shape; preserved the mandatory sections and structured-v1 block
  by converting the section and field layout back to the Tier 4 auxiliary form.

### [DONE] Agent-quality contract closure slice

- Added `.github/AGENT_QUALITY_CONTRACT.md`,
  `scripts/agent-customization/validate-agent-quality.mjs`, and
  `scripts/agent-customization/gates/agent-quality.gate.mjs`, then registered
  the new gate in `scripts/agent-customization/mcp/neataptic-gate-mcp.mjs`.
- Fixed the strict-tier coordinator path omission in
  `scripts/agent-customization/validate-agent-frontmatter.mjs`, restored
  `.github/agents/docs-example-writer.agent.md` to Tier 4, and normalized all
  57 `.agent.md` files to tier-appropriate `structured-v1` `Output Format`
  blocks.
- Current validation state: `validate-agent-quality.mjs --json`,
  `agent-quality.gate.mjs --json`, `neataptic-gate-mcp.mjs --self-check`, and
  `npx tsc --noEmit -p tsconfig.json` are green; `validate-agent-graph.mjs`
  and `validate-agent-frontmatter.mjs --strict` remain red only because of 82
  pre-existing unknown-subagent naming mismatches across 27 unique title-form
  names.

## Controls and evidence

- Active-workstream validation covered
  `validate-agent-frontmatter.mjs --strict`,
  `validate-skill-frontmatter.mjs --strict`,
  `validate-agent-graph.mjs`,
  `validate-plan-phase-packets.mjs`, and
  `validate-plan-sync.mjs`.
- Follow-up green rerun for this maintenance slice should start with
  `node scripts/agent-customization/validate-agent-graph.mjs --json` and
  `node scripts/agent-customization/validate-agent-quality.mjs --json`.
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
