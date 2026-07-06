# External Tool Assimilation — Repeatable Skill/Agent

**Status:** [DONE]

## Phase 1 — External Tool Assimilation

[DONE] Phase 1 completed and compressed.

- Step 01: Defined skill name (`external-tool-assimilation`), agent name (`assimilator`), tier (3), trigger phrase (`/assimilate <repo-url>`), comparison dimensions, and the full Step 02-07 packet sequence.
- Step 02: Surveyed existing fetch utilities and confirmed none fetch arbitrary public GitHub repo trees + files; documented fetch hierarchy (raw → contents API → tree API → git clone fallback) with rate-limit and 404 handling.
- Step 03: Authored deterministic red tests in `testing/assimilation/assimilate-repo.test.ts` with mocked fixtures; initial run failed for the right reason (stub functions threw `not implemented`).
- Step 04: Implemented `.github/skills/external-tool-assimilation/SKILL.md`, `.github/agents/assimilator.agent.md`, eight per-area templates under `.github/templates/assimilation/`, `scripts/assimilation/assimilate-repo.mjs`, and regenerated `.github/agent-skill-routing-table.md`.
- Step 05: Green validation passed; fixed portable main-module detection and synthesis filename (`${repo}.md`); sample assimilation of `github/choosealicense.com` produced the expected artifacts.
- Step 06: Added worked example and output contract to SKILL.md and agent file; updated canonical example attribution (`spec-kit/notes/license-attribution.md`, `spec-kit/verbatim/LICENSE`); appended `external-tool-assimilated` learning event.
- Step 07: Compressed Phase 1 history, created this log, and archived the plan/log pair to `plans/completed/`.

### Validation evidence

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=testing/assimilation/assimilate-repo.test.ts` → PASS (10/10 tests)
- `node scripts/agent-customization/validate-agent-frontmatter.mjs --json --agent .github/agents/assimilator.agent.md` → PASS (ok=true, warnings only for real tool aliases)
- `node scripts/agent-customization/validate-skill-frontmatter.mjs --json --strict` → PASS (0 errors, 0 warnings)
- `node scripts/agent-customization/validate-agent-graph.mjs --json` → PASS (0 errors, 0 warnings)
- `npm run agents:routing-table` / `npm run agents:routing-table:gate` → PASS (agents=66, skills=61)
- `npm run lint` → PASS
- `node scripts/assimilation/assimilate-repo.mjs https://github.com/github/choosealicense.com C:\NeatapticTS\tmp\green-sample-choosealicense` → PASS (149 files downloaded, 0 failures)
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md` → PASS
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/assimilate-repeatable-skill.plans.md` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=plan-sync` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=step-packet` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=agent-graph` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=routing-table-freshness` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=agent-quality` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=learning-event` → PASS
- `neataptic-gate-mcp:run_gate_check --gate=cortex-first-search` → PASS after rebuild
- `neataptic-gate-mcp:run_gate_check --gate=cortex-index` → FAIL (`workflow_mcp_alive: false`) due to stale workflow-MCP binding, not a content issue.

### Files changed

- `.github/skills/external-tool-assimilation/SKILL.md` — durable workflow: trigger, process, comparison dimensions, license rules.
- `.github/agents/assimilator.agent.md` — Tier-3 hidden specialist agent with structured-v1 output contract.
- `.github/templates/assimilation/00-overview.template.md` — overview summary template.
- `.github/templates/assimilation/01-sdd-workflow.template.md` — workflow/commands analysis template.
- `.github/templates/assimilation/02-validation-cycles.template.md` — validation/gate analysis template.
- `.github/templates/assimilation/03-planning-triage.template.md` — planning/triage analysis template.
- `.github/templates/assimilation/04-templates-artifacts.template.md` — templates/artifact analysis template.
- `.github/templates/assimilation/05-extensions-ecosystem.template.md` — extensions/ecosystem analysis template.
- `.github/templates/assimilation/06-iso42001-learning-overlap.template.md` — ISO/learning-event overlap template.
- `.github/templates/assimilation/07-verbatim-phrases-and-verbs.template.md` — verbatim phrase/verb adoption template.
- `.github/templates/assimilation/99-synthesis.template.md` — final condensed synthesis template.
- `scripts/assimilation/assimilate-repo.mjs` — ESM script that fetches tree/files and scaffolds the output folder.
- `scripts/assimilation/assimilate-repo.d.mts` — TypeScript declarations for the ESM exports.
- `testing/assimilation/assimilate-repo.test.ts` — deterministic Jest tests with mocked fixtures.
- `.github/agent-skill-routing-table.md` — regenerated after adding the new skill/agent.
- `spec-kit/notes/license-attribution.md` — MIT attribution for the canonical `github/spec-kit` example.
- `spec-kit/verbatim/LICENSE` — original `github/spec-kit` LICENSE.
- `.github/ai-learning/learning-log.jsonl` — `external-tool-assimilated` and skill/agent creation events appended.

### Risks / out-of-scope notes

- The `cortex-index` gate reported `workflow_mcp_alive: false` during Step 06 documentation checks. This is an environment-side stale workflow-MCP binding, not a code or documentation issue; resolving it requires restarting the workflow MCP server before the next Cortex-dependent session.
- Pre-existing `examples/racing_curriculum/` TypeScript errors remain unrelated to this workstream.

### Reopen conditions

- Reopen only if the `/assimilate` skill/agent contract, templates, or script need amendment.
- For new external repo studies, prefer a fresh plan that references this archive or invoke `/assimilate <repo-url>` directly.
