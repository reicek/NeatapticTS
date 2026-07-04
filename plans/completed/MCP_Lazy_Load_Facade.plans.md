# MCP Lazy-Load Facade

**Status:** [DONE]

## Scope

Reduce per-session MCP context tax by replacing the heavy `devtools` and `cortex` server registrations with lightweight lazy-load router facades that expose one tool each and spawn the real heavy servers only on first use. Preserve the existing tool-name surface and agent/skill routing.

## Final state

Phase 1 completed. All planned artifacts delivered and validated:

- `scripts/agent-customization/mcp/cortex-facade.mjs`, `devtools-facade.mjs`, and `lazy-facade-core.mjs` implement single router-tool facades.
- `scripts/agent-customization/mcp/cortex-tool-snapshot.json` and `devtools-tool-snapshot.json` advertise one minimal router tool each.
- `scripts/agent-customization/mcp/__tests__/lazy-facade.red.test.ts` passes 51/51 with 100% coverage on the touched facade files.
- `.mcp.json` and `.vscode/mcp.json` restored all 6 server registrations, with `cortex` and `devtools` as the lazy facades.
- All 57 agent frontmatter files, 9 skill files, `.github/copilot-instructions.md`, and `CLAUDE.md` migrated to `cortex({ operation, args })` / `devtools({ operation, args })` router envelope.
- `scripts/agent-customization/customization-utils.mjs` updated with `cortex` and `devtools` in `knownAgentTools`.
- `README.md` and `plans/mcp-active-binding.plans.md` wording corrected to name `.mcp.json` as the config authority.
- `scripts/agent-customization/gates/devtools-coverage.gate.mjs` and its test updated for the new `devtools` key.
- `.github/skills/devtools/SKILL.md` created/updated.
- Semantic index rebuilt.
- Detailed milestone evidence moved to `plans/completed/MCP_Lazy_Load_Facade.logs.md`.
- Plan and log pair archived to `plans/completed/`.

## Phase 1 — Deploy lightweight lazy-load MCP facades [DONE]

[DONE] Phase 1: implemented router-tool facades for `cortex` and `devtools`, migrated all callers, updated MCP configs, regenerated routing table and semantic index, and passed green validation. Detailed evidence moved to `plans/completed/MCP_Lazy_Load_Facade.logs.md`.

### Step 07 — Close and archive [DONE]

[DONE] Plan compressed to `plans/completed/MCP_Lazy_Load_Facade.logs.md`; closed plan and log pair archived to `plans/completed/`.

## Decision records

- DR-015: Tool facade chosen over hook/script alternatives.
- DR-016: Short server keys `cortex` / `devtools` selected.
- DR-017 (superseded): Wrapper approach initially chosen.
- DR-018: Router-tool approach adopted because wrapper snapshots still paid the per-session tool-list cost.

## Reopen conditions

- Reopen only if the facade contract, registration, or caller migration needs amendment.
- For new downstream consumers (e.g., a third facade), prefer a fresh plan that references this archive.
