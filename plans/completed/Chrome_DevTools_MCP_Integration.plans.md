# Chrome DevTools MCP Integration

**Status:** [DONE]

## Scope

Integrate the Chrome DevTools MCP server (https://github.com/ChromeDevTools/chrome-devtools-mcp)
into the NeatapticTS multi-tier agent orchestration system to enable direct performance
measurements, UI testing, and browser-based validation of the library and its demos.

This plan delivered:

1. **New specialist agents (Tier 3):** `performance-trace-specialist`, `browser-ui-specialist`,
   `browser-memory-specialist` — all callable by any agent for Chrome DevTools MCP operations.
2. **New `/delegate` skill:** mandatory delegation enforcement skill added to all Tier 0, 1, and 2
   agents, containing the tier graph, routing table, delegation rules, goal-to-agent mapping, and
   the slice-based implementation orchestration protocol.
3. **New `chrome-devtools-mcp` skill:** durable knowledge for Chrome DevTools MCP tool usage,
   decision trees for direct MCP tools vs. specialist delegation, and token-efficient strategies.
4. **Sliced implementation loop enforcement:** strict RED → IMPLEMENT → GREEN loop managed by the
   orchestrator, replacing the loose flow.
5. **Updated testing agents (03-red-testing, 05-green-testing):** Chrome DevTools MCP awareness,
   specialist references, and decision trees.
6. **Trace analysis infrastructure:** scripts for compressing, parsing, and summarizing large
   Chrome DevTools performance traces into concise agent-context-friendly metric summaries.
7. **Validation gates:** `chrome-devtools-mcp-coverage` and `delegate-skill-coverage`.
8. **Updated flows, copilot-instructions.md, agent table scripts, and plan registrations.**

This was a standalone meta-workflow lane. It modified `.github/agents/`, `.github/skills/`,
`.github/flows/`, `.github/copilot-instructions.md`, `scripts/agent-customization/`,
`scripts/analyze-trace/`, `.vscode/mcp.json`, `.gitignore`, and plan index files. It did not
modify `src/` library code.

## Final state

All 8 phases complete. All gates pass. Plan registered, compressed, and archived.

- **Agents:** 65 total (3 new Tier 3 specialists: performance-trace-specialist,
  browser-ui-specialist, browser-memory-specialist).
- **Skills:** 58 total (2 new: `execute`, `chrome-devtools-mcp`).
- **Gates:** 2 new validation gates (`chrome-devtools-mcp-coverage`, `delegate-skill-coverage`)
  registered in `neataptic-gate-mcp` and callable via `run_gate_check`.
- **Trace scripts:** `scripts/trace-compress.mjs`, `scripts/trace-summarize.mjs`, and
  `analyze-trace.io.ts` extensions (`detectTraceFormat`/`normalizeTraceFile`).
- **Tests:** 52 new tests, 100% coverage on all new code.
- **Flows:** All 11 flow files (03._, 04._, 05.\*) updated with sliced loop-back protocol.
- **copilot-instructions.md:** Updated (§0 loop management, §2 Tier 3=42, §3 strict
  RED→IMPLEMENT→GREEN loop, Chrome DevTools MCP classification, execute skill refs).
- **execute skill:** All 19 Tier 1/2 agents (8 Tier 1 + 11 Tier 2) carry the `execute` skill.
- **MCP config:** Chrome DevTools MCP server registered in `.vscode/mcp.json` (5th server).
- **Routing table:** Regenerated (agents=65, skills=58, hash matches).

## Audit summary

### Phase 1 — Infrastructure & Scripts [DONE]

`.gitignore` updated for `tmp/traces/`; `tmp/traces/.gitkeep` created;
`scripts/trace-compress.mjs` (12 tests, 100% coverage); `scripts/trace-summarize.mjs`
(18 tests, 100% coverage); `analyze-trace.io.ts` extended with `detectTraceFormat`/
`normalizeTraceFile` for Chrome DevTools MCP trace format (22 tests, 100% coverage);
`.vscode/mcp.json` 5th server `chrome-devtools-mcp` added; `jest.config.mjs` and
`package.json` updated with new test projects (`analyze-trace-scripts`, `trace-scripts-mjs`).
tsc: OK; lint: 0 issues; prettier: OK; JSON parse: OK; git check-ignore: OK.

### Phase 2 — New Skills (execute, chrome-devtools-mcp) [DONE]

`.github/skills/execute/SKILL.md` created (tier graph, routing table, delegation rules,
goal-to-agent mapping, sliced implementation orchestration protocol).
`.github/skills/chrome-devtools-mcp/SKILL.md` created (tool usage, decision trees for direct
MCP vs. specialist delegation, token-efficient strategies).
`validate-skill-frontmatter.mjs`: PASS (0 errors, 0 warnings) for both. lint: PASS.

### Phase 3 — New Specialist Agents [DONE]

Created `.github/agents/performance-trace-specialist.agent.md`,
`.github/agents/browser-ui-specialist.agent.md`,
`.github/agents/browser-memory-specialist.agent.md` (all Tier 3, `user-invocable: false`,
`agents: []`, `chrome-devtools-mcp/*` tools namespace).
agent-graph gate: pass (65 agents, Tier 3=42, 0 issues); tier-enforcement gate: pass;
lint: PASS.

### Phase 4 — Update Testing Agents [DONE]

`03-red-testing.agent.md` and `05-green-testing.agent.md` updated with Chrome DevTools MCP
Decision Tree sections, browser-related red/green test patterns, and Sliced Implementation
Loop-Back protocol.
agent-graph gate: PASS (0 issues, 65 agents); plan-sync gate: PASS; frontmatter (non-strict):
PASS; lint: PASS.

### Phase 5 — Add /execute skill to All Tier 1 and Tier 2 Agents [DONE]

`execute` skill added to all 8 Tier 1 agents and all 11 Tier 2 agents
(flappy-architecture-polish, green-test-failure-triage-coordinator,
helping-agent-maintenance-coordinator, helping-gap-resolution-coordinator,
implementation-executor, implementation-pattern-coordinator, planning-context-coordinator,
planning-risk-coordinator, planning-test-strategy-coordinator,
research-codebase-coordinator, solid-split). `03-red-testing` and `05-green-testing` also
received `chrome-devtools-mcp` skill, `chrome-devtools-mcp/*` tools, and
performance-trace/browser-ui/browser-memory-specialist delegation entries.
Routing table regenerated (agents=65, skills=58). agent-graph gate: pass; tier-enforcement
gate: pass; routing-table-freshness gate: pass; lint: pass.

### Phase 6 — Sliced Implementation Loop Enforcement [DONE]

`copilot-instructions.md` updated (§0 loop management, §2 Tier 3=42, §3 strict
RED→IMPLEMENT→GREEN loop, Chrome DevTools MCP classification, execute skill refs).
All `04.*` flows updated (loop-back declarations, execute skill, Chrome DevTools MCP
specialists). All `05.*` flows updated (loop-back declarations, Chrome DevTools MCP
specialists). All `03.*` flows updated (Chrome DevTools MCP specialists, browser-red-tests
notes). YAML validation: PASS (js-yaml); lint: PASS.

### Phase 7 — Validation Gates & Agent Table Scripts [DONE]

Created `scripts/agent-customization/gates/chrome-devtools-mcp-coverage.gate.mjs` (validates
03/05 agents have `chrome-devtools-mcp` skill + 3 specialists) and
`scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs` (validates all Tier 1/2
agents have `execute` skill). Both gates registered in `neataptic-gate-mcp` server
`TIER_1_GATES` registry and callable via `run_gate_check` MCP tool (pass=true).
jest: 7 tests passed (4 chrome-devtools-mcp-coverage + 3 delegate-skill-coverage).
inventory-customizations.mjs: 65 agents, 58 skills. Routing table regenerated; freshness
gate: PASS. lint: PASS.

### Phase 8 — Documentation, Plan Registration, and Final Validation [DONE]

Final gate suite run by 05-green-testing. All gates pass:

- `agent-graph` gate: pass=true (65 agents, 0 issues, Tier 1=8 / Tier 2=11 / Tier 3=42 /
  Tier 4=4)
- `tier-enforcement` gate: pass=true (0 violations, 8 user-invocable)
- `routing-table-freshness` gate: pass=true (hash
  `613a377ab5892f239038460e39c9f252b0cb441379fcb73cdd5aca8cd0b1dba6` matches, agents=65,
  skills=58)
- `plan-sync` gate: pass=true (3 WIP plans registered, 6 plans checked, 0 missing)
- `chrome-devtools-mcp-coverage` gate: pass=true (03-red-testing and 05-green-testing both
  have `chrome-devtools-mcp` skill + all 3 specialists)
- `delegate-skill-coverage` gate: pass=true (19 Tier 1/2 agents all have `execute` skill)
- lint: PASS (0 errors)

All 10 new files verified to exist. Plan registration confirmed in `plans/README.md` and
`plans/Roadmap.md`. READY for plan closure (compression + archival) by 07-logging.

## Reopen conditions

Reopen this plan if:

- A fourth browser-interaction specialist is needed (e.g., separate console/network specialists),
  per DR-20260620-01 rollback plan.
- `browser-memory-specialist` proves too thin to warrant a separate agent and should be merged
  into `performance-trace-specialist`, per DR-20260620-02 rollback plan.
- The strict sliced RED→IMPLEMENT→GREEN loop needs relaxation or restructuring.
- The Chrome DevTools MCP server requires a Node.js version higher than the current repo
  baseline (deferred question, resolved as compatible during Phase 1).

## Audit log

- 2026-06-20: Plan created by 01-planning (8 phases, 41 steps).
- 2026-06-20: Phases 1–8 executed and marked [DONE] across multiple sessions.
- 2026-06-20: Final gate suite green; plan registered; 65 agents / 58 skills.
- 2026-06-20: Plan compressed and archived to `plans/completed/` by 07-logging.

## Decision Records

- **DR-20260620-01:** Combine browser DOM interaction, console monitoring, and network
  inspection into a single `browser-ui-specialist` (chosen) rather than splitting into
  separate agents. Rationale: all three share the same browser session state and Chrome
  DevTools MCP connection. Rollback: split if console/network grows complex.
- **DR-20260620-02:** Keep `browser-memory-specialist` separate from
  `performance-trace-specialist` (chosen). Rationale: memory profiling is a distinct
  discipline with different tool surfaces (heap snapshots vs. performance traces). Rollback:
  merge if memory profiling proves too thin.
