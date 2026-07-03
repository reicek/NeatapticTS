# Chrome MCP Browser Tests

**Status:** [DONE]

**Claim:** `07-logging` @ 2026-07-02 — all seven phases complete; plan and log
archived to `plans/completed/`.

## Scope

Add a reusable, Chrome-MCP-powered browser testing harness to NeatapticTS so
maintainers can run focused browser scenarios locally against `http://localhost:8080`
(`npm start`) and capture performance traces, DOM state, console output, and
memory metrics via the existing Chrome DevTools MCP server and its three
specialist agents.

The harness must be skill-driven and specialist-backed: a new
`browser-testing-harness` skill owns the durable orchestration patterns, and a new
Tier-3 `browser-harness-specialist` agent performs multi-step scenario execution
on behalf of red-testing, green-testing, implementing, and documenting agents.

The first real consumer of the harness will be the WebGPU inference path in
`src/architecture/network/gpu/` (delivered by
`plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`). Example test
URLs will live under `docs/browser-tests/` and be parseable by the docs
generation system while remaining hidden from public navigation indexes via a
`docs.order.json` `hiddenFiles` marker.

This is a meta-workflow/infrastructure lane. It touches `.github/skills/`,
`.github/agents/`, `.github/copilot-instructions.md`, `scripts/agent-customization/`,
`docs/browser-tests/`, and plan index files. It does not change core `src/`
library code unless a discovered WebGPU gap requires it in a later, explicitly
scoped phase.

## Acceptance criteria

1. A maintainer can run `npm start` and then visit a documented local URL such
   as `http://localhost:8080/docs/browser-tests/webgpu-inference-smoke.html` to
   execute a browser scenario that asserts WebGPU inference produces the same
   outputs as the CPU path.
2. The `browser-testing-harness` skill contains durable workflows for:
   - starting the local server,
   - navigating to a scenario URL,
   - waiting for a readiness signal,
   - delegating to `performance-trace-specialist`, `browser-ui-specialist`, or
     `browser-memory-specialist` as appropriate,
   - capturing and returning a concise metric summary.
3. The `browser-harness-specialist` agent is callable from `03-red-testing`,
   `04-implementing`, `05-green-testing`, and `06-documenting` and owns
   multi-step browser scenario execution.
4. `03-red-testing`, `04-implementing`, `05-green-testing`, and `06-documenting`
   frontmatter reference the new skill/specialist, and the three existing
   browser specialists know when to loop in the harness specialist.
5. The routing table (`npm run agents:routing-table`) and relevant gates
   (`chrome-devtools-mcp-coverage`, `delegate-skill-coverage`) pass after the
   updates.
6. At least one real WebGPU scenario runs under Chrome MCP tracing and reports
   deterministic output parity and frame/time metrics.
7. Docs generation parses the test-URL sources but does not surface them in
   `docs/index.html` navigation or README indexes; maintainers discover them via
   this plan and the skill.

## Upstream context and constraints

- `package.json` `start` script: `npx http-server . -p 8080 -c-1`.
- Existing Chrome DevTools MCP skill: `.github/skills/chrome-devtools-mcp/SKILL.md`.
- Existing browser specialists:
  `performance-trace-specialist`, `browser-ui-specialist`, `browser-memory-specialist`.
- WebGPU fast path lives in `src/architecture/network/gpu/`.
- Completed WebGPU plan:
  `plans/completed/NEAT_Genesis_EvoDevo_GPU_Acceleration.plans.md`.
- Docs generation supports `docs.order.json` with a `hiddenFiles` key, but that
  key only hides source `.ts` files from generated per-directory READMEs and
  `docs/FOLDERS.md`. The primary mechanism for hiding maintainer-only test pages
  is placing them under `docs/browser-tests/`, which is outside the source-driven
  docs pipeline and therefore never rendered into navigation indexes. A
  defensive `docs/browser-tests/docs.order.json` can still be added for
  future-proofing if folder-docs generation is ever extended to the directory.
- No deferred cleanup: any new skill/agent wiring must remove deprecated
  references in the same step that introduces the new ones.

## Decision records

- **DR-2026-07-02-01 — New skill name:** `browser-testing-harness` (chosen).
  Rationale: the skill orchestrates the full browser-test lifecycle, not merely
  a single tool. Alternatives considered: `browser-test-harness` (too close to
  generic Jest terminology), `chrome-mcp-browser-tests` (too tool-specific). The
  chosen name aligns with the existing `chrome-devtools-mcp` skill as a
  higher-level consumer.
- **DR-2026-07-02-02 — New specialist name:** `browser-harness-specialist`
  (chosen). Rationale: it is the single point of delegation for multi-step
  browser scenarios, distinct from the three lower-level tool specialists. It
  will be Tier 3, `user-invocable: false`, and carry the new skill plus the
  `chrome-devtools-mcp` skill.
- **DR-2026-07-02-03 — Test URL hiding strategy:** Place test pages under
  `docs/browser-tests/`. Rationale: the source-driven docs pipeline only targets
  `src`, `asciiMaze`, `flappy-bird`, and `racing-curriculum`; it never scans
  `docs/browser-tests/`, so hand-maintained HTML pages there are rendered by the
  local HTTP server but are absent from generated `README.md` indexes and the
  `docs/index.html` sidebar. `docs.order.json` `hiddenFiles` only filters source
  `.ts` files from generated per-directory READMEs and is therefore a secondary
  safeguard, not the primary hiding mechanism.

## Implementation phases

### Phase 1 — Planning the harness workstream [DONE]

```yaml
phase: 1
title: 'Planning the harness workstream'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Research docs generation hiding and WebGPU test surfaces'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
  - planning-acceptance-criteria
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/completed/Chrome_MCP_Browser_Tests.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Chrome_MCP_Browser_Tests.plans.md'
acceptance_criteria:
  - 'Phase/step YAML blocks conform to the schema and pass validation.'
  - 'Plan is registered in plans/README.md and plans/Roadmap.md with trigger phrases: chrome mcp, browser tests, webgpu tests, browser harness.'
placeholder_steps:
  - 'Step 01 — Author Phase 2-7 step packets'
```

[DONE] Phase 1 Step 01 — authored Phase 2-7 step packets, registered the plan,
and passed plan-phase-packets and plan-sync gates.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 2 — Research docs generation hiding and WebGPU test surfaces [DONE]

```yaml
phase: 2
title: 'Research docs generation hiding and WebGPU test surfaces'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 3 — Red tests and contracts for the harness'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'Cortex-first search reports on docs.order.json hiddenFiles behavior and WebGPU demo entry points.'
  - 'Confirm at least one concrete scenario URL and the bundles it must load.'
acceptance_criteria:
  - 'Docs generation hiddenFiles behavior is documented in this plan.'
  - 'A concrete WebGPU smoke-test URL and required bundle(s) are identified.'
  - 'No production files are changed.'
placeholder_steps:
  - 'Step 02 — Research docs generation hiding and WebGPU test surfaces'
```

[DONE] Phase 2 Step 02 — confirmed docs-order hiding behavior, identified the
WebGPU smoke scenario entry point/bundle, and recorded the research brief.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 3 — Red tests and contracts for the harness [DONE]

```yaml
phase: 3
title: 'Red tests and contracts for the harness'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 4 — Implement the browser testing harness skill and specialist agent'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests'
  - 'Red tests fail for the right reason before implementation.'
acceptance_criteria:
  - 'Red tests exist for the harness launcher, scenario parser, and trace-summary consumer.'
  - 'A red contract exists for the WebGPU smoke scenario parity assertion.'
placeholder_steps:
  - 'Step 03 — Write red tests and contracts for the harness'
```

[DONE] Phase 3 Step 03 — authored red tests/contracts for the harness launcher,
scenario parser, trace summary, and WebGPU smoke scenario; confirmed failures
were due to missing implementation.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 4 — Implement the browser testing harness skill and specialist agent [DONE]

```yaml
phase: 4
title: 'Implement the browser testing harness skill and specialist agent'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 5 — Green validation and agent frontmatter wiring'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests'
  - 'npm run lint'
  - 'npm run agents:routing-table'
acceptance_criteria:
  - 'browser-testing-harness skill exists and passes skill-frontmatter validation.'
  - 'browser-harness-specialist agent exists with correct tier, skills, and delegation allow-list.'
  - 'Existing browser specialists know when to call the harness specialist.'
  - '03-red-testing, 04-implementing, 05-green-testing, 06-documenting frontmatter are updated.'
  - 'Routing table is regenerated and freshness gate passes.'
placeholder_steps:
  - 'Step 04 — Implement the browser testing harness skill and specialist agent'
```

[DONE] Phase 4 Step 04 — implemented the `browser-testing-harness` skill,
`browser-harness-specialist` agent, wired the skill/specialist into existing
agents, regenerated the routing table, and made all Phase 3 red tests pass.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 5 — Green validation and agent frontmatter wiring [DONE]

```yaml
phase: 5
title: 'Green validation and agent frontmatter wiring'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 6 — Document the harness and WebGPU scenario'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests'
  - 'node scripts/agent-customization/validate-agent-frontmatter.mjs --json'
  - 'node scripts/agent-customization/validate-skill-frontmatter.mjs --json'
  - 'npm run agents:routing-table:gate'
  - 'node scripts/agent-customization/gates/chrome-devtools-mcp-coverage.gate.mjs --json'
  - 'node scripts/agent-customization/gates/delegate-skill-coverage.gate.mjs --json'
acceptance_criteria:
  - 'All focused tests pass.'
  - 'New code has 100% coverage on touched src/ files (harness scripts are treated as src/ for coverage).'
  - 'Routing table and required gates pass.'
placeholder_steps:
  - 'Step 05 — Green validation and agent frontmatter wiring'
```

[DONE] Phase 5 Step 05 — green-validated the harness unit tests, agent
frontmatter/routing wiring, and the first real WebGPU smoke scenario under
Chrome MCP; pre-existing `devtools-coverage.gate.mjs` naming mismatch recorded
as a non-regression exception.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 6 — Document the harness and WebGPU scenario [DONE]

```yaml
phase: 6
title: 'Document the harness and WebGPU scenario'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: 'Phase 7 — Compress phase history and close the workstream'
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'npm run docs:quality:gate'
  - 'npm run lint'
acceptance_criteria:
  - 'JSDoc and generated READMEs explain the browser-testing-harness skill.'
  - 'WebGPU scenario has inline documentation suitable for educational standards.'
  - 'Docs quality gate passes with no new high-complexity findings.'
placeholder_steps:
  - 'Step 06 — Document the harness and WebGPU scenario'
```

[DONE] Phase 6 Step 06 — documented the `browser-testing-harness` skill,
authored `Browser_Tests.md`, updated `WebGPU.md`, aligned JSDoc, and verified
all docs-quality, lint, Mermaid, frontmatter, and stale-wip gates pass.
See `plans/completed/Chrome_MCP_Browser_Tests.logs.md` for detailed evidence.

### Phase 7 — Compress phase history and close the workstream [DONE]

```yaml
phase: 7
title: 'Compress phase history and close the workstream'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_phase: null
skills:
  - plan-alignment
  - tracker-handoff
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Chrome_MCP_Browser_Tests.plans.md'
acceptance_criteria:
  - 'Completed phases are compressed into the matching .logs.md file.'
  - 'Plan and logs are moved to plans/completed/.'
  - 'plans/README.md and plans/Roadmap.md point to the archived pair and mark it [DONE].'
placeholder_steps:
  - 'Step 07 — Compress phase history and close the workstream'
```

**Phase objective:** Use `07-logging` to compress the completed phase histories,
create the matching `.logs.md` record, and archive both files under
`plans/completed/`.

**Stop conditions:**

- Done: plan compressed, logs created, archive moved, all closure gates pass.
- Blocked: if a gate fails, fix before claiming closure.

**Required validation:**

- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Chrome_MCP_Browser_Tests.plans.md`

[DONE] Phase 7 Step 07 — compressed Phase 1–6 histories into
`plans/completed/Chrome_MCP_Browser_Tests.logs.md`, moved the plan/log pair to
`plans/completed/`, and updated `plans/README.md` and `plans/Roadmap.md` to
[DONE]. Closure gates run on the archived plan.

#### Step 07 — Compress phase history and close the workstream [DONE]

```yaml
phase: 7
step: 7
title: 'Compress phase history and close the workstream'
status: '[DONE]'
goal: 'logging'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
copy_paste: true
next_step: null
skills:
  - tracker-handoff
  - plan-sync-validation
  - summarizing-session-log
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
  - 'node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Chrome_MCP_Browser_Tests.plans.md'
acceptance_criteria:
  - 'Phase history compressed; plan and logs moved to plans/completed/.'
  - 'Stale-wip-plans gate passes.'
```

## Validation gates

- `plan-sync`: run after plan registration changes.
- `step-packet`: run after authoring or revising step packets.
- `agent-graph`: run after any agent delegation change.
- `stale-wip-plans`: run after changing a plan's active status.

## Final state / Audit summary

```yaml
closure:
  status: done
  closer: 07-logging
  timestamp: 2026-07-02T16:07:23-04:00
  archived_plan: 'plans/completed/Chrome_MCP_Browser_Tests.plans.md'
  archived_log: 'plans/completed/Chrome_MCP_Browser_Tests.logs.md'
  phases: [1, 2, 3, 4, 5, 6, 7]
  note: >
    All phases complete. Plan and log archived. README/Roadmap updated to [DONE].
    Lingering exception: devtools-coverage.gate.mjs naming mismatch
    (expects skill 'devtools', canonical skill is 'chrome-devtools-mcp')
    remains unresolved and is carried as a pre-existing defect.
```

## Reopen conditions

Reopen only for new browser-harness infrastructure work not covered by the
completed scope. Start from `plans/completed/Chrome_MCP_Browser_Tests.plans.md`
and its matching log.
