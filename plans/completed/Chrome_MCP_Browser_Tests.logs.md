# Chrome MCP Browser Tests Log

**Status:** [DONE]

Archive log for the Chrome-MCP-powered browser testing harness workstream.
Detailed evidence for Phases 1–6 is preserved below so the closed plan can stay
compact while retaining durable audit coverage.

---

## Phase 1 — Planning the harness workstream

**Step 01 — Author Phase 2-7 step packets**

- Specialist/agent: `01-planning`
- Outcome: authored the full set of step packets for Phases 2–7 using the canonical
  YAML schema; registered the plan in `plans/README.md` and `plans/Roadmap.md`.
- Validation evidence:
  - `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/Chrome_MCP_Browser_Tests.plans.md` → PASS
  - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Chrome_MCP_Browser_Tests.plans.md` → PASS

---

## Phase 2 — Research docs generation hiding and WebGPU test surfaces

**Step 02 — Research docs generation hiding and WebGPU test surfaces**

- Research date: 2026-07-02
- Specialists used: `docs-scout`, `browser-runtime-scout`, `plan-scout`

### 1. Docs generation and hiding

- `generate-docs.js` is source-driven: it scans `.ts` files in configured targets
  (`src`, `asciiMaze`, `flappy-bird`, `racing-curriculum`) using `fast-glob` with
  `SOURCE_FILE_GLOBS = ['**/*.ts']`, ignoring `.d.ts` and `.test.ts`.
- Each directory may contain a `docs.order.json`. `loadDirectoryDocsOrderConfig`
  loads only that directory's file; there is no parent/child merge.
- `hiddenFiles` is an array of source `.ts` basenames.
  `resolveVisibleDirectoryFiles` filters them out of the sorted file list used by
  `buildDirectoryReadme` and `buildFolderIndexMarkdown`.
- Hidden source files are still parsed by ts-morph and can serve as `introFile`,
  but they do **not** get a `## <file.ts>` section in the generated README and are
  not counted in `docs/FOLDERS.md`.
- The HTML renderer (`render-docs-html.js`) turns every `README.md` under `docs/`
  into a directory page and builds the sidebar from those pages. There are no
  per-file pages.
- **Implication for `docs/browser-tests/`:** `docs/browser-tests/` is not a source
  target and is not managed by `generate-docs.js` or `copy-examples.js`. A
  hand-maintained HTML page there is naturally absent from generated `README.md`
  indexes and from `docs/index.html` navigation. `docs.order.json` `hiddenFiles`
  is therefore **not the primary hiding mechanism** for these pages; it only
  becomes relevant if folder-docs generation is later extended to scan the
  directory. A defensive `docs/browser-tests/docs.order.json` with
  `"hiddenFiles": ["webgpu-inference-smoke.html"]` can still be added for
  future-proofing.

### 2. Browser-facing examples and bundles

- Root browser artifacts: `npm run build:browser` produces
  `dist/neataptic.browser.esm.js`, `dist/neataptic.browser.iife.js`,
  `dist/neataptic.browser.iife.min.js` from `src/browser-entry.ts`.
- Example bundles live in `docs/assets/` (`hello-network.bundle.js`,
  `evolve-xor.bundle.js`, `sequence-reset.bundle.js`, `ascii-maze.bundle.js`,
  `flappy-bird.bundle.js`, `neat-chat.bundle.js`, `racing-curriculum.bundle.js`).
- Existing `examples/*/index.html` pages detect `/examples/` vs `/docs/examples/`
  serving and adjust bundle paths.

### 3. WebGPU test/demo surfaces

- No real browser WebGPU HTML demo exists today.
- Node/Jest GPU tests: `src/architecture/network/gpu/*.test.ts` including
  `network.gpu.parity.test.ts` (CPU/GPU parity: max abs diff ≤ 0.5, MAE ≤ 0.1),
  `network.gpu.fallback.test.ts`, `network.gpu.racing.test.ts`.
- Browser-oriented source: `examples/racing_curriculum/gpu-enabled-racing.example.ts`
  builds a network and calls `network.activate(observation, { useGPU: true })`
  with a real `navigator.gpu` device, but has no CPU parity assertion.
- `Network.createMLP(2, [3], 1)` uses logistic activation (index 0), which is in
  `SUPPORTED_ACTIVATION_INDICES`, so it is GPU-eligible.

### 4. Chrome DevTools MCP tooling

- Existing specialists: `performance-trace-specialist`,
  `browser-ui-specialist`, `browser-memory-specialist`.
- They are already referenced in `03-red-testing` and `05-green-testing`
  frontmatter `agents:` lists and carry the `chrome-devtools-mcp` skill.
- Proposed `browser-harness-specialist` would be the cross-cutting orchestrator:
  start `npm start`, navigate to hidden docs URLs, parse scenario result objects,
  decide which lower-level specialist to delegate to, and return a concise metric
  summary.

### 5. HTTP server behavior

- `npm start` runs `npx http-server . -p 8080 -c-1` from repo root.
- `docs/browser-tests/webgpu-inference-smoke.html` will be reachable at
  `http://localhost:8080/docs/browser-tests/webgpu-inference-smoke.html`.
- `dist/neataptic.browser.iife.js` will be reachable at
  `http://localhost:8080/dist/neataptic.browser.iife.js`, but only after
  `npm run build:browser` because `/dist/` is gitignored and not present in a clean
  checkout.
- `docs/assets/*.bundle.js` are reachable once they exist locally.

### 6. Concrete first scenario recommendation

- File: `docs/browser-tests/webgpu-inference-smoke.html`
- Loads: `../../dist/neataptic.browser.iife.js` (requires `npm run build:browser`)
- Logic:
  - Build `Neataptic.Network.createMLP(2, [3], 1)`.
  - Run CPU reference: `network.activate([0.1, 0.2])`.
  - Request `navigator.gpu` adapter/device, assign `network.gpuDevice = device`.
  - Run GPU path: `await network.activate([0.1, 0.2], { useGPU: true })`.
  - Assert parity: max abs diff ≤ 0.5, MAE ≤ 0.1.
  - Emit `window.webgpuSmokeResult = { success, cpuOutput, gpuOutput, maxAbsDiff,
    meanAbsDiff, gpuDeviceBound }`.
- This reuses the parity contract already in `WebGPU.md` and
  `network.gpu.parity.test.ts`.

### 7. Risks/gaps recorded

- `dist/` must be generated before the smoke page can load the root browser
  bundle.
- No existing real browser WebGPU demo; the first scenario must be authored from
  scratch.
- `docs/browser-tests/` is outside the automated docs/examples pipeline, so pages
  there will not be refreshed or pruned automatically.
- `docs.order.json` `hiddenFiles` only hides source `.ts` files from generated
  source READMEs, not arbitrary HTML pages. The actual hiding comes from
  `docs/browser-tests/` not being a docs generation target.

---

## Phase 3 — Red tests and contracts for the harness

**Step 03 — Write red tests and contracts for the harness**

- Red date: 2026-07-02
- Specialist used: `unit-test-writer`

### Files created

- `scripts/agent-customization/browser-tests/__tests__/harness-launcher.test.ts`
- `scripts/agent-customization/browser-tests/__tests__/scenario-parser.test.ts`
- `scripts/agent-customization/browser-tests/__tests__/trace-summary.test.ts`
- `docs/browser-tests/webgpu-inference-smoke.html`
- `docs/browser-tests/scenarios/webgpu-smoke.mjs`

### Red-test evidence

Command:
`npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests`

Result: 3 failed, 2 passed, 5 total.

Failures:

1. `harness-launcher.test.ts`: `Cannot find module '../harness-launcher'` — expected,
   the launcher module is not implemented yet.
2. `trace-summary.test.ts`: `Cannot find module '../trace-summary'` — expected,
   the trace-summary module is not implemented yet.
3. `scenario-parser.test.ts`: the smoke HTML page does not yet contain the required
   `window.webgpuSmokeResult = { success, cpuOutput, gpuOutput, maxAbsDiff, meanAbsDiff, gpuDeviceBound }`
   emission statement.

The two passing tests confirm the stub page loads the correct bundle path and
constructs the expected MLP shape, proving the fixtures are not the cause of the
red failures.

### Lint evidence

- `npx eslint scripts/agent-customization/browser-tests/ docs/browser-tests/` exits
  with code 2 because `eslint.config.mjs` globally ignores `docs/` and `scripts/`.
- `npx eslint --no-ignore scripts/agent-customization/browser-tests/ docs/browser-tests/`
  exits with code 0 and reports no errors; the new files are lint-clean.

### Handoff to Phase 4

- Implement `scripts/agent-customization/browser-tests/harness-launcher.ts` so
  `launchLocalServer({ cwd, port })` returns `{ serverUrl, scenarioUrl }`.
- Implement `scripts/agent-customization/browser-tests/trace-summary.ts` so
  `createTraceSummary(input)` returns `{ scenarioUrl, durationMs, success, metrics }`.
- Complete `docs/browser-tests/scenarios/webgpu-smoke.mjs` and update
  `docs/browser-tests/webgpu-inference-smoke.html` to emit the required
  `window.webgpuSmokeResult` object after the CPU/GPU parity check.

---

## Phase 4 — Implement the browser testing harness skill and specialist agent

**Step 04 — Implement the browser testing harness skill and specialist agent**

- Implementation date: 2026-07-02
- Specialists used: `implementation-pattern-scout` (via `04-implementing`),
  `browser-harness-specialist`

### Files created

- `.github/skills/browser-testing-harness/SKILL.md`
- `.github/agents/browser-harness-specialist.agent.md`
- `scripts/agent-customization/browser-tests/harness-launcher.ts`
- `scripts/agent-customization/browser-tests/trace-summary.ts`
- `docs/browser-tests/index.html`
- `docs/browser-tests/docs.order.json`
- `.prettierignore`

### Files changed

- Completed the WebGPU smoke HTML/JS emission.
- Added `teardown()` to launcher tests.
- Switched the server to a built-in Node static server on `127.0.0.1`.
- Wired the new skill/specialist into `03-red-testing`, `04-implementing`,
  `05-green-testing`, `06-documenting`, and the three DevTools specialists.
- Regenerated `.github/agent-skill-routing-table.md`.

### Validation evidence

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests` → pass (3 suites, 5 tests).
- `npx tsc --noEmit -p tsconfig.json` → pass.
- `npm run lint` → pass.
- `npx prettier --check <touched-files>` → pass.
- `validate-skill-frontmatter.mjs --strict --json` → PASS.
- `validate-agent-frontmatter.mjs --json` → PASS (warnings are pre-existing).
- `validate-plan-sync.mjs` / `validate-plan-phase-packets.mjs` → PASS.
- `npm run agents:routing-table` → regenerated (67 agents, 62 skills).
- `npm run agents:routing-table:gate` → pass.
- `delegate-skill-coverage.gate.mjs` → pass.
- `devtools-coverage.gate.mjs` → **FAIL (pre-existing):** expects skill `devtools`
  but the canonical skill is `chrome-devtools-mcp`. Recorded as a pre-existing
  planning defect; Phase 5 treated it as a gate exception rather than a
  regression.

### Handoff to Phase 5

- Confirm the focused Jest slice and frontmatter/routing gates.
- The devtools-coverage naming mismatch was out of slice scope.

---

## Phase 5 — Green validation and agent frontmatter wiring

**Step 05 — Green validation and agent frontmatter wiring**

### Validation evidence (compressed)

- Focused Jest: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripts/agent-customization/browser-tests` → pass (3 suites, 5 tests).
- `npm run lint` → pass.
- `npx tsc --noEmit -p tsconfig.test.json` → pass.
- `validate-agent-frontmatter.mjs --json` → PASS (pre-existing assimilator warnings).
- `validate-skill-frontmatter.mjs --strict --json` → PASS.
- `npm run agents:routing-table` → unchanged (67 agents, 62 skills); gate → pass.
- `delegate-skill-coverage.gate.mjs` → pass.
- `devtools-coverage.gate.mjs` → fail (pre-existing naming mismatch: gate expects
  `devtools`, canonical skill is `chrome-devtools-mcp`). Escalated to
  `00-helping` / `01-planning`; not a Phase 5 regression.
- Plan phase packets, plan sync, and stale-wip gates → PASS.

---

## Phase 6 — Document the harness and WebGPU scenario

**Step 06 — Document the harness and WebGPU scenario**

### Validation evidence

- Skill README expanded: `.github/skills/browser-testing-harness/SKILL.md` now
  includes DevTools comparison table, runnable example, hidden-URL note,
  Mermaid workflow/decision-tree diagrams, and explicit workflow guidance.
- Root guide created: `Browser_Tests.md` with hidden URLs, quick start, Mermaid
  workflow diagram, and explanation of why test pages stay out of generated
  navigation.
- WebGPU guide updated: `WebGPU.md` links to the smoke test and the harness
  skill; parity and opt-in contracts unchanged.
- Agent bodies already reference the harness/skill: `03-red-testing`,
  `04-implementing`, `05-green-testing`, `06-documenting`, and
  `browser-harness-specialist` list `browser-testing-harness` in skills/agents
  and use it in their browser decision trees.
- JSDoc aligned on `harness-launcher.ts` and `trace-summary.ts`.
- Note: `docs/browser-tests/README.md` was intentionally not added; the existing
  `docs/browser-tests/index.html` already serves as the hidden local index, and a
  Markdown README would risk surfacing in generated docs navigation.
- `npm run lint` → pass.
- Mermaid validation (extracted from each Markdown file and run with
  `node ./dist-docs/scripts/mermaid-cli.js validate --input <path>`):
  - `Browser_Tests.md` harness workflow → "Valid diagram".
  - `WebGPU.md` GPU eligibility flow → "Valid diagram".
  - `SKILL.md` workflow + decision tree → "Valid diagram" (×2).
- `npx prettier --check/write` → all touched Markdown/TypeScript files formatted.
- `validate-plan-phase-packets.mjs --json --plan=plans/Chrome_MCP_Browser_Tests.plans.md`
  → PASS (0 errors, 0 warnings).
- `validate-plan-sync.mjs --json --plan=plans/Chrome_MCP_Browser_Tests.plans.md`
  → PASS (0 errors, 0 warnings).
- `npm run plans:stale-wip:gate` → `{ "pass": true }`.
- `validate-agent-frontmatter.mjs --json` → PASS (pre-existing assimilator warnings).
- `validate-skill-frontmatter.mjs --strict --json` → PASS (0 errors, 0 warnings).
- `npm run docs:quality:gate` → `{ "pass": true }` (mechanism-only gate).
- `academic-docs-auditor` → PASS for `Browser_Tests.md`, `WebGPU.md`, and
  `SKILL.md` (Mermaid, citations, atemporal language).
- `license-attribution-auditor` → PASS for the same files (no unlicensed media,
  only internal/Wikipedia background links).

---

## Closure summary

- Plan compressed and moved to `plans/completed/Chrome_MCP_Browser_Tests.plans.md`
- Log moved to `plans/completed/Chrome_MCP_Browser_Tests.logs.md`
- `plans/README.md` and `plans/Roadmap.md` updated to point to the archived pair
  and marked [DONE].
- All closure gates run on the archived plan.
- Lingering exception: `devtools-coverage.gate.mjs` naming mismatch
  (expects `devtools`, canonical skill is `chrome-devtools-mcp`) remains unresolved
  and is carried as a known pre-existing defect outside this workstream's scope.
