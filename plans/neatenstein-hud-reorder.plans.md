# Neatenstein HUD Element Reorder

**Status:** [WIP]
**Plan ID:** NEATENSTEIN_HUD_REORDER
**Created:** 2026-08-09
**Source of truth:** `plans/neatenstein-hud-reorder.plans.md`

## Mandates

- `model: glm-5.2:cloud` for all dispatches under this plan.
- Pragmatic mode: broad slices (one per phase), bypass legacy ceremony (skip plan-verification green-light cycle, skip per-AC gate calls, skip fix-packet YAML). Green-only TDD sequence — tests already exist. Reuse the same idle agent via `write_agent` for follow-up passes rather than spawning fresh instances.
- This plan is **separate** from `neatenstein-hud-face-cannon-waves.plans.md`. Do NOT add steps to the old plan.
- Remove obsolete build artifacts if encountered; do not leave stale bundles after rebuild.

## Scope

Reorder the Neatenstein HUD status bar elements from the current **built** order:

```
K: → heat bar (health segments) → portrait (mugshot) → ammo bar (ammo segments) → D:
```

to the desired order:

```
heat bar → K: → portrait → D: → ammo bar
```

The source code in `examples/neatenstein/browser-entry/host/hud.ts` (`createNeonStatusBar()`) already contains the desired DOM append order (health segments → HIVE density → K: → mugshot → D: → ammo segments) from a prior phase, but the **built bundle** (`docs/assets/neatenstein.bundle.js`) is stale and still reflects the old order. This plan verifies the source, rebuilds the bundle, and confirms the visual order in a real browser.

### Element mapping

| User-visible label | DOM elements                              | Source location (hud.ts) |
| ------------------ | ----------------------------------------- | ------------------------ |
| Heat bar           | `health-segment` divs + `hive-density-track` | Lines ~613–637           |
| K:                 | killsPrefix span + killsLabel div          | Lines ~639–659           |
| Portrait           | mugshot canvas (absolutely centered)       | Lines ~661–668           |
| D:                 | deathsPrefix span + deathsLabel div         | Lines ~670–690           |
| Ammo bar           | `ammo-segment` divs                       | Lines ~692–702           |

## Non-goals

- No changes to the core NeatapticTS library (`src/`).
- No new HUD features, no styling changes, no new elements — only element ordering.
- No changes to the mugshot rendering logic, input handling, or game state.
- No changes to robot sprite data, gun sprite data, or worker logic.

## Current state

- Source `hud.ts` `createNeonStatusBar()` append order: health segments → HIVE density → K: → mugshot → D: → ammo segments (desired order — already correct in source).
- Built bundle `docs/assets/neatenstein.bundle.js` (stale, last built 5:56 PM): K: → health segments → mugshot → ammo segments → HIVE density → D: (old order — does NOT match source).
- Test `hud-status-bar.test.ts` AC-1006 (lines 378–458): verifies health-first, HIVE-before-K:, K:-before-mugshot, D:-after-mugshot, ammo-last — already matches desired order.
- The old plan `neatenstein-hud-face-cannon-waves.plans.md` has been archived; this plan is standalone.

## Implementation phases

### Phase 1 — Plan lock and verification [DONE]

**Phase objective:** Lock the implementation plan, confirm the source/test state, and prepare the single implementation step.

```yaml
phase: 1
title: 'Plan lock and verification'
status: 'DONE'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hud-reorder.plans.md'
copy_paste: true
next_phase: 'Step 01 — Verify source order and rebuild bundle'
skills:
  - 'plan-alignment'
  - 'planning-acceptance-criteria'
validation:
  - 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hud-reorder.plans.md'
acceptance_criteria:
  - id: AC-001
    text: 'Phase and step packets are authored and pass slice-advancement gate'
    validation: 'neataptic-gate-mcp:run_gate_check --gate=slice-advancement --json --args.slice-id=01-plan --args.changed-files=plans/neatenstein-hud-reorder.plans.md'
constitution_check:
  - 'principle-4-small-slices'
placeholder_steps:
  - 'Step 01 — Verify source order, rebuild bundle, browser smoke test'
```

### Phase 2 — Verify source, rebuild bundle, browser smoke test [WIP]

**Phase objective:** Confirm the `createNeonStatusBar()` DOM append order matches the desired sequence, run the existing test suite, rebuild the neatenstein bundle, and verify the visual order in a real browser.

**Phase progression rule:** Step 01 is the only step. It is a green-only slice — tests already exist (AC-1006).

#### Step 01: Verify source order, rebuild bundle, browser smoke test [WIP]

```yaml
phase: 2
step: 1
title: 'Verify source order, rebuild bundle, browser smoke test'
status: '[WIP]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/neatenstein-hud-reorder.plans.md'
copy_paste: true
next_step: 'null'
skills:
  - 'implementation-standards'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar'
  - 'npm run build:neatenstein'
  - 'npm run lint'
acceptance_criteria:
  - id: AC-002
    text: 'createNeonStatusBar() DOM append order is: health segments, HIVE density track, K: prefix+label, mugshot canvas, D: prefix+label, ammo segments'
    validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar'
  - id: AC-003
    text: 'Built bundle docs/assets/neatenstein.bundle.js reflects the same append order as the source (heat bar before K:, D: before ammo bar)'
    validation: 'npm run build:neatenstein'
  - id: AC-004
    text: 'Lint passes with no new errors on hud.ts'
    validation: 'npm run lint'
  - id: AC-005
    text: 'Browser smoke test: loading the page in a visible browser shows the HUD status bar with elements in order heat bar, K:, portrait, D:, ammo bar — no console errors'
    validation: 'browser-harness-specialist or Chrome DevTools MCP real-browser smoke test'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '01-verify-rebuild'
    title: 'Verify source order, run tests, rebuild bundle'
    status: '[WIP]'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/neatenstein/browser-entry/host/hud.ts'
      - 'examples/neatenstein/browser-entry/host/hud-status-bar.test.ts'
      - 'docs/assets/neatenstein.bundle.js'
    acceptance_criteria:
      - id: AC-006
        text: 'hud-status-bar.test.ts AC-1006 tests pass (element order verified)'
        validation: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar'
      - id: AC-007
        text: 'neatenstein bundle rebuilt successfully (npm run build:neatenstein exit 0)'
        validation: 'npm run build:neatenstein'
      - id: AC-008
        text: 'Lint passes (npm run lint exit 0, no new errors in hud.ts)'
        validation: 'npm run lint'
    parallelizable: false
    dependencies: []
    next_slice: '01-green'
  - slice_id: '01-green'
    title: 'Green validation and browser smoke test'
    status: '[PLANNED]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'docs/assets/neatenstein.bundle.js'
    acceptance_criteria:
      - id: AC-009
        text: 'Real browser smoke test on visible window: HUD shows heat bar, K:, portrait, D:, ammo bar order, no console errors'
        validation: 'browser-harness-specialist or Chrome DevTools MCP real-browser smoke test'
    parallelizable: false
    dependencies:
      - '01-verify-rebuild'
    next_slice: 'null'
```

**User instruction:** Paste this full step packet.

**Step objective:** Verify the source `createNeonStatusBar()` has the correct element append order, run the existing test suite to confirm, rebuild the neatenstein bundle so the browser reflects the updated order, and run a real browser smoke test.

**Context the agent must know:**
- The source code in `hud.ts` already has the desired DOM append order (from a prior phase). This step primarily needs to **verify** the order is correct, **rebuild** the stale bundle, and **confirm** in a browser.
- The built bundle (`docs/assets/neatenstein.bundle.js`, last built 5:56 PM) is stale — it still has the old K:-first order. The source was updated at 6:19 PM.
- Test `AC-1006` in `hud-status-bar.test.ts` (lines 378–458) already verifies the desired order.
- The mugshot canvas is absolutely positioned at `left: 50%` with `translateX(-50%)`, so its visual position is centered regardless of DOM order — only the flex-item elements (health segments, HIVE density, K:, D:, ammo segments) are affected by DOM order.
- This is a DEMO/UI slice — real browser visible-window validation is MANDATORY per the execute skill.

**Execution steps:**
1. Read `examples/neatenstein/browser-entry/host/hud.ts` and verify the `createNeonStatusBar()` function appends elements in this order: health segments → HIVE density track → K: prefix + label → mugshot canvas → D: prefix + label → ammo segments. If the order is already correct, no source changes are needed.
2. Run the focused test suite: `npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar`. All AC-1006 tests must pass.
3. Run lint: `npm run lint`. Confirm no new errors in `hud.ts`.
4. Rebuild the bundle: `npm run build:neatenstein`. Confirm exit 0 and that `docs/assets/neatenstein.bundle.js` is regenerated.
5. Verify the rebuilt bundle contains the correct append order by searching the bundle for the element creation/append sequence (health segments before K:, D: before ammo segments).
6. Run a real browser smoke test: load the Neatenstein page in a visible browser window, inspect the HUD status bar at the bottom of the screen, confirm the visual order is: heat bar → K: → portrait → D: → ammo bar, and check the console for runtime errors. Use the `browser-harness-specialist` or Chrome DevTools MCP for the browser test.

**Stop conditions:**
- DONE: All tests pass, bundle rebuilt, lint clean, browser smoke confirms correct visual order with no console errors.
- BLOCKED: Source order is wrong and cannot be fixed within slice scope, or build fails, or browser smoke reveals a runtime error.
- ROUTE-BACK: If the source order is incorrect, fix `createNeonStatusBar()` append order and re-run tests before rebuilding the bundle.

**Required validation:**
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar` — exit 0
- `npm run build:neatenstein` — exit 0
- `npm run lint` — exit 0
- Browser smoke: visible window, HUD order confirmed, no console errors

**Plan update requirement:** Update this plan with the validation evidence (test output, build result, lint result, browser smoke result) and set Step 01 status to [DONE] before ending. If the bundle was rebuilt, note the new bundle timestamp.

**Traceability:**

```yaml
traceability:
  - id: AC-002
    criterion: 'createNeonStatusBar() DOM append order matches desired sequence'
    files_changed:
      - 'examples/neatenstein/browser-entry/host/hud.ts'
    validation_command: 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=hud-status-bar'
  - id: AC-003
    criterion: 'Built bundle reflects same append order as source'
    files_changed:
      - 'docs/assets/neatenstein.bundle.js'
    validation_command: 'npm run build:neatenstein'
  - id: AC-004
    criterion: 'Lint passes with no new errors on hud.ts'
    files_changed:
      - 'examples/neatenstein/browser-entry/host/hud.ts'
    validation_command: 'npm run lint'
  - id: AC-005
    criterion: 'Browser smoke test shows correct visual order, no console errors'
    files_changed:
      - 'docs/assets/neatenstein.bundle.js'
    validation_command: 'browser-harness-specialist or Chrome DevTools MCP real-browser smoke test'
```

## Clarifications

- Q: Source already has desired order — is this just a rebuild? → A: Yes, primarily. Verify source, rebuild bundle, confirm in browser. If source order is wrong for any reason, fix it first.

## Latest validation evidence

- slice-advancement gate: pass (all 4 sub-gates passed: plan-sync, step-packet, plan-slice-quality, plan-command-lint) — slice 01-plan, TRIVIAL severity, 0 specialists required.
- Plan registered in plans/README.md and plans/Roadmap.md.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.

Execute Step 01 of plans/neatenstein-hud-reorder.plans.md: Verify the createNeonStatusBar() DOM append order in examples/neatenstein/browser-entry/host/hud.ts is health segments → HIVE density → K: → mugshot → D: → ammo segments. Run hud-status-bar tests. Rebuild the neatenstein bundle (npm run build:neatenstein). Run lint. Run a real browser smoke test to confirm the HUD visual order is heat bar → K: → portrait → D: → ammo bar with no console errors. Model mandate: glm-5.2:cloud.
```