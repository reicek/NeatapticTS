# Neatenstein Scroll + Ammo Pickup — Workstream Log

**Status:** [DONE]

**Source of truth:** `plans/neatenstein-scroll-ammo.plans.md`

**Scope:** Add natural scroll/turn rate smoothing and AI ammo-pickup awareness to the Neatenstein demo.

---

## Plan Registration

- Created `plans/neatenstein-scroll-ammo.plans.md` with pragmatic mandates, phase/step packets, and gate evidence.
- Registered in `plans/README.md` and `plans/Roadmap.md`.
- `slice-advancement` gate passed for plan registration: plan-sync, step-packet, plan-slice-quality, plan-command-lint.

---

## Phase 1 — Natural Scroll/Turn Rate Limit + Accel/Decel [DONE]

**Goal:** Add shared rate-limit and smoothing constants, integrate them into both the champion-network and fallback auto tick builders, and verify with focused unit tests.

### Step 01: Plan Phase 1 implementation [DONE]

- Authored Step 02 implementation packet with one broad slice and clear acceptance criteria.
- Confirmed smoothing boundary and mapped each acceptance criterion to a focused Jest command.

### Step 02: Implement scroll/turn smoothing [DONE]

- Slice: `P1S1-smoothing`
- Files changed:
  - `examples/neatenstein/browser-entry/harness/neat-io-config.ts` — added shared turn-rate cap and movement smoothing constants
  - `examples/neatenstein/browser-entry/worker/display.worker.ts` — `buildAutoTickInput` and `buildFallbackAutoTickInput` now use smoothed move/look values
  - `examples/neatenstein/browser-entry/harness/neat-io-config.test.ts` — updated tests
  - `examples/neatenstein/browser-entry/worker/display.worker.test.ts` — updated tests
- Validation:
  - `npx tsc --noEmit -p tsconfig.json` → OK
  - `npm run lint` → 0 issues
  - `npx prettier --check .` → OK
  - Targeted Jest with `--testPathPatterns`:
    - `neatenstein/.*neat-io-config` → 34 passed, 0 failed
    - `neatenstein/.*display.worker` → 137 passed, 0 failed
  - `npm run build` → exit 0
- Green validation evidence:
  - Coverage on touched files: 100% statements/branches/functions/lines for `neat-io-config.ts` and `display.worker.ts`
  - `slice-advancement` gate: content gates passed; `shared-validation` sub-gate errored with `spawnSync node ETIMEDOUT` (tooling failure), then re-ran manually and passed
  - Specialist review: APPROVE

---

## Phase 2 — AI Hero Ammo Pickup [DONE]

**Goal:** Expose the three closest active ammo pickups by path distance, add new NEAT sensors, reward ammo collection, and keep enemy combat the primary objective.

### Step 01: Plan Phase 2 implementation [DONE]

- Authored the single broad implementation step for AI ammo pickup.
- Confirmed acceptance criteria and left Phase 2 ready for `04-implementing`.

### Step 02: Implement AI ammo pickup sensors and reward [DONE]

- Slice: `P2S1-ammo-pickup`
- Files changed:
  - `examples/neatenstein/scripts/enemy-navigation.ts` — added `findNearestAmmoPickups` using BFS path distance with Euclidean fallback
  - `examples/neatenstein/browser-entry/harness/neat-io-config.ts` — expanded NEAT input count from 15 to 22, added ammo-pickup sensor constants
  - `examples/neatenstein/browser-entry/worker/display.worker.ts` — wired new ammo-pickup sensors into observation vector
  - `examples/neatenstein/browser-entry/host/game/state.ts` — added `telemetry.ammoPickupsCollected`
  - `examples/neatenstein/browser-entry/host/game/tick.ts` — ammo pickup collection and telemetry increment
  - `examples/neatenstein/browser-entry/harness/fitness.ts` — added small ammo-pickup fitness bonus (`NEATENSTEIN_WEIGHT_AMMO_PICKUP_BONUS = 1`)
  - `examples/neatenstein/browser-entry/harness/constants.ts`, `types.ts`
  - `examples/neatenstein/browser-entry/harness/main-runner.ts`, `worker/eval.worker.ts` — threaded collision map into `extractSensors`
  - Test files updated: `enemy-navigation.test.ts`, `neat-io-config.test.ts`, `fitness.test.ts`, `display.worker.test.ts`
- Validation (04-implementing preflight):
  - `npx tsc --noEmit -p tsconfig.json` → exit 0
  - `npm run lint` → exit 0
  - `npx prettier --check .` → exit 0
  - `npm run build` → exit 0
  - Targeted Jest with `--testPathPatterns`:
    - `neatenstein/scripts/enemy-navigation` → 70 passed, 0 failed
    - `neatenstein/browser-entry/harness/neat-io-config` → 34 passed, 0 failed
    - `neatenstein/browser-entry/harness/fitness` → 11 passed, 0 failed
    - `neatenstein/browser-entry/host/game/tick` → 76 passed, 0 failed
    - `neatenstein/browser-entry/worker/display.worker -t "sensor|champion"` → 16 passed, 0 failed, 119 skipped
- Fix packet `fix-packet-P2S1-ammo-pickup-iteration-1`:
  - Trigger: specialist-review REQUEST_CHANGES
  - Observations addressed:
    - Wired `NEATENSTEIN_AMMO_PICKUP_START_INDEX` into `enemy-navigation.ts` sensor loop
    - Added tests for `findNearestAmmoPickups` (BFS path distance, Euclidean fallback, equal-distance tie-break)
    - Added tests for ammo sensor slots (bearing, distance, low-ammo gate)
    - Added test for `NEATENSTEIN_WEIGHT_AMMO_PICKUP_BONUS` in `fitness.ts`
    - Added tests for new constants in `neat-io-config.ts`
    - Added doc comment for equal-distance tie-break rule
    - Cleaned up stale test comments in `display.worker.test.ts`
  - Post-fix validation:
    - `tsc`: OK
    - `lint`: 0 issues
    - `prettier`: formatted changed files
    - `enemy-navigation` tests: 77 passed
    - `fitness` + `neat-io-config` tests: 50 passed
    - `display.worker` tests: 135 passed
    - Specialist review: APPROVE
- Green validation evidence (05-green-testing):
  - `npm run build` → exit 0
  - `npm run lint` → exit 0
  - Targeted Jest with coverage:
    - `neatenstein/scripts/enemy-navigation` → 77 passed, 0 failed
    - `neatenstein/browser-entry/harness/neat-io-config` → 38 passed, 0 failed
    - `neatenstein/browser-entry/harness/fitness` → 12 passed, 0 failed
    - `neatenstein/browser-entry/host/game/tick` → 76 passed, 0 failed
    - `neatenstein/browser-entry/worker/display.worker` → 137 passed, 0 failed (2 suites)
  - `shared-validation` gate: passed (10 suites, 397 tests)
  - `slice-advancement` gate: content gates passed; `shared-validation` had earlier tooling timeout but re-ran green
  - Coverage summary (touched files):
    - `enemy-navigation.ts`: statements 100%, branches 97.91%, functions 100%, lines 100%
    - `neat-io-config.ts`: 100% across all categories
    - `fitness.ts`: 100% across all categories
    - `display.worker.ts`: 100% across all categories
    - `tick.ts`: statements 99.41%, branches 98.15%, functions 100%, lines 99.41% (uncovered line 436 is telemetry increment path)
  - Verdict: GREEN_OK_WITH_GAP (single reachable uncovered line in `tick.ts`)

---

## Residual Risks / Gaps

- `examples/neatenstein/browser-entry/host/game/tick.ts` line 436 (`telemetry.ammoPickupsCollected` increment) is reachable but not covered by the focused `tick.test.ts` run.
- The plan file still contains 15 stale singular `--testPathPattern` flags. All green validation used plural `--testPathPatterns`. A plan-format patch by `01-planning` is recommended before the plan is used as a template.
- AC-ID numbering has minor mismatches between the top-level summary and slice-level criteria.
- The validation MCP workflow snapshot still pointed to a non-existent `plans/neatenstein-hunter-bugfix.plans.md` during the green phase; validations were run directly from the plan's test list.

---

## Reopen Conditions

- New gameplay/AI improvements beyond scroll smoothing and ammo pickup.
- Browser smoke-test failures or regression in `humanMode` playability.
- Coverage closure for the reachable `tick.ts` line 436 telemetry path.
