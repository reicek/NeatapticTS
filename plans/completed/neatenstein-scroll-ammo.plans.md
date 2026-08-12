**Status:** [DONE]

**Plan ID:** NEATENSTEIN_SCROLL_AMMO

**Source of truth:** `plans/neatenstein-scroll-ammo.plans.md`

## Mandates

- `model: kimi-k2.7-code:cloud` for every dispatch under this plan. Every agent dispatch overrides its frontmatter `model` with `kimi-k2.7-code:cloud`.
- Pragmatic mode: broad slices (one slice per feature), bypass legacy ceremony (skip plan-verification green-light cycle, skip per-AC gate calls, skip fix-packet YAML ceremony). Ship working software.
- Targeted tests only: never run the full test suite in a single shell invocation; use `--testPathPatterns` focused on the touched module.
- No deferred cleanup: when replacing code, remove the old code in the same slice that introduces the new code. No backward-compatibility wrappers, no dual-path code.
- Human-played mode (`humanMode`) remains fully functional — the AI-path changes must not break the human input router.

## Scope

Add two independent gameplay/AI improvements to the Neatenstein demo:

1. **Natural scroll/turn rate limit + accel/decel movement weight.** Cap the rate of look and movement change on the NEAT/auto paths and apply per-tick acceleration / deceleration smoothing so the AI player does not instantaneously snap between full reverse and full forward or between zero and max turn.
2. **AI hero ammo pickup awareness.** Give the NEAT/main agent (and the fallback auto-mode AI) the three closest active ammo pickups by path distance, expose them as new sensors, add a small fitness/reward bonus for collecting ammo, and keep enemies as the primary target (pickups are opportunistic, enemies are not ignored).

**Non-goals:** New weapons, enemy evolution, rendering changes, level generation, WebGPU/perf work, full regression matrices.

## File Impact Map

| Feature                        | Primary Files                                                                                                                                                                                                                              | Secondary Files                                                                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scroll/turn rate + accel/decel | `examples/neatenstein/browser-entry/worker/display.worker.ts`, `examples/neatenstein/browser-entry/harness/neat-io-config.ts`                                                                                                              | `examples/neatenstein/browser-entry/host/game/tick.ts` (look cap already imported), `examples/neatenstein/browser-entry/host/game/movement.ts` (if speed scaling changes) |
| AI ammo pickup                 | `examples/neatenstein/scripts/enemy-navigation.ts`, `examples/neatenstein/browser-entry/worker/display.worker.ts`, `examples/neatenstein/browser-entry/harness/neat-io-config.ts`, `examples/neatenstein/browser-entry/host/game/state.ts` | `examples/neatenstein/browser-entry/harness/fitness.ts` (pickup bonus)                                                                                                    |

All paths are relative to the repository root unless otherwise noted.

## Acceptance Criteria Summary

- id: AC-001
  text: Shared turn-rate cap and movement smoothing constants live in `neat-io-config.ts` and are imported by the worker controller.
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/.*neat-io-config`
- id: AC-002
  text: `buildAutoTickInput` and `buildFallbackAutoTickInput` use smoothed move/look values so 0 -> max transitions are damped by per-tick accel/decel.
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/.*display.worker`
- id: AC-003
  text: Any old direct move/look mapping in the NEAT and fallback paths is removed in the same smoothing slice.
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/.*display.worker`
- id: AC-004
  text: `enemy-navigation.ts` exposes a function that returns the 3 closest active ammo pickups ordered by path distance, with Euclidean fallback when unreachable.
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/scripts/enemy-navigation`
- id: AC-005
  text: New ammo-pickup sensors are wired into the NEAT observation vector and the main input count in `neat-io-config.ts` is updated (with champion extinction guard if input size changes).
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/(scripts/enemy-navigation|browser-entry/worker/display.worker)`
- id: AC-006
  text: Fitness/reward receives a small bonus when ammo is collected and enemy combat remains the dominant objective (ammo is opportunistic).
  validation: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neatenstein/.*fitness`
- id: AC-007
  text: `humanMode` and fallback auto-mode both remain playable; no rendering or input router regressions.
  validation: manual smoke check via browser harness (or `npm run build` when no browser is available)
- id: AC-008
  text: Build and lint pass on touched files.
  validation: `npm run build` and `npm run lint`

---

### Phase 1 — Natural Scroll/Turn Rate Limit + Accel/Decel [DONE]

[DONE] Phase 1: scroll/turn smoothing implemented and green-validated. Slice `P1S1-smoothing` touched `neat-io-config.ts` and `display.worker.ts`; focused Jest suites passed, coverage 100% on touched files, build/lint clean. Detailed history in `plans/neatenstein-scroll-ammo.logs.md`.

### Phase 2 — AI Hero Ammo Pickup [DONE]

[DONE] Phase 2: AI ammo-pickup sensors and fitness bonus implemented and green-validated. Slice `P2S1-ammo-pickup` touched `enemy-navigation.ts`, `neat-io-config.ts`, `display.worker.ts`, `fitness.ts`, `tick.ts`, and supporting files; focused Jest suites passed, coverage ≥99.4% on touched files, build/lint clean. Detailed history in `plans/neatenstein-scroll-ammo.logs.md`.

---

## Final State

Both phases are complete and green-validated. The Neatenstein demo now has natural scroll/turn smoothing and AI ammo-pickup awareness.

## Audit Summary

- Files changed: see `plans/neatenstein-scroll-ammo.logs.md` for the full list.
- Validations:
  - `slice-advancement` content gates passed for both phases.
  - `shared-validation` sub-gate initially errored with `spawnSync node ETIMEDOUT` (tooling failure) during both implementation phases; it was re-run manually and passed.
  - Specialist review: APPROVE for both implementation slices.
  - Green testing: focused Jest suites passed, build/lint clean, coverage green on touched files.
- Residual gaps:
  - 15 stale singular `--testPathPattern` flags remain in the plan; all validation commands used plural `--testPathPatterns` and passed. A plan-format patch by `01-planning` is recommended if this plan is reopened as a template.
  - `examples/neatenstein/browser-entry/host/game/tick.ts` line 436 (the `telemetry.ammoPickupsCollected` increment) is reachable but uncovered by the focused `tick.test.ts` run.
  - Minor AC-ID numbering mismatch between the top-level summary and slice-level criteria.

## Reopen Conditions

- New gameplay/AI improvements beyond scroll smoothing and ammo pickup.
- Browser smoke-test failures or `humanMode` playability regressions.
- Coverage closure for the reachable `tick.ts` line 436 telemetry path.
