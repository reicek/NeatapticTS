# Neatenstein vision-system wiring fixes

**Status:** [DONE]
**Plan ID:** NEATENSTEIN_VISION_WIRING_FIXES
**Created:** 2026-08-10
**Closed:** 2026-08-10
**Source of truth:** `plans/completed/neatenstein-vision-wiring-fixes.plans.md`
**Log:** `plans/completed/neatenstein-vision-wiring-fixes.logs.md`
**Research artifact:** `plans/neatenstein-auto-neat-mode.research.md`

Pragmatic fix plan for five wiring gaps in the Neatenstein demo's vision/shooter integration.

## Scope

- Vision-aware fallback auto-tick input (LOS, vision range, fire gate).
- Correct wave-clear detection.
- Thread fire gate through fitness evaluation paths.
- Delete dead shooter harness code.
- Remove unused PBRS/sensorHistory plumbing.

## Final state

All five slices implemented and green-validated:

- `02-vision-fallback` — `display.worker.ts` and `enemy-navigation.ts` wired for vision-aware fallback.
- `02-wave-clear` — `display.worker.ts` wave-clear detection uses alive-enemy predicate.
- `02-fire-gate` — `main-runner.ts` and `eval.worker.ts` pass fire-gate config into fitness episodes.
- `02-dead-shooter` — removed `enemy-runner.ts`, `main-agent.ts`, `barrier.ts` and unused fitness/constants/types exports.
- `02-pbrs-cleanup-green` — removed `sensorHistory`/`SensorSnapshot` and PBRS helpers; green validation passed.

Cross-slice validation: 1512/1512 Neatenstein tests pass, `tsc` and `npm run lint` clean.

## Latest validation evidence

- `npx jest --config=jest.config.mjs --no-cache --runInBand --testPathPatterns=examples/neatenstein` → 73/73 suites, 1512/1512 tests pass (1 skipped).
- `npx tsc --noEmit -p tsconfig.json` → pass.
- `npx tsc --noEmit -p tsconfig.neatenstein.json` → pass.
- `npm run lint` → pass.
- `slice-advancement` gate passed for `02-fire-gate` and `02-pbrs-cleanup-green`; remaining slices accepted by `05-green-testing` after tooling-timeout / pre-existing coverage triage.

## Implementation phases

### Phase 1 — Vision wiring fixes [DONE]

**Phase objective:** Author and execute five focused fixes that close vision-system wiring gaps in Neatenstein: vision-aware fallback auto-tick input, correct wave-clear detection, fire-gate threading through fitness evaluation, deletion of dead shooter code, and removal of unused PBRS/sensorHistory plumbing.

**Status:** All five slices implemented and green-validated; plan ready for compression and archival.

**Stop conditions:** Any fix reveals a deeper architecture conflict (e.g., fire-gate config not available in eval worker), scope expands beyond the five listed gaps, or the user requests a different model mandate.

## Audit summary

- 5/5 slices implemented and green-validated.
- 1512/1512 Neatenstein tests pass; `tsc` (main + Neatenstein configs) and `npm run lint` clean.
- Two `slice-advancement` gate calls (vision-fallback, wave-clear) failed only on `shared-validation` tooling timeout and pre-existing `display.worker.ts` eval-helper coverage gaps, both triaged and accepted by the final `05-green-testing` run.
- Dead-code deletion slice left `fitness.ts`/`constants.ts` below 100% on residual unreachable/old branches; full suite green accepts this pragmatically.
- No library (`src/`) changes, no dual paths left in examples.

## Reopen conditions

- Regression in Neatenstein suite >0 failures attributable to these changes.
- New consumer discovered for removed symbols (`sensorHistory`, PBRS helpers, enemy-team fitness helpers).
- Parent plan `neatenstein-auto-neat-mode.plans.md` requires a back-merge of these fixes.

## Audit log

- 2026-08-10 - Created pragmatic child plan to honor `kimi` model mandate.
- 2026-08-10 - All five slices implemented.
- 2026-08-10 - Full Neatenstein green run (1512/1512) accepted by `05-green-testing`.
- 2026-08-10 - Compressed to `.logs.md` and plan trimmed to `[DONE]` closure shape.
- 2026-08-10 - Archived to `plans/completed/`.

## Handoff

This workstream is complete. To resume the parent Neatenstein work, load `plans/neatenstein-auto-neat-mode.plans.md` and merge the closure pointer from this plan's log.
