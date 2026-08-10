# Neatenstein HUD, Robot Mugshot, Voxel Cannon & Infinite Waves

**Status:** [DONE]
**Plan ID:** NEATENSTEIN_HUD_FACE_CANNON_WAVES
**Created:** 2026-08-07
**Closed:** 2026-08-10
**Source of truth:** `plans/completed/neatenstein-hud-face-cannon-waves.plans.md`
**Research artifact:** `plans/completed/neatenstein-hud-face-cannon-waves.research.md`
**Compressed log:** `plans/completed/neatenstein-hud-face-cannon-waves.logs.md`

## Scope

Update the `examples/neatenstein/` demo so that:

1. Its HUD / status indicators evoke Wolfenstein 3D but use the project's neon palette.
2. A front-view robot mugshot is derived from the existing `robot-sprite-data.json`, showing `frontLeft` / `frontRight` when strafing and tinting from neon teal to neon gray by damage.
3. The on-screen cannon is rebuilt as a 2D palette-indexed sprite asset (`examples/neatenstein/gun-sprite-data.js`) analogous to `robot-sprite-data.js`, evoking a Wolfenstein 3D chaingun in the project neon palette, and wired into the existing render pipeline.
4. Player death respawns the hero at the maze center and increments a `deaths` counter.
5. Enemy waves are infinite: when all 8 current enemies die, the next 8 respawn on their initial edge spots.

## Non-goals

- No changes to the core NeatapticTS library (`src/`).
- No changes to the robot sprite source JSON (`examples/neatenstein/robot-sprite-data.json`); it is read-only source-of-truth.
- No new enemy AI behavior, maze generation, WebGPU tier, or NGE harness integration.
- No sound asset changes.

## Final state

All 10 phases completed and validated.

- [DONE] Phase 1: Plan lock and acceptance criteria.
- [DONE] Phase 2: Neon Wolfenstein-style HUD status bar overlay, old flex HUD factories removed, render frame extended with scalar HUD fields. 100% coverage, browser smoke PASS.
- [DONE] Phase 3: Robot mugshot canvas overlay with strafe-driven frame selection and damage tint. Shared decode module extracted from `sprites.ts`. 100% coverage, browser smoke PASS.
- [DONE] Phase 4: Voxel cannon rebuilt through 3 iterations (procedural voxel → Doom-style neon → Wolfenstein palette-indexed chaingun). Final: `gun-sprite-data.js` + `gun-sprite-decode.ts` + `gun.ts`. GUN_BODY_ASPECT_RATIO 1.6. 40/40 tests, 100% coverage, browser smoke PASS.
- [DONE] Phase 5: Death / respawn / kill counter — superseded by Phase 8 bug fix 2b-02. No separate implementation.
- [DONE] Phase 6: Infinite enemy waves — superseded by Phase 8 bug fix 2b-04. No separate implementation.
- [DONE] Phase 7: Integration and final review — superseded by Phase 8 validation (342 tests, 100% coverage) and Phase 9 archive.
- [DONE] Phase 8: Four game-logic bug fixes (stale enemy index crash, hero respawn, spawn-at-corners, infinite waves). 342 tests, 100% coverage, browser smoke PASS.
- [DONE] Phase 9: HUD layout (K: left, D: right, mugshot centered) and mouse-driven mugshot heading (yawDelta replaces keyboard strafe). 66/66 tests, 100% coverage, browser smoke PASS.
- [DONE] Phase 10: HUD status bar element reorder (health → HIVE track → K: → mugshot → D: → ammo). 69/69 tests, ESLint clean, tsc clean.

## Audit summary

- All 10 phases marked [DONE] with recorded validation evidence.
- All step/slice YAML blocks, validation evidence, and implementation details compressed to `neatenstein-hud-face-cannon-waves.logs.md`.
- No open steps, no stale WIP markers, no unresolved validation gaps.
- Pre-existing residual: GPU type errors in `src/architecture/network/gpu/` (unrelated to this plan), 28 ESLint warnings in non-touched files.
- slice-advancement gate: PASS for all executed slices.
- No learning events needed — no agent-system gaps, routing changes, or output-contract fixes occurred during this plan.

## Reopen conditions

This plan is terminally closed. New HUD reorder work should go in a **separate new plan**, not this one. To reopen:

1. Move the plan + log pair from `plans/completed/` back to `plans/`.
2. Add a fresh `Handoff query` section.
3. Do not reuse stale closure-era prompts.

## Audit log

- 2026-08-07: Plan created, registered in README and Roadmap.
- 2026-08-09: Phases 1-4 completed. Phase 4 reopened for Step 04c (palette-indexed chaingun redesign).
- 2026-08-10: Plan corruption fixed (status mismatches corrected). Phases 5-10 completed.
- 2026-08-10: Plan compressed and closed. All phase details moved to `.logs.md`. Plan + log pair archived to `plans/completed/`.
