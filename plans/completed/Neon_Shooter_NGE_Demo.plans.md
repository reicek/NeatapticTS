# Neatenstein NGE Demo (alias "Neat Shooter")

**Status:** [DONE] - Phase 1 [DONE] | Phase 2 [DONE] | Phase 3 [DONE] | Phase 4 [DONE] | Phase 5 [DONE] | Phase 6 [DONE] | Phase 7 [DONE] | Phase 8 [DONE] | **Plan ID:** NEATENSTEIN_NGE_DEMO | **Created:** 2026-07-17 | **Closed:** 2026-08-07
**Consensus:** 4 specialists (NGE Core, NGE Benchmark, Visualizer, Game Director) � all APPROVED after 2 review rounds.
**Downstream of:** `plans/completed/NEAT_Genesis_EvoDevo.md` (NGE core), `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` (co-evolution harness reference, not duplicated).
**Engine research:** `plans/completed/Neon_Shooter_NGE_Demo.research.md` � DOOM/raycasting algorithm notes, neon renderer design (Lineage B grid DDA, locked), Flappy ground grid reuse, license attribution, and reuse map. **Read this before implementing Phase 1.**
**Rendering direction:** Lineage B (grid DDA raycasting) � locked. See research file �1.

**Sprite rendering resolution mandate:** All robot/enemy sprites are authored and rendered at a logical resolution of **48�48 pixels**, then scaled 4� to 192�192 for display. The renderer, sprite projection, raycasting collision checks, and voxel calculations must operate on the 48�48 logical grid and only scale at the final blit. This preserves the reference artwork exactly while keeping CPU cost ~16� lower than native 192�192 per-pixel operations. The `examples/neatenstein/robot-sprite-data.js` module is the source-of-truth encoded sprite set (8 directions � 4 poses: `stand`, `walk1`, `walk2`, `shoot`). Walk cycle: `stand ? walk1 ? stand ? walk2`. The `shoot` frame uses semitransparent muzzle-blast palette indices 7/8 so the blast can be overlaid on any walk frame with a natural glow; recoil and cannon pixels remain opaque. Art is user-approved and locked.

**Model mandate:** All dispatches under this plan use `glm-5.2:cloud`. No Chrome MCP / browser DevTools MCP � validation is jest-based only (no visible-browser smoke tests via MCP specialists).

---

## Final state

All 8 phases [DONE]. Full step packets, fix packets, and validation evidence compressed to `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

- Phase 8 [DONE]: Enemy death derez animation fix, HUD health/ammo display, ammo drops from dying enemies. 234/234 tests pass, tsc/lint clean, 100% coverage on all 4 touched source files (combat.ts, tick.ts, bolt-render.ts, display.worker.ts). 4 fix-loop iterations closed all coverage gaps.
- Deferred items: 10.2 perf observation 9 addressed in Step 10.3 (`10.3-perf-cleanup`). No outstanding deferred items.

## Implementation phases

### Phase 1 � Arena + Hero FPS controls (game-director-owned) [DONE]

[DONE] Phase 1 complete. Full step packet and validation evidence archived in `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

### Phase 2 � Raycast Renderer + WebGL VFX (visualizer-owned) [DONE]

[DONE] Phase 2 complete. Full step packet and validation evidence archived in `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

### Phase 3 � Live Enemy Rendering + Polish (visualizer + benchmark-owned) [DONE]

Steps 01�10.3 [DONE] � compressed to `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

#### Step 10.3: Render cap visual fixes, perf analysis, rAF clock [DONE]

Step 10.3 [DONE] � compressed to logs. Render cap visual fixes (fog, enemy cull, floor/ceiling tunnel), rAF clock with backpressure, worker-paced render loop. 4 iterations. 195 tests pass, 100% coverage. See `plans/completed/Neon_Shooter_NGE_Demo.logs.md` �Phase 3 Step 10.3 final compression.

#### Step 10.4: Fix enemy wall spawn, maze-aware pathfinding, 30-cell fog [DONE]

Step 10.4 [DONE] � compressed to logs. Fix enemy wall spawn, maze-aware BFS pathfinding, 30-cell fog, walk animation, collision radius 0.25, corridor centering, pre-collision centering, turn centering nudge, wall-aware flanking, stall fallback, framerate-scaled nudge, one-sided diagonal gap centering, BFS stall-recovery, retry-after-block. 6 iterations. 174 tests, 100% coverage on all touched files. User e2e approved. See plans/completed/Neon_Shooter_NGE_Demo.logs.md �Phase 3 Step 10.4 final compression.

#### Step 10.5: Real MLP neural network enemy AI � replacing stub BFS-only navigation [DONE]

Step 10.5 [DONE] � compressed to logs. Real MLP neural network enemy AI: replace stub BFS-only with MLP-re-ranked BFS, real bounded rollouts, Lamarckian warm-start, composite fitness. 5 slices: vision-inputs, mlp-wiring, episode-rollouts, warm-start, fitness-shaping. 1095/1097 neatenstein tests pass (2 pre-existing failures: arms-race timing flake + generate-enemy-sprites ENOENT). tsc/lint clean, 100% coverage on fitness.ts. See `plans/completed/Neon_Shooter_NGE_Demo.logs.md` �Phase 3 Step 10.5 final compression.

#### Step 11: Gameplay adjustments � view distance, combat, enemy fire, death effects [DONE]

- **Status:** All 5 slices [DONE] � 11-view-distance, 11-combat-rebalance, 11-enemy-impact, 11-enemy-fire, 11-death-effects.
- **Coverage:** Compressed to `plans/completed/Neon_Shooter_NGE_Demo.logs.md` ? Step 11.
- **Compressed by:** 07-logging phase agent.

---

#### Step 12: Enhance cannon overlay � fix horizontal stretch, add detail, voxel 3D look via sprite projection [DONE]

Step 12 [DONE] � compressed to `plans/completed/Neon_Shooter_NGE_Demo.logs.md` �Phase 3 Step 12 final compression. Cannon overlay improvements: `gun.ts` uses aspect-correct `gunWidth = gunHeight * GUN_BODY_ASPECT_RATIO`, added barrel bands / side vents / top sight / energy-core rings, and integrated `gun-sprite.ts` voxel projection. Four slices (`12-red-gun`, `12-aspect-detail`, `12-voxel-sprite`, `12-green`) all [DONE]. 14 tests pass, `gun.ts` and `gun-sprite.ts` 100% coverage, tsc/lint/prettier clean, visible-browser smoke pass. Parent README route tables updated.

### Phase 4 � NGE Main Agent + Enemy MLPs (core + benchmark-owned) [DONE]

Coverage notes: Phase 4 Steps 01-07 completed; full step packets and validation evidence archived to `plans/completed/Neon_Shooter_NGE_Demo.logs.md` (Phase 4 final compression section).

Required gates at compression: `phase-compression` plan-level gate; `slice-advancement` Phase4-Step07 recorded pass (tooling failure noted as warning per policy).

### Phase 5 � SWARM Mode (core + benchmark-owned) [DONE]

**Goal:** WeightSharedCohort swarm + HIVE DENSITY legibility.

[DONE] Phase 5 complete. Full step packets and validation evidence archived in `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

Required gates at compression: `phase-compression` plan-level gate; `slice-advancement` Phase5-Step07 recorded pass.

### Phase 6 � Human Modes + Replay Buffer (benchmark + game-director-owned) [DONE]

[DONE] Phase 6 complete. Steps 01-07 all [DONE]. Full step packets, PlanUpdate blocks, and validation evidence archived in `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

Required gates at compression: `phase-compression` plan-level gate; `slice-advancement` Phase6-Step07 recorded pass.

---

## Phase 7: Visual Polish & Gameplay Features [DONE]

[DONE] Phase 7 complete. Research findings in `plans/completed/Neon_Shooter_NGE_Demo.research.md`. Full step packets archived in `plans/completed/Neon_Shooter_NGE_Demo.logs.md`.

Required gates at compression: `phase-compression` plan-level gate; `slice-advancement` Phase7-Step01 recorded pass.

---

## Phase 8: Visual Polish & Gameplay Features Implementation [DONE]

[DONE] Phase 8 complete. All 7 steps [DONE] (derez fix, HUD health/ammo, ammo drops). 234/234 tests pass, tsc/lint clean, 100% coverage on all 4 touched source files (combat.ts, tick.ts, bolt-render.ts, display.worker.ts). 4 fix-loop iterations closed all coverage gaps. Full step packets and validation evidence archived in plans/completed/Neon_Shooter_NGE_Demo.logs.md.

Required gates at compression: phase-compression plan-level gate; slice-advancement Phase8-Step07 recorded pass (4/4 sub-gates, TRIVIAL severity); code-coverage PASS (all 4 files at 100%).
