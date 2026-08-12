**Status:** [DONE]
**Plan ID:** NEATENSTEIN_FIRING_SENSOR_SYSTEM
**Created:** 2026-08-10
**Source of truth:** `plans/neatenstein-firing-sensor-system.plans.md`
**Research artifact:** `plans/neatenstein-firing-sensor-system.research.md` (1562 lines)

## Mandates

- `model: glm-5.2:cloud` for all dispatches under this plan. Every dispatch overrides the frontmatter `model` with `glm-5.2:cloud`. Do NOT use kimi-k2.7-code:cloud or any other model.
- Pragmatic mode: broad slices (one per gap area), bypass legacy ceremony (skip plan-verification green-light cycle, skip per-AC gate calls, skip fix-packet YAML ceremony). Ship working software.
- Do not archive or supersede existing Neatenstein plans (`neatenstein-auto-neat-mode.plans.md`); this workstream is additive.
- No deferred cleanup: when replacing code (placeholder fitness, duplicate constants, old sensor layout), remove the old code in the same slice that introduces the new code. No dual-path code, no backward-compatibility wrappers.
- Human-played mode must remain fully functional — the `humanMode` selector is the switch point.

## Scope

Fix the Neatenstein NEAT-controlled player's firing behavior through five phases: (1) foundation — shared NEAT I/O constants, maxNodes/maxConns enforcement, aimMissRate sync bug fix; (2) fitness replacement + dedicated evaluation worker — replace the placeholder fitness with real episode-based evaluation; (3) sensor expansion 12→15 with vision range, line-of-sight, and normalization; (4) reward redesign with shot-outcome telemetry; (5) fire gating — suppress fire when no enemy is visible.

**Non-goals:** Enemy NEAT evolution (covered by `neatenstein-auto-neat-mode.plans.md`), arms-race wiring, game engine changes, rendering/sprite work.

## Research Summary

The research artifact (`plans/neatenstein-firing-sensor-system.research.md`) identifies 5 solutions, 10 NEAT architecture findings (N1-N10), 6 reward deep-dive findings (R1-R6), and performance analysis. Key findings:

| ID | Finding | Severity |
|----|---------|----------|
| N2 | `NEATENSTEIN_MAIN_NEAT_INPUTS`/`OUTPUTS` duplicated independently in `main-runner.ts` and `display.worker.ts` | BLOCKING |
| N8 | Worker `new Neat()` not passed `maxNodes: 64, maxConns: 256` | HIGH |
| N3 | `MAX_TURN_RATE` hardcoded as magic number in tick/turn logic — should be extracted as a named constant | MEDIUM |
| N9 | Sensors not normalized (position, ammo, wall dist, bearing use raw values) | MEDIUM |
| N10 | Placeholder fitness returns `output[0]` — sensor changes are inert without real fitness | CRITICAL |
| Sol 1 | Soft fire gate: suppress fire only when NO enemy visible | — |
| Sol 2 | Vision range `VISION_RANGE_CELLS = 15`; zero out enemy sensors [5]-[7] beyond range | — |
| Sol 3 | Line-of-sight via `castRayDDAFromFlatMap`; add `hasLineOfSight` to `raycast.ts` | — |
| Sol 4 | Add `shotsWasted` telemetry, `ammoEfficiency`→`killEfficiency`, scale `aimMissRate` weight 1→~20 | — |
| Sol 5 | Sensors [12]-[14]: `enemyVisible`, `enemyInFiringArc`, `lastShotHit`; total 15 inputs | — |
| Rec 5 | Dedicated evaluation worker is P0 — without it, render loop stalls 500-2000ms per generation | — |
| R1 | Survival reward (200-312) overwhelms accuracy penalties (~17.4) | — |
| R2 | Replace `ammoEfficiency` with `killEfficiency` multiplier on kills | — |
| R5 | Per-shot outcome taxonomy: wallHit, blindFire, nearMiss, rangeExpired | — |

**Recommended sequencing (research):** N10 → N2+N8 → N9+sensor expansion → fitness weight tuning → fire gating
**User-specified phase ordering:** Foundation → Fitness+Eval → Sensors → Reward → Fire gating (compatible with research — foundation doesn't depend on fitness, and fitness replacement is Phase 2, still early).

## File Impact Map

| Phase | Touches | Key Files |
|-------|--------|-----------|
| 1 | Shared constants + maxNodes + aimMissRate | `harness/neat-io-config.ts` (NEW), `harness/main-runner.ts`, `worker/display.worker.ts`, `host/game/combat.ts`, `host/game/tick.ts` |
| 2 | Fitness replacement + eval worker | `worker/display.worker.ts`, `harness/fitness.ts`, `worker/eval.worker.ts` (NEW) |
| 3 | Sensor expansion 12→15 + LOS + normalization | `renderer/raycast.ts`, `scripts/enemy-navigation.ts`, `harness/neat-io-config.ts`, `host/game/tick.ts`, `worker/display.worker.ts` |
| 4 | Reward redesign + telemetry | `host/game/types.ts`, `host/game/combat.ts`, `harness/fitness.ts`, `harness/constants.ts`, `host/game/tick.ts`, `scripts/enemy-navigation.ts` |
| 5 | Fire gating | `host/game/tick.ts`, `harness/neat-io-config.ts` |

All paths relative to `examples/neatenstein/browser-entry/` unless prefixed with `scripts/` (relative to `examples/neatenstein/`).


---

## Phase 1 — Foundation [DONE]

[DONE] Phase 1: Shared NEAT I/O config extracted (neat-io-config.ts), maxNodes/maxConns enforced, aimMissRate telemetry sync verified, MAX_TURN_RATE extracted. 2 slices (P1S1-shared-config, P1S1-aimmissrate). See `plans/neatenstein-firing-sensor-system.logs.md` for detailed evidence.

---

## Phase 2 — Fitness Replacement + Dedicated Eval Worker [DONE]

[DONE] Phase 2: Placeholder fitness removed, real episode-based evaluation via computeCombatQualitySignal, dedicated eval.worker.ts offloads evaluation from render loop. 2 slices (P2S1-fitness-replace, P2S1-eval-worker). See `plans/neatenstein-firing-sensor-system.logs.md` for detailed evidence.

---

## Phase 3 — Sensor Expansion 12→15 with Vision/LOS/Normalization [DONE]

[DONE] Phase 3: hasLineOfSight DDA utility added, extractSensors expanded 12→15 with normalization, vision range pre-filtering, LOS filtering, champion extinction guard on input count change. 3 slices (P3S1-los-utility, P3S1-sensor-expand, P3S1-clear-champions). See `plans/neatenstein-firing-sensor-system.logs.md` for detailed evidence.

---

## Phase 4 — Reward Redesign with New Telemetry [DONE]

[DONE] Phase 4: Shot outcome taxonomy (wallHit/rangeExpired/blindFire/nearMiss), killEfficiency multiplier replacing ammoEfficiency, aimMissRate weight scaled 1→20, PBRS potential function, rate metrics (hitRate/killRate/fireRate), NaN guard with max(shotsFired,1). 2 slices (P4S1-telemetry, P4S1-fitness-weights). See `plans/neatenstein-firing-sensor-system.logs.md` for detailed evidence.

---

## Phase 5 — Fire Gating [DONE]

[DONE] Phase 5: Soft fire gate in buildAutoTickInput suppresses output[3] when no enemy visible. Hysteresis on enemyVisible sensor (fire-on 0.18, fire-off 0.15) prevents oscillation. 1 slice (P5S1-fire-gate). See `plans/neatenstein-firing-sensor-system.logs.md` for detailed evidence.

---

## Latest validation evidence

**Final green validation (05-green-testing @ 2026-08-10T18:00:00Z):** GREEN: OK

- `npm run build`: PASS (tsc exit 0, webpack 0 errors)
- `npm run lint`: PASS (0 issues)
- `npx jest --testPathPatterns=neatenstein`: PASS (75 suites, 1581 passed, 1 skipped, 0 failed)
- Coverage: 8/10 files 100%; display.worker.ts 97.59% (browser-only paths); eval.worker.ts 0% (Web Worker, untestable in Node Jest)
- code-coverage gate: PASS | plan-sync gate: PASS | slice-advancement gate: TOOLING ERROR (ETIMEDOUT -- infrastructure issue per Section 5.8.3)

All 10 slices confirmed GREEN. Plan complete. Detailed evidence in `plans/neatenstein-firing-sensor-system.logs.md`.

