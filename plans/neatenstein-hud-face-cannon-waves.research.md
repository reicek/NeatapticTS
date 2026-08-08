# Neatenstein HUD / Face / Cannon / Waves — Research Synthesis

**Plan:** `neatenstein-hud-face-cannon-waves`  
**Research date:** 2026-08-07  
**Sources inspected:** `examples/neatenstein/` host, renderer, game-logic, worker, and asset files.

## 1. HUD redesign

- Current HUD factories live in `examples/neatenstein/browser-entry/host/hud.ts`:
  - `createHiveDensityHud`
  - `createHealthAmmoHud`
  - `createDeathFeedbackIndicator`
  - `createHumanModeSelector`
- They append bare `<div>` elements to `#neatenstein-output` **below** the canvas.
- Existing color tokens (locked) are in `examples/neatenstein/browser-entry/constants.ts`:
  - `NEATENSTEIN_GUN_BODY_COLOR = '#FBFFFF'` (neon white)
  - `NEATENSTEIN_GUN_ACCENT_COLOR = '#00f0ff'` (neon teal)
  - `NEATENSTEIN_HEALTH_COLOR_CYAN = 'rgb(0, 240, 255)'`
  - `NEATENSTEIN_HEALTH_COLOR_AMBER = 'rgb(240, 160, 0)'`
  - `NEATENSTEIN_HEALTH_COLOR_MAGENTA = 'rgb(255, 0, 85)'`
  - HIVE density thresholds/colors already defined.
- Test pattern: `createHudFixture` mounts `#neatenstein-hud-output` in jsdom, dynamically imports `hud.ts`, and asserts `style.backgroundColor` / `textContent`.
- Direction: convert the HUD into a single bottom status-bar overlay with neon borders, monospace/CRT labels, segmented vitals bars, and a left/right split for the robot mugshot / kill-death counters.

## 2. Robot mugshot / face

- Robot sprite source-of-truth: `examples/neatenstein/robot-sprite-data.json`.
  - Atlas frames: `front`, `frontRight`, `right`, `backRight`, `back`, `backLeft`, `left`, `frontLeft`.
  - Each frame contains `stand`, `walk1`, `walk2`, `shoot` (48×48 palette-indexed grids).
  - Palette indices of interest for the head region: `1` outline, `2` helmet shell, `5` eye-stripe/accent.
- Head-only crop: logical rows roughly `0–14`, x approximately `19–29` of the 48×48 grid. Apply the same crop to `front`, `frontLeft`, and `frontRight` frames so the mugshot is consistently head-only in every direction.
- Mugshot frame selection:
  - `front.stand` when idle (neither or both strafe keys held).
  - `frontLeft.stand` when strafing left (`InputSnapshot.movement.left` true and right false).
  - `frontRight.stand` when strafing right (`InputSnapshot.movement.right` true and left false).
  - Left takes precedence when both keys are held simultaneously; no state strobe if both/neither.
  - Maintain a short anti-flicker cooldown (≈100 ms) per direction; reset cooldown when the active frame changes, not on every key press.
- Damage tint:
  - Healthy eye-stripe/accent (`palette index 5`) → neon teal `rgb(0,240,255)`.
  - At death → neon gray `rgb(180,190,210)` (same neutral used by `NEATENSTEIN_ENEMY_DEATH_COLOR`).
  - Interpolate by `playerHealth / playerMaxHealth` for the eye-stripe only; outline/helmet shell stay unchanged.
- Implementation location: host DOM `<canvas>` overlay (not worker) so it can be driven cheaply from host input state.
- Existing decode/tint helpers (`decodeRobotSpriteFrame`, `buildTeamColorPalette`) in `examples/neatenstein/browser-entry/renderer/sprites.ts` are not exported and live inside the worker sprite renderer. Plan a new pure shared module `examples/neatenstein/browser-entry/renderer/robot-sprite-decode.ts` that exports the helpers and constants; both `sprites.ts` and the host mugshot import from it.
- The mugshot tint targets **only** palette index `5` (eye-stripe/accent); indices `6`/`7` are inert in the head region but are swapped by `buildTeamColorPalette`, so the helper must accept a per-index tint map.

## 3. Voxel cannon

- `examples/neatenstein/browser-entry/renderer/gun.ts` draws the gun body with vector gradients and side planes.
- `examples/neatenstein/browser-entry/renderer/gun-sprite.ts` only projects a 5×5 barrel cap (`GUN_BARREL_VOXEL_GRID`).
- **Finding: no full voxel player-weapon descriptor exists.** Only the small barrel cap is wired into `renderGunOverlay`.
- Plan: create a new procedural weapon descriptor module (`examples/neatenstein/scripts/voxel-gun.ts`) that imports `Voxel`/`VoxelGrid` from `scripts/voxel-enemy.ts` and describes a chunky rotary receiver + barrel cluster in neon white/teal/dark vent colors, with an optional muzzle-flash burst on fire.
- Wire into `renderGunOverlay` so the vector body can be optionally replaced/augmented by the voxel projection.
- The `GunState` firing signal must be tick-derived: set true in the `tick.ts` fire block and reset false in `decayGunRecoil`; this keeps the muzzle-flash deterministic and avoids wall-clock animation state in `GameState`.

## 4. Death / respawn / kill counter

- `GameState` (`examples/neatenstein/browser-entry/host/game/types.ts`) already has `kills: number` and `spawnCount: number`.
- Add `deaths: number` to `GameState`, initialized to `0` in `createGameState`.
- Death occurs when `state.player.health <= 0`.
- `examples/neatenstein/browser-entry/host/game/episode.ts#isEpisodeComplete` currently terminates on player death OR after `NEATENSTEIN_ENEMY_MAX_CONCURRENT * NEATENSTEIN_ENEMY_WAVE_COUNT` kills.
- For infinite waves, the player-death condition must be replaced by a respawn hook and `deaths++`; the `allEnemiesKilled` wave-count terminal condition must be removed.
- Respawn behavior:
  - Detect `playerDead` at the **end** of `gameTick`, then increment `deaths`, reset player position to `(NEATENSTEIN_SPAWN_CENTER_X, NEATENSTEIN_SPAWN_CENTER_Y)`, restore `health`/`ammo` to max, clear residual `contactIFrameMs`, and grant a short post-respawn invulnerability window (e.g. `NEATENSTEIN_RESPAWN_INVULN_MS`) so enemies adjacent to the center do not immediately kill the player again.
  - Update `state.ts#isInvulnerable()` to return true while `respawnInvulnMs > 0`.
  - Clear `bolts`, `enemyBolts`, `ammoPickups`, `impacts`, `enemyImpacts`; preserve the live `enemies` array (it is not reset on respawn).
  - Preserve `seed`, `kills`, `deaths`, `spawnCount`, `generation` across respawn. `spawnCount` must remain monotonic and unbounded for infinite-wave determinism.
- Episode completion: replace the `playerDead` terminal condition with a time-based guard (`episodeTimeMs >= episodeDurationMs`). The episode is considered active while the timer has not expired; respawns happen within the same episode. Update existing `episode.test.ts` expectations accordingly.
- The kill counter is already incremented in `examples/neatenstein/browser-entry/host/game/combat.ts#applyEnemyDamage`.
- Forward `kills` and `deaths` in `NeatensteinRenderFrame` so the HUD can display them.

## 5. Infinite enemy waves

- `examples/neatenstein/browser-entry/host/game/waves.ts#spawnWaveTick` spawns one enemy per tick until `maxSpawnCount = NEATENSTEIN_ENEMY_MAX_CONCURRENT * NEATENSTEIN_ENEMY_WAVE_COUNT`.
- It also pauses a full batch until all 8 enemies are dead, then starts a new batch — behavior we can reuse.
- For infinite waves: remove the `maxSpawnCount` cap so `spawnCount` can grow without bound.
- Keep the existing 8-enemy batch semantics; when the roster is cleared, the next tick begins respawning the next 8 on their initial edge spots.
- Because `spawnCount` is used as a monotonic RNG seed offset (`state.seed + state.spawnCount`), removing the cap preserves determinism as long as the seed arithmetic stays unchanged.

## 6. Worker / host boundary

- Worker: `examples/neatenstein/browser-entry/worker/display.worker.ts` owns simulation and packs `NeatensteinRenderFrame`.
- Host: `examples/neatenstein/browser-entry/browser-entry.ts` consumes frames via `bridge.setFrameConsumer`.
- `NeatensteinRenderState` already forwards `movement.left/right` to the worker.
- For the mugshot, the host can read `InputSnapshot.movement.left/right` directly; no worker round-trip is required for frame selection.
- The CPU/GPU fallback path already populates scalar HUD fields (`playerHealth`, `playerAmmo`, etc.). The primary worker-tier OffscreenCanvas path currently posts back only a minimal `{ type: 'frame', frame: { requestId } }` ack; it must be extended to forward scalar HUD fields (`playerHealth`, `playerMaxHealth`, `playerAmmo`, `playerMaxAmmo`, `playerKills`, `playerDeaths`) so the host HUD can read them.
- New frame fields needed: `playerKills`, `playerDeaths` (numeric). Optional: `mugshotFrame` if the worker is chosen to author the frame, but consensus is host-side.

## 7. Frame protocol

- `examples/neatenstein/browser-entry/renderer/frame.ts` defines `NeatensteinRenderFrame`.
- Add optional numeric fields: `playerKills`, `playerDeaths`.
- Update `buildNeatensteinRenderFrame` and `resolveNeatensteinRenderFrameTransferList` only if new typed arrays are added; plain number fields do not require transfer entries.

## 8. Test patterns

- HUD host tests: `examples/neatenstein/browser-entry/host/hud-health-ammo.test.ts`, `hud-death-feedback.test.ts`, `hud-human-mode.test.ts`.
- Game-logic tests: co-located `*.test.ts` files using `createGameState`, `gameTick`, etc.
- Renderer canvas tests: mock 2D context with `jest.fn()` stubs and assert `fillRect` / `fillStyle` calls.

## 9. Open assumptions

- The attached Doom-style chaingun reference image is not readable by agents; planning is based on the textual description and repo evidence.
- Wave respawn timing: existing trickle is one enemy per tick after batch clear. A simultaneous 8-enemy burst is not currently implemented and is called out as an implementation decision in the plan.
- Mugshot placement and status-bar pixel sizes will be finalized during red-phase host tests, not in this research file.

## 10. Plan revisions from review cycle

- **Phase 2 HUD protocol slice (02-protocol)** added because the worker-tier OffscreenCanvas path only posts `requestId`; scalar HUD fields must be forwarded on the worker path before the mugshot/HUD can consume them.
- **New `host/game/respawn.ts` module** extracted so both live and headless paths can call a single `respawnPlayer(state)` implementation, instead of duplicating reset logic in `tick.ts` and episode tests.
- **`constants.ts` expanded** with `NEATENSTEIN_RESPAWN_INVULN_MS` and center-spawn constants, rather than hard-coding values inside the tick loop.
- **Episode terminal-condition ownership split:**
  - Phase 5 removes `playerDead` from `isEpisodeComplete` (episode now ends on time only).
  - Phase 6 removes `allEnemiesKilled` and the wave-count cap from `waves.ts`.
- **Voxel cannon cleanup and firing signal:** Phase 4 splits into asset, renderer-wiring, and simulation-wiring slices. The old 5×5 monochrome `GUN_BARREL_VOXEL_GRID` dense height grid, monochrome projector, and obsolete `gun-sprite.test.ts` are deleted in the same renderer-wiring slice that rewires `gun.ts`. A tick-derived `firing` boolean is added to `GunState` in a separate simulation slice (`types.ts`, `tick.ts`).
- **Voxel type ownership:** `scripts/voxel-gun.ts` imports `Voxel`/`VoxelGrid` from `scripts/voxel-enemy.ts` rather than redefining them.
- **Mugshot implementation helper split:** Phase 3 now creates `renderer/robot-sprite-decode.ts` first, then `host/hud-mugshot.ts` consumes it; both are covered by red tests and the cleanup of duplicate private helpers happens in the same step.
- **Respawn helper extraction:** Phase 5 now has a dedicated `05-respawn-module` slice for `host/game/respawn.ts` plus a `05-respawn-wiring` slice for `tick.ts` and `episode.ts`, keeping each slice at ≤3 files.
