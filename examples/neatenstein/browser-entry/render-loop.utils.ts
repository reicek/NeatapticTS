/**
 * Render-loop executors for the Neatenstein browser entrypoint.
 *
 * Pure leaf functions extracted from the `tick` render loop and frame
 * consumer in `browser-entry.ts`. Each executor is stateless and deterministic.
 *
 * @module render-loop.utils
 */

import {
  ADAPTATION_STRONGER,
  ADAPTATION_WEAKER,
  ADAPTATION_SHIFTED,
} from './constants';
import type { NeatensteinRenderState } from './renderer/frame';

// ---------------------------------------------------------------------------
// Timing executors
// ---------------------------------------------------------------------------

/**
 * Compute clamped delta-time from consecutive `requestAnimationFrame`
 * timestamps.
 *
 * On the first frame there is no previous timestamp, so `deltaMs` is 0.
 * The delta is clamped to `maxDeltaMs` so a suspended tab or long frame
 * cannot produce a single physics step large enough to tunnel through walls.
 *
 * @param lastTimestamp - Previous rAF timestamp in ms, or `null` on first frame.
 * @param timestamp - Current rAF timestamp in ms.
 * @param maxDeltaMs - Upper bound for the delta in milliseconds.
 * @returns Clamped delta-time in milliseconds.
 */
export function computeDeltaMs(
  lastTimestamp: number | null,
  timestamp: number,
  maxDeltaMs: number,
): number {
  return Math.min(
    lastTimestamp === null ? 0 : timestamp - lastTimestamp,
    maxDeltaMs,
  );
}

/**
 * Advance the simulation tick proportionally to the frame delta-time.
 *
 * FPS-scaled stepping ensures simulation progress is consistent across
 * varying refresh rates: a 60 Hz display yields ~1 tick/frame, 30 Hz
 * yields ~2, etc. Always advances by at least 1 tick per frame.
 *
 * @param simTick - Current simulation tick value.
 * @param deltaMs - Frame delta-time in milliseconds.
 * @param referenceTimestepMs - Reference timestep (16 ms ≈ one 60 Hz frame).
 * @returns New simulation tick value.
 */
export function advanceSimTick(
  simTick: number,
  deltaMs: number,
  referenceTimestepMs: number,
): number {
  return simTick + Math.max(1, Math.round(deltaMs / referenceTimestepMs));
}

// ---------------------------------------------------------------------------
// Density + wave executors
// ---------------------------------------------------------------------------

/**
 * Compute the hive density ratio (enemy population relative to the
 * concurrency cap), clamped to [0, 1].
 *
 * @param enemyCount - Current number of enemies in the game state.
 * @param maxConcurrent - Maximum concurrent enemies (wave spawn cap).
 * @returns Density ratio in [0, 1].
 */
export function computeHiveDensityRatio(
  enemyCount: number,
  maxConcurrent: number,
): number {
  return Math.min(1, Math.max(0, enemyCount / maxConcurrent));
}

/**
 * Resolve the wave number from the cumulative spawn count.
 *
 * Wave 1 = spawnCount 0 to `maxConcurrent`, Wave 2 = `maxConcurrent` to
 * `2 * maxConcurrent`, etc. Using `(spawnCount - 1)` ensures the wave
 * number only advances when the first enemy of the new wave actually spawns.
 *
 * @param spawnCount - Cumulative enemy spawn count.
 * @param maxConcurrent - Maximum concurrent enemies per wave.
 * @returns 1-based wave number.
 */
export function resolveWaveNumber(
  spawnCount: number,
  maxConcurrent: number,
): number {
  return Math.floor(Math.max(0, spawnCount - 1) / maxConcurrent) + 1;
}

// ---------------------------------------------------------------------------
// Render state executor
// ---------------------------------------------------------------------------

/**
 * Build a {@link NeatensteinRenderState} snapshot from render-loop parameters.
 *
 * Assembles the render state object posted to the display worker on each
 * animation frame. The `enemies` array is always empty on the host side
 * (the worker maintains the live population).
 *
 * @param params - Render state fields.
 * @returns A `NeatensteinRenderState` ready to post to the worker bridge.
 */
export function buildRenderState(params: {
  canvasWidth: number;
  canvasHeight: number;
  simTick: number;
  cameraX: number;
  cameraY: number;
  cameraYaw: number;
  mapSeed: number;
  movement: {
    forward: boolean;
    backward: boolean;
    left: boolean;
    right: boolean;
  };
  deltaMs: number;
  hiveDensity: number;
  humanMode: 'auto' | 'human';
}): NeatensteinRenderState {
  return {
    canvasWidth: params.canvasWidth,
    canvasHeight: params.canvasHeight,
    simTick: params.simTick,
    cameraX: params.cameraX,
    cameraY: params.cameraY,
    cameraYaw: params.cameraYaw,
    mapSeed: params.mapSeed,
    movement: params.movement,
    enemies: [],
    deltaMs: params.deltaMs,
    hiveDensity: params.hiveDensity,
    humanMode: params.humanMode,
  };
}

// ---------------------------------------------------------------------------
// Death feedback executor
// ---------------------------------------------------------------------------

/**
 * Derive the death-feedback adaptation direction from the hive-density delta.
 *
 * The signal mirrors the `AdaptationSignal` shape produced by
 * `computeAdaptationSignal` in the harness death-feedback module so the HUD
 * indicator stays consistent with the arms-race result wiring.
 *
 * @param densityDelta - Change in hive density between consecutive frames.
 * @returns `'stronger'` when density increased, `'weaker'` when it decreased,
 *   `'shifted'` when it stayed roughly the same.
 */
export function deriveDeathFeedbackDirection(
  densityDelta: number,
): 'stronger' | 'weaker' | 'shifted' {
  if (densityDelta > 0.01) return ADAPTATION_STRONGER;
  if (densityDelta < -0.01) return ADAPTATION_WEAKER;
  return ADAPTATION_SHIFTED;
}
