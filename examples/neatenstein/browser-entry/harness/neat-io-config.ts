/**
 * Shared NEAT I/O configuration for the Neatenstein main agent.
 *
 * This module centralises the network input/output sizes, the maximum
 * turn-rate constant, and the {@link networkOutputToTickInput} mapping so
 * that both the headless main-runner and the live worker controller use
 * identical configuration without drift.
 *
 * @module
 */

import type { GameTickInputSnapshot } from '../host/game/tick';

/**
 * Number of main-agent NEAT network inputs (sensor vector length).
 *
 * @see AC-039
 */
export const NEATENSTEIN_MAIN_NEAT_INPUTS = 15;

/**
 * Number of main-agent NEAT network outputs (action vector length).
 *
 * @see AC-039
 */
export const NEATENSTEIN_MAIN_NEAT_OUTPUTS = 5;

/**
 * Maximum look-delta per tick, in radians.
 *
 * The NEAT network's third output is passed through `tanh` (range [-1, 1])
 * and multiplied by this constant to produce the final `lookDelta`.
 * At 16 ms per tick, π/4 ≈ 45° per tick gives a fast but controllable turn
 * rate suitable for the neon raycasting arena.
 *
 * Also imported by {@link gameTick}'s `applyLook` to enforce the turn-rate
 * cap at the simulation level for all callers.
 *
 * @see AC-066
 */
export const MAX_TURN_RATE = Math.PI / 4;

/**
 * Maximum turn rate for the fallback auto-mode AI, in radians per tick.
 *
 * The fallback AI uses this bound when no enemy is visible and it must
 * steer away from an imminent wall (wall-bounce exploration) or turn
 * toward a visible enemy. Keeping it in the shared I/O config ensures the
 * worker controller and any future headless fallback use the same cap.
 *
 * @see buildFallbackAutoTickInput in display.worker.ts
 */
export const NEATENSTEIN_FALLBACK_TURN_RATE = Math.PI / 12;

// ---------------------------------------------------------------------------
// P5S1 — Soft fire gate with hysteresis on enemyVisible sensor
// ---------------------------------------------------------------------------

/**
 * Sensor index for `enemyVisible` in the 15-element observation vector.
 *
 * Binary: 1 if a visible enemy exists within vision range, 0 otherwise.
 *
 * @see AC-P5S1a-001
 */
export const ENEMY_VISIBLE_SENSOR_INDEX = 12;

/**
 * Hysteresis floor — fire gate deactivates (closes) when the enemyVisible
 * sensor drops below this value.
 *
 * @see AC-P5S1a-001, AC-P5S1a-003
 */
export const FIRE_GATE_HYSTERESIS_LOW = 0.15;

/**
 * Hysteresis ceiling — fire gate activates (opens) when the enemyVisible
 * sensor rises above this value.
 *
 * @see AC-P5S1a-002, AC-P5S1a-003
 */
export const FIRE_GATE_HYSTERESIS_HIGH = 0.18;

/**
 * Mutable hysteresis state for the soft fire gate.
 *
 * Maintained between ticks to prevent rapid on/off oscillation at the
 * vision boundary. The `fireActive` flag tracks whether the gate is
 * currently open (enemy was recently visible).
 *
 * @see AC-P5S1a-003
 */
export interface FireGateState {
  fireActive: boolean;
}

/**
 * Create a fresh fire-gate hysteresis state with the gate closed.
 *
 * @returns A new `FireGateState` with `fireActive = false`.
 * @see AC-P5S1a-003
 */
export function createFireGateState(): FireGateState {
  return { fireActive: false };
}

/**
 * Apply the soft fire gate with hysteresis to a raw fire output.
 *
 * Hysteresis logic:
 * - When `enemyVisible >= FIRE_GATE_HYSTERESIS_HIGH` (0.18), the gate opens
 *   and fire is allowed (subject to the normal `rawFireOutput > 0` check).
 * - When `enemyVisible < FIRE_GATE_HYSTERESIS_LOW` (0.15), the gate closes
 *   and fire is suppressed regardless of `rawFireOutput`.
 * - Between the two thresholds, the gate maintains its current state
 *   (hysteresis band — prevents boundary oscillation).
 *
 * The `state` object is mutated in-place to persist hysteresis between ticks.
 *
 * @param state - Mutable hysteresis state (updated in-place).
 * @param enemyVisible - Current value of the enemyVisible sensor (sensor[12]).
 * @param rawFireOutput - Raw network output[3] (fire activation).
 * @returns `true` if fire should be emitted, `false` if suppressed.
 * @see AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003
 */
export function applyFireGate(
  state: FireGateState,
  enemyVisible: number,
  rawFireOutput: number,
): boolean {
  // Update hysteresis state based on enemyVisible sensor.
  if (enemyVisible >= FIRE_GATE_HYSTERESIS_HIGH) {
    state.fireActive = true;
  } else if (enemyVisible < FIRE_GATE_HYSTERESIS_LOW) {
    state.fireActive = false;
  }
  // Between thresholds: maintain current state (hysteresis band).

  // If the gate is closed (no enemy visible), suppress fire entirely.
  if (!state.fireActive) {
    return false;
  }

  // Gate is open — apply the normal fire threshold.
  return rawFireOutput > 0;
}

/**
 * Optional fire-gate configuration passed to {@link networkOutputToTickInput}.
 */
export interface FireGateConfig {
  /** Mutable hysteresis state, persisted between ticks by the caller. */
  state: FireGateState;
  /** Current enemyVisible sensor value (sensor[12]). */
  enemyVisible: number;
}

/**
 * Map a raw NEAT network output vector to a {@link GameTickInputSnapshot}.
 *
 * The mapping uses `tanh` for continuous outputs (move, look) to produce
 * values in [-1, 1] regardless of the network's raw output range, and
 * threshold comparisons for discrete outputs (fire, dash):
 *
 * - `move.x     = tanh(outputs[0])` — strafe
 * - `move.y     = tanh(outputs[1])` — forward/back
 * - `lookDelta  = tanh(outputs[2]) * MAX_TURN_RATE`
 * - `fire       = outputs[3] > 0` (or gated by fire gate when provided)
 * - `dash       = outputs[4] > 0.5`
 *
 * Shorter output vectors are zero-padded to {@link NEATENSTEIN_MAIN_NEAT_OUTPUTS}.
 *
 * When `fireGate` is provided, the fire output is passed through
 * {@link applyFireGate} which suppresses fire when no enemy is visible
 * (enemyVisible sensor below hysteresis floor) and applies hysteresis
 * to prevent boundary oscillation.
 *
 * @param outputs - Raw network activation outputs.
 * @param fireGate - Optional fire-gate configuration for soft fire suppression.
 * @returns A game-tick input snapshot derived from the network output.
 * @see AC-065, AC-066, AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003
 */
export function networkOutputToTickInput(
  outputs: number[],
  fireGate?: FireGateConfig,
): GameTickInputSnapshot {
  const out =
    outputs.length >= NEATENSTEIN_MAIN_NEAT_OUTPUTS
      ? outputs
      : [
          ...outputs,
          ...new Array<number>(
            NEATENSTEIN_MAIN_NEAT_OUTPUTS - outputs.length,
          ).fill(0),
        ];

  const fire = fireGate
    ? applyFireGate(fireGate.state, fireGate.enemyVisible, out[3])
    : out[3] > 0;

  return {
    move: {
      x: Math.tanh(out[0]),
      y: Math.tanh(out[1]),
    },
    lookDelta: Math.tanh(out[2]) * MAX_TURN_RATE,
    fire,
    dash: out[4] > 0.5,
  };
}
