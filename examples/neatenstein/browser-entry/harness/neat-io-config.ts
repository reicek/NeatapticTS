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
import type { FireGateState, FireGateConfig, FireGateResult } from './types';
import {
  DASH_THRESHOLD,
  NEAT_OUTPUT_INDEX_DASH,
  NEAT_OUTPUT_INDEX_FIRE,
} from './enemy-mlp.constants';

/**
 * Mutable hysteresis state for the soft fire gate.
 *
 */
export type { FireGateState, FireGateResult } from './types';

/**
 * Optional fire-gate configuration passed to {@link networkOutputToTickInput}.
 *
 */
export type { FireGateConfig } from './types';

/**
 * Number of main-agent NEAT network inputs (sensor vector length).
 *
 * The main-agent sensor vector grew from 15 to 22 in P2S1 to expose the
 * three nearest active ammo pickups by path distance (bearing + distance
 * per pickup, plus a low-ammo gate). Consumers must use this constant
 * rather than hard-coding the vector length so the genome-extinction guard
 * stays synchronized with the sensor layout.
 *
 * @see AC-039, AC-P2S1-001
 */
export const NEATENSTEIN_MAIN_NEAT_INPUTS = 22;

/**
 * Number of main-agent NEAT network outputs (action vector length).
 *
 * @see AC-039
 */
export const NEATENSTEIN_MAIN_NEAT_OUTPUTS = 5;

// ---------------------------------------------------------------------------
// P2S1 — Ammo-pickup awareness sensors
// ---------------------------------------------------------------------------

/**
 * Fraction of max ammo below which the low-ammo gate sensor activates.
 *
 * Sensor [21] emits `1` when `player.ammo / player.maxAmmo` is strictly
 * below this threshold, and `0` otherwise. The 0.25 value keeps the gate
 * off until the hero is genuinely low, avoiding noisy activation near full
 * ammo.
 *
 * @see AC-P2S1-001
 */
export const NEATENSTEIN_LOW_AMMO_RATIO = 0.25;

/**
 * Number of consecutive ammo-pickup sensors appended to the main sensor
 * vector in P2S1.
 *
 * Each of the three nearest active pickups contributes a bearing and a
 * distance (6 sensors), and a single low-ammo gate adds one more, for a
 * total of 7 extra inputs on top of the original 15.
 *
 * @see AC-P2S1-001
 */
export const NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT = 7;

/**
 * Starting index of the ammo-pickup sensor block in the 22-element vector.
 *
 * Sensors [15]–[20] hold the three nearest pickups (bearing/distance pairs),
 * and sensor [21] is the low-ammo gate. Keeping this index in shared config
 * lets sensor consumers and tests refer to the ammo block without magic
 * numbers.
 *
 * @see AC-P2S1-001
 */
export const NEATENSTEIN_AMMO_PICKUP_START_INDEX = 15;

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
// P1S1 — Scroll/turn rate cap + accel/decel movement smoothing
// ---------------------------------------------------------------------------

/**
 * Maximum per-tick increase in the main-agent move/look command.
 *
 * Damps 0 → max transitions so the AI does not snap instantly from
 * standstill to full forward/strafe/turn.
 *
 * @see AC-002, AC-003
 */
export const NEATENSTEIN_MOVE_ACCEL_PER_TICK = 0.2;

/**
 * Maximum per-tick decrease in the main-agent move/look command.
 *
 * Damps max → 0 transitions so the AI decelerates smoothly when it
 * stops or reverses.
 *
 * @see AC-002, AC-003
 */
export const NEATENSTEIN_MOVE_DECEL_PER_TICK = 0.25;

/**
 * Smooth a scalar command toward its target using per-tick accel/decel limits.
 *
 * @param current - Current smoothed value.
 * @param target - Desired raw value.
 * @param accel - Maximum increase per tick (defaults to {@link NEATENSTEIN_MOVE_ACCEL_PER_TICK}).
 * @param decel - Maximum decrease per tick (defaults to {@link NEATENSTEIN_MOVE_DECEL_PER_TICK}).
 * @returns Updated smoothed value.
 * @see AC-003, AC-004
 */
export function smoothCommand(
  current: number,
  target: number,
  accel = NEATENSTEIN_MOVE_ACCEL_PER_TICK,
  decel = NEATENSTEIN_MOVE_DECEL_PER_TICK,
): number {
  const delta = target - current;
  if (delta === 0) return current;
  const limit = delta > 0 ? accel : decel;
  return Math.abs(delta) <= limit ? target : current + Math.sign(delta) * limit;
}

// ---------------------------------------------------------------------------
// P5S1 — Soft fire gate with hysteresis on enemyVisible sensor
// ---------------------------------------------------------------------------

/**
 * Sensor index for `enemyVisible` in the 22-element observation vector.
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
 * The `state` object is NOT mutated — a new {@link FireGateResult} is returned
 * containing the updated state and fire decision. Callers must persist
 * `result.state` between ticks to maintain hysteresis.
 *
 * @param state - Current hysteresis state (not mutated).
 * @param enemyVisible - Current value of the enemyVisible sensor (sensor[12]).
 * @param rawFireOutput - Raw network output[3] (fire activation).
 * @returns A {@link FireGateResult} with the new state and fire decision.
 * @see AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003
 */
export function applyFireGate(
  state: FireGateState,
  enemyVisible: number,
  rawFireOutput: number,
): FireGateResult {
  // Compute new hysteresis state based on enemyVisible sensor.
  let fireActive = state.fireActive;
  if (enemyVisible >= FIRE_GATE_HYSTERESIS_HIGH) {
    fireActive = true;
  } else if (enemyVisible < FIRE_GATE_HYSTERESIS_LOW) {
    fireActive = false;
  }
  // Between thresholds: maintain current state (hysteresis band).

  const newState: FireGateState = { fireActive };

  // If the gate is closed (no enemy visible), suppress fire entirely.
  if (!fireActive) {
    return { state: newState, shouldFire: false };
  }

  // Gate is open — apply the normal fire threshold.
  return { state: newState, shouldFire: rawFireOutput > 0 };
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
    ? (() => {
        const result = applyFireGate(
          fireGate.state,
          fireGate.enemyVisible,
          out[NEAT_OUTPUT_INDEX_FIRE],
        );
        fireGate.state = result.state;
        return result.shouldFire;
      })()
    : out[NEAT_OUTPUT_INDEX_FIRE] > 0;

  return {
    move: {
      x: Math.tanh(out[0]),
      y: Math.tanh(out[1]),
    },
    lookDelta: Math.tanh(out[2]) * MAX_TURN_RATE,
    fire,
    dash: out[NEAT_OUTPUT_INDEX_DASH] > DASH_THRESHOLD,
  };
}
