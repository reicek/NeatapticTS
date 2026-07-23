/**
 * Deterministic fixed-timestep game tick for the Neatenstein host-side
 * simulation.
 *
 * This module owns the top-level world update pipeline: one input snapshot in,
 * one immutable {@link GameState} out, advanced by exactly one fixed timestep.
 * It is intentionally thin — the heavy lifting lives in the movement, combat,
 * collision, and episode helpers so each subsystem can be tested in isolation.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../../constants';
import { fireNeonBeam } from './combat';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
import { updateEpisode } from './episode';
import { updatePlayerMovement } from './movement';
import {
  buildNeatensteinMap,
  createCollisionMap,
  type CollisionMap,
} from '../../renderer/map';
import { applyDash, createGameState } from './state';
import type { GameState, ImpactSpot, TracerState, Vector2 } from './types';

export { createGameState };
export { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';

/**
 * Normalized input snapshot consumed by {@link gameTick}.
 *
 * The shape is deliberately clone-safe so the same snapshot can be forwarded
 * from the host input router, across the worker boundary, and replayed for
 * deterministic evolution evaluation.
 */
export interface GameTickInputSnapshot {
  /** Player-local movement vector; `x=-1` left, `x=1` right, `y=1` forward. */
  move: Vector2;
  /** Horizontal look delta to apply this tick, in radians. */
  lookDelta: number;
  /** `true` when the fire action is held this tick. */
  fire: boolean;
  /** `true` when the dash action is requested this tick. */
  dash: boolean;
}

/**
 * Advance the deterministic game state by one fixed timestep.
 *
 * The pipeline is intentionally ordered so timing, spawning, and contact
 * damage are resolved by {@link updateEpisode}, then player input (look, dash,
 * movement, fire) is applied for the new tick. This keeps the episode clock,
 * enemy waves, and melee damage in one place while making the player-driven
 * transitions explicit.
 *
 * @param state - Snapshot before this tick.
 * @param snapshot - Input snapshot for this tick; missing fields default to
 *   zero/no-action so the worker can forward partial snapshots safely.
 * @param collisionMap - Optional collision map; when omitted a map is built
 *   deterministically from {@link GameState.seed}.
 * @param dtMs - Duration of this tick in milliseconds; defaults to the fixed
 *   simulation timestep.
 * @returns A new immutable {@link GameState} advanced by `dtMs`.
 *
 * @example
 * ```ts
 * const state = createGameState({ seed: 7 });
 * const next = gameTick(state, {
 *   move: { x: 0, y: 1 },
 *   lookDelta: 0.05,
 *   fire: true,
 *   dash: false,
 * });
 * expect(next.simTimeMs).toBe(state.simTimeMs + NEATENSTEIN_FIXED_TIMESTEP_MS);
 * ```
 */
export function gameTick(
  state: GameState,
  snapshot: Partial<GameTickInputSnapshot>,
  collisionMap?: CollisionMap,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  const map =
    collisionMap ??
    createCollisionMap(buildNeatensteinMap(state.seed), NEATENSTEIN_MAP_SIZE);

  let next = updateEpisode(state, dtMs);
  next = applyLook(next, snapshot.lookDelta ?? 0);
  if (snapshot.dash) {
    next = applyDash(next);
  }
  next = updatePlayerMovement(
    next,
    snapshotToMovement(snapshot.move),
    map,
    dtMs,
  );
  next = { ...next, tracers: ageTracers(next.tracers, dtMs) };
  next = { ...next, impacts: ageImpacts(next.impacts, dtMs) };
  if (snapshot.fire) {
    const fireResult = fireNeonBeam(next);
    next = fireResult.state;
  }
  return next;
}

/**
 * Rotate the player by a yaw delta for one tick.
 */
function applyLook(state: GameState, lookDelta: number): GameState {
  if (lookDelta === 0) {
    return { ...state, player: { ...state.player } };
  }
  return {
    ...state,
    player: {
      ...state.player,
      angleRad: state.player.angleRad + lookDelta,
    },
  };
}

/**
 * Age active tracers by one tick and remove any that have expired.
 *
 * Tracers are immutable snapshots; each surviving tracer gets its remaining
 * duration reduced by the elapsed timestep.
 *
 * @param tracers - Active tracer snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of tracers still visible after aging.
 */
export function ageTracers(
  tracers: TracerState[],
  dtMs: number,
): TracerState[] {
  return tracers
    .map((tracer) => ({ ...tracer, durationMs: tracer.durationMs - dtMs }))
    .filter((tracer) => tracer.durationMs > 0);
}

/**
 * Age active wall-impact spots by one tick and remove any that have expired.
 *
 * Impact spots are immutable snapshots; each surviving impact gets its
 * remaining lifetime reduced by the elapsed timestep.
 *
 * @param impacts - Active wall-impact snapshots before this tick.
 * @param dtMs - Elapsed time in milliseconds.
 * @returns New array of impact spots still visible after aging.
 */
export function ageImpacts(impacts: ImpactSpot[], dtMs: number): ImpactSpot[] {
  return impacts
    .map((impact) => ({ ...impact, lifetimeMs: impact.lifetimeMs - dtMs }))
    .filter((impact) => impact.lifetimeMs > 0);
}

/**
 * Convert a movement vector into the directional boolean intent used by the
 * keyboard-based movement helper.
 *
 * A non-zero axis value is treated as pressed on that side. Diagonal vectors
 * preserve both components so {@link updatePlayerMovement} can normalize the
 * final world-space step.
 */
function snapshotToMovement(move?: Vector2): {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
} {
  const x = move?.x ?? 0;
  const y = move?.y ?? 0;
  return {
    forward: y > 0,
    backward: y < 0,
    left: x < 0,
    right: x > 0,
  };
}
