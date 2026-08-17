/**
 * Deterministic fixed-timestep game tick for the Neatenstein host-side
 * simulation.
 *
 * This module owns the top-level world update pipeline: one input snapshot in,
 * one immutable {@link GameState} out, advanced by one simulation timestep.
 *
 * The tick function deliberately remains thin. Movement, combat, collision,
 * dashing, and episode progression are delegated to focused subsystem helpers
 * so each part can be tested independently.
 *
 * @module
 */

import type { CollisionMap } from '../../renderer/map';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
import { updateEpisode } from './episode';
import { updatePlayerMovement } from './movement';
import { applyDash } from './state';
import type { GameState } from './types';

import { resolveCollisionMap } from './tick.collision-map.utils';
import { resolveTickDurationMs } from './tick.time.utils';
import {
  normalizeGameTickInput,
  snapshotToMovement,
  type GameTickInputSnapshot,
} from './tick.input.utils';
import { applyLook } from './tick.look.utils';
import { updateBolts } from './tick.bolt.utils';
import { updateEnemyBolts } from './tick.enemy-bolt.utils';
import { ageImpacts, ageEnemyImpacts } from './tick.impact.utils';
import { decayGunRecoil } from './tick.gun.utils';
import { updateAmmoPickups } from './tick.pickups.utils';
import {
  applyBoltImpactResults,
  respawnHeroOnDeath,
  applyFireAndRecoil,
} from './tick.lifecycle.utils';

export { createGameState } from './state';
export {
  NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
  NEATENSTEIN_ENEMY_BOLT_SPEED_CELLS_PER_SECOND,
} from './constants';
export { NEATENSTEIN_FIXED_TIMESTEP_MS };
export { updateBolts } from './tick.bolt.utils';
export { updateEnemyBolts } from './tick.enemy-bolt.utils';
export { updateAmmoPickups } from './tick.pickups.utils';
export { decayGunRecoil } from './tick.gun.utils';
export { ageImpacts, ageEnemyImpacts } from './tick.impact.utils';
export type { GameTickInputSnapshot };

/**
 * Advance the deterministic game state by one simulation timestep.
 *
 * Declarative pipeline:
 *
 * 1. Resolve timestep, collision map, and normalized input.
 * 2. Advance episode systems (spawning, timers, contact damage).
 * 3. Apply player look.
 * 4. Apply dash if requested.
 * 5. Resolve player movement against collision.
 * 6. Move player bolts and process enemy hits.
 * 7. Age impacts and decay gun recoil.
 * 8. Move enemy bolts and apply player damage on hit.
 * 9. Respawn hero on death.
 * 10. Update ammo pickups.
 * 11. Fire if requested (after bolt movement so new bolts start at muzzle).
 *
 * @param state - Snapshot before this tick.
 * @param snapshot - Partial input snapshot for this tick.
 * @param collisionMap - Optional collision map; when omitted a deterministic
 *   map is resolved from {@link GameState.seed}.
 * @param dtMs - Duration of this tick in milliseconds.
 * @returns A new immutable {@link GameState} advanced by the resolved timestep.
 */
export function gameTick(
  state: GameState,
  snapshot: Partial<GameTickInputSnapshot>,
  collisionMap?: CollisionMap,
  dtMs: number = NEATENSTEIN_FIXED_TIMESTEP_MS,
): GameState {
  // Step 1: Resolve timestep, collision map, and normalized input.
  const resolvedDtMs = resolveTickDurationMs(dtMs);
  const input = normalizeGameTickInput(snapshot);
  const map = resolveCollisionMap(state, collisionMap);

  // Step 2: Advance deterministic world/episode systems.
  let next = updateEpisode(state, resolvedDtMs, map);

  // Step 3: Apply yaw input before movement so movement uses the new facing.
  next = applyLook(next, input.lookDelta);

  // Step 4: Apply dash before movement so movement can consume updated player
  // state such as dash velocity, cooldown, or flags.
  if (input.dash) {
    next = applyDash(next);
  }

  // Step 5: Resolve player movement against the collision map.
  next = updatePlayerMovement(
    next,
    snapshotToMovement(input.move),
    map,
    resolvedDtMs,
  );

  // Step 6: Move active player bolts and process enemy hits.
  const updatedBolts = updateBolts(
    next.bolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next.enemies,
  );
  const boltImpact = applyBoltImpactResults(next, updatedBolts, next.simTimeMs);
  next = boltImpact.state;

  // Step 7: Age impacts and decay gun recoil.
  next = {
    ...next,
    bolts: updatedBolts.filter((bolt) => bolt.active),
    impacts: ageImpacts(next.impacts, resolvedDtMs),
    enemyImpacts: ageEnemyImpacts(next.enemyImpacts ?? [], resolvedDtMs),
    gun: decayGunRecoil(
      next.gun ?? { recoilOffset: 0, firing: false },
      resolvedDtMs,
    ),
    lastShotHit: boltImpact.boltHitEnemy,
  };

  // Step 8: Move active enemy bolts and apply player damage on hit.
  const enemyBoltResult = updateEnemyBolts(
    next.enemyBolts ?? [],
    resolvedDtMs,
    next.simTimeMs,
    map,
    next,
  );
  next = enemyBoltResult.state;
  next = {
    ...next,
    enemyBolts: enemyBoltResult.bolts.filter((bolt) => bolt.active),
  };

  // Step 9: Respawn the hero at the map center if health has been depleted.
  next = respawnHeroOnDeath(next);

  // Step 10: Update ammo pickups — collect by proximity and expire by lifetime.
  next = updateAmmoPickups(next, next.simTimeMs);

  // Step 11: Fire after updating bolts so newly spawned bolts start at the
  // muzzle and are advanced on the following tick.
  if (input.fire) {
    const fireResult = applyFireAndRecoil(next);
    next = fireResult.state;
  }

  return next;
}
