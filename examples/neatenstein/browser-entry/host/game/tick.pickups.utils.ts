/**
 * Ammo-pickup collection and expiry executors for the Neatenstein tick
 * pipeline.
 *
 * @module
 */

import {
  NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS,
  NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS,
} from './constants';
import { restoreAmmo } from './state';
import type { GameState } from './types';

/**
 * Update ammo pickups for one tick: collect by proximity and expire by lifetime.
 *
 * Active pickups within {@link NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS}
 * of the player are collected — the player's ammo is restored via
 * {@link restoreAmmo} and the pickup is marked inactive. Pickups whose
 * lifetime has elapsed are also marked inactive. Inactive pickups are filtered
 * out of the returned state.
 *
 * @param state - Snapshot before the pickup update.
 * @param simTimeMs - Current simulation time in milliseconds.
 * @returns New snapshot with collected/expired pickups removed and ammo
 *   restored for any collected pickups.
 */
export function updateAmmoPickups(
  state: GameState,
  simTimeMs: number,
): GameState {
  const pickups = state.ammoPickups ?? [];
  if (pickups.length === 0) {
    return state;
  }

  // In-place mutation with single-pass compaction (A2 Fix 5).
  // Mutate pickup.active in place, compact active pickups in a single pass.
  let ammoGain = 0;
  let writeIndex = 0;

  for (let readIndex = 0; readIndex < pickups.length; readIndex += 1) {
    const pickup = pickups[readIndex];

    if (!pickup.active) {
      continue;
    }

    const lifetimeMs = pickup.lifetimeMs ?? NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS;
    const expired = simTimeMs - pickup.createdAtMs >= lifetimeMs;
    if (expired) {
      pickup.active = false;
      continue;
    }

    const dx = pickup.position.x - state.player.position.x;
    const dy = pickup.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS) {
      ammoGain += pickup.amount;
      pickup.active = false;
      continue;
    }

    // Keep active pickup in compacted array.
    pickups[writeIndex] = pickup;
    writeIndex += 1;
  }

  pickups.length = writeIndex;

  let next: GameState = {
    ...state,
    ammoPickups: pickups,
  };

  if (ammoGain > 0) {
    next = restoreAmmo(next, ammoGain);
    if (next.telemetry) {
      next = {
        ...next,
        telemetry: {
          ...next.telemetry,
          ammoPickupsCollected: (next.telemetry.ammoPickupsCollected ?? 0) + 1,
        },
      };
    }
  }

  return next;
}
