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

  let ammoGain = 0;
  const updatedPickups = pickups.map((pickup) => {
    if (!pickup.active) {
      return pickup;
    }

    const lifetimeMs = pickup.lifetimeMs ?? NEATENSTEIN_AMMO_PICKUP_LIFETIME_MS;
    const expired = simTimeMs - pickup.createdAtMs >= lifetimeMs;
    if (expired) {
      return { ...pickup, active: false };
    }

    const dx = pickup.position.x - state.player.position.x;
    const dy = pickup.position.y - state.player.position.y;
    const dist = Math.hypot(dx, dy);
    if (dist <= NEATENSTEIN_AMMO_PICKUP_COLLECTION_RADIUS_CELLS) {
      ammoGain += pickup.amount;
      return { ...pickup, active: false };
    }

    return pickup;
  });

  const activePickups = updatedPickups.filter((pickup) => pickup.active);

  let next: GameState = {
    ...state,
    ammoPickups: activePickups,
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
