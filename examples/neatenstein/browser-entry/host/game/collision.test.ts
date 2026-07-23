import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_CONTACT_DAMAGE,
  NEATENSTEIN_CONTACT_IFRAME_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_MS_PER_SECOND,
  NEATENSTEIN_TEST_ENEMY_BEYOND_CONTACT_RANGE_CELLS,
  NEATENSTEIN_TEST_ENEMY_DEAD_HEALTH,
  NEATENSTEIN_TEST_ENEMY_HEALTH,
  NEATENSTEIN_TEST_SEED,
} from './constants';
import { resolveContactDamage } from './collision';
import { applyDash, createGameState } from './state';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/collision.ts.
 *
 * Covers AC-202 and AC-207: enemy/player contact damage respects i-frames and
 * never drives health negative.
 */

describe('Neatenstein game collision', () => {
  describe('AC-202 / AC-207: contact damage contract', () => {
    it('exports resolveContactDamage', async () => {
      const mod = (await import('./collision.ts')) as Record<string, unknown>;
      expect(typeof mod.resolveContactDamage).toBe('function');
    });

    it('does not damage the player when no enemy is nearby', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.health).toBe(before.player.health);
    });

    it('applies contact damage when an enemy overlaps the player', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.health).toBe(
        before.player.health - NEATENSTEIN_CONTACT_DAMAGE,
      );
    });

    it('clamps health at zero from oversized repeated contact damage', () => {
      let state = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      state.enemies.push({
        position: { ...state.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      for (let i = 0; i < 20; i++) {
        state = resolveContactDamage(state, NEATENSTEIN_MS_PER_SECOND);
      }
      expect(state.player.health).toBe(0);
    });

    it('starts the configured contact i-frame timer when damage occurs', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.contactIFrameMs).toBe(NEATENSTEIN_CONTACT_IFRAME_MS);
    });

    it('ignores contact damage while the contact i-frame timer is active', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const first = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      const second = resolveContactDamage(first, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(second.player.health).toBe(first.player.health);
    });

    it('ignores contact damage while the player is dashing', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const dashed = applyDash(before);
      const after = resolveContactDamage(dashed, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.health).toBe(before.player.health);
    });

    it('returns a new state reference even when nothing happens', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after).not.toBe(before);
    });

    it('returns a new player reference even when nothing happens', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player).not.toBe(before.player);
    });

    it('blocks contact damage while the i-frame timer is still active', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const damaged = resolveContactDamage(
        before,
        NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
      const partial = resolveContactDamage(
        damaged,
        NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
      expect(partial.player.health).toBe(damaged.player.health);
    });

    it('decrements the contact i-frame timer by the elapsed timestep', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const damaged = resolveContactDamage(
        before,
        NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
      const partial = resolveContactDamage(
        damaged,
        NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
      expect(partial.player.contactIFrameMs).toBe(
        NEATENSTEIN_CONTACT_IFRAME_MS - NEATENSTEIN_FIXED_TIMESTEP_MS,
      );
    });

    it('ignores a dead enemy overlapping the player', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_DEAD_HEALTH,
      });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.health).toBe(before.player.health);
    });

    it('applies contact damage only from enemies within range', () => {
      const before = createGameState({ seed: NEATENSTEIN_TEST_SEED });
      before.enemies.push({
        position: { ...before.player.position },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      before.enemies.push({
        position: {
          x:
            before.player.position.x +
            NEATENSTEIN_TEST_ENEMY_BEYOND_CONTACT_RANGE_CELLS,
          y: before.player.position.y,
        },
        health: NEATENSTEIN_TEST_ENEMY_HEALTH,
      });
      const after = resolveContactDamage(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(after.player.health).toBe(
        before.player.health - NEATENSTEIN_CONTACT_DAMAGE,
      );
    });
  });
});
