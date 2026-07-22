import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_CONTACT_DAMAGE,
  NEATENSTEIN_CONTACT_IFRAME_MS,
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
      const before = createGameState({ seed: 1 });
      const after = resolveContactDamage(before, 16);
      expect(after.player.health).toBe(before.player.health);
    });

    it('applies contact damage when an enemy overlaps the player', () => {
      const before = createGameState({ seed: 1 });
      before.enemies.push({
        position: { ...before.player.position },
        health: 10,
      });
      const after = resolveContactDamage(before, 16);
      expect(after.player.health).toBe(
        before.player.health - NEATENSTEIN_CONTACT_DAMAGE,
      );
    });

    it('clamps health at zero from oversized repeated contact damage', () => {
      let state = createGameState({ seed: 1 });
      state.enemies.push({
        position: { ...state.player.position },
        health: 10,
      });
      for (let i = 0; i < 20; i++) {
        state = resolveContactDamage(state, 1_000);
      }
      expect(state.player.health).toBe(0);
    });

    it('starts the configured contact i-frame timer when damage occurs', () => {
      const before = createGameState({ seed: 1 });
      before.enemies.push({
        position: { ...before.player.position },
        health: 10,
      });
      const after = resolveContactDamage(before, 16);
      expect(after.player.contactIFrameMs).toBe(NEATENSTEIN_CONTACT_IFRAME_MS);
    });

    it('ignores contact damage while the contact i-frame timer is active', () => {
      const before = createGameState({ seed: 1 });
      before.enemies.push({
        position: { ...before.player.position },
        health: 10,
      });
      const first = resolveContactDamage(before, 16);
      const second = resolveContactDamage(first, 16);
      expect(second.player.health).toBe(first.player.health);
    });

    it('ignores contact damage while the player is dashing', () => {
      const before = createGameState({ seed: 1 });
      before.enemies.push({
        position: { ...before.player.position },
        health: 10,
      });
      const dashed = applyDash(before);
      const after = resolveContactDamage(dashed, 16);
      expect(after.player.health).toBe(before.player.health);
    });

    it('returns a new state and player reference even when nothing happens', () => {
      const before = createGameState({ seed: 1 });
      const after = resolveContactDamage(before, 16);
      expect(after).not.toBe(before);
      expect(after.player).not.toBe(before.player);
    });
  });
});
