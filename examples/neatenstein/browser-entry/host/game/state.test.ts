import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_DASH_COOLDOWN_MS,
  NEATENSTEIN_DASH_INVULNERABILITY_MS,
  NEATENSTEIN_PLAYER_MAX_AMMO,
} from './constants';

/**
 * Contract tests for examples/neatenstein/browser-entry/host/game/state.ts.
 *
 * Covers AC-202 (health/ammo invariants) and AC-207 (dash invulnerability).
 * The source module now exists; these tests verify the exported state helpers
 * behave according to the game-logic contracts.
 */

describe('Neatenstein game state', () => {
  describe('AC-202 / AC-207: exported state functions', () => {
    it('exports createGameState', async () => {
      const mod = (await import('./state.ts')) as Record<string, unknown>;
      expect(typeof mod.createGameState).toBe('function');
    });

    it('exports applyDamage', async () => {
      const mod = (await import('./state.ts')) as Record<string, unknown>;
      expect(typeof mod.applyDamage).toBe('function');
    });

    it('exports consumeAmmo', async () => {
      const mod = (await import('./state.ts')) as Record<string, unknown>;
      expect(typeof mod.consumeAmmo).toBe('function');
    });

    it('exports applyDash', async () => {
      const mod = (await import('./state.ts')) as Record<string, unknown>;
      expect(typeof mod.applyDash).toBe('function');
    });

    it('initializes player health and ammo to their documented maximums', async () => {
      const { createGameState } = (await import('./state.ts')) as Record<
        string,
        any
      >;
      const state = createGameState({ seed: 1 });
      expect({
        health: state.player.health,
        ammo: state.player.ammo,
        hasMaxHealth: typeof state.player.maxHealth === 'number',
        hasMaxAmmo: typeof state.player.maxAmmo === 'number',
      }).toEqual({
        health: state.player.maxHealth,
        ammo: state.player.maxAmmo,
        hasMaxHealth: true,
        hasMaxAmmo: true,
      });
    });

    it('returns identical canonical state for the same seed', async () => {
      const { createGameState } = (await import('./state.ts')) as Record<
        string,
        any
      >;
      const first = createGameState({ seed: 42 });
      const second = createGameState({ seed: 42 });
      expect({
        seed: second.seed,
        health: second.player.health,
        ammo: second.player.ammo,
        position: second.player.position,
        angleRad: second.player.angleRad,
      }).toEqual({
        seed: first.seed,
        health: first.player.health,
        ammo: first.player.ammo,
        position: first.player.position,
        angleRad: first.player.angleRad,
      });
    });

    it('produces different canonical state for different seeds', async () => {
      const { createGameState } = (await import('./state.ts')) as Record<
        string,
        any
      >;
      const first = createGameState({ seed: 1 });
      const second = createGameState({ seed: 2 });
      expect(second.player.angleRad).not.toBe(first.player.angleRad);
    });

    it('defaults to seed 1 when createGameState is called with no options', async () => {
      const { createGameState } = (await import('./state.ts')) as Record<
        string,
        any
      >;
      const state = createGameState();
      expect(state.seed).toBe(1);
    });

    it('decrements ammo by exactly one when consumeAmmo is called', async () => {
      const { createGameState, consumeAmmo } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = consumeAmmo(before);
      expect(after.player.ammo).toBe(before.player.ammo - 1);
    });

    it('does not allow health to drop below zero from oversized damage', async () => {
      const { createGameState, applyDamage } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = applyDamage(before, before.player.health + 50);
      expect(after.player.health).toBe(0);
    });

    it('grants exactly 200 ms of dash invulnerability when applyDash is called', async () => {
      const { createGameState, applyDash } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = applyDash(before);
      expect(after.player.dashTimeRemainingMs).toBe(
        NEATENSTEIN_DASH_INVULNERABILITY_MS,
      );
    });

    it('reduces player health by the exact damage amount', async () => {
      const { createGameState, applyDamage } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = applyDamage(before, 23);
      expect(after.player.health).toBe(before.player.health - 23);
    });

    it('ignores negative damage and leaves health unchanged', async () => {
      const { createGameState, applyDamage } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = applyDamage(before, -10);
      expect(after.player.health).toBe(before.player.health);
    });

    it('clamps ammo at zero after repeated consumeAmmo calls', async () => {
      const { createGameState, consumeAmmo } =
        (await import('./state.ts')) as Record<string, any>;
      let state = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_PLAYER_MAX_AMMO + 5; i++) {
        state = consumeAmmo(state);
      }
      expect(state.player.ammo).toBe(0);
    });

    it('prevents damage while the player is invulnerable from a dash', async () => {
      const { createGameState, applyDash, applyDamage } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const dashed = applyDash(before);
      const after = applyDamage(dashed, before.player.health);
      expect(after.player.health).toBe(before.player.health);
    });

    it('treats contact i-frames as invulnerable and ignores damage', async () => {
      const { createGameState, applyDamage, isInvulnerable } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const contactState = {
        ...before,
        player: {
          ...before.player,
          contactIFrameMs: 100,
          dashTimeRemainingMs: 0,
        },
      };
      const after = applyDamage(contactState, 10);
      expect({
        invulnerable: isInvulnerable(contactState),
        health: after.player.health,
      }).toEqual({
        invulnerable: true,
        health: before.player.health,
      });
    });

    it('returns a new state and player reference when damage is ignored during invulnerability', async () => {
      const { createGameState, applyDash, applyDamage } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const dashed = applyDash(before);
      const after = applyDamage(dashed, before.player.health);
      expect({
        stateNew: after !== dashed,
        playerNew: after.player !== dashed.player,
      }).toEqual({ stateNew: true, playerNew: true });
    });

    it('starts the configured cooldown when applyDash is called', async () => {
      const { createGameState, applyDash } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const after = applyDash(before);
      expect(after.player.dashCooldownMs).toBe(NEATENSTEIN_DASH_COOLDOWN_MS);
    });

    it('does not refresh dash while the cooldown is active', async () => {
      const { createGameState, applyDash } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const first = applyDash(before);
      const second = applyDash(first);
      expect({
        invulnerability: second.player.dashTimeRemainingMs,
        cooldown: second.player.dashCooldownMs,
      }).toEqual({
        invulnerability: first.player.dashTimeRemainingMs,
        cooldown: first.player.dashCooldownMs,
      });
    });

    it('returns a new state and player reference when dash is gated by cooldown', async () => {
      const { createGameState, applyDash } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const first = applyDash(before);
      const second = applyDash(first);
      expect({
        stateNew: second !== first,
        playerNew: second.player !== first.player,
      }).toEqual({ stateNew: true, playerNew: true });
    });

    it('exports canDash that reflects cooldown state', async () => {
      const { createGameState, applyDash, canDash } =
        (await import('./state.ts')) as Record<string, any>;
      const before = createGameState({ seed: 1 });
      const dashed = applyDash(before);
      expect({
        beforeDash: canDash(before),
        afterDash: canDash(dashed),
      }).toEqual({
        beforeDash: true,
        afterDash: false,
      });
    });

    it('spawns the player at the center of a 60x60 map', async () => {
      const { createGameState } = (await import('./state.ts')) as Record<
        string,
        any
      >;
      const state = createGameState({ seed: 42 });
      expect(state.player.position).toEqual({
        x: 30.5,
        y: 30.5,
      });
    });
  });
});
