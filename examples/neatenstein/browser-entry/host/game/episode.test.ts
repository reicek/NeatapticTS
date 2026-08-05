import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
import { createGameState } from './state';
import {
  createEpisode,
  endEpisode,
  isEpisodeComplete,
  runEpisode,
  startEpisode,
  updateEpisode,
} from './episode';
import type { BoltState, EnemyState, GameState, ImpactSpot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/host/game/episode.ts.
 *
 * Covers AC-208: a default episode ends within 15-25 seconds and replays
 * deterministically for the same seed.
 */

describe('Neatenstein game episode', () => {
  describe('AC-208: deterministic episode contract', () => {
    it('exports createEpisode', async () => {
      const mod = (await import('./episode.ts')) as Record<string, unknown>;
      expect(typeof mod.createEpisode).toBe('function');
    });

    it('exports runEpisode', async () => {
      const mod = (await import('./episode.ts')) as Record<string, unknown>;
      expect(typeof mod.runEpisode).toBe('function');
    });

    it('exports startEpisode', async () => {
      const mod = (await import('./episode.ts')) as Record<string, unknown>;
      expect(typeof mod.startEpisode).toBe('function');
    });

    it('returns a fresh game state from startEpisode', () => {
      const state = startEpisode(42);
      expect(state.seed).toBe(42);
      expect(state.episodeDurationMs).toBeGreaterThan(0);
    });

    it('preserves bolts and impact spots in endEpisode', () => {
      const base = createGameState({ seed: 1 });
      const bolt: BoltState = {
        position: { x: 1, y: 1 },
        direction: { x: 1, y: 0 },
        speedCellsPerSecond: 1,
        active: true,
        createdAtMs: 0,
        origin: { x: 1, y: 1 },
      };
      const impact: ImpactSpot = {
        wallHit: { mapX: 1, mapY: 1, side: 0, wallX: 0.5 },
        position: { x: 2, y: 2 },
        createdAtMs: 0,
        lifetimeMs: 100,
        perpWallDist: 1,
        boltTravelTimeMs: 0,
      };
      const state: GameState = {
        ...base,
        bolts: [bolt],
        impacts: [impact],
      };
      const next = endEpisode(state);

      expect(next.bolts).toBeDefined();
      expect(next.bolts!.length).toBe(1);
      expect(next.bolts![0]).not.toBe(bolt);
      expect(next.bolts![0].position).not.toBe(bolt.position);
      expect(next.bolts![0].direction).not.toBe(bolt.direction);
      expect(next.impacts).toHaveLength(1);
      expect(next.impacts[0]).not.toBe(impact);
      expect(next.impacts[0].position).not.toBe(impact.position);
      expect(next.impacts[0].wallHit).not.toBe(impact.wallHit);
    });

    it('completes a default episode within 15-25 seconds', async () => {
      const { createEpisode, runEpisode } =
        (await import('./episode.ts')) as typeof import('./episode.ts');
      const episode = createEpisode({ seed: 1 });
      const final = runEpisode(episode);
      expect({
        withinMin: final.episodeTimeMs >= 15000,
        withinMax: final.episodeTimeMs <= 25000,
      }).toEqual({
        withinMin: true,
        withinMax: true,
      });
    });

    it('produces identical final state when replayed with the same seed', async () => {
      const { createEpisode, runEpisode } =
        (await import('./episode.ts')) as typeof import('./episode.ts');
      const run1 = runEpisode(createEpisode({ seed: 7 }));
      const run2 = runEpisode(createEpisode({ seed: 7 }));
      expect({
        health: run1.player.health,
        ammo: run1.player.ammo,
        kills: run1.kills,
        enemyCount: run1.enemies.length,
      }).toEqual({
        health: run2.player.health,
        ammo: run2.player.ammo,
        kills: run2.kills,
        enemyCount: run2.enemies.length,
      });
    });
  });
});

describe('Coverage: edge cases', () => {
  it('falls back to the default seed for non-finite seeds', () => {
    const episode1 = createEpisode({ seed: Number.NaN });
    const episode2 = createEpisode({ seed: Number.POSITIVE_INFINITY });
    expect(episode1.state.seed).toBe(1);
    expect(episode2.state.seed).toBe(1);
  });

  it('falls back to the default duration for invalid durations', () => {
    const episode1 = createEpisode({ durationMs: -1 });
    const episode2 = createEpisode({ durationMs: Number.NaN });
    expect(episode1.durationMs).toBe(20_000);
    expect(episode2.durationMs).toBe(20_000);
  });

  it('uses the fixed timestep when updateEpisode receives invalid dt', () => {
    const base = createGameState({ seed: 1 });
    const before: GameState = { ...base, simTimeMs: 0 };
    const next = updateEpisode(before, Number.NaN);
    expect(next.simTimeMs).toBe(NEATENSTEIN_FIXED_TIMESTEP_MS);
  });

  it('treats non-finite dash timers as already expired', () => {
    const base = createGameState({ seed: 1 });
    const before: GameState = {
      ...base,
      player: {
        ...base.player,
        dashTimeRemainingMs: Number.NaN,
        dashCooldownMs: Number.NEGATIVE_INFINITY,
      },
    };
    const next = updateEpisode(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(next.player.dashTimeRemainingMs).toBe(0);
    expect(next.player.dashCooldownMs).toBe(0);
  });

  it('restarts non-finite episode timers from zero', () => {
    const base = createGameState({ seed: 1 });
    const before: GameState = {
      ...base,
      simTimeMs: Number.NaN,
      episodeTimeMs: Number.NaN,
    };
    const next = updateEpisode(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(next.simTimeMs).toBe(NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(next.episodeTimeMs).toBe(NEATENSTEIN_FIXED_TIMESTEP_MS);
  });

  it('reports the episode as not complete before any enemies spawn', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = { ...base, enemies: [], spawnCount: 0 };
    expect(isEpisodeComplete(state)).toBe(false);
  });

  it('reports the episode as not complete when kills precede the total spawn cap', () => {
    const base = createGameState({ seed: 1 });
    const enemies: EnemyState[] = [
      { position: { x: 30, y: 30 }, health: 0, active: false } as EnemyState,
    ];
    const state: GameState = {
      ...base,
      enemies,
      spawnCount: 1,
    };
    expect(isEpisodeComplete(state)).toBe(false);
  });

  it('preserves a previous player position in endEpisode', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      player: {
        ...base.player,
        previousPosition: { x: 1, y: 2 },
      },
    };
    const next = endEpisode(state);
    expect(next.player.previousPosition).toEqual({ x: 1, y: 2 });
    expect(next.player.previousPosition).not.toBe(
      state.player.previousPosition,
    );
  });

  it('handles a missing bolts array in endEpisode', () => {
    const base = createGameState({ seed: 1 });
    const state = {
      ...base,
      bolts: undefined,
    } as unknown as GameState;
    const next = endEpisode(state);
    expect(next.bolts).toEqual([]);
  });

  it('completes by time when every enemy has an invalid position', () => {
    const base = createGameState({ seed: 1 });
    const invalidEnemies: EnemyState[] = Array.from(
      { length: 8 },
      () =>
        ({
          position: { x: Number.NaN, y: Number.NaN },
          active: true,
        }) as EnemyState,
    );
    const state: GameState = {
      ...base,
      enemies: invalidEnemies,
      spawnCount: 72,
      episodeDurationMs: 15_000,
    };
    const episode = { state, durationMs: 15_000 };
    const final = runEpisode(episode);
    expect(final.episodeTimeMs).toBeGreaterThanOrEqual(15_000);
    expect(final.episodeTimeMs).toBeLessThanOrEqual(25_000);
  });

  it('uses the default options when createEpisode is called without arguments', () => {
    const episode = createEpisode();
    expect(episode.state.seed).toBe(1);
    expect(episode.durationMs).toBe(20_000);
  });

  it('advances a positive dash timer in updateEpisode', () => {
    const base = createGameState({ seed: 1 });
    const before: GameState = {
      ...base,
      player: {
        ...base.player,
        dashTimeRemainingMs: 100,
        dashCooldownMs: 100,
      },
    };
    const next = updateEpisode(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(next.player.dashTimeRemainingMs).toBeLessThan(100);
    expect(next.player.dashCooldownMs).toBeLessThan(100);
  });

  it('handles non-finite episode and player state in isEpisodeComplete', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      episodeTimeMs: Number.NaN,
      player: {
        ...base.player,
        health: Number.NaN,
      },
      enemies: [],
      spawnCount: 0,
    };
    expect(isEpisodeComplete(state)).toBe(true);
  });

  it('endEpisode handles a missing previous position', () => {
    const base = createGameState({ seed: 1 });
    const state: GameState = {
      ...base,
      player: {
        ...base.player,
        previousPosition: undefined,
      },
    };
    const next = endEpisode(state);
    expect(next.player.previousPosition).toBeUndefined();
  });
});

describe('AC-10.2d-003: episode waves advance only after all enemies cleared', () => {
  it('spawns a new wave once all existing enemies are dead', () => {
    const base = createGameState({ seed: 42 });
    const deadEnemies: EnemyState[] = Array.from({ length: 8 }, () => ({
      position: { x: 30, y: 30 },
      health: 0,
      active: false,
    }));
    const before: GameState = {
      ...base,
      enemies: deadEnemies,
      spawnCount: 8,
    };

    const next = updateEpisode(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    const spawnedNew =
      next.enemies.length > 8 ||
      next.enemies.some((enemy) => enemy.health > 0) ||
      next.spawnCount > 8;

    expect(spawnedNew).toBe(true);
  });

  it('ends by clearing all spawned enemies, not by time-out', () => {
    const final = runEpisode(createEpisode({ seed: 42 }));
    expect(final.kills).toBe(final.spawnCount);
    expect(final.episodeDurationMs).toBeDefined();
    expect(final.episodeTimeMs).toBeLessThan(final.episodeDurationMs as number);
  });
});
