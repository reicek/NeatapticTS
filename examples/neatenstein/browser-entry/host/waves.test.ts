import { describe, expect, it } from '@jest/globals';
import { createMlpEnemyPopulation } from '../harness/enemy-mlp';
import { NEATENSTEIN_ENEMY_MAX_CONCURRENT } from './game/constants';
import { createGameState } from './game/state';
import type { GameState } from './game/types';
import { advanceWave } from './waves';

/**
 * Contract tests for examples/neatenstein/browser-entry/host/waves.ts.
 *
 * Covers AC-705I: the host-level wave loop clears the arena, advances the
 * enemy evolution harness, and spawns up to
 * {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT} concurrent enemies.
 */

describe('Neatenstein host wave loop', () => {
  function createPopulatedState(seed: number): GameState {
    const state = createGameState({ seed });
    return {
      ...state,
      enemies: [
        { position: { x: 10, y: 10 }, health: 1 },
        { position: { x: 12, y: 12 }, health: 1 },
      ],
      bolts: [
        {
          position: { x: 5, y: 5 },
          direction: { x: 1, y: 0 },
          speedCellsPerSecond: 10,
          active: true,
          createdAtMs: 0,
        },
      ],
      impacts: [
        {
          wallHit: { mapX: 1, mapY: 1, side: 0, wallX: 0.5 },
          position: { x: 6, y: 6 },
          createdAtMs: 0,
          lifetimeMs: 1000,
          perpWallDist: 1,
          boltTravelTimeMs: 0,
        },
      ],
      kills: 3,
      spawnCount: 5,
    };
  }

  describe('AC-705I: wave loop API', () => {
    it('exports advanceWave', () => {
      expect(typeof advanceWave).toBe('function');
    });

    it('returns a new state, snapshot and spawnedCount', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population });

      expect({
        hasState: typeof result.state === 'object',
        hasSnapshot: typeof result.snapshot === 'object',
        hasSpawnedCount: Number.isFinite(result.spawnedCount),
      }).toEqual({
        hasState: true,
        hasSnapshot: true,
        hasSpawnedCount: true,
      });
    });
  });

  describe('AC-705I: clear arena', () => {
    it('removes all pre-existing enemies', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createPopulatedState(1);
      const result = advanceWave(before, { population, spawnCount: 0 });

      expect(result.state.enemies).toEqual([]);
    });

    it('removes all active bolts', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createPopulatedState(1);
      const result = advanceWave(before, { population, spawnCount: 0 });

      expect(result.state.bolts).toEqual([]);
    });

    it('removes all impact spots', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createPopulatedState(1);
      const result = advanceWave(before, { population, spawnCount: 0 });

      expect(result.state.impacts).toEqual([]);
    });

    it('preserves the player, seed and episode timing', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createPopulatedState(1);
      const result = advanceWave(before, { population, spawnCount: 0 });

      expect({
        player: result.state.player,
        seed: result.state.seed,
        simTimeMs: result.state.simTimeMs,
        episodeTimeMs: result.state.episodeTimeMs,
      }).toEqual({
        player: before.player,
        seed: before.seed,
        simTimeMs: before.simTimeMs,
        episodeTimeMs: before.episodeTimeMs,
      });
    });

    it('does not reset the persistent spawn counter', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createPopulatedState(1);
      const result = advanceWave(before, { population, spawnCount: 0 });

      expect(result.state.spawnCount).toBe(before.spawnCount);
    });
  });

  describe('AC-705I: evolution harness', () => {
    it('advances the generation by one', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population });

      expect(result.state.generation).toBe(before.generation + 1);
    });

    it('returns the population champion snapshot', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population });

      expect(result.snapshot).toBe(population.snapshot());
    });

    it('advances the population snapshot when crossing a refresh boundary', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const first = advanceWave(before, { population });

      // Force another wave so generation moves from 2 to 3; refresh happens on
      // multiples of NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS (5), so this
      // alone does not refresh, but the snapshot must remain the one returned
      // by update().
      const second = advanceWave(first.state, { population });
      expect(second.snapshot).toBe(population.snapshot());
      expect(second.state.generation).toBe(before.generation + 2);
    });
  });

  describe('AC-705I: spawn concurrency cap', () => {
    it('spawns up to NEATENSTEIN_ENEMY_MAX_CONCURRENT enemies by default', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population });

      expect(result.state.enemies.length).toBeLessThanOrEqual(
        NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      );
      expect(result.spawnedCount).toBe(result.state.enemies.length);
    });

    it('spawns exactly the requested number when below the cap', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population, spawnCount: 3 });

      expect({
        spawnedCount: result.spawnedCount,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedCount: 3,
        enemyCount: 3,
      });
    });

    it('clamps a requested count above the cap down to the cap', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, {
        population,
        spawnCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT + 10,
      });

      expect(result.state.enemies.length).toBe(
        NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      );
    });

    it('treats a negative spawnCount as zero', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population, spawnCount: -5 });

      expect({
        spawnedCount: result.spawnedCount,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedCount: 0,
        enemyCount: 0,
      });
    });

    it('treats a non-finite spawnCount as the default cap', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population, spawnCount: NaN });

      expect(result.state.enemies.length).toBeLessThanOrEqual(
        NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      );
      expect(result.spawnedCount).toBeGreaterThan(0);
    });

    it('clears an existing full roster when advancing with spawnCount 0', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const first = advanceWave(before, { population });

      const second = advanceWave(first.state, {
        population,
        spawnCount: 0,
      });

      expect({
        spawnedCount: second.spawnedCount,
        enemyCount: second.state.enemies.length,
        previousRosterCleared:
          second.state.enemies !== first.state.enemies &&
          second.state.enemies.length === 0,
      }).toEqual({
        spawnedCount: 0,
        enemyCount: 0,
        previousRosterCleared: true,
      });
    });
  });

  describe('AC-705I: deterministic wave identity', () => {
    it('produces identical results from the same starting state and population', () => {
      const populationA = createMlpEnemyPopulation({ seed: 1 });
      const populationB = createMlpEnemyPopulation({ seed: 1 });
      const beforeA = createGameState({ seed: 1 });
      const beforeB = createGameState({ seed: 1 });

      const resultA = advanceWave(beforeA, { population: populationA });
      const resultB = advanceWave(beforeB, { population: populationB });

      expect(resultA.state.enemies.map((e) => e.position)).toEqual(
        resultB.state.enemies.map((e) => e.position),
      );
      expect(resultA.spawnedCount).toBe(resultB.spawnedCount);
    });

    it('places spawned enemies outside the player contact-damage range', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const before = createGameState({ seed: 1 });
      const result = advanceWave(before, { population });

      const tooClose = result.state.enemies.some(
        (enemy) =>
          Math.hypot(
            enemy.position.x - before.player.position.x,
            enemy.position.y - before.player.position.y,
          ) < 0.5,
      );
      expect(tooClose).toBe(false);
    });
  });
});
