import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_CONTACT_RANGE_CELLS,
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS,
  NEATENSTEIN_ENEMY_SPAWN_RADIUS,
  NEATENSTEIN_EPISODE_MAX_DURATION_MS,
  NEATENSTEIN_EPISODE_MIN_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import { NEATENSTEIN_MAP_SIZE } from '../../constants';
import { createGameState } from './state';
import { spawnWaveTick } from './waves';
import { createEpisode, runEpisode } from './episode';
import type { GameState } from './types';

/**
 * Contract tests for examples/neatenstein/browser-entry/host/game/waves.ts.
 *
 * Covers AC-203: enemy waves spawn as a continuous trickle with at most one
 * new enemy per tick and no more than {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT}
 * concurrent enemies.
 */

describe('Neatenstein game waves', () => {
  describe('AC-203: continuous-trickle spawn contract', () => {
    it('exports spawnWaveTick', () => {
      expect(typeof spawnWaveTick).toBe('function');
    });

    it('spawns exactly one enemy on a tick that is below the cap', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect({
        spawnedThisTick: result.spawnedThisTick,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedThisTick: 1,
        enemyCount: 1,
      });
    });

    it('grows the enemy roster by exactly one per tick under the cap', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < 5; i++) {
        const result = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
        state = result.state;
      }
      expect(state.enemies.length).toBe(5);
    });

    it('returns a new immutable state even when no enemy spawns', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      const result = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect({
        newReference: result.state !== state,
        newEnemiesArray: result.state.enemies !== state.enemies,
        spawnedThisTick: result.spawnedThisTick,
      }).toEqual({
        newReference: true,
        newEnemiesArray: true,
        spawnedThisTick: 0,
      });
    });

    it('stops spawning once the concurrent cap is reached', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT + 2; i++) {
        state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      expect(state.enemies.length).toBe(NEATENSTEIN_ENEMY_MAX_CONCURRENT);
    });

    it('reports zero spawns when the enemy roster is already at the cap', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      const result = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect({
        spawnedThisTick: result.spawnedThisTick,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedThisTick: 0,
        enemyCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      });
    });

    it('spawns enemies at least the minimum distance from the map center', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      const [enemy] = result.state.enemies;
      const radius = Math.hypot(
        enemy.position.x - NEATENSTEIN_SPAWN_CENTER_X,
        enemy.position.y - NEATENSTEIN_SPAWN_CENTER_Y,
      );
      expect(radius).toBeGreaterThanOrEqual(
        NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS,
      );
    });

    it('spawns enemies no farther than the configured spawn radius', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      const [enemy] = result.state.enemies;
      const radius = Math.hypot(
        enemy.position.x - NEATENSTEIN_SPAWN_CENTER_X,
        enemy.position.y - NEATENSTEIN_SPAWN_CENTER_Y,
      );
      expect(radius).toBeLessThanOrEqual(NEATENSTEIN_ENEMY_SPAWN_RADIUS);
    });
  });

  describe('AC-203: deterministic spawn identity and anti-collision', () => {
    it('produces identical enemy sequences from the same seed', () => {
      let stateA: GameState = createGameState({ seed: 42 });
      let stateB: GameState = createGameState({ seed: 42 });
      for (let i = 0; i < 5; i++) {
        stateA = spawnWaveTick(stateA, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
        stateB = spawnWaveTick(stateB, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      expect({
        enemies: stateA.enemies,
        spawnCount: stateA.spawnCount,
      }).toEqual({
        enemies: stateB.enemies,
        spawnCount: stateB.spawnCount,
      });
    });

    it('ignores kill history when deriving the first spawn position', () => {
      const base = createGameState({ seed: 7 });
      const beforeA: GameState = { ...base, kills: 5 };
      const beforeB: GameState = { ...base, kills: 10 };
      const resultA = spawnWaveTick(beforeA, NEATENSTEIN_FIXED_TIMESTEP_MS);
      const resultB = spawnWaveTick(beforeB, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(resultA.state.enemies[0].position).toEqual(
        resultB.state.enemies[0].position,
      );
    });

    it('increments spawnCount after a successful spawn', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect(result.state.spawnCount).toBe(before.spawnCount + 1);
    });
  });

  describe('AC-216 / 03-red: map spawn bounds', () => {
    it('spawns every active enemy inside the map bounds', () => {
      let state: GameState = createGameState({ seed: 42 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      const outOfBounds = state.enemies.some(
        (enemy) =>
          Math.floor(enemy.position.x) < 0 ||
          Math.floor(enemy.position.x) >= NEATENSTEIN_MAP_SIZE ||
          Math.floor(enemy.position.y) < 0 ||
          Math.floor(enemy.position.y) >= NEATENSTEIN_MAP_SIZE,
      );
      expect(outOfBounds).toBe(false);
    });
  });

  describe('AC-230 / 03-map: enemy spawn separation from player spawn', () => {
    it('spawns every active enemy outside the player contact-damage range', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
      }
      const tooClose = state.enemies.some(
        (enemy) =>
          Math.hypot(
            enemy.position.x - NEATENSTEIN_SPAWN_CENTER_X,
            enemy.position.y - NEATENSTEIN_SPAWN_CENTER_Y,
          ) < NEATENSTEIN_CONTACT_RANGE_CELLS,
      );
      expect(tooClose).toBe(false);
    });

    it('lets the player survive the default episode', () => {
      const final = runEpisode(createEpisode({ seed: 1 }));
      expect(final.player.health).toBeGreaterThan(0);
    });

    it('reaches at least the minimum episode duration', () => {
      const final = runEpisode(createEpisode({ seed: 1 }));
      expect(final.episodeTimeMs).toBeGreaterThanOrEqual(
        NEATENSTEIN_EPISODE_MIN_DURATION_MS,
      );
    });

    it('does not exceed the maximum episode duration', () => {
      const final = runEpisode(createEpisode({ seed: 1 }));
      expect(final.episodeTimeMs).toBeLessThanOrEqual(
        NEATENSTEIN_EPISODE_MAX_DURATION_MS,
      );
    });
  });
});
