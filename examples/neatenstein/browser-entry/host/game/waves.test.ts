import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_ENEMY_MAX_CONCURRENT } from './constants';
import { createGameState, spawnWaveTick } from './waves';
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
      const result = spawnWaveTick(before, 16);
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
        const result = spawnWaveTick(state, 16);
        state = result.state;
      }
      expect(state.enemies.length).toBe(5);
    });

    it('returns a new immutable state even when no enemy spawns', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, 16).state;
      }
      const result = spawnWaveTick(state, 16);
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
        state = spawnWaveTick(state, 16).state;
      }
      expect(state.enemies.length).toBe(NEATENSTEIN_ENEMY_MAX_CONCURRENT);
    });

    it('reports zero spawns when the enemy roster is already at the cap', () => {
      let state: GameState = createGameState({ seed: 1 });
      for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i++) {
        state = spawnWaveTick(state, 16).state;
      }
      const result = spawnWaveTick(state, 16);
      expect({
        spawnedThisTick: result.spawnedThisTick,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedThisTick: 0,
        enemyCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      });
    });

    it('spawns enemies within the configured spawn radius', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, 16);
      const [enemy] = result.state.enemies;
      const radius = Math.hypot(enemy.position.x, enemy.position.y);
      expect(radius).toBeLessThanOrEqual(8);
    });
  });

  describe('AC-203: deterministic spawn identity and anti-collision', () => {
    it('produces identical enemy sequences from the same seed', () => {
      let stateA: GameState = createGameState({ seed: 42 });
      let stateB: GameState = createGameState({ seed: 42 });
      for (let i = 0; i < 5; i++) {
        stateA = spawnWaveTick(stateA, 16).state;
        stateB = spawnWaveTick(stateB, 16).state;
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
      const resultA = spawnWaveTick(beforeA, 16);
      const resultB = spawnWaveTick(beforeB, 16);
      expect(resultA.state.enemies[0].position).toEqual(
        resultB.state.enemies[0].position,
      );
    });

    it('increments spawnCount after a successful spawn', () => {
      const before = createGameState({ seed: 1 });
      const result = spawnWaveTick(before, 16);
      expect(result.state.spawnCount).toBe(before.spawnCount + 1);
    });
  });
});
