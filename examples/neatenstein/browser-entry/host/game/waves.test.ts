import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_CONTACT_RANGE_CELLS,
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_MAX_HEALTH,
  NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS,
  NEATENSTEIN_ENEMY_SPAWN_RADIUS,
  NEATENSTEIN_EPISODE_MAX_DURATION_MS,
  NEATENSTEIN_EPISODE_MIN_DURATION_MS,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import { NEATENSTEIN_MAP_SIZE } from '../../constants';
import { createCollisionMap, type CollisionMap } from '../../renderer/map';
import { createGameState } from './state';
import { allEnemiesCleared, spawnWaveTick } from './waves';
import { createEpisode, runEpisode } from './episode';
import type { EnemyState, GameState } from './types';

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
      // The state object is always a new immutable snapshot, but the enemies
      // array reference is preserved when no enemy spawns (dead enemies stay
      // in the array for display-worker index alignment).
      expect({
        newReference: result.state !== state,
        newEnemiesArray: result.state.enemies !== state.enemies,
        spawnedThisTick: result.spawnedThisTick,
      }).toEqual({
        newReference: true,
        newEnemiesArray: false,
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

  describe('AC-203: concurrent alive limit guard', () => {
    it('blocks spawning via the concurrent alive limit when the batch gate does not apply', () => {
      const base = createGameState({ seed: 42 });
      // Create NEATENSTEIN_ENEMY_MAX_CONCURRENT alive enemies with
      // spawnCount=1 so batchComplete is false (1 % 8 !== 0). The aliveCount
      // guard should block spawning without the batch gate.
      const aliveEnemies: EnemyState[] = Array.from(
        { length: NEATENSTEIN_ENEMY_MAX_CONCURRENT },
        () =>
          ({
            position: { x: 30, y: 30 },
            health: NEATENSTEIN_ENEMY_MAX_HEALTH,
            active: true,
          }) as EnemyState,
      );
      const before: GameState = {
        ...base,
        enemies: aliveEnemies,
        spawnCount: 1,
      };
      const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
      expect({
        spawnedThisTick: result.spawnedThisTick,
        enemyCount: result.state.enemies.length,
      }).toEqual({
        spawnedThisTick: 0,
        enemyCount: NEATENSTEIN_ENEMY_MAX_CONCURRENT,
      });
    });
  });
});

describe('AC-10.2d-003/004/007: edge-based waves and batch gating', () => {
  function createSolidEdgeCollisionMap(): CollisionMap {
    return createCollisionMap(
      new Uint8Array(NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE).fill(1),
      NEATENSTEIN_MAP_SIZE,
    );
  }

  function distanceToNearestEdge(position: { x: number; y: number }): number {
    return Math.min(
      position.x,
      position.y,
      NEATENSTEIN_MAP_SIZE - position.x,
      NEATENSTEIN_MAP_SIZE - position.y,
    );
  }

  function classifyEdgeDirection(position: { x: number; y: number }): string {
    const threshold = 1.5;
    const nearLeft = position.x <= threshold;
    const nearRight = NEATENSTEIN_MAP_SIZE - position.x <= threshold;
    const nearTop = position.y <= threshold;
    const nearBottom = NEATENSTEIN_MAP_SIZE - position.y <= threshold;

    let direction = '';
    if (nearTop) direction += 'N';
    if (nearBottom) direction += 'S';
    if (nearLeft) direction += 'W';
    if (nearRight) direction += 'E';

    return direction || 'none';
  }

  it('spawns enemies near a map edge', () => {
    const before = createGameState({ seed: 42 });
    const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    const distance = distanceToNearestEdge(result.state.enemies[0].position);
    expect(distance).toBeLessThanOrEqual(1.5);
  });

  it('covers all eight edge directions across a full batch', () => {
    // Seed 15334 produces an open cell at every cardinal and intercardinal edge,
    // so all eight directions can be observed without falling back to the center.
    let state: GameState = createGameState({ seed: 15334 });
    const directions = new Set<string>();

    for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
      const result = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
      state = result.state;
      directions.add(classifyEdgeDirection(state.enemies[i].position));
    }

    expect(Array.from(directions).sort()).toEqual([
      'E',
      'N',
      'NE',
      'NW',
      'S',
      'SE',
      'SW',
      'W',
    ]);
  });

  it('starts a new batch once all current enemies are dead', () => {
    let state: GameState = createGameState({ seed: 42 });
    for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
      state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
    }

    const allDead: GameState = {
      ...state,
      enemies: state.enemies.map((enemy) => ({ ...enemy, health: 0 })),
    };

    const result = spawnWaveTick(allDead, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.spawnedThisTick).toBeGreaterThan(0);
  });

  it('starts a new batch once all current enemies are deactivated', () => {
    let state: GameState = createGameState({ seed: 15334 });
    for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
      state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
    }

    const allInactive: GameState = {
      ...state,
      enemies: state.enemies.map((enemy) => ({
        ...enemy,
        health: 1,
        active: false,
      })),
    };

    const result = spawnWaveTick(allInactive, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.spawnedThisTick).toBeGreaterThan(0);
    expect(result.state.enemies.some((enemy) => enemy.health > 0)).toBe(true);
  });

  it('starts a new batch when current enemies are missing health values', () => {
    let state: GameState = createGameState({ seed: 15334 });
    for (let i = 0; i < NEATENSTEIN_ENEMY_MAX_CONCURRENT; i += 1) {
      state = spawnWaveTick(state, NEATENSTEIN_FIXED_TIMESTEP_MS).state;
    }

    const missingHealth: GameState = {
      ...state,
      enemies: state.enemies.map((enemy) => ({
        ...enemy,
        active: true,
      })) as EnemyState[],
    };
    missingHealth.enemies.forEach((enemy) => {
      delete (enemy as { health?: number }).health;
    });

    const result = spawnWaveTick(missingHealth, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.spawnedThisTick).toBeGreaterThan(0);
    expect(result.state.enemies.length).toBeGreaterThan(0);
  });

  it('falls back to the map center when every edge cell is solid', () => {
    const base = createGameState({ seed: 42 });
    const before: GameState = {
      ...base,
      enemies: [],
      spawnCount: 1,
    };
    const solidMap = createSolidEdgeCollisionMap();
    const result = (
      spawnWaveTick as unknown as (
        state: GameState,
        dtMs: number,
        collisionMap: CollisionMap,
      ) => { spawnedThisTick: number; state: GameState }
    )(before, NEATENSTEIN_FIXED_TIMESTEP_MS, solidMap);
    const [enemy] = result.state.enemies;
    const centerDistance = Math.hypot(
      enemy.position.x - NEATENSTEIN_SPAWN_CENTER_X,
      enemy.position.y - NEATENSTEIN_SPAWN_CENTER_Y,
    );
    expect(centerDistance).toBeLessThanOrEqual(2.0);
  });
});

describe('allEnemiesCleared helper', () => {
  it('returns true for an empty roster', () => {
    expect(allEnemiesCleared([])).toBe(true);
  });

  it('returns true when all enemies are dead', () => {
    const enemies: EnemyState[] = [
      { position: { x: 0, y: 0 }, health: 0, active: true } as EnemyState,
      { position: { x: 1, y: 1 }, health: 0, active: true } as EnemyState,
    ];
    expect(allEnemiesCleared(enemies)).toBe(true);
  });

  it('returns true when all enemies are deactivated', () => {
    const enemies: EnemyState[] = [
      { position: { x: 0, y: 0 }, health: 1, active: false } as EnemyState,
    ];
    expect(allEnemiesCleared(enemies)).toBe(true);
  });

  it('treats missing health as dead', () => {
    const enemies: EnemyState[] = [
      { position: { x: 0, y: 0 }, active: true } as EnemyState,
    ];
    expect(allEnemiesCleared(enemies)).toBe(true);
  });

  it('returns false when any enemy is alive and active', () => {
    const enemies: EnemyState[] = [
      { position: { x: 0, y: 0 }, health: 0, active: true } as EnemyState,
      { position: { x: 1, y: 1 }, health: 1, active: true } as EnemyState,
    ];
    expect(allEnemiesCleared(enemies)).toBe(false);
  });
});

describe('AC-10.4-r-001: resolveEdgeSpawn returns open-cell positions only', () => {
  /**
   * Fixture: a collision map where the first two cells inward from the N edge
   * at the spawn center X are solid, but the third cell is open.
   *
   * NEATENSTEIN_SPAWN_CENTER_X = Math.floor(120 / 2) + 0.5 = 60.5, so
   * floor(centerX) = 60. Direction N (spawnCount=0) starts at (60.5, 0.5)
   * and scans inward with dx=0, dy=1.
   *
   * The off-by-one bug in the scan loop checks `Math.floor(y + step * dy)`
   * AFTER y has already been incremented `step` times, so at step=1 it
   * checks cell (60, 2) while returning position (60.5, 1.5) — which is on
   * solid cell (60, 1). After the fix, it should return (60.5, 2.5) on
   * open cell (60, 2).
   */
  it('does not spawn an enemy on a solid cell when the edge cell is blocked (AC-10.4-r-001)', () => {
    const flatMap = new Uint8Array(
      NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE,
    ).fill(0);
    const setSolid = (x: number, y: number): void => {
      flatMap[y * NEATENSTEIN_MAP_SIZE + x] = 1;
    };
    // Solid cells at (60, 0) and (60, 1); cell (60, 2) is open.
    setSolid(60, 0);
    setSolid(60, 1);
    const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);

    // spawnCount=0 → directionIndex=0 → direction N (dx=0, dy=1).
    const before = createGameState({ seed: 1 });
    const result = spawnWaveTick(
      before,
      NEATENSTEIN_FIXED_TIMESTEP_MS,
      collisionMap,
    );
    const [enemy] = result.state.enemies;
    const cellX = Math.floor(enemy.position.x);
    const cellY = Math.floor(enemy.position.y);

    // The spawned enemy must be on an open cell, not inside a wall.
    expect(collisionMap.isSolid(cellX, cellY)).toBe(false);
  });
});

describe('AC-11b-001: enemies spawn with health 100 and stun fields', () => {
  it('spawns an enemy with health = NEATENSTEIN_ENEMY_MAX_HEALTH', () => {
    const before = createGameState({ seed: 1 });
    const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.state.enemies[0].health).toBe(NEATENSTEIN_ENEMY_MAX_HEALTH);
  });

  it('spawns an enemy with maxHealth = NEATENSTEIN_ENEMY_MAX_HEALTH', () => {
    const before = createGameState({ seed: 1 });
    const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.state.enemies[0].maxHealth).toBe(
      NEATENSTEIN_ENEMY_MAX_HEALTH,
    );
  });

  it('spawns an enemy with stunTimerMs = 0', () => {
    const before = createGameState({ seed: 1 });
    const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
    expect(result.state.enemies[0].stunTimerMs).toBe(0);
  });

  it('NEATENSTEIN_ENEMY_MAX_HEALTH is 100', () => {
    expect(NEATENSTEIN_ENEMY_MAX_HEALTH).toBe(100);
  });
});
