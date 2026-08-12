import { describe, expect, it } from '@jest/globals';
import type {
  BoltState,
  CreateGameStateOptions,
  EnemyBoltState,
  EnemyState,
  EpisodeTelemetry,
  GameState,
  PlayerState,
  Vector2,
} from './types';

/**
 * Sibling test file for examples/neatenstein/browser-entry/host/game/types.ts.
 *
 * These tests exercise the exported type shapes at compile time and perform
 * lightweight runtime assertions so the folder quality gate sees a sibling
 * test file for every source module.
 */

describe('Neatenstein game types', () => {
  it('accepts a Vector2 shape', () => {
    const vector: Vector2 = { x: 1, y: 2 };
    expect(vector).toEqual({ x: 1, y: 2 });
  });

  it('accepts a PlayerState shape', () => {
    const player: PlayerState = {
      position: { x: 0, y: 0 },
      angleRad: 0,
      health: 100,
      maxHealth: 100,
      ammo: 30,
      maxAmmo: 30,
      dashTimeRemainingMs: 0,
      dashCooldownMs: 0,
    };
    expect(player.health).toBe(100);
  });

  it('accepts an EnemyState shape', () => {
    const enemy: EnemyState = { position: { x: 1, y: 1 }, health: 10 };
    expect(enemy.health).toBe(10);
  });

  it('accepts CreateGameStateOptions with an optional seed', () => {
    const options: CreateGameStateOptions = { seed: 7 };
    expect(options.seed).toBe(7);
  });

  it('accepts a complete GameState shape', () => {
    const state: GameState = {
      seed: 1,
      simTimeMs: 0,
      episodeTimeMs: 0,
      player: {
        position: { x: 0, y: 0 },
        angleRad: 0,
        health: 100,
        maxHealth: 100,
        ammo: 30,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
      enemies: [],
      impacts: [],
      bolts: [],
      kills: 0,
      spawnCount: 0,
      generation: 1,
    };
    expect(state.seed).toBe(1);
  });

  it('accepts a BoltState shape', () => {
    const bolt: BoltState = {
      position: { x: 0, y: 0 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 10,
      active: true,
      createdAtMs: 0,
    };
    expect(bolt.active).toBe(true);
  });

  it('accepts an EnemyBoltState shape', () => {
    const enemyBolt: EnemyBoltState = {
      position: { x: 1, y: 2 },
      direction: { x: -1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 500,
      origin: { x: 0, y: 2 },
      damage: 10,
      hitPlayer: false,
    };
    expect(enemyBolt.damage).toBe(10);
    expect(enemyBolt.hitPlayer).toBe(false);
  });

  it('accepts an EnemyBoltState without optional fields', () => {
    const enemyBolt: EnemyBoltState = {
      position: { x: 5, y: 5 },
      direction: { x: 1, y: 0 },
      speedCellsPerSecond: 36,
      active: true,
      createdAtMs: 0,
      damage: 10,
    };
    expect(enemyBolt.origin).toBeUndefined();
    expect(enemyBolt.hitPlayer).toBeUndefined();
  });

  it('accepts a GameState shape with enemyBolts', () => {
    const state: GameState = {
      seed: 1,
      simTimeMs: 0,
      episodeTimeMs: 0,
      player: {
        position: { x: 0, y: 0 },
        angleRad: 0,
        health: 100,
        maxHealth: 100,
        ammo: 30,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
      enemies: [],
      impacts: [],
      bolts: [],
      enemyBolts: [
        {
          position: { x: 1, y: 1 },
          direction: { x: 0, y: 1 },
          speedCellsPerSecond: 36,
          active: true,
          createdAtMs: 0,
          damage: 10,
        },
      ],
      kills: 0,
      spawnCount: 0,
      generation: 1,
    };
    expect(state.enemyBolts).toHaveLength(1);
    expect(state.enemyBolts![0].damage).toBe(10);
  });

  it('accepts a GameState shape with lastShotHit flag (AC-P3S1c-003)', () => {
    const state: GameState = {
      seed: 1,
      simTimeMs: 0,
      episodeTimeMs: 0,
      player: {
        position: { x: 0, y: 0 },
        angleRad: 0,
        health: 100,
        maxHealth: 100,
        ammo: 30,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
      enemies: [],
      impacts: [],
      bolts: [],
      kills: 0,
      spawnCount: 0,
      generation: 1,
      lastShotHit: true,
    };
    expect(state.lastShotHit).toBe(true);
  });

  it('accepts a GameState shape without lastShotHit (optional field)', () => {
    const state: GameState = {
      seed: 1,
      simTimeMs: 0,
      episodeTimeMs: 0,
      player: {
        position: { x: 0, y: 0 },
        angleRad: 0,
        health: 100,
        maxHealth: 100,
        ammo: 30,
        maxAmmo: 30,
        dashTimeRemainingMs: 0,
        dashCooldownMs: 0,
      },
      enemies: [],
      impacts: [],
      bolts: [],
      kills: 0,
      spawnCount: 0,
      generation: 1,
    };
    expect(state.lastShotHit).toBeUndefined();
  });

  // AC-P4S1a-001: EpisodeTelemetry includes shot outcome taxonomy fields
  it('accepts an EpisodeTelemetry shape with shot outcome taxonomy fields', () => {
    const telemetry: EpisodeTelemetry = {
      damageDealt: 100,
      shotsFired: 20,
      shotsHit: 10,
      aimMissRate: 0.5,
      shotsWallHit: 3,
      shotsRangeExpired: 2,
      shotsBlindFire: 4,
      shotsNearMiss: 1,
    };
    expect(telemetry.shotsWallHit).toBe(3);
    expect(telemetry.shotsRangeExpired).toBe(2);
    expect(telemetry.shotsBlindFire).toBe(4);
    expect(telemetry.shotsNearMiss).toBe(1);
  });
});
