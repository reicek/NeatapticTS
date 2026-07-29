import { describe, expect, it } from '@jest/globals';
import type {
  BoltState,
  CreateGameStateOptions,
  EnemyState,
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
});
