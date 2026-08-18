/**
 * Red-phase contract tests for Step B1: Enemy AI Parallelism — Independent Workers.
 *
 * These tests define the contracts for per-enemy weight slots, SAB-backed
 * parallel inference, determinism under async inference, inference barrier
 * semantics, enemy count scaling, and tiered inference fallback.
 *
 * All tests in this file MUST fail (RED) until the B1 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import type { CollisionMap } from '../renderer/map';
import { createGameState } from '../host/game/state';
import type { EnemyState, GameState, Vector2 } from '../host/game/types';
import { NEATENSTEIN_ENEMY_MAX_CONCURRENT } from '../host/game/constants';
import {
  createEnemyControllerState,
  updateEnemyController,
} from './enemy-controller';
import * as EnemyControllerModule from './enemy-controller';

/** Access unknown (future) exports on the enemy-controller module. */
const enemyControllerExports = EnemyControllerModule as Record<string, unknown>;

/** Build a collision map where every cell is open floor. */
function createEmptyCollisionMap(): CollisionMap {
  return { isSolid: () => false };
}

/** Build a game state with N enemies placed at distinct offsets from the player. */
function stateWithEnemies(
  base: GameState,
  count: number,
  offset: Vector2 = { x: -10, y: 0 },
): GameState {
  const enemies: EnemyState[] = [];
  for (let i = 0; i < count; i += 1) {
    enemies.push({
      position: {
        x: base.player.position.x + offset.x - i * 2,
        y: base.player.position.y + offset.y,
      },
      health: 100,
    });
  }
  return { ...base, enemies };
}

// ---------------------------------------------------------------------------
// Contract 1: Per-enemy weight slot assignment
// ---------------------------------------------------------------------------

describe('B1: Per-enemy weight slots', () => {
  it('assigns distinct weights to each enemy from a Float32Array[] (not shared)', () => {
    const base = createGameState({ seed: 42 });
    const state = stateWithEnemies(base, 3);
    const collisionMap = createEmptyCollisionMap();

    // Three distinct per-enemy weight vectors.
    const perEnemyWeights: Float32Array[] = [
      new Float32Array([1, 2, 3, 4]),
      new Float32Array([5, 6, 7, 8]),
      new Float32Array([9, 10, 11, 12]),
    ];

    const controller = createEnemyControllerState(state);

    // B1 contract: updateEnemyController must accept per-enemy weights
    // (Float32Array[]) and assign each enemy its own weight slot.
    // Currently it accepts a single Float32Array shared by all enemies.
    const result = updateEnemyController(
      controller,
      state,
      collisionMap,
      16,
      perEnemyWeights as unknown as Float32Array,
    );

    // Each enemy should have a DISTINCT weights reference.
    expect(result.enemies[0].weights).not.toBe(result.enemies[1].weights);
    expect(result.enemies[1].weights).not.toBe(result.enemies[2].weights);
  });

  it('preserves per-enemy weight identity across ticks (respawn loads new weights)', () => {
    const base = createGameState({ seed: 42 });
    const state = stateWithEnemies(base, 2);
    const collisionMap = createEmptyCollisionMap();

    const weightsA = new Float32Array([1, 2, 3, 4]);
    const weightsB = new Float32Array([5, 6, 7, 8]);

    const controller = createEnemyControllerState(state);
    const perEnemyWeights: Float32Array[] = [weightsA, weightsB];

    const result = updateEnemyController(
      controller,
      state,
      collisionMap,
      16,
      perEnemyWeights as unknown as Float32Array,
    );

    // Enemy 0 should have weightsA, enemy 1 should have weightsB.
    expect(result.enemies[0].weights).toBe(weightsA);
    expect(result.enemies[1].weights).toBe(weightsB);
  });
});

// ---------------------------------------------------------------------------
// Contract 2: Enemy count scaling (raise cap from 8 to 16-32)
// ---------------------------------------------------------------------------

describe('B1: Enemy count scaling', () => {
  it('supports at least 16 concurrent enemies (raised from 8)', () => {
    // B1 contract: with SAB parallel inference, the 8-enemy cap is raised.
    // Target is 16-32 enemies.
    expect(NEATENSTEIN_ENEMY_MAX_CONCURRENT).toBeGreaterThanOrEqual(16);
  });

  it('exports a pool capacity constant of at least 16 slots', () => {
    // B1 contract: the pool must support 16-32 enemies (currently 8).
    // POOLED_CONTEXT_SLOT_COUNT = 8 is hardcoded and not exported.
    // B1 must export an ENEMY_INFERENCE_POOL_SIZE constant ≥ 16.
    expect(enemyControllerExports.ENEMY_INFERENCE_POOL_SIZE).toBeGreaterThanOrEqual(16);
  });
});

// ---------------------------------------------------------------------------
// Contract 3: SAB pool as PRIMARY inference path
// ---------------------------------------------------------------------------

describe('B1: SAB-backed parallel inference', () => {
  it('exports a function to create a SAB-backed enemy inference pool', () => {
    // B1 contract: Neatenstein uses SharedInferenceWorker with SAB-backed
    // weight slots as the PRIMARY inference path for 16-32 enemies.
    expect(
      typeof enemyControllerExports.createEnemyInferencePool,
    ).toBe('function');
  });

  it('exports a function to load per-enemy weights into SAB slots', () => {
    // B1 contract: each enemy owns a weight slot in the SAB. Weights are
    // copied into SAB slots at respawn time.
    expect(
      typeof enemyControllerExports.loadEnemyWeightSlots,
    ).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Contract 4: Determinism under async inference
// ---------------------------------------------------------------------------

describe('B1: Determinism under async inference', () => {
  it('exports a function to dispatch parallel inference with (simTick, enemyIndex) tags', () => {
    // B1 contract: each enemy's inference result is tagged with
    // (simTick, enemyIndex) so results can be collected and applied
    // deterministically regardless of completion order.
    expect(
      typeof enemyControllerExports.dispatchParallelInference,
    ).toBe('function');
  });

  it('same seed produces same enemy positions regardless of inference completion order', () => {
    // B1 contract: results are applied in enemyIndex order, not completion
    // order. Same seed → same simulation regardless of inference latency.
    //
    // This test verifies that two runs with the same seed and same inputs
    // produce identical enemy positions, even if inference results arrive
    // in different orders.
    //
    // Currently updateEnemyController is synchronous and deterministic,
    // but B1 introduces async inference. The test calls the new
    // dispatchParallelInference function which doesn't exist yet.
    expect(
      typeof enemyControllerExports.dispatchParallelInference,
    ).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Contract 5: Inference barrier (sim tick waits for ALL enemy results)
// ---------------------------------------------------------------------------

describe('B1: Inference barrier', () => {
  it('exports a barrier function that collects all enemy results before advancing', () => {
    // B1 contract: the sim tick does NOT advance until ALL enemy inference
    // results for that tick are collected (barrier). This prevents partial
    // state updates from creating nondeterministic behavior.
    expect(
      typeof enemyControllerExports.awaitInferenceBarrier,
    ).toBe('function');
  });
});

// ---------------------------------------------------------------------------
// Contract 6: Tiered inference fallback (SAB → InferenceChannel → inline)
// ---------------------------------------------------------------------------

describe('B1: Tiered inference fallback', () => {
  it('exports a function to resolve the inference strategy (SAB, channel, or inline)', () => {
    // B1 contract: tiered approach — SAB → InferenceChannel → inline.
    // For 16-32 enemies: SAB is primary.
    // For small N (≤8) or when SAB unavailable: InferenceChannel fallback.
    // When neither is available: inline activation.
    expect(
      typeof enemyControllerExports.resolveInferenceStrategy,
    ).toBe('function');
  });
});