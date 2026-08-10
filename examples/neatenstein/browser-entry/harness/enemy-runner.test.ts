import { describe, expect, it, jest } from '@jest/globals';

import * as enemyNavigation from '../../scripts/enemy-navigation';
import { activateMlp, createMlpEnemyPopulation } from './enemy-mlp';
import { runEnemyWaveRunner, simulateEnemyEpisode } from './enemy-runner';
import { getEnemySnapshot, refreshEnemySnapshots } from './snapshot';
import type { EnemyEpisodeTelemetry, MlpSnapshot } from './types';

const VARIANT_COUNT = 32;

describe('Neatenstein headless enemy wave runner', () => {
  describe('runEnemyWaveRunner(population, seed, config?)', () => {
    it('exports runEnemyWaveRunner as a function', async () => {
      const mod = (await import('./enemy-runner.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.runEnemyWaveRunner).toBe('function');
    });

    it('returns a champion, per-variant scores, generation, and seed pack', () => {
      const population = createMlpEnemyPopulation({ seed: 1 });
      const result = runEnemyWaveRunner(population, 123);

      expect(result).toHaveProperty('champion');
      expect(result).toHaveProperty('scores');
      expect(result).toHaveProperty('generation');
      expect(result).toHaveProperty('seedPack');
      expect(result.scores).toHaveLength(VARIANT_COUNT);
      expect(result.seedPack.seeds).toHaveLength(VARIANT_COUNT);
      expect(result.generation).toBe(123);
      expect(result.champion.id).toBeGreaterThanOrEqual(0);
      expect(result.champion.id).toBeLessThan(VARIANT_COUNT);
      expect(typeof result.champion.score).toBe('number');
    });

    it('evaluates all 32 variants with unique ids in population order', () => {
      const population = createMlpEnemyPopulation({ seed: 2 });
      const result = runEnemyWaveRunner(population, 456);
      const ids = result.scores.map((entry) => entry.id);

      expect(new Set(ids).size).toBe(VARIANT_COUNT);
      expect(ids).toEqual(
        Array.from({ length: VARIANT_COUNT }, (_, index) => index),
      );
    });

    it('selects the champion with deterministic lowest-id tie-breaking', () => {
      const population = createMlpEnemyPopulation({ seed: 3 });
      const result = runEnemyWaveRunner(population, 789, {
        fitness: {
          navigationWeight: 0,
          combatWeight: 0,
          damageWeight: 0,
          survivalWeight: 0,
        },
      });

      // With zero weights every score is zero, so the lowest id wins.
      expect(result.champion.id).toBe(0);
      expect(result.scores.every((entry) => entry.score === 0)).toBe(true);
    });

    it('uses the generation override from config when provided', () => {
      const population = createMlpEnemyPopulation({ seed: 4 });
      const result = runEnemyWaveRunner(population, 999, { generation: 42 });

      expect(result.generation).toBe(42);
      expect(result.seedPack.generation).toBe(999);
    });

    it('returns a frozen champion snapshot that does not alias live weights', () => {
      const population = createMlpEnemyPopulation({ seed: 5 });
      const result = runEnemyWaveRunner(population, 111);

      expect(Object.isFrozen(result.champion.snapshot)).toBe(true);
      expect(Object.isFrozen(result.champion.snapshot.weights)).toBe(true);

      const liveVariant = population.sample(result.champion.id) as {
        weights: Float32Array;
      };
      const before = liveVariant.weights[0];
      liveVariant.weights[0] = before + 9_999;

      expect(result.champion.snapshot.weights[0]).not.toBe(
        liveVariant.weights[0],
      );
    });

    it('is deterministic for the same seed and population', () => {
      const populationA = createMlpEnemyPopulation({ seed: 6 });
      const populationB = createMlpEnemyPopulation({ seed: 6 });

      const resultA = runEnemyWaveRunner(populationA, 222);
      const resultB = runEnemyWaveRunner(populationB, 222);

      expect(resultA.champion.id).toBe(resultB.champion.id);
      expect(resultA.scores.map((entry) => entry.score)).toEqual(
        resultB.scores.map((entry) => entry.score),
      );
      expect(resultA.seedPack.seeds).toEqual(resultB.seedPack.seeds);
    });

    it('honours custom fitness weights through the optional config', () => {
      const population = createMlpEnemyPopulation({ seed: 7 });

      const defaultRun = runEnemyWaveRunner(population, 333);
      const weightedRun = runEnemyWaveRunner(population, 333, {
        fitness: { damageWeight: 10, survivalWeight: 0 },
      });

      expect(weightedRun.scores.length).toBe(VARIANT_COUNT);
      expect(
        weightedRun.scores.some(
          (entry, index) => entry.score !== defaultRun.scores[index].score,
        ),
      ).toBe(true);
    });

    it('returns different champions for different seeds', () => {
      const populationA = createMlpEnemyPopulation({ seed: 8 });
      const populationB = createMlpEnemyPopulation({ seed: 8 });

      const resultA = runEnemyWaveRunner(populationA, 444);
      const resultB = runEnemyWaveRunner(populationB, 555);

      expect(resultA.seedPack.seeds).not.toEqual(resultB.seedPack.seeds);
    });
  });

  describe('simulateEnemyEpisode(snapshot, episodeSeed)', () => {
    it('exports simulateEnemyEpisode as a function', async () => {
      const mod = (await import('./enemy-runner.ts')) as Record<
        string,
        unknown
      >;
      expect(typeof mod.simulateEnemyEpisode).toBe('function');
    });

    it('returns EnemyEpisodeTelemetry with all required fields', () => {
      const snapshot = { kind: 'mlp' as const, weights: new Float32Array(90) };
      const telemetry: EnemyEpisodeTelemetry = simulateEnemyEpisode(
        snapshot,
        42,
      );

      expect(telemetry).toHaveProperty('position');
      expect(telemetry).toHaveProperty('bfsDistances');
      expect(telemetry).toHaveProperty('damageDealt');
      expect(telemetry).toHaveProperty('enemiesSurvived');
      expect(telemetry).toHaveProperty('cellsVisited');
      expect(telemetry).toHaveProperty('stagnationTicks');
      expect(telemetry).toHaveProperty('finalDistance');
      expect(typeof telemetry.position.x).toBe('number');
      expect(typeof telemetry.position.y).toBe('number');
      expect(Array.isArray(telemetry.bfsDistances)).toBe(true);
      expect(typeof telemetry.damageDealt).toBe('number');
      expect(typeof telemetry.enemiesSurvived).toBe('number');
      expect(typeof telemetry.cellsVisited).toBe('number');
      expect(typeof telemetry.stagnationTicks).toBe('number');
      expect(typeof telemetry.finalDistance).toBe('number');
    });

    it('is deterministic for the same snapshot and seed', () => {
      const snapshot = { kind: 'mlp' as const, weights: new Float32Array(90) };
      const a = simulateEnemyEpisode(snapshot, 100);
      const b = simulateEnemyEpisode(snapshot, 100);

      expect(a).toEqual(b);
    });

    it('produces different telemetry for different weights', () => {
      const population = createMlpEnemyPopulation({ seed: 10 });
      refreshEnemySnapshots(population);
      const snapshotA = getEnemySnapshot(0) as MlpSnapshot;

      const zeroSnapshot = {
        kind: 'mlp' as const,
        weights: new Float32Array(90),
      };

      // Use seed=1: enemy spawns at offset (-1, +1) from player, 2 cells away.
      // Zero weights → all MLP outputs 0 → no movement (cellsVisited = 1).
      // Population weights → non-zero outputs → movement (cellsVisited > 1).
      const telemetryA = simulateEnemyEpisode(snapshotA, 1);
      const telemetryZero = simulateEnemyEpisode(zeroSnapshot, 1);

      expect(telemetryA.cellsVisited).not.toBe(telemetryZero.cellsVisited);
    });

    it('bounds the rollout: cellsVisited ≤ max ticks and ≥ 1', () => {
      const population = createMlpEnemyPopulation({ seed: 11 });
      refreshEnemySnapshots(population);
      const snapshot = getEnemySnapshot(0) as MlpSnapshot;
      const telemetry = simulateEnemyEpisode(snapshot, 300);

      expect(telemetry.cellsVisited).toBeGreaterThanOrEqual(1);
      expect(telemetry.cellsVisited).toBeLessThanOrEqual(240);
    });

    it('reports per-step BFS distances in the telemetry', () => {
      const population = createMlpEnemyPopulation({ seed: 12 });
      refreshEnemySnapshots(population);
      const snapshot = getEnemySnapshot(0) as MlpSnapshot;
      const telemetry = simulateEnemyEpisode(snapshot, 400);

      expect(telemetry.bfsDistances.length).toBeGreaterThan(0);
      expect(telemetry.bfsDistances.every((d) => typeof d === 'number')).toBe(
        true,
      );
    });

    it('reports finalDistance = -1 when the BFS distance is unreachable', () => {
      const distanceSpy = jest
        .spyOn(enemyNavigation, 'getDistance')
        .mockReturnValue(-1);
      const snapshot = { kind: 'mlp' as const, weights: new Float32Array(90) };
      const telemetry = simulateEnemyEpisode(snapshot, 1);

      expect(telemetry.finalDistance).toBe(-1);
      distanceSpy.mockRestore();
    });

    it('always reports enemiesSurvived = 1 (simplified single-enemy rollout)', () => {
      const population = createMlpEnemyPopulation({ seed: 13 });
      refreshEnemySnapshots(population);
      const snapshot = getEnemySnapshot(0) as MlpSnapshot;
      const telemetry = simulateEnemyEpisode(snapshot, 500);

      expect(telemetry.enemiesSurvived).toBe(1);
    });

    it('uses snapshot weights directly with activateMlp (AC-10.5c-004)', () => {
      const population = createMlpEnemyPopulation({ seed: 14 });
      refreshEnemySnapshots(population);
      const snapshot = getEnemySnapshot(0) as MlpSnapshot;

      // Weights from getEnemySnapshot must work directly with activateMlp
      // without any INetwork materialization.
      const vision = new Float32Array(6);
      const outputs = activateMlp(snapshot.weights, vision);

      expect(outputs).toBeInstanceOf(Float32Array);
      expect(outputs.length).toBe(4);
    });
  });
});
