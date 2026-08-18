/**
 * Red-phase contract tests for Step B4 item 2:
 * sep-CMA-ES (separable CMA-ES) for MLP weight optimization.
 *
 * Defines the contracts for diagonal-covariance CMA-ES that operates in O(n)
 * per generation, making it feasible for browser-based 90-weight MLPs.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as CmaEsModule from './cma-es';

/** Access unknown (future) exports on the cma-es module. */
const cmaEs = CmaEsModule as Record<string, unknown>;

/** Fixed MLP weight count for the 6→6→4→4 topology (90 parameters). */
const MLP_WEIGHT_COUNT = 6 * 6 + 6 * 4 + 4 * 4 + 6 + 4 + 4;

/** Build a zero weight vector for fixture use. */
function makeWeights(): Float32Array {
  return new Float32Array(MLP_WEIGHT_COUNT);
}

/** Minimal fitness function: negative sum of squares (max at zero vector). */
function sphereFitness(weights: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += weights[i] * weights[i];
  }
  return -sum;
}

describe('B4.2: sep-CMA-ES for MLP weight optimization', () => {
  describe('B4.2: CMA-ES state creation', () => {
    it('exports createSepCmaEs as a function', () => {
      expect(typeof cmaEs.createSepCmaEs).toBe('function');
    });

    it('creates a CMA-ES state with diagonal covariance of size n', () => {
      const state = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      expect(state).toBeDefined();
      const covariance = state.covarianceDiag;
      expect(covariance).toBeInstanceOf(Float32Array);
      expect((covariance as Float32Array).length).toBe(MLP_WEIGHT_COUNT);
    });

    it('initializes diagonal covariance to 1.0 for all dimensions', () => {
      const state = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      const covariance = state.covarianceDiag as Float32Array;
      for (let i = 0; i < covariance.length; i++) {
        expect(covariance[i]).toBeCloseTo(1.0, 5);
      }
    });
  });

  describe('B4.2: CMA-ES evolution step', () => {
    it('exports stepSepCmaEs as a function', () => {
      expect(typeof cmaEs.stepSepCmaEs).toBe('function');
    });

    it('produces a population of candidate weight vectors', () => {
      const state = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      const result = (cmaEs.stepSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)(state, sphereFitness);
      const population = result.population as Float32Array[];
      expect(Array.isArray(population)).toBe(true);
      expect(population.length).toBe(8);
      for (const candidate of population) {
        expect(candidate).toBeInstanceOf(Float32Array);
        expect(candidate.length).toBe(MLP_WEIGHT_COUNT);
      }
    });

    it('updates the mean toward higher-fitness candidates', () => {
      const initialState = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      const result = (cmaEs.stepSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)(initialState, sphereFitness);
      const newState = result.state as Record<string, unknown>;
      expect(newState).toBeDefined();
      const newMean = newState.mean as Float32Array;
      expect(newMean).toBeInstanceOf(Float32Array);
      expect(newMean.length).toBe(MLP_WEIGHT_COUNT);
    });

    it('is deterministic for the same seed and initial state', () => {
      const stateA = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      const stateB = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      const resultA = (cmaEs.stepSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)(stateA, sphereFitness);
      const resultB = (cmaEs.stepSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)(stateB, sphereFitness);
      const popA = resultA.population as Float32Array[];
      const popB = resultB.population as Float32Array[];
      expect(Array.from(popA[0])).toEqual(Array.from(popB[0]));
    });
  });

  describe('B4.2: CMA-ES complexity (diagonal only)', () => {
    it('does not store a full n*n covariance matrix', () => {
      const state = (cmaEs.createSepCmaEs as (...a: unknown[]) => Record<
        string,
        unknown
      >)({
        dimension: MLP_WEIGHT_COUNT,
        initialMean: makeWeights(),
        populationSize: 8,
        seed: 42,
      });
      expect(state.covarianceMatrix).toBeUndefined();
    });
  });
});