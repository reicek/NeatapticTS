/**
 * Red-phase contract tests for Step B4 items 3 + 9:
 * Unified league structure (AlphaStar-style) and NSNE forgetting prevention.
 *
 * Replaces the separate hall-of-fame (A3.6) and opponent pool (B4.3) with a
 * single league containing current champions, past champions (main
 * exploiters), and diverse strategy samples from MAP-Elites. Past champions
 * serve as the forgetting-prevention curriculum.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as LeagueModule from './league';

/** Access unknown (future) exports on the league module. */
const league = LeagueModule as Record<string, unknown>;

/** Fixed MLP weight count for the 6→6→4→4 topology (90 parameters). */
const MLP_WEIGHT_COUNT = 6 * 6 + 6 * 4 + 4 * 4 + 6 + 4 + 4;

/** Build a deterministic weight vector for fixture use. */
function makeWeights(seed: number): Float32Array {
  const weights = new Float32Array(MLP_WEIGHT_COUNT);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (seed + i) * 0.01;
  }
  return weights;
}

describe('B4.3+B4.9: Unified league structure', () => {
  describe('B4.3: league creation', () => {
    it('exports createLeague as a function', () => {
      expect(typeof league.createLeague).toBe('function');
    });

    it('creates an empty league with bounded capacity', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 10, maxDiverseSamples: 10 });
      expect(leagueState).toBeDefined();
      expect(leagueState.pastChampions).toEqual([]);
      expect(leagueState.diverseSamples).toEqual([]);
    });
  });

  describe('B4.3: champion management', () => {
    it('exports addCurrentChampion as a function', () => {
      expect(typeof league.addCurrentChampion).toBe('function');
    });

    it('exports archiveCurrentChampion as a function', () => {
      expect(typeof league.archiveCurrentChampion).toBe('function');
    });

    it('moves the current champion to past champions when a new one is added', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 10, maxDiverseSamples: 10 });
      const championA = {
        weights: makeWeights(1),
        generation: 1,
        fitness: 10,
      };
      const championB = {
        weights: makeWeights(2),
        generation: 2,
        fitness: 20,
      };
      (league.addCurrentChampion as (...a: unknown[]) => unknown)(
        leagueState,
        championA,
      );
      (league.addCurrentChampion as (...a: unknown[]) => unknown)(
        leagueState,
        championB,
      );
      const past = leagueState.pastChampions as Record<string, unknown>[];
      expect(past.length).toBe(1);
      expect(past[0].generation).toBe(1);
      const current = leagueState.currentChampion as Record<string, unknown>;
      expect(current.generation).toBe(2);
    });

    it('enforces the past-champion capacity bound (FIFO eviction)', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 3, maxDiverseSamples: 10 });
      for (let gen = 1; gen <= 5; gen++) {
        (league.addCurrentChampion as (...a: unknown[]) => unknown)(leagueState, {
          weights: makeWeights(gen),
          generation: gen,
          fitness: gen * 10,
        });
      }
      const past = leagueState.pastChampions as Record<string, unknown>[];
      expect(past.length).toBe(3);
      expect(past[0].generation).toBe(3);
    });
  });

  describe('B4.3: diverse samples from MAP-Elites', () => {
    it('exports addDiverseSample as a function', () => {
      expect(typeof league.addDiverseSample).toBe('function');
    });

    it('stores diverse strategy samples up to capacity', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 10, maxDiverseSamples: 5 });
      for (let i = 0; i < 7; i++) {
        (league.addDiverseSample as (...a: unknown[]) => unknown)(leagueState, {
          weights: makeWeights(i + 100),
          behaviorMetrics: {
            aggression: i * 0.1,
            positioning: 0.5,
            movementPattern: 0.5,
          },
        });
      }
      const samples = leagueState.diverseSamples as unknown[];
      expect(samples.length).toBeLessThanOrEqual(5);
    });
  });

  describe('B4.3: opponent sampling', () => {
    it('exports sampleOpponents as a function', () => {
      expect(typeof league.sampleOpponents).toBe('function');
    });

    it('returns a mix of past champions and diverse samples', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 10, maxDiverseSamples: 10 });
      (league.addCurrentChampion as (...a: unknown[]) => unknown)(leagueState, {
        weights: makeWeights(1),
        generation: 1,
        fitness: 10,
      });
      (league.addCurrentChampion as (...a: unknown[]) => unknown)(leagueState, {
        weights: makeWeights(2),
        generation: 2,
        fitness: 20,
      });
      (league.addDiverseSample as (...a: unknown[]) => unknown)(leagueState, {
        weights: makeWeights(101),
        behaviorMetrics: {
          aggression: 0.9,
          positioning: 0.1,
          movementPattern: 0.5,
        },
      });
      const opponents = (league.sampleOpponents as (...a: unknown[]) => unknown[])(
        leagueState,
        3,
        42,
      );
      expect(Array.isArray(opponents)).toBe(true);
      expect(opponents.length).toBeGreaterThan(0);
    });
  });

  describe('B4.9: forgetting prevention via past champion curriculum', () => {
    it('exports getCurriculumOpponents as a function', () => {
      expect(typeof league.getCurriculumOpponents).toBe('function');
    });

    it('returns past champions for periodic re-evaluation', () => {
      const leagueState = (league.createLeague as (...a: unknown[]) => Record<
        string,
        unknown
      >)({ maxPastChampions: 10, maxDiverseSamples: 10 });
      for (let gen = 1; gen <= 3; gen++) {
        (league.addCurrentChampion as (...a: unknown[]) => unknown)(leagueState, {
          weights: makeWeights(gen),
          generation: gen,
          fitness: gen * 10,
        });
      }
      const curriculum = (league.getCurriculumOpponents as (...a: unknown[]) => unknown[])(
        leagueState,
      );
      expect(Array.isArray(curriculum)).toBe(true);
      expect(curriculum.length).toBe(2);
    });
  });
});