import { describe, expect, it } from '@jest/globals';

import { isMlpSnapshot } from './arms-race';
import type * as ArmsRace from './arms-race';
import type { MlpSnapshot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/arms-race.ts.
 *
 * Covers:
 * - AC-408: ARMS RACE mode advances at interactive rates with a deterministic
 *   generation barrier.
 * - AC-409: Main agent fitness is evaluated against a frozen MLP snapshot, not
 *   the live enemy MLP population.
 */

interface ArmsRaceModule {
  runArmsRaceGeneration: typeof ArmsRace.runArmsRaceGeneration;
}

describe('Neatenstein harness arms-race', () => {
  describe('AC-408: interactive deterministic generation barrier', () => {
    it('exports runArmsRaceGeneration as a function', async () => {
      const mod = (await import('./arms-race.ts')) as Record<string, unknown>;
      expect(typeof mod.runArmsRaceGeneration).toBe('function');
    });

    it('advances the generation number', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const first = runArmsRaceGeneration({ seed: 1, generation: 1 });
      const second = runArmsRaceGeneration({ seed: 1, generation: 2 });
      expect(second.generation).toBe(first.generation + 1);
    });

    it('returns a synchronous well-formed generation result', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const result = runArmsRaceGeneration({ seed: 1, generation: 1 });
      expect(result.generation).toBe(2);
      expect(result.mainSnapshot).toBeDefined();
      expect(result.enemySnapshot).toBeDefined();
      expect(result.quality).toBeDefined();
      expect(result.enemyBehaviorMetrics).toBeDefined();
    });

    it('produces deterministic barrier state for the same seed', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const first = runArmsRaceGeneration({ seed: 123, generation: 1 });
      const second = runArmsRaceGeneration({ seed: 123, generation: 1 });
      expect({
        generation: first.generation,
        mainSnapshot: first.mainSnapshot,
        enemySnapshot: first.enemySnapshot,
      }).toEqual({
        generation: second.generation,
        mainSnapshot: second.mainSnapshot,
        enemySnapshot: second.enemySnapshot,
      });
    });
  });

  describe('AC-409: frozen MLP snapshot evaluation', () => {
    it('evaluates main fitness against an enemy snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const enemySnapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      const result = runArmsRaceGeneration({
        seed: 5,
        generation: 1,
        enemySnapshot,
      });
      expect(typeof result.quality.survivalTicks).toBe('number');
    });

    it('returns the supplied frozen snapshot instead of generating a new one', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const frozen: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      frozen.weights[0] = 0.5;
      const result = runArmsRaceGeneration({
        seed: 5,
        generation: 5,
        enemySnapshot: frozen,
      });
      if (!isMlpSnapshot(result.enemySnapshot)) {
        throw new Error('Expected result.enemySnapshot to be an MlpSnapshot');
      }
      expect(result.enemySnapshot.weights).toEqual(frozen.weights);
    });

    it('uses a default quality signal when championNetwork is provided without championQuality', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const championNetwork = {
        activate: () => [],
      } as unknown as import('neataptic').Network;
      const result = runArmsRaceGeneration({
        seed: 7,
        generation: 1,
        championNetwork,
      });
      expect(result.quality).toEqual({
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      });
    });
  });
});
