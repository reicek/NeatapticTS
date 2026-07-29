import { describe, expect, it } from '@jest/globals';

import { isMlpSnapshot } from './arms-race';
import type * as ArmsRace from './arms-race';
import type * as EnemyMlp from './enemy-mlp';
import type { MlpSnapshot } from './types';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/arms-race.ts
 * focused on the frozen MLP snapshot guarantee.
 *
 * Covers AC-409: Main agent fitness is evaluated against a frozen MLP snapshot,
 * not the live enemy MLP population.
 */

interface ArmsRaceModule {
  runArmsRaceGeneration: typeof ArmsRace.runArmsRaceGeneration;
}

interface EnemyMlpModule {
  createMlpEnemyPopulation: typeof EnemyMlp.createMlpEnemyPopulation;
}

describe('Neatenstein harness enemy-mlp snapshot', () => {
  describe('AC-409: frozen snapshot vs live population', () => {
    it('exports runArmsRaceGeneration from the arms-race module', async () => {
      const mod = (await import('./arms-race.ts')) as Record<string, unknown>;
      expect(typeof mod.runArmsRaceGeneration).toBe('function');
    });

    it('evaluates main fitness against the supplied enemy snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const snapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      const result = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      if (!isMlpSnapshot(result.enemySnapshot)) {
        throw new Error('Expected result.enemySnapshot to be an MlpSnapshot');
      }
      expect(result.enemySnapshot.weights).toEqual(snapshot.weights);
    });

    it('returns the same main fitness when replayed with the same frozen snapshot', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const snapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      const first = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      const second = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: snapshot,
      });
      expect(first.quality).toEqual(second.quality);
    });

    it('produces different main fitness for different frozen snapshots', async () => {
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const firstSnapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      firstSnapshot.weights[0] = 0.1;
      const secondSnapshot: MlpSnapshot = {
        kind: 'mlp',
        weights: new Float32Array(80),
      };
      secondSnapshot.weights[0] = 0.9;
      const first = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: firstSnapshot,
      });
      const second = runArmsRaceGeneration({
        seed: 1,
        generation: 1,
        enemySnapshot: secondSnapshot,
      });
      expect(first.quality).not.toEqual(second.quality);
    });

    it('does not mutate the live MLP population when evaluating from a snapshot', async () => {
      const { createMlpEnemyPopulation } =
        (await import('./enemy-mlp.ts')) as EnemyMlpModule;
      const { runArmsRaceGeneration } =
        (await import('./arms-race.ts')) as ArmsRaceModule;
      const population = createMlpEnemyPopulation({ seed: 1 });
      const liveSnapshot = population.snapshot();
      if (!isMlpSnapshot(liveSnapshot)) {
        throw new Error(
          'Expected population.snapshot() to return an MlpSnapshot',
        );
      }
      const liveBefore = liveSnapshot.weights[0];
      const frozen = population.snapshot();
      runArmsRaceGeneration({
        seed: 1,
        generation: 5,
        enemySnapshot: frozen,
      });
      const afterSnapshot = population.snapshot();
      if (!isMlpSnapshot(afterSnapshot)) {
        throw new Error(
          'Expected population.snapshot() to return an MlpSnapshot',
        );
      }
      expect(afterSnapshot.weights[0]).toBe(liveBefore);
    });
  });
});
