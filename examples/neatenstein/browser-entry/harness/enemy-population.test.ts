import { describe, expect, it } from '@jest/globals';
import type { EnemyPopulation } from './enemy-population';

/**
 * Red-phase type-shape tests for examples/neatenstein/browser-entry/harness/enemy-population.ts.
 *
 * Covers AC-306: a common EnemyPopulation interface shared by the MLP and
 * WeightSharedCohort backends. Because the source module does not exist yet,
 * the type-only import fails at compile time until the 03-enemy-mlp slice
 * creates it.
 */

describe('Neatenstein harness enemy-population', () => {
  describe('AC-306: EnemyPopulation interface', () => {
    it('accepts an MLP implementation of EnemyPopulation', () => {
      const population: EnemyPopulation = {
        kind: 'mlp',
        size: 32,
        sample: () => ({ weights: new Float32Array(8) }),
        snapshot: () => ({ kind: 'mlp', weights: new Float32Array(8) }),
      };
      expect(population.kind).toBe('mlp');
    });

    it('accepts a SWARM implementation of EnemyPopulation', () => {
      const population: EnemyPopulation = {
        kind: 'swarm',
        size: 8,
        sample: () => ({ dna: 'abc', coordinates: [0, 0, 0] }),
        snapshot: () => ({ kind: 'swarm', dna: 'abc', coordinates: [] }),
      };
      expect(population.kind).toBe('swarm');
    });
  });
});
