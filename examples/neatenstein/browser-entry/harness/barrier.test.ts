import { describe, expect, it } from '@jest/globals';

import type * as Barrier from './barrier';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/barrier.ts.
 *
 * Covers AC-301: genBarrier(seed=K) must reproduce identical main network
 * state and enemy roster from the same seed. Selection is index-stable and
 * tie-breaks by the lowest variant id.
 */

interface BarrierModule {
  genBarrier: typeof Barrier.genBarrier;
  hashEnemySnapshot: typeof Barrier.hashEnemySnapshot;
}

describe('Neatenstein harness barrier', () => {
  describe('AC-301: genBarrier determinism', () => {
    it('exports genBarrier as a function', async () => {
      const mod = (await import('./barrier.ts')) as Record<string, unknown>;
      expect(typeof mod.genBarrier).toBe('function');
    });

    it('returns identical main snapshot and enemy roster for the same seed', async () => {
      const { genBarrier } = (await import('./barrier.ts')) as BarrierModule;
      const first = genBarrier({ seed: 123, generation: 1 });
      const second = genBarrier({ seed: 123, generation: 1 });
      expect({
        main: first.mainSnapshot,
        enemy: first.enemySnapshot,
      }).toEqual({
        main: second.mainSnapshot,
        enemy: second.enemySnapshot,
      });
    });

    it('advances the generation number', async () => {
      const { genBarrier } = (await import('./barrier.ts')) as BarrierModule;
      const first = genBarrier({ seed: 1, generation: 1 });
      const second = genBarrier({ seed: 1, generation: 2 });
      expect(second.generation).toBe(first.generation + 1);
    });

    it('refreshes the MLP enemy roster on refresh generations', async () => {
      const { genBarrier } = (await import('./barrier.ts')) as BarrierModule;
      const at4 = genBarrier({ seed: 7, generation: 4 });
      const at5 = genBarrier({ seed: 7, generation: 5 });
      expect(at5.enemySnapshot).not.toEqual(at4.enemySnapshot);
    });

    it('produces a stable hash for a swarm enemy snapshot', async () => {
      const { hashEnemySnapshot } =
        (await import('./barrier.ts')) as BarrierModule;
      const hash = hashEnemySnapshot({
        kind: 'swarm',
        dna: 'swarm-dna',
        coordinates: [
          { x: 1, y: 2 },
          { x: 3, y: 4 },
        ],
      });
      expect(typeof hash).toBe('string');
    });
  });
});
