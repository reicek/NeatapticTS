import { describe, expect, it } from '@jest/globals';

import type * as Snapshot from './snapshot';

/**
 * Red-phase contract tests for examples/neatenstein/browser-entry/harness/snapshot.ts.
 *
 * Covers AC-302: the rolling opponent snapshot must refresh every 5
 * generations for MLP enemies and every 3 generations for SWARM enemies.
 */

interface SnapshotModule {
  shouldRefreshMlpSnapshot: typeof Snapshot.shouldRefreshMlpSnapshot;
  shouldRefreshSwarmSnapshot: typeof Snapshot.shouldRefreshSwarmSnapshot;
}

describe('Neatenstein harness snapshot', () => {
  describe('AC-302: refresh cadence', () => {
    it('exports shouldRefreshMlpSnapshot as a function', async () => {
      const mod = (await import('./snapshot.ts')) as Record<string, unknown>;
      expect(typeof mod.shouldRefreshMlpSnapshot).toBe('function');
    });

    it('exports shouldRefreshSwarmSnapshot as a function', async () => {
      const mod = (await import('./snapshot.ts')) as Record<string, unknown>;
      expect(typeof mod.shouldRefreshSwarmSnapshot).toBe('function');
    });

    it('refreshes MLP snapshot only every 5 generations', async () => {
      const { shouldRefreshMlpSnapshot } =
        (await import('./snapshot.ts')) as SnapshotModule;
      expect({
        gen4: shouldRefreshMlpSnapshot(4),
        gen5: shouldRefreshMlpSnapshot(5),
        gen6: shouldRefreshMlpSnapshot(6),
        gen10: shouldRefreshMlpSnapshot(10),
      }).toEqual({
        gen4: false,
        gen5: true,
        gen6: false,
        gen10: true,
      });
    });

    it('refreshes SWARM snapshot only every 3 generations', async () => {
      const { shouldRefreshSwarmSnapshot } =
        (await import('./snapshot.ts')) as SnapshotModule;
      expect({
        gen2: shouldRefreshSwarmSnapshot(2),
        gen3: shouldRefreshSwarmSnapshot(3),
        gen4: shouldRefreshSwarmSnapshot(4),
        gen6: shouldRefreshSwarmSnapshot(6),
      }).toEqual({
        gen2: false,
        gen3: true,
        gen4: false,
        gen6: true,
      });
    });
  });
});
