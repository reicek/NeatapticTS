/**
 * Sibling smoke tests for `simulation-worker.opponent-snapshot.service.ts`.
 *
 * Full contract tests live in `simulation-worker.coevolution.test.ts`
 * (authored in Step 03).  This file satisfies the folder quality gate's
 * sibling-test-file requirement.
 *
 * Single-expect rule enforced throughout.
 */
import { createOpponentSnapshotStore } from './simulation-worker.opponent-snapshot.service';

// ---------------------------------------------------------------------------
// Locally-defined contract for the missing opponent snapshot sampler.
// Step 04 implements the actual sampling logic; these interfaces let the
// red-phase tests type-check against the future API.
// ---------------------------------------------------------------------------

type SnapshotSample = {
  readonly snapshotId: string;
  readonly source: 'hall-of-fame' | 'recent';
};

interface OpponentSnapshotSampler {
  sampleOpponentSnapshots(options: {
    hallOfFameIds: readonly string[];
    recentIds: readonly string[];
    sampleCount: number;
  }): readonly SnapshotSample[];
}

type SamplingOpponentSnapshotStore = ReturnType<
  typeof createOpponentSnapshotStore
> &
  OpponentSnapshotSampler;

describe('simulation-worker.opponent-snapshot.service module exports', () => {
  describe('createOpponentSnapshotStore', () => {
    it('is exported as a function', () => {
      expect(typeof createOpponentSnapshotStore).toBe('function');
    });

    it('returns a store with frozenSnapshotId initially null', () => {
      const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });

      expect(store.frozenSnapshotId).toBeNull();
    });

    it('returns a store with isEvaluationActive as a function', () => {
      const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });

      expect(typeof store.isEvaluationActive).toBe('function');
    });

    it('returns a store with beginEvaluation as a function', () => {
      const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });

      expect(typeof store.beginEvaluation).toBe('function');
    });

    it('returns a store with endEvaluation as a function', () => {
      const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });

      expect(typeof store.endEvaluation).toBe('function');
    });

    it('returns a store with tryUpdateSnapshot as a function', () => {
      const store = createOpponentSnapshotStore({ updateEveryNGenerations: 5 });

      expect(typeof store.tryUpdateSnapshot).toBe('function');
    });
  });

  describe('opponent snapshot sampling contract', () => {
    const hallOfFameIds = ['hof-1', 'hof-2', 'hof-3'];
    const recentIds = ['recent-1', 'recent-2', 'recent-3'];

    it('samples only from hall-of-fame and recent pools', () => {
      // Arrange
      const store = createOpponentSnapshotStore({
        updateEveryNGenerations: 5,
      }) as unknown as SamplingOpponentSnapshotStore;

      // Act
      const samples = store.sampleOpponentSnapshots({
        hallOfFameIds,
        recentIds,
        sampleCount: 4,
      });

      // Assert — no sample may come from an unknown source
      expect(
        samples.every(
          (sample) =>
            sample.source === 'hall-of-fame' || sample.source === 'recent',
        ),
      ).toBe(true);
    });

    it('never returns a snapshot id that is not in the provided frozen pools', () => {
      // Arrange
      const store = createOpponentSnapshotStore({
        updateEveryNGenerations: 5,
      }) as unknown as SamplingOpponentSnapshotStore;
      const allowedIds = [...hallOfFameIds, ...recentIds];

      // Act
      const samples = store.sampleOpponentSnapshots({
        hallOfFameIds,
        recentIds,
        sampleCount: 4,
      });

      // Assert — every sampled id must belong to one of the input pools
      expect(
        samples.every((sample) => allowedIds.includes(sample.snapshotId)),
      ).toBe(true);
    });

    it('includes at least one hall-of-fame sample when both pools are non-empty', () => {
      // Arrange
      const store = createOpponentSnapshotStore({
        updateEveryNGenerations: 5,
      }) as unknown as SamplingOpponentSnapshotStore;

      // Act
      const samples = store.sampleOpponentSnapshots({
        hallOfFameIds,
        recentIds,
        sampleCount: 4,
      });

      // Assert — the sample must mix in at least one hall-of-fame entry
      expect(samples.some((sample) => sample.source === 'hall-of-fame')).toBe(
        true,
      );
    });
  });
});
