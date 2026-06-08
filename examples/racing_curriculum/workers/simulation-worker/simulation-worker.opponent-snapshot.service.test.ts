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
});
