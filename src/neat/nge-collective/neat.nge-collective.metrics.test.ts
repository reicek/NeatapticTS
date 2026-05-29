/**
 * Red-phase test contract for the NGE collective observability metrics shelf.
 *
 * Target production module: `./neat.nge-collective.metrics`
 * Phase G Step 03 — intended failures before any implementation exists.
 *
 * Covered behaviors:
 * 1. `computeRoleDivergenceMetric` returns 0 for two identical module-size distributions.
 * 2. `computeRoleDivergenceMetric` returns a positive value for clearly divergent distributions.
 * 3. `createOpponentSnapshotPool` creates an empty pool with the specified capacity.
 * 4. `addOpponentSnapshot` freezes the snapshot so post-registration mutations to the
 *    original object are not reflected in the stored snapshot.
 * 5. `addOpponentSnapshot` at capacity rotates out the oldest snapshot to make room.
 */

import {
  addOpponentSnapshot,
  computeRoleDivergenceMetric,
  createOpponentSnapshotPool,
} from './neat.nge-collective.metrics';

describe('neat.nge-collective observability metrics', () => {
  describe('computeRoleDivergenceMetric', () => {
    describe('given two identical module-size distributions', () => {
      it('returns 0 when both distributions are equal', () => {
        // Arrange — two agents with the same module size breakdown (worker/soldier/scout)
        const distributionA = [10, 5, 3];
        const distributionB = [10, 5, 3];

        // Act
        const divergence = computeRoleDivergenceMetric(
          distributionA,
          distributionB,
        );

        // Assert
        expect(divergence).toBe(0);
      });
    });

    describe('given completely divergent module-size distributions', () => {
      it('returns a positive value when one agent is all-worker and the other is all-soldier', () => {
        // Arrange — agent A is entirely workers; agent B is entirely soldiers
        const allWorkerDistribution = [20, 0, 0];
        const allSoldierDistribution = [0, 20, 0];

        // Act
        const divergence = computeRoleDivergenceMetric(
          allWorkerDistribution,
          allSoldierDistribution,
        );

        // Assert
        expect(divergence).toBeGreaterThan(0);
      });

      it('returns a higher divergence for fully opposite distributions than for partially shifted ones', () => {
        // Arrange
        const baseDistribution = [10, 5, 3];
        const slightlyDifferentDistribution = [10, 6, 3]; // one unit shifted
        const totallyOppositeDistribution = [0, 18, 0]; // all mass moved

        // Act
        const smallDivergence = computeRoleDivergenceMetric(
          baseDistribution,
          slightlyDifferentDistribution,
        );
        const largeDivergence = computeRoleDivergenceMetric(
          baseDistribution,
          totallyOppositeDistribution,
        );

        // Assert — total opposite must be strictly more divergent
        expect(largeDivergence).toBeGreaterThan(smallDivergence);
      });
    });

    describe('given a single-element distribution', () => {
      it('returns 0 when both single-element distributions are equal', () => {
        // Arrange
        const distributionA = [7];
        const distributionB = [7];

        // Act
        const divergence = computeRoleDivergenceMetric(
          distributionA,
          distributionB,
        );

        // Assert
        expect(divergence).toBe(0);
      });
    });

    describe('given distributions of different lengths', () => {
      it('treats out-of-bounds slots as 0 and returns the L1 difference', () => {
        // Arrange — distributionA has one element, distributionB has two; slot [1] is missing from A
        const distributionA = [10];
        const distributionB = [10, 5];

        // Act
        const divergence = computeRoleDivergenceMetric(
          distributionA,
          distributionB,
        );

        // Assert — slot [1] contributes |0 - 5| = 5
        expect(divergence).toBe(5);
      });

      it('treats missing slots in the shorter second distribution as 0', () => {
        // Arrange — distributionB is shorter; slot [1] is missing from B
        const distributionA = [10, 5];
        const distributionB = [10];

        // Act
        const divergence = computeRoleDivergenceMetric(
          distributionA,
          distributionB,
        );

        // Assert — slot [1] contributes |5 - 0| = 5
        expect(divergence).toBe(5);
      });
    });
  });

  describe('createOpponentSnapshotPool', () => {
    describe('given capacity = 3', () => {
      it('returns a pool with the declared capacity', () => {
        // Arrange / Act
        const pool = createOpponentSnapshotPool(3);

        // Assert
        expect(pool.capacity).toBe(3);
      });

      it('starts with an empty snapshots list', () => {
        // Arrange / Act
        const pool = createOpponentSnapshotPool(3);

        // Assert
        expect(pool.snapshots.length).toBe(0);
      });
    });
  });

  describe('addOpponentSnapshot', () => {
    describe('given a pool below capacity', () => {
      it('increases the snapshot count by one', () => {
        // Arrange
        const pool = createOpponentSnapshotPool(3);
        const agentPayload = { moduleCount: 12 };

        // Act
        const updatedPool = addOpponentSnapshot(
          pool,
          'agent:alpha',
          agentPayload,
          1,
        );

        // Assert
        expect(updatedPool.snapshots.length).toBe(1);
      });

      it('records the correct agentId on the stored snapshot', () => {
        // Arrange
        const pool = createOpponentSnapshotPool(3);

        // Act
        const updatedPool = addOpponentSnapshot(
          pool,
          'agent:beta',
          { moduleCount: 8 },
          2,
        );

        // Assert
        expect(updatedPool.snapshots[0]?.agentId).toBe('agent:beta');
      });

      it('records the generation tick passed at registration time', () => {
        // Arrange
        const pool = createOpponentSnapshotPool(3);

        // Act
        const updatedPool = addOpponentSnapshot(pool, 'agent:gamma', {}, 5);

        // Assert
        expect(updatedPool.snapshots[0]?.frozenAt).toBe(5);
      });
    });

    describe('snapshot immutability (freeze contract)', () => {
      it('does not reflect post-registration mutations to the original payload', () => {
        // Arrange
        const pool = createOpponentSnapshotPool(3);
        const mutablePayload = { fitness: 42 };

        // Act — register snapshot then mutate the original
        const updatedPool = addOpponentSnapshot(
          pool,
          'agent:delta',
          mutablePayload,
          3,
        );
        mutablePayload.fitness = 999; // mutate after registration

        // Assert — stored snapshot preserves the value at registration time
        expect(
          (updatedPool.snapshots[0]?.snapshot as { fitness: number }).fitness,
        ).toBe(42);
      });
    });

    describe('rolling rotation at capacity', () => {
      it('removes the oldest snapshot when the pool is at capacity and a new one is added', () => {
        // Arrange — fill pool to capacity with agents alpha, beta, gamma (in order)
        const emptyPool = createOpponentSnapshotPool(2);
        const poolWithAlpha = addOpponentSnapshot(
          emptyPool,
          'agent:alpha',
          {},
          1,
        );
        const poolWithBeta = addOpponentSnapshot(
          poolWithAlpha,
          'agent:beta',
          {},
          2,
        );

        // Act — add a third snapshot, which should rotate out 'agent:alpha'
        const rotatedPool = addOpponentSnapshot(
          poolWithBeta,
          'agent:gamma',
          {},
          3,
        );

        // Assert — 'agent:alpha' (oldest) is no longer in the pool
        expect(
          (rotatedPool.snapshots as Array<{ agentId: string }>).some(
            (snap) => snap.agentId === 'agent:alpha',
          ),
        ).toBe(false);
      });

      it('retains the most recently added snapshot after rotation', () => {
        // Arrange — fill a capacity-2 pool
        const emptyPool = createOpponentSnapshotPool(2);
        const poolWithAlpha = addOpponentSnapshot(
          emptyPool,
          'agent:alpha',
          {},
          1,
        );
        const poolWithBeta = addOpponentSnapshot(
          poolWithAlpha,
          'agent:beta',
          {},
          2,
        );

        // Act — rotation: alpha evicted, gamma added
        const rotatedPool = addOpponentSnapshot(
          poolWithBeta,
          'agent:gamma',
          {},
          3,
        );

        // Assert — 'agent:gamma' is present
        expect(
          (rotatedPool.snapshots as Array<{ agentId: string }>).some(
            (snap) => snap.agentId === 'agent:gamma',
          ),
        ).toBe(true);
      });

      it('keeps the snapshot count at capacity after rotation', () => {
        // Arrange
        const emptyPool = createOpponentSnapshotPool(2);
        const poolWithAlpha = addOpponentSnapshot(
          emptyPool,
          'agent:alpha',
          {},
          1,
        );
        const poolWithBeta = addOpponentSnapshot(
          poolWithAlpha,
          'agent:beta',
          {},
          2,
        );

        // Act
        const rotatedPool = addOpponentSnapshot(
          poolWithBeta,
          'agent:gamma',
          {},
          3,
        );

        // Assert — still exactly 2 snapshots (capacity was not exceeded)
        expect(rotatedPool.snapshots.length).toBe(2);
      });
    });
  });
});
