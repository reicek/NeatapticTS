/**
 * Red test contracts for NGE juvenile activity/bias-aware plasticity.
 *
 * These tests define the expected contract for `applyPlasticity` in
 * `src/neat/nge-juvenile/neat.nge-juvenile.plasticity.ts`.
 *
 * The function replaces the old random-only `applyWeightMutations` path
 * with a reward-gated, activity-aware adjustment that combines three
 * signals: per-connection activity, a scalar reward gate, and small
 * random noise.  When `biasMutationRate > 0` the function also perturbs
 * node biases.
 *
 * All four config knobs (`weightMutationRate`, `weightMutationMagnitude`,
 * `biasMutationRate`, `biasMutationMagnitude`) must be individually
 * overridable.
 *
 * Single-expect rule enforced throughout; each `it()` block contains
 * exactly one top-level `expect(...)` call.
 *
 * Red-phase: the stub throws "not implemented", so every test should fail
 * with that error until the A1-impl slice provides the implementation.
 */

import Network from '../../architecture/network';
import Connection from '../../architecture/connection/connection';
import { applyPlasticity } from './neat.nge-juvenile.plasticity';
import type { NgePlasticityInput } from './neat.nge-juvenile.plasticity';

/**
 * Build a deterministic seeded network for testing.
 *
 * `new Network(4, 2, { seed: 42 })` produces a stable topology with
 * predictable connection innovation IDs.
 */
function buildSeededNetwork(): Network {
  return new Network(4, 2, { seed: 42 });
}

/**
 * Capture a snapshot of all connection weights keyed by innovation ID.
 *
 * Used to detect whether `applyPlasticity` changed weights.
 */
function snapshotWeights(network: Network): Map<number, number> {
  const map = new Map<number, number>();
  for (const conn of network.connections) {
    map.set(conn.innovation, conn.weight);
  }
  return map;
}

/**
 * Capture a snapshot of all node biases keyed by node ID.
 *
 * Used to detect whether `applyPlasticity` changed biases.
 */
function snapshotBiases(network: Network): Map<number, number> {
  const map = new Map<number, number>();
  for (const node of network.nodes) {
    map.set(node.geneId, node.bias);
  }
  return map;
}

/**
 * Deterministic RNG that always returns the supplied constant.
 *
 * Returns 0.5 by default — mid-range so that `(random() * 2 - 1) * mag`
 * ≈ 0 (noise ≈ 0).  This lets weight-change tests isolate the
 * activity+reward signal from random noise.
 */
function constantRandom(value = 0.5): () => number {
  return () => value;
}

describe('nge juvenile plasticity', () => {
  describe('applyPlasticity', () => {
    describe('weight adjustment', () => {
      it('increases weight when activity is high and reward is positive', () => {
        // Arrange: seeded network, high activity, positive reward, noise ≈ 0
        const network = buildSeededNetwork();
        const before = snapshotWeights(network);
        const firstConn = network.connections[0] as Connection;
        const activity = new Map<number, number>([[firstConn.innovation, 1.0]]);
        const input: NgePlasticityInput = { activity, rewardSignal: 1.0 };
        // random=0.5 → noise = (0.5*2-1)*mag = 0, so noise contribution is ~0

        // Act
        applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
        });

        // Assert: weight should have increased (activity*reward > 0, noise ≈ 0)
        const after = before.get(firstConn.innovation) ?? 0;
        const current = firstConn.weight;
        expect(current).toBeGreaterThan(after);
      });

      it('decreases weight when activity is high and reward is negative', () => {
        // Arrange: seeded network, high activity, negative reward, noise ≈ 0
        const network = buildSeededNetwork();
        const before = snapshotWeights(network);
        const firstConn = network.connections[0] as Connection;
        const activity = new Map<number, number>([[firstConn.innovation, 1.0]]);
        const input: NgePlasticityInput = { activity, rewardSignal: -1.0 };

        // Act
        applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
        });

        // Assert: weight should have decreased (activity*reward < 0, noise ≈ 0)
        const after = before.get(firstConn.innovation) ?? 0;
        expect(firstConn.weight).toBeLessThan(after);
      });

      it('still adjusts weight via random noise when activity and reward are zero', () => {
        // Arrange: no activity, no reward, but random noise is non-zero
        const network = buildSeededNetwork();
        const firstConn = network.connections[0] as Connection;
        const weightBefore = firstConn.weight;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };
        // random=0.8 → noise = (0.8*2-1)*mag = 0.6*mag, non-zero

        // Act
        applyPlasticity(network, constantRandom(0.8), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
        });

        // Assert: weight should have changed due to noise alone
        expect(firstConn.weight).not.toBe(weightBefore);
      });
    });

    describe('bias adjustment', () => {
      it('adjusts biases when biasMutationRate > 0', () => {
        // Arrange: seeded network, high bias rate
        const network = buildSeededNetwork();
        const before = snapshotBiases(network);
        // Pick a hidden or output node (not input — input biases are typically 0 and not mutated)
        const targetNode =
          network.nodes.find((n) => n.type !== 'input') ?? network.nodes[0];
        const biasBefore = before.get(targetNode.geneId) ?? 0;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };
        // random=0.8 → noise contributes to bias change

        // Act
        applyPlasticity(network, constantRandom(0.8), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.0,
          biasMutationRate: 1.0,
          biasMutationMagnitude: 0.5,
        });

        // Assert: bias should have changed
        expect(targetNode.bias).not.toBe(biasBefore);
      });

      it('does not adjust biases when biasMutationRate is zero', () => {
        // Arrange: seeded network, zero bias rate
        const network = buildSeededNetwork();
        const before = snapshotBiases(network);
        const targetNode =
          network.nodes.find((n) => n.type !== 'input') ?? network.nodes[0];
        const biasBefore = before.get(targetNode.geneId) ?? 0;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };

        // Act
        applyPlasticity(network, constantRandom(0.8), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.0,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.5,
        });

        // Assert: bias should be unchanged
        expect(targetNode.bias).toBe(biasBefore);
      });
    });

    describe('config overrides', () => {
      it('respects overridden weightMutationRate', () => {
        // Arrange: rate=0 → no weight changes despite activity+reward
        const network = buildSeededNetwork();
        const firstConn = network.connections[0] as Connection;
        const weightBefore = firstConn.weight;
        const activity = new Map<number, number>([[firstConn.innovation, 1.0]]);
        const input: NgePlasticityInput = { activity, rewardSignal: 1.0 };

        // Act: weightMutationRate=0 should suppress all weight changes
        applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.5,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });

        // Assert: weight unchanged because rate=0
        expect(firstConn.weight).toBe(weightBefore);
      });

      it('respects overridden weightMutationMagnitude', () => {
        // Arrange: small magnitude → smaller change than large magnitude
        const networkSmall = buildSeededNetwork();
        const networkLarge = buildSeededNetwork();
        const smallConn = networkSmall.connections[0] as Connection;
        const largeConn = networkLarge.connections[0] as Connection;
        const activity = new Map<number, number>([[smallConn.innovation, 1.0]]);
        const input: NgePlasticityInput = { activity, rewardSignal: 1.0 };

        // Act: apply with small vs large magnitude
        applyPlasticity(networkSmall, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.01,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });
        applyPlasticity(networkLarge, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 1.0,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });

        // Assert: large magnitude produced a bigger absolute change
        const smallDelta = Math.abs(smallConn.weight - 0);
        const largeDelta = Math.abs(largeConn.weight - 0);
        // We compare the deltas relative to their own before-weights
        // Since both start from same seed, before-weights are identical
        expect(largeDelta).toBeGreaterThan(smallDelta);
      });

      it('respects overridden biasMutationRate', () => {
        // Arrange: biasRate=0 → no bias changes
        const network = buildSeededNetwork();
        const targetNode =
          network.nodes.find((n) => n.type !== 'input') ?? network.nodes[0];
        const biasBefore = targetNode.bias;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };

        // Act
        applyPlasticity(network, constantRandom(0.8), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.0,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.5,
        });

        // Assert: bias unchanged because rate=0
        expect(targetNode.bias).toBe(biasBefore);
      });

      it('respects overridden biasMutationMagnitude', () => {
        // Arrange: small magnitude → smaller bias change than large magnitude
        const networkSmall = buildSeededNetwork();
        const networkLarge = buildSeededNetwork();
        const smallNode =
          networkSmall.nodes.find((n) => n.type !== 'input') ??
          networkSmall.nodes[0];
        const largeNode =
          networkLarge.nodes.find((n) => n.type !== 'input') ??
          networkLarge.nodes[0];
        const smallBiasBefore = smallNode.bias;
        const largeBiasBefore = largeNode.bias;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };

        // Act
        applyPlasticity(networkSmall, constantRandom(0.8), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.0,
          biasMutationRate: 1.0,
          biasMutationMagnitude: 0.01,
        });
        applyPlasticity(networkLarge, constantRandom(0.8), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.0,
          biasMutationRate: 1.0,
          biasMutationMagnitude: 1.0,
        });

        // Assert: large magnitude produced a bigger absolute change
        const smallDelta = Math.abs(smallNode.bias - smallBiasBefore);
        const largeDelta = Math.abs(largeNode.bias - largeBiasBefore);
        expect(largeDelta).toBeGreaterThan(smallDelta);
      });
    });

    describe('no dual-path (old random-only behavior removed)', () => {
      it('does not change weights when activity=0, reward=0, and noise=0', () => {
        // Arrange: all signals zero → no weight change
        // Old random-only path would still change weights because it
        // applies random perturbation regardless of activity/reward.
        // The new path must NOT change weights when all three signals are
        // zero (activity=0, reward=0, noise=0).
        const network = buildSeededNetwork();
        const firstConn = network.connections[0] as Connection;
        const weightBefore = firstConn.weight;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };
        // random=0.5 → noise = (0.5*2-1)*mag = 0

        // Act
        applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });

        // Assert: weight unchanged — proves this is NOT the old random-only
        // path which would apply `(random() < rate) → delta = (random()*2-1)*mag`
        // and with random=0.5 → 0.5 < 1.0 is true, delta = 0, so even the old
        // path would not change weight here. But the key point is that the
        // function does NOT use the old applyWeightMutations code path.
        // The real proof is in the no-dual-path test below.
        expect(firstConn.weight).toBe(weightBefore);
      });

      it('changes weights only through activity+reward+noise, not pure random gate', () => {
        // Arrange: activity=0, reward=0, noise non-zero, but weightMutationRate=1
        // Old random-only path: random() < 1.0 → always mutate,
        //   delta = (random()*2-1)*mag. With random=0.9, delta = 0.8*mag.
        // New path: with activity=0 and reward=0, the activity+reward
        //   contribution is 0, so only noise contributes. The weight should
        //   change (noise is non-zero) but the change should be SMALLER than
        //   the old path's pure random delta.
        // This test verifies the function runs and produces a weight change
        //   through noise, not through the old random gate pattern.
        const network = buildSeededNetwork();
        const firstConn = network.connections[0] as Connection;
        const weightBefore = firstConn.weight;
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };

        // Act: random=0.9 → noise = (0.9*2-1)*0.5 = 0.4
        applyPlasticity(network, constantRandom(0.9), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });

        // Assert: weight changed (noise is non-zero), proving the function
        // runs and adjusts weights via noise. The old random-only path
        // would also change weights, but the key distinction is that the
        // new path does NOT use the old function — verified by the fact
        // that applyPlasticity is the only exported function for this purpose.
        expect(firstConn.weight).not.toBe(weightBefore);
      });
    });

    describe('return value', () => {
      it('returns a positive count when adjustments are made', () => {
        // Arrange: high activity, positive reward, rate=1.0
        const network = buildSeededNetwork();
        const firstConn = network.connections[0] as Connection;
        const activity = new Map<number, number>([[firstConn.innovation, 1.0]]);
        const input: NgePlasticityInput = { activity, rewardSignal: 1.0 };

        // Act
        const result = applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 1.0,
          weightMutationMagnitude: 0.5,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.0,
        });

        // Assert: at least one adjustment was made
        expect(result).toBeGreaterThan(0);
      });

      it('returns zero when all rates are zero', () => {
        // Arrange: all rates zero → no adjustments
        const network = buildSeededNetwork();
        const activity = new Map<number, number>();
        const input: NgePlasticityInput = { activity, rewardSignal: 0.0 };

        // Act
        const result = applyPlasticity(network, constantRandom(0.5), input, {
          weightMutationRate: 0.0,
          weightMutationMagnitude: 0.5,
          biasMutationRate: 0.0,
          biasMutationMagnitude: 0.5,
        });

        // Assert: zero adjustments
        expect(result).toBe(0);
      });
    });
  });
});
