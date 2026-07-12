/**
 * Red-phase E2E contracts for the NGE Core Growth Engine wiring.
 *
 * These tests verify that the full NGE growth pipeline
 * (`adaptOnTick` → `computeGrowthThrottle` → `runNgeLifecycle` →
 * `computeFocusScores` → `planGrowthMorphs` → `applyMorphDeltas` →
 * `commitGrowth`) operates end-to-end against a live `Network` instance.
 *
 * A custom `trendOnlyEvaluator` isolates the growth commit decision from the
 * default evaluator's size penalty, so committed growth is exercised
 * deterministically.  Tests target Phase 5 Step 17 of the NGE Core Growth
 * Engine Wiring plan and will fail until the pipeline is fully wired.
 *
 * Single-expect rule enforced throughout.  AAA structure in every test.
 */
import { Network } from '../../../src/browser-entry';
import {
  createRuntimeAdaptationEngine,
  type RacingQualitySignal,
} from './runtime.adaptation';

/**
 * Deterministic mulberry32 PRNG factory.
 *
 * Produces a `() => number` function that yields the same sequence for a
 * given seed on every invocation, ensuring reproducible NGE lifecycle
 * morph decisions in tests.
 *
 * @param seed - Unsigned 32-bit integer seed.
 * @returns A deterministic `() => number` returning floats in [0, 1).
 */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

/**
 * Trend-only evaluator that ignores the network's size penalty.
 *
 * Returns `last - first` from the rolling score history so baseline and
 * candidate scores are identical for the same evidence window — keeping
 * `improvement >= improvementThreshold (0)` true and growth committed.
 */
const trendOnlyEvaluator = (
  _network: Network,
  scoreHistory: readonly (number | RacingQualitySignal)[],
): number => {
  if (scoreHistory.length === 0) return 0;
  const last = scoreHistory.at(-1);
  if (last === undefined) return 0;
  const first = scoreHistory[0];
  const lastValue = typeof last === 'number' ? last : (last.score ?? 0);
  const firstValue = typeof first === 'number' ? first : (first.score ?? 0);
  return lastValue - firstValue;
};

describe('NGE Core Growth Engine E2E pipeline', () => {
  describe('seed-to-growth proof', () => {
    it('grows the network beyond its initial seed size after sustained ticks', () => {
      // Arrange
      const network = new Network(4, 2, { seed: 42 });
      const initialNodeCount = network.nodes.length;
      const initialConnCount = network.connections.length;
      const engine = createRuntimeAdaptationEngine({
        evaluateScore: trendOnlyEvaluator,
      });
      const scoreHistory = [1, 2, 3, 4];

      // Act
      for (let tick = 0; tick < 20; tick++) {
        engine.adaptOnTick({ tick, network, scoreHistory });
      }

      // Assert
      const networkGrew =
        network.nodes.length > initialNodeCount ||
        network.connections.length > initialConnCount;
      expect(networkGrew).toBe(true);
    });
  });

  describe('committed telemetry', () => {
    it('emits committed=true with at least one operation on the first eligible tick', () => {
      // Arrange
      const network = new Network(4, 2, { seed: 42 });
      const engine = createRuntimeAdaptationEngine({
        evaluateScore: trendOnlyEvaluator,
      });

      // Act
      const telemetry = engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory: [1, 2, 3, 4],
      });

      // Assert
      expect(telemetry.committed && telemetry.operations.length > 0).toBe(true);
    });
  });

  describe('overall growth trend across committed ticks', () => {
    it('grows the total network size (nodes + connections) beyond the initial size after sustained ticks', () => {
      // Arrange — seeded PRNG ensures deterministic morph decisions across
      // runs. The assertion checks the live network's total size (all nodes
      // plus connections) after N ticks, independent of committed-size
      // tracking, so prune morphs cannot produce false negatives.
      const seed = 12_345;
      const network = new Network(4, 2, { seed });
      const initialTotalSize =
        network.nodes.length + network.connections.length;
      const engine = createRuntimeAdaptationEngine({
        cadence: { mode: 'every_tick' },
        evaluateScore: trendOnlyEvaluator,
        limits: { mutationCooldownTicks: 0 },
        random: mulberry32(seed),
      });
      const scoreHistory = [1, 2, 3, 4];

      // Act
      for (let tick = 0; tick < 10; tick++) {
        engine.adaptOnTick({ tick, network, scoreHistory });
      }
      const finalTotalSize = network.nodes.length + network.connections.length;

      // Assert — the live network must have grown beyond its initial total
      // size after sustained adaptation ticks.
      expect(finalTotalSize > initialTotalSize).toBe(true);
    });
  });

  describe('growth throttle', () => {
    it('engages growth_throttled reason on tick 1 when the network exceeds the large-network threshold', () => {
      // Arrange — 1002 nodes exceeds the LARGE_NETWORK_NODE_THRESHOLD (1000).
      const network = new Network(1001, 1, { seed: 42 });
      const engine = createRuntimeAdaptationEngine({
        cadence: { mode: 'every_tick' },
        evaluateScore: trendOnlyEvaluator,
        limits: { mutationCooldownTicks: 0 },
      });
      const scoreHistory = [1, 2, 3, 4];

      // Act
      engine.adaptOnTick({ tick: 0, network, scoreHistory });
      const secondTelemetry = engine.adaptOnTick({
        tick: 1,
        network,
        scoreHistory,
      });

      // Assert
      expect(secondTelemetry.reason).toBe('growth_throttled');
    });
  });

  describe('hysteresis cooldown after committed growth', () => {
    it('returns mutation_cooldown_active on the tick immediately after a committed morph', () => {
      // Arrange
      const network = new Network(4, 2, { seed: 42 });
      const engine = createRuntimeAdaptationEngine({
        cadence: { mode: 'every_tick' },
        evaluateScore: trendOnlyEvaluator,
        limits: { mutationCooldownTicks: 5 },
      });
      const scoreHistory = [1, 2, 3, 4];

      // Act
      const commitTelemetry = engine.adaptOnTick({
        tick: 0,
        network,
        scoreHistory,
      });
      const cooldownTelemetry = engine.adaptOnTick({
        tick: 1,
        network,
        scoreHistory,
      });

      // Assert
      expect(
        commitTelemetry.committed &&
          cooldownTelemetry.reason === 'mutation_cooldown_active',
      ).toBe(true);
    });
  });

  describe('capacity limits', () => {
    it('respects the configured maxNodes and maxConnections bounds across sustained growth', () => {
      // Arrange
      const network = new Network(4, 2, { seed: 42 });
      const maxNodes = 25;
      const maxConnections = 50;
      const engine = createRuntimeAdaptationEngine({
        evaluateScore: trendOnlyEvaluator,
        limits: { maxNodes, maxConnections },
      });
      const scoreHistory = [1, 2, 3, 4];

      // Act
      for (let tick = 0; tick < 50; tick++) {
        engine.adaptOnTick({ tick, network, scoreHistory });
      }

      // Assert
      const withinBounds =
        network.nodes.length <= maxNodes &&
        network.connections.length <= maxConnections;
      expect(withinBounds).toBe(true);
    });
  });
});
