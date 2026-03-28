/**
 * Tests covering telemetry selection filtering and structural entropy caching.
 * Single expectation per test; AAA pattern applied.
 */
import {
  applyTelemetrySelect,
  structuralEntropy as teleEntropy,
} from '../../src/neat/telemetry/recorder/telemetry.recorder';

type TelemetrySelectStub = Record<string, unknown> & {
  generation: number;
  _getRNG: () => () => number;
  options: Record<string, unknown>;
  _telemetrySelect?: Set<string>;
};

type GraphStub = {
  nodes: Array<{ geneId: number }>;
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>;
  _entropyGen?: number;
  _entropyVal?: number;
};

/** Minimal NeatLike stub implementing generation + RNG */
const makeStub = (): TelemetrySelectStub => ({
  generation: 0,
  _getRNG: () => () => Math.random(),
  options: {},
});

describe('Telemetry selection & entropy cache', () => {
  describe('applyTelemetrySelect filtering', () => {
    test('filters out non-whitelisted keys', () => {
      // Arrange
      const ctx = makeStub();
      /** keys to retain besides core */
      ctx._telemetrySelect = new Set(['keep']);
      /** raw telemetry entry containing extra keys */
      const entry: Record<string, unknown> = {
        gen: 1,
        best: 0.5,
        species: 3,
        keep: 1,
        drop: 2,
      };
      // Act
      applyTelemetrySelect.call(ctx, entry);
      // Assert
      expect('drop' in entry).toBe(false);
    });
    test('returns same object when no select set', () => {
      // Arrange
      const ctx = makeStub();
      /** entry with several keys */
      const entry: Record<string, unknown> = {
        gen: 2,
        best: 1,
        species: 1,
        a: 1,
      };
      // Act
      const out = applyTelemetrySelect.call(ctx, entry);
      // Assert
      expect(out).toBe(entry);
    });
  });
  describe('structuralEntropy cache behavior', () => {
    test('returns cached value for same generation', () => {
      // Arrange
      const ctx = makeStub();
      /** simple graph with two nodes and one enabled connection */
      const graph: GraphStub = {
        nodes: [{ geneId: 1 }, { geneId: 2 }],
        connections: [
          { from: { geneId: 1 }, to: { geneId: 2 }, enabled: true },
        ],
      };
      // Act
      const first = teleEntropy.call(ctx, graph);
      // mutate structure after caching (would change entropy if recomputed)
      graph.connections.push({
        from: { geneId: 1 },
        to: { geneId: 2 },
        enabled: true,
      });
      const second = teleEntropy.call(ctx, graph);
      // Assert
      expect(second).toBe(first);
    });
  });
});
