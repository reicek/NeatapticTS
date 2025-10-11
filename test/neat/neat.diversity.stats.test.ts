import { computeDiversityStats } from '../../src/neat/neat.diversity';

describe('computeDiversityStats', () => {
  const mockGenome = (nodes: number, conns: number, depth?: number) => ({
    nodes: Array.from({ length: nodes }, (_, i) => ({
      id: i,
      connections: { out: [] as unknown[] },
    })),
    connections: Array.from({ length: conns }, (_, i) => ({ id: i })),
    _depth: depth,
  });
  const compat = {
    _compatibilityDistance: (
      a: { nodes: unknown[]; connections: unknown[] },
      b: { nodes: unknown[]; connections: unknown[] }
    ) =>
      Math.abs(a.nodes.length - b.nodes.length) +
      Math.abs(a.connections.length - b.connections.length),
  };

  it('returns undefined for empty population', () => {
    expect(computeDiversityStats([], compat)).toBeUndefined();
  });

  it('computes expected keys', () => {
    const pop = [mockGenome(3, 2, 1), mockGenome(5, 4, 2), mockGenome(4, 3, 4)];
    const stats = computeDiversityStats(pop, compat)!;
    // Keys required by telemetry/tests
    const keys = [
      'lineageMeanDepth',
      'lineageMeanPairDist',
      'meanNodes',
      'meanConns',
      'nodeVar',
      'connVar',
      'meanCompat',
      'graphletEntropy',
      'population',
    ];
    for (const k of keys) expect(stats).toHaveProperty(k);
    expect(stats.population).toBe(pop.length);
  });
});
