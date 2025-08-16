import { computeDiversityStats } from '../../src/neat/neat.diversity';

describe('computeDiversityStats', () => {
  function mockGenome(nodes: number, conns: number, depth?: number) {
    return {
      nodes: new Array(nodes)
        .fill(0)
        .map((_, i) => ({ id: i, connections: { out: [] as any[] } })),
      connections: new Array(conns).fill(0).map((_, i) => ({ id: i })),
      _depth: depth,
    } as any;
  }
  const compat = {
    _compatibilityDistance(a: any, b: any) {
      return (
        Math.abs(a.nodes.length - b.nodes.length) +
        Math.abs(a.connections.length - b.connections.length)
      );
    },
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
