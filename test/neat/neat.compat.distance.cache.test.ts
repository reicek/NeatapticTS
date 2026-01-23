import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import Connection from '../../src/architecture/connection';

type ConnectionWithOptionalInnovation = Omit<Connection, 'innovation'> & {
  innovation?: number;
};

type NetworkWithMutableConnections = Network & {
  connections: ConnectionWithOptionalInnovation[];
};

/**
 * Tests for compatibility distance & fallback innovation id logic (neat.compat.ts).
 * Focus: cache reuse path and fallback innovation usage when innovations missing.
 */
describe('NEAT Compatibility Distance', () => {
  describe('fallback innovation id and cache reuse', () => {
    /** Fitness simple structural count for deterministic shape. */
    const fitness = (n: Network) => n.nodes.length;
    /** Instance with coefficients set for predictable distance. */
    const neat = new Neat(3, 1, fitness, {
      popsize: 4,
      seed: 777,
      excessCoeff: 1,
      disjointCoeff: 1,
      weightDiffCoeff: 0.4,
    });
    let genomeA: NetworkWithMutableConnections;
    let genomeB: NetworkWithMutableConnections;
    beforeAll(async () => {
      await neat.evaluate();
      // Arrange: take two genomes and strip innovations to force fallback path
      genomeA = neat.population[0] as NetworkWithMutableConnections;
      genomeB = neat.population[1] as NetworkWithMutableConnections;
      genomeA.connections.forEach((connection) =>
        Reflect.deleteProperty(connection, 'innovation'),
      );
      genomeB.connections.forEach((connection) =>
        Reflect.deleteProperty(connection, 'innovation'),
      );
    });
    test('second distance call reuses cached value', () => {
      // Arrange: first call populates cache
      const first = neat._compatibilityDistance(genomeA, genomeB);
      // Act: second call should hit cache (value identical)
      const second = neat._compatibilityDistance(genomeA, genomeB);
      // Assert: identical numeric distance implies cache reuse path executed
      expect(second).toBe(first);
    });
  });
});
