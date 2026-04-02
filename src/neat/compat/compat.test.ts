import Network from '../../architecture/network';
import Neat from '../../neat';

type ConnectionWithOptionalInnovation = {
  from: { index: number };
  to: { index: number };
  weight: number;
  innovation?: number;
};

type NetworkWithMutableConnections = Network & {
  connections: ConnectionWithOptionalInnovation[];
  _compatCache?: Array<[number, number]>;
};

describe('neat compat chapter', () => {
  describe('_compatibilityDistance', () => {
    describe('given two evaluated genomes missing explicit innovation ids', () => {
      const fitness = (network: Network) => network.nodes.length;
      const neat = new Neat(3, 1, fitness, {
        popsize: 4,
        seed: 777,
        excessCoeff: 1,
        disjointCoeff: 1,
        weightDiffCoeff: 0.4,
      });

      let genomeA: NetworkWithMutableConnections;
      let genomeB: NetworkWithMutableConnections;
      let expectedFallbackInnovation: number;
      let firstDistance: number;
      let secondDistance: number;

      beforeAll(async () => {
        // Arrange
        await neat.evaluate();
        genomeA = neat.population[0] as NetworkWithMutableConnections;
        genomeB = neat.population[1] as NetworkWithMutableConnections;

        genomeA.connections.forEach((connection) => {
          Reflect.deleteProperty(connection, 'innovation');
        });
        genomeB.connections.forEach((connection) => {
          Reflect.deleteProperty(connection, 'innovation');
        });
        expectedFallbackInnovation =
          genomeA.connections[0].from.index * 100_000 +
          genomeA.connections[0].to.index;

        // Act
        firstDistance = neat._compatibilityDistance(genomeA, genomeB);
        secondDistance = neat._compatibilityDistance(genomeA, genomeB);
      });

      describe('when fallback innovations are used for the first comparison', () => {
        it('still produces a finite compatibility distance', () => {
          // Assert
          expect(Number.isFinite(firstDistance)).toBe(true);
        });
      });

      describe('when the same pair is compared twice in one generation', () => {
        it('reuses the cached distance value', () => {
          // Assert
          expect(secondDistance).toBe(firstDistance);
        });
      });

      describe('when missing innovations are normalized into the cache', () => {
        it('stores the fallback-derived innovation ids on the genome cache', () => {
          // Assert
          expect(
            genomeA._compatCache?.some(
              ([innovation]) => innovation === expectedFallbackInnovation,
            ),
          ).toBe(true);
        });
      });

      describe('when the first comparison normalizes sorted innovations', () => {
        it('stores a reusable compatibility cache on the genome', () => {
          // Assert
          expect(Array.isArray(genomeA._compatCache)).toBe(true);
        });
      });
    });
  });
});
