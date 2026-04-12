import Network from '../../architecture/network';
import Neat from '../../neat';
import type { SpeciesLike } from '../shared/neat.shared.types';

type ConnectionWithOptionalInnovation = {
  from: { index: number };
  to: { index: number };
  weight: number;
  innovation?: number;
};

type NetworkWithMutableConnections = Network & {
  connections: ConnectionWithOptionalInnovation[];
  _compatInnovationMode?: 'allow-fallback';
};

function readSpeciesRegistry(neat: Neat): SpeciesLike[] {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Owner-local root speciation coverage needs the internal registry.
  return (neat as any)._species as SpeciesLike[];
}

function rebuildSpeciesRegistry(neat: Neat): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Owner-local root speciation coverage needs the internal controller hook.
  (neat as any)._speciate();
}

describe('neat speciation root chapter', () => {
  describe('given a native population member loses its explicit innovation id', () => {
    it('throws during species rebuild instead of silently using fallback ids', () => {
      // Arrange
      const scoreByConnectionCount = (network: Network) =>
        network.connections.length;
      const neat = new Neat(2, 1, scoreByConnectionCount, {
        popsize: 4,
        seed: 41,
        speciation: true,
      });
      const malformedGenome = neat.population[0] as NetworkWithMutableConnections;
      Reflect.deleteProperty(malformedGenome.connections[0], 'innovation');

      // Assert
      expect(() => rebuildSpeciesRegistry(neat)).toThrow(
        /Compatibility distance requires explicit connection innovations/,
      );
    });
  });

  describe('given a population that already accumulated some structural diversity', () => {
    const scoreByConnectionCount = (network: Network) =>
      network.connections.length;
    const neat = new Neat(2, 1, scoreByConnectionCount, {
      popsize: 20,
      seed: 42,
      speciation: true,
      mutationRate: 0.3,
      mutationAmount: 1,
    });

    beforeAll(async () => {
      for (let generationIndex = 0; generationIndex < 3; generationIndex++) {
        await neat.evolve();
      }

      rebuildSpeciesRegistry(neat);
    });

    describe('when the root lifecycle rebuilds the live species registry', () => {
      it('keeps at least one live species entry', () => {
        // Arrange
        const liveSpeciesRegistry = readSpeciesRegistry(neat);

        // Assert
        expect(liveSpeciesRegistry.length).toBeGreaterThan(0);
      });

      it('assigns every population member into the rebuilt registry', () => {
        // Arrange
        const assignedMemberCount = readSpeciesRegistry(neat).reduce(
          (memberCount: number, species: SpeciesLike) =>
            memberCount + species.members.length,
          0,
        );

        // Assert
        expect(assignedMemberCount).toBe(neat.population.length);
      });
    });
  });
});
