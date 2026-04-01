import Network from '../../architecture/network';
import Neat from '../../neat';

type HelperMetadataNetwork = Network & {
  _id?: number;
  _parents?: number[];
  _depth?: number;
};

type NeatHelperHarness = Neat & {
  spawnFromParent: (
    parentGenome: HelperMetadataNetwork,
    mutateCount: number,
  ) => Promise<HelperMetadataNetwork>;
  addGenome: (genome: HelperMetadataNetwork, parents?: number[]) => void;
};

function cloneGenome(genome: HelperMetadataNetwork): HelperMetadataNetwork {
  if (genome.clone) {
    return genome.clone();
  }

  const serializedGenome = genome.toJSON?.();
  if (!serializedGenome) {
    throw new Error('Expected genome serialization support');
  }

  return Network.fromJSON(serializedGenome) as HelperMetadataNetwork;
}

function countHiddenNodes(genome: Network): number {
  return genome.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden').length;
}

describe('neat helpers chapter', () => {
  describe('spawnFromParent', () => {
    const fitness = (network: Network) => network.nodes.length;

    function createHelperHarness(): NeatHelperHarness {
      return new Neat(3, 2, fitness, {
        popsize: 6,
        seed: 444,
      }) as NeatHelperHarness;
    }

    describe('given a live parent genome from the active population', () => {
      it('assigns a fresh genome id to the provisional child', async () => {
        // Arrange
        const neat = createHelperHarness();
        const parentGenome = neat.population[0] as HelperMetadataNetwork;

        // Act
        const childGenome = await neat.spawnFromParent(parentGenome, 1);

        // Assert
        expect(childGenome._id).not.toBe(parentGenome._id);
      });

      it('records the single parent id and increments the child depth', async () => {
        // Arrange
        const neat = createHelperHarness();
        const parentGenome = neat.population[0] as HelperMetadataNetwork;

        // Act
        const childGenome = await neat.spawnFromParent(parentGenome, 1);

        // Assert
        expect({
          parents: childGenome._parents,
          depth: childGenome._depth,
        }).toEqual({
          parents: [parentGenome._id],
          depth: (parentGenome._depth ?? 0) + 1,
        });
      });

      it('keeps the provisional child structurally connected after normalization', async () => {
        // Arrange
        const neat = createHelperHarness();
        const parentGenome = neat.population[0] as HelperMetadataNetwork;

        // Act
        const childGenome = await neat.spawnFromParent(parentGenome, 1);

        // Assert
        expect(childGenome.connections.length).toBeGreaterThan(0);
      });
    });
  });

  describe('addGenome', () => {
    const fitness = (network: Network) => network.connections.length;

    function createHelperHarness(): NeatHelperHarness {
      return new Neat(3, 2, fitness, {
        popsize: 6,
        seed: 555,
      }) as NeatHelperHarness;
    }

    describe('given an externally sourced genome candidate', () => {
      it('appends the candidate into the live population', () => {
        // Arrange
        const neat = createHelperHarness();
        const parentGenome = neat.population[0] as HelperMetadataNetwork;
        const beforePopulationLength = neat.population.length;
        const externalGenome = cloneGenome(parentGenome);

        // Act
        neat.addGenome(externalGenome, [parentGenome._id ?? 0]);

        // Assert
        expect(neat.population).toHaveLength(beforePopulationLength + 1);
      });

      it('copies the provided parent ids onto the admitted genome', () => {
        // Arrange
        const neat = createHelperHarness();
        const parentGenome = neat.population[0] as HelperMetadataNetwork;
        const externalGenome = cloneGenome(parentGenome);

        // Act
        neat.addGenome(externalGenome, [parentGenome._id ?? 0]);
        const addedGenome = neat.population.at(-1) as
          | HelperMetadataNetwork
          | undefined;

        // Assert
        expect(addedGenome?._parents).toEqual([parentGenome._id ?? 0]);
      });

      it('derives admitted depth from the deepest known parent', async () => {
        // Arrange
        const neat = createHelperHarness();
        const firstParentGenome = neat.population[0] as HelperMetadataNetwork;
        const secondParentGenome = (await neat.spawnFromParent(
          firstParentGenome,
          1,
        )) as HelperMetadataNetwork;
        neat.addGenome(secondParentGenome, [firstParentGenome._id ?? 0]);
        const externalGenome = cloneGenome(firstParentGenome);
        const deepestParentDepth = Math.max(
          firstParentGenome._depth ?? 0,
          secondParentGenome._depth ?? 0,
        );

        // Act
        neat.addGenome(externalGenome, [
          firstParentGenome._id ?? 0,
          secondParentGenome._id ?? 0,
        ]);
        const addedGenome = neat.population.at(-1) as
          | HelperMetadataNetwork
          | undefined;

        // Assert
        expect(addedGenome?._depth).toBe(deepestParentDepth + 1);
      });
    });
  });

  describe('createPool', () => {
    const fitness = (network: Network) => network.connections.length;

    describe('given a seed network for generation-zero cloning', () => {
      it('keeps every cloned genome on the same input-output signature', () => {
        // Arrange
        const seedNetwork = new Network(2, 1);
        const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 333 });

        // Act
        neat.createPool(seedNetwork);

        // Assert
        expect(
          new Set(
            neat.population.map(
              (genome: Network) => `${genome.input}-${genome.output}`,
            ),
          ).size,
        ).toBe(1);
      });
    });

    describe('given a minimum hidden size is configured for fresh generation-zero genomes', () => {
      it('creates every fresh genome at or above that hidden floor', () => {
        // Arrange
        const minimumHiddenCount = 5;
        const neat = new Neat(2, 1, fitness, {
          popsize: 4,
          minHidden: minimumHiddenCount,
          seed: 334,
        });

        // Act
        neat.createPool(null);
        const hiddenNodeCounts = neat.population.map((genome: Network) =>
          countHiddenNodes(genome),
        );

        // Assert
        expect(
          hiddenNodeCounts.every(
            (hiddenNodeCount) => hiddenNodeCount >= minimumHiddenCount,
          ),
        ).toBe(true);
      });
    });
  });
});
