import Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';
import { createGenomeFromNetwork } from '../genome/genome';
import { validateNativeGenome } from '../validate/neat.validate';

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

function collectGenerationZeroSignature(genome: Network): string {
  return JSON.stringify({
    nodeGeneIds: genome.nodes.map((nodeEntry) => nodeEntry.geneId),
    connectionInnovations: genome.connections.map(
      (connection) => connection.innovation,
    ),
    topologyIntent: genome.getTopologyIntent(),
    rngState: genome.getRNGState(),
  });
}

function resolveMaxPopulationInnovation(population: Network[]): number {
  return population
    .flatMap((genome) =>
      genome.connections.map((connection) => connection.innovation),
    )
    .reduce(
      (currentMaxInnovation, innovation) =>
        Math.max(currentMaxInnovation, innovation),
      -1,
    );
}

function createSeedNetworkWithDisconnectedInput(): Network {
  const seedNetwork = new Network(2, 1, { seed: 1_204 });
  const disconnectedConnection = seedNetwork.connections[0];

  if (!disconnectedConnection) {
    throw new Error('Expected seed network to expose one connection');
  }

  seedNetwork.disconnect(
    disconnectedConnection.from,
    disconnectedConnection.to,
  );

  return seedNetwork;
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
          HelperMetadataNetwork | undefined;

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
          HelperMetadataNetwork | undefined;

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

      it('keeps every seeded clone aligned on history, topology intent, and rng state', () => {
        // Arrange
        const seedNetwork = new Network(2, 1, { seed: 1_203 });
        seedNetwork.setTopologyIntent('unconstrained');
        const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 336 });

        // Act
        neat.createPool(seedNetwork);

        // Assert
        expect(
          new Set(neat.population.map(collectGenerationZeroSignature)).size,
        ).toBe(1);
        expect(neat.population[0].getRNGState()).toBe(
          seedNetwork.getRNGState(),
        );
        expect(neat.population[0].getTopologyIntent()).toBe(
          seedNetwork.getTopologyIntent(),
        );
      });

      it('keeps every seeded generation-zero genome validator-clean', () => {
        // Arrange
        const seedNetwork = new Network(2, 1);
        const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 335 });

        // Act
        neat.createPool(seedNetwork);
        const allGenomesValidate = neat.population.every(
          (genome: Network) => validateNativeGenome(genome).isValid,
        );

        // Assert
        expect(allGenomesValidate).toBe(true);
      });

      it('keeps repaired seeded generation-zero genomes validator-clean when the seed needs dead-end repair', () => {
        // Arrange
        const seedNetwork = createSeedNetworkWithDisconnectedInput();
        const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 340 });

        // Act
        neat.createPool(seedNetwork);
        const allGenomesValidate = neat.population.every(
          (genome: Network) => validateNativeGenome(genome).isValid,
        );

        // Assert
        expect(allGenomesValidate).toBe(true);
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

      it('builds one homologous fresh template for the entire starting population', () => {
        // Arrange
        const neat = new Neat(2, 1, fitness, {
          popsize: 4,
          minHidden: 5,
          seed: 337,
        });

        // Act
        neat.createPool(null);

        // Assert
        expect(
          new Set(neat.population.map(collectGenerationZeroSignature)).size,
        ).toBe(1);
      });

      it('seeds the innovation tracker above the starting population history', () => {
        // Arrange
        const neat = new Neat(2, 1, fitness, {
          popsize: 4,
          minHidden: 3,
          seed: 338,
        });
        const innovationTrackerHost = neat as unknown as {
          _innovationTracker: {
            nextInnovationId: number;
          };
        };

        // Act
        neat.createPool(null);
        const maxPopulationInnovation = resolveMaxPopulationInnovation(
          neat.population,
        );

        // Assert
        expect(innovationTrackerHost._innovationTracker.nextInnovationId).toBe(
          maxPopulationInnovation + 1,
        );
      });

      it('keeps every fresh generation-zero genome validator-clean', () => {
        // Arrange
        const neat = new Neat(2, 1, fitness, {
          popsize: 4,
          minHidden: 3,
          seed: 339,
        });

        // Act
        neat.createPool(null);
        const allGenomesValidate = neat.population.every(
          (genome: Network) => validateNativeGenome(genome).isValid,
        );

        // Assert
        expect(allGenomesValidate).toBe(true);
      });

      it('keeps every fresh feed-forward starter genome strict-genome clean before evolve runs', () => {
        // Arrange
        const neat = new Neat(2, 1, fitness, {
          fastMode: true,
          mutation: methods.mutation.FFW,
          popsize: 4,
          seed: 341,
        });

        // Act
        const allGenomesConvertToStrictGenome = neat.population.every(
          (genome: Network) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          },
        );

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });
    });
  });
});
