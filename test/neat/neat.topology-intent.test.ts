import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';

type FastSlabPredicate = (training: boolean) => boolean;

const getFastSlabPredicate = (network: Network): FastSlabPredicate => {
  return Reflect.get(network, '_canUseFastSlab') as FastSlabPredicate;
};

describe('Neat topology intent', () => {
  it('promotes fresh FFW populations to feed-forward intent', () => {
    // Arrange
    const neat = new Neat(2, 1, () => 0, {
      popsize: 1,
      mutation: methods.mutation.FFW,
    });

    // Act
    const topologyIntent = neat.population[0].getTopologyIntent();

    // Assert
    expect(topologyIntent).toBe('feed-forward');
  });

  it('promotes nested FFW mutation wrappers to feed-forward intent', () => {
    // Arrange
    const neat = new Neat(2, 1, () => 0, {
      popsize: 1,
      mutation: [methods.mutation.FFW],
    });

    // Act
    const topologyIntent = neat.population[0].getTopologyIntent();

    // Assert
    expect(topologyIntent).toBe('feed-forward');
  });

  it('restores fast slab eligibility for fresh FFW populations after activation prepares topology', () => {
    // Arrange
    const neat = new Neat(2, 1, () => 0, {
      popsize: 1,
      mutation: methods.mutation.FFW,
    });
    const genome = neat.population[0];
    genome.activate([0, 0]);

    // Act
    const canUseFastSlab = getFastSlabPredicate(genome).call(genome, false);

    // Assert
    expect(canUseFastSlab).toBe(true);
  });

  it('promotes eligible seed networks to feed-forward intent when FFW is configured', () => {
    // Arrange
    const seedNetwork = new Network(2, 1);
    const neat = new Neat(2, 1, () => 0, {
      popsize: 1,
      network: seedNetwork,
      mutation: methods.mutation.FFW,
    });

    // Act
    const topologyIntent = neat.population[0].getTopologyIntent();

    // Assert
    expect(topologyIntent).toBe('feed-forward');
  });

  it('does not promote recurrent seed networks to feed-forward intent', () => {
    // Arrange
    const seedNetwork = new Network(2, 1);
    const outputNode = seedNetwork.nodes.find((node) => node.type === 'output');
    if (!outputNode) {
      throw new Error('Expected an output node for recurrent seed setup');
    }
    seedNetwork.connect(outputNode, outputNode, 1);
    const neat = new Neat(2, 1, () => 0, {
      popsize: 1,
      network: seedNetwork,
      mutation: methods.mutation.FFW,
    });

    // Act
    const topologyIntent = neat.population[0].getTopologyIntent();

    // Assert
    expect(topologyIntent).toBe('unconstrained');
  });

  it('promotes fresh provenance genomes to feed-forward intent when FFW is configured', async () => {
    // Arrange
    const neat = new Neat(2, 1, () => 0, {
      popsize: 2,
      provenance: 1,
      mutation: methods.mutation.FFW,
    });

    // Act
    await neat.evolve();
    const topologyIntent = neat.population[0].getTopologyIntent();

    // Assert
    expect(topologyIntent).toBe('feed-forward');
  });

  it('restores fast slab eligibility for fresh provenance genomes after activation prepares topology', async () => {
    // Arrange
    const neat = new Neat(2, 1, () => 0, {
      popsize: 2,
      provenance: 1,
      mutation: methods.mutation.FFW,
    });

    // Act
    await neat.evolve();
    const provenanceGenome = neat.population[0];
    provenanceGenome.activate([0, 0]);
    const canUseFastSlab = getFastSlabPredicate(provenanceGenome).call(
      provenanceGenome,
      false,
    );

    // Assert
    expect(canUseFastSlab).toBe(true);
  });
});
