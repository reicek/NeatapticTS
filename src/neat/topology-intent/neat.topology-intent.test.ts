import Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';

type FastSlabPredicate = (training: boolean) => boolean;

function getFastSlabPredicate(network: Network): FastSlabPredicate {
  return Reflect.get(network, '_canUseFastSlab') as FastSlabPredicate;
}

describe('neat topology-intent chapter', () => {
  describe('fresh feed-forward populations', () => {
    describe('when the canonical FFW mutation pool is configured directly', () => {
      it('promotes the fresh population to feed-forward intent', () => {
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
    });

    describe('when the canonical FFW mutation pool is wrapped in a legacy single-item array', () => {
      it('still promotes the fresh population to feed-forward intent', () => {
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
    });

    describe('when activation prepares the runtime topology', () => {
      it('restores fast slab eligibility for the promoted genome', () => {
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
    });
  });

  describe('seed networks', () => {
    describe('when the seed already satisfies feed-forward eligibility', () => {
      it('promotes the seed genome to feed-forward intent', () => {
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
    });

    describe('when the seed contains a recurrent self-connection', () => {
      it('preserves unconstrained topology intent', () => {
        // Arrange
        const seedNetwork = new Network(2, 1);
        const outputNode = seedNetwork.nodes.find(
          (node) => node.type === 'output',
        );

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
    });
  });

  describe('provenance genomes', () => {
    describe('when evolve inserts fresh provenance genomes under FFW mutation', () => {
      it('promotes the inserted provenance genome to feed-forward intent', async () => {
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
    });

    describe('when activation prepares the promoted provenance genome', () => {
      it('restores fast slab eligibility for the provenance genome', async () => {
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
  });
});
