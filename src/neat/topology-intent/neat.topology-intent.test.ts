import Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';
import {
  allowsRecurrentConnectionMutation,
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from './neat.topology-intent';

type FastSlabPredicate = (training: boolean) => boolean;

function getFastSlabPredicate(network: Network): FastSlabPredicate {
  return Reflect.get(network, '_canUseFastSlab') as FastSlabPredicate;
}

describe('neat topology-intent chapter', () => {
  describe('feed-forward policy detection', () => {
    describe('when the configured mutation option is one custom object instead of a pool', () => {
      it('does not report feed-forward mutation intent', () => {
        // Arrange
        const mutationConfig = { name: 'ADD_NODE' };

        // Act
        const usesFeedForwardPolicy =
          usesFeedForwardMutationPolicy(mutationConfig);

        // Assert
        expect(usesFeedForwardPolicy).toBe(false);
      });
    });

    describe('when the configured mutation pool is a flattened canonical FFW copy', () => {
      it('recognizes the canonical feed-forward operator order', () => {
        // Arrange
        const mutationConfig = (
          methods.mutation.FFW as Array<{ name?: string }>
        ).map((mutationMethod) => ({ name: mutationMethod.name }));

        // Act
        const usesFeedForwardPolicy =
          usesFeedForwardMutationPolicy(mutationConfig);

        // Assert
        expect(usesFeedForwardPolicy).toBe(true);
      });
    });

    describe('when the configured mutation pool omits one canonical FFW operator', () => {
      it('rejects the pool as a feed-forward policy signal', () => {
        // Arrange
        const mutationConfig = (
          methods.mutation.FFW as Array<{ name?: string }>
        )
          .slice(0, -1)
          .map((mutationMethod) => ({ name: mutationMethod.name }));

        // Act
        const usesFeedForwardPolicy =
          usesFeedForwardMutationPolicy(mutationConfig);

        // Assert
        expect(usesFeedForwardPolicy).toBe(false);
      });
    });
  });

  describe('direct promotion guardrails', () => {
    describe('when the active mutation policy does not request promotion', () => {
      it('returns without applying feed-forward topology intent', () => {
        // Arrange
        const setTopologyIntent = jest.fn();
        const inputNode = { id: 'input' };
        const outputNode = { id: 'output' };
        const genome = {
          nodes: [inputNode, outputNode],
          connections: [{ from: inputNode, to: outputNode }],
          gates: [],
          selfconns: [],
          setTopologyIntent,
        };

        // Act
        promoteGenomeToFeedForwardIntentWhenEligible(genome, false);

        // Assert
        expect(setTopologyIntent).not.toHaveBeenCalled();
      });
    });

    describe('when the genome is missing one topology collection', () => {
      it('does not apply feed-forward topology intent', () => {
        // Arrange
        const setTopologyIntent = jest.fn();
        const genome = {
          setTopologyIntent,
        };

        // Act
        promoteGenomeToFeedForwardIntentWhenEligible(genome, true);

        // Assert
        expect(setTopologyIntent).not.toHaveBeenCalled();
      });
    });
  });

  describe('recurrent mutation policy seam', () => {
    describe('when recurrent growth is enabled for one unconstrained genome', () => {
      it('allows recurrent connection mutation', () => {
        // Arrange
        const genome = {
          getTopologyIntent: () => 'unconstrained' as const,
        };

        // Act
        const canAddRecurrentConnection = allowsRecurrentConnectionMutation(
          genome,
          true,
        );

        // Assert
        expect(canAddRecurrentConnection).toBe(true);
      });
    });
  });

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
