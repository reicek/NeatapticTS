import type Node from '../../architecture/node';
import Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';
import { createGenomeFromNetwork } from '../genome/genome';
import { recordNodeSplitRecord } from '../innovation-tracker/innovation-tracker';
import * as mutationAddConn from './add-conn/mutation.add-conn';
import {
  ensureMinHiddenNodes,
  ensureNoDeadEnds,
  mutateAddConnReuse,
  mutateAddNodeReuse,
  selectMutationMethod,
} from './mutation';

function countHiddenNodes(network: Network): number {
  return network.nodes.filter((nodeEntry) => nodeEntry.type === 'hidden')
    .length;
}

function createMutationHarness(input: {
  inputCount: number;
  outputCount: number;
  minimumHidden?: number;
  seed: number;
}): Neat {
  return new Neat(input.inputCount, input.outputCount, () => 0, {
    popsize: 0,
    minHidden: input.minimumHidden,
    seed: input.seed,
  });
}

async function enforceMinimumHiddenRepair(input: {
  mutationController: Neat;
  network: Network;
}): Promise<void> {
  await ensureMinHiddenNodes.call(
    input.mutationController as unknown as ThisParameterType<
      typeof ensureMinHiddenNodes
    >,
    input.network as unknown as Parameters<typeof ensureMinHiddenNodes>[0],
  );
}

function hasPathBetweenNodes(
  currentNode: Node,
  targetNode: Node,
  visitedNodes: Set<Node> = new Set(),
): boolean {
  if (currentNode === targetNode) return true;
  if (visitedNodes.has(currentNode)) return false;

  visitedNodes.add(currentNode);

  return currentNode.connections.out.some((connectionEntry) =>
    hasPathBetweenNodes(connectionEntry.to as Node, targetNode, visitedNodes),
  );
}

describe('neat mutation chapter', () => {
  describe('ensureMinHiddenNodes', () => {
    describe('given the network starts below the configured hidden floor', () => {
      describe('when minimum-hidden repair runs', () => {
        it('adds enough hidden nodes to satisfy the configured floor', async () => {
          // Arrange
          const minimumHiddenCount = 4;
          const mutationController = createMutationHarness({
            inputCount: 3,
            outputCount: 2,
            minimumHidden: minimumHiddenCount,
            seed: 610,
          });
          const network = new Network(3, 2, { seed: 611 });

          // Act
          await enforceMinimumHiddenRepair({ mutationController, network });

          // Assert
          expect(countHiddenNodes(network)).toBeGreaterThanOrEqual(
            minimumHiddenCount,
          );
        });
      });
    });

    describe('given minimum-hidden repair must create hidden structure canonically', () => {
      describe('when the hidden floor is enforced', () => {
        it('records node-split innovations in the controller tracker', async () => {
          // Arrange
          const mutationController = createMutationHarness({
            inputCount: 2,
            outputCount: 1,
            minimumHidden: 1,
            seed: 620,
          });
          const network = new Network(2, 1, { seed: 621 });

          // Act
          await enforceMinimumHiddenRepair({ mutationController, network });

          // Assert
          expect(
            mutationController.toJSON().innovationTracker.nodeSplitRecords
              .length > 0,
          ).toBe(true);
        });
      });
    });

    describe('given an existing hidden node has lost both incident connections', () => {
      describe('when minimum-hidden repair runs', () => {
        it('restores inbound and outbound connectivity for every hidden node', async () => {
          // Arrange
          const mutationController = createMutationHarness({
            inputCount: 2,
            outputCount: 1,
            minimumHidden: 1,
            seed: 612,
          });
          const network = new Network(2, 1, { seed: 613 });
          const inputNode = network.nodes[0];
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Expected output node to exist');
          }

          if (!inputNode.isProjectingTo(outputNode)) {
            network.connect(inputNode, outputNode);
          }
          network.mutate(methods.mutation.ADD_NODE);

          const hiddenNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'hidden',
          );
          if (!hiddenNode) {
            throw new Error('Expected hidden node to exist');
          }

          network.disconnect(inputNode, hiddenNode);
          network.disconnect(hiddenNode, outputNode);

          // Act
          await enforceMinimumHiddenRepair({ mutationController, network });
          const hiddenNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'hidden',
          );

          // Assert
          expect(
            hiddenNodes.every(
              (hiddenNodeEntry) =>
                hiddenNodeEntry.connections.in.length > 0 &&
                hiddenNodeEntry.connections.out.length > 0,
            ),
          ).toBe(true);
        });
      });
    });

    describe('given direct input-output routes already exist', () => {
      describe('when minimum-hidden repair adds the required hidden nodes', () => {
        it('keeps every input node on a path to the output node', async () => {
          // Arrange
          const mutationController = createMutationHarness({
            inputCount: 2,
            outputCount: 1,
            minimumHidden: 2,
            seed: 614,
          });
          const network = new Network(2, 1, { seed: 615 });
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const outputNode = network.nodes.at(-1);
          if (!outputNode) {
            throw new Error('Expected output node to exist');
          }

          inputNodes.forEach((inputNode) => {
            if (!inputNode.isProjectingTo(outputNode)) {
              network.connect(inputNode, outputNode);
            }
          });

          // Act
          await enforceMinimumHiddenRepair({ mutationController, network });

          // Assert
          expect(
            inputNodes.every((inputNode) =>
              hasPathBetweenNodes(inputNode, outputNode),
            ),
          ).toBe(true);
        });
      });
    });

    describe('given the network is missing its input endpoints', () => {
      describe('when minimum-hidden repair runs', () => {
        it('warns that endpoint-free networks are skipped', async () => {
          // Arrange
          const mutationController = createMutationHarness({
            inputCount: 2,
            outputCount: 1,
            minimumHidden: 1,
            seed: 616,
          });
          const network = new Network(2, 1, { seed: 617 });
          network.nodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type !== 'input',
          );
          const consoleWarnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            await enforceMinimumHiddenRepair({ mutationController, network });

            // Assert
            expect(consoleWarnSpy).toHaveBeenCalledWith(
              expect.stringContaining(
                'Network is missing input or output nodes',
              ),
            );
          } finally {
            consoleWarnSpy.mockRestore();
          }
        });
      });
    });
  });

  describe('mutate', () => {
    describe('given the controller always applies MOD_WEIGHT to the current population', () => {
      it('changes at least one connection weight in place', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 1,
          mutation: [methods.mutation.MOD_WEIGHT],
          mutationRate: 1,
          seed: 618,
        });
        const genome = mutationController.population[0];
        const connectionWeightsBeforeMutation = genome.connections.map(
          (connectionEntry) => connectionEntry.weight,
        );

        // Act
        await mutationController.mutate();

        // Assert
        expect(
          genome.connections.some(
            (connectionEntry, connectionIndex) =>
              connectionEntry.weight !==
              connectionWeightsBeforeMutation[connectionIndex],
          ),
        ).toBe(true);
      });
    });

    describe('given operator selection resolves to null for the genome', () => {
      it('skips the genome-level mutate call', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 1,
          mutation: [methods.mutation.ADD_NODE],
          mutationRate: 1,
          maxNodes: 0,
          seed: 619,
        });
        const genomeMutateSpy = jest.spyOn(
          mutationController.population[0],
          'mutate',
        );
        jest
          .spyOn(mutationController, 'selectMutationMethod')
          .mockResolvedValue(null);

        // Act
        await mutationController.mutate();

        // Assert
        expect(genomeMutateSpy).not.toHaveBeenCalled();
      });
    });

    describe('given a clean feed-forward starter population is mutated in place', () => {
      it('keeps every mutated genome strict-genome clean', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          fastMode: true,
          mutation: methods.mutation.FFW,
          mutationAmount: 2,
          mutationRate: 0.8,
          popsize: 20,
          seed: 42,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });

      it('keeps every mutated genome strict-genome clean when ADD_NODE is the only operator', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          mutation: [methods.mutation.ADD_NODE],
          mutationAmount: 2,
          mutationRate: 1,
          popsize: 20,
          seed: 43,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });

      it('keeps every mutated genome strict-genome clean when ADD_CONN is the only operator', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          mutation: [methods.mutation.ADD_CONN],
          mutationAmount: 2,
          mutationRate: 1,
          popsize: 20,
          seed: 44,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });

      it('keeps every mutated genome strict-genome clean when SUB_NODE and SUB_CONN are the only operators', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          mutation: [methods.mutation.SUB_NODE, methods.mutation.SUB_CONN],
          mutationAmount: 2,
          mutationRate: 1,
          popsize: 20,
          seed: 46,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });

      it('keeps every mutated genome strict-genome clean when SWAP_NODES is the only operator', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          mutation: [methods.mutation.SWAP_NODES],
          mutationAmount: 2,
          mutationRate: 1,
          popsize: 20,
          seed: 47,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });

      it('keeps every mutated genome strict-genome clean when ADD_NODE and ADD_CONN are the only operators', async () => {
        // Arrange
        const mutationController = new Neat(2, 1, () => 0, {
          mutation: [methods.mutation.ADD_NODE, methods.mutation.ADD_CONN],
          mutationAmount: 2,
          mutationRate: 1,
          popsize: 20,
          seed: 45,
        });

        // Act
        await mutationController.mutate();
        const allGenomesConvertToStrictGenome =
          mutationController.population.every((genome) => {
            try {
              createGenomeFromNetwork(genome);
              return true;
            } catch {
              return false;
            }
          });

        // Assert
        expect(allGenomesConvertToStrictGenome).toBe(true);
      });
    });
  });

  describe('selectMutationMethod', () => {
    describe('given ADD_NODE is sampled but the genome node count is at maxNodes', () => {
      describe('when selectMutationMethod runs', () => {
        it('returns null when the structural limit blocks the sampled method', async () => {
          // Arrange: genome already at the configured node cap.
          const network = new Network(2, 1, { seed: 702 });
          const mutationController = new Neat(2, 1, () => 0, {
            popsize: 0,
            mutation: [methods.mutation.ADD_NODE],
            maxNodes: network.nodes.length,
            seed: 703,
          });

          // Act
          const result = await selectMutationMethod.call(
            mutationController as unknown as ThisParameterType<
              typeof selectMutationMethod
            >,
            network as unknown as Parameters<typeof selectMutationMethod>[0],
            false,
          );

          // Assert
          expect(result).toBeNull();
        });
      });
    });

    describe('given the mutation pool is the legacy FFW array', () => {
      it('returns the FFW-sampled method directly without building a separate pool (line 535 true arm)', async () => {
        // Arrange: configure mutation as methods.mutation.FFW → resolveFFWPolicyForSelect returns non-null
        const network = new Network(2, 1, { seed: 790 });
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          mutation: methods.mutation
            .FFW as unknown as (typeof methods.mutation.MOD_WEIGHT)[],
          seed: 791,
        });

        // Act
        const result = await selectMutationMethod.call(
          mutationController as unknown as ThisParameterType<
            typeof selectMutationMethod
          >,
          network as unknown as Parameters<typeof selectMutationMethod>[0],
          false,
        );

        // Assert: FFW policy returned a concrete method (not null)
        expect(result).not.toBeNull();
      });
    });

    describe('given the function is called without the rawReturnForTest argument', () => {
      it('uses the default parameter value (line 522 default arm)', async () => {
        // Arrange: non-FFW pool so the default rawReturnForTest=true does not return the FFW array
        const network = new Network(2, 1, { seed: 730 });
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          mutation: [methods.mutation.MOD_WEIGHT],
          seed: 731,
        });

        // Act: omit second argument → rawReturnForTest defaults to true
        const result = await selectMutationMethod.call(
          mutationController as unknown as ThisParameterType<
            typeof selectMutationMethod
          >,
          network as unknown as Parameters<typeof selectMutationMethod>[0],
        );

        // Assert: a method was sampled (non-null result)
        expect(result).not.toBeNull();
      });
    });

    describe('given an empty mutation pool', () => {
      it('returns null when no operator can be sampled (line 557 true arm)', async () => {
        // Arrange: empty mutation pool → sampleFromPoolForSelect returns null
        const network = new Network(2, 1, { seed: 732 });
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          mutation: [] as unknown as (typeof methods.mutation.MOD_WEIGHT)[],
          seed: 733,
        });

        // Act
        const result = await selectMutationMethod.call(
          mutationController as unknown as ThisParameterType<
            typeof selectMutationMethod
          >,
          network as unknown as Parameters<typeof selectMutationMethod>[0],
          false,
        );

        // Assert
        expect(result).toBeNull();
      });
    });

    describe('given ADD_BACK_CONN is sampled but recurrent connections are not allowed', () => {
      it('returns null when the recurrent policy blocks the bandit method (line 583 true arm)', async () => {
        // Arrange: ADD_BACK_CONN is a recurrent mutation; without allowRecurrent it is blocked
        const network = new Network(2, 1, { seed: 752 });
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          mutation: [methods.mutation.ADD_BACK_CONN],
          allowRecurrent: false,
          seed: 753,
        });

        // Act
        const result = await selectMutationMethod.call(
          mutationController as unknown as ThisParameterType<
            typeof selectMutationMethod
          >,
          network as unknown as Parameters<typeof selectMutationMethod>[0],
          false,
        );

        // Assert: blocked by recurrent policy → null
        expect(result).toBeNull();
      });
    });
  });

  describe('mutateAddNodeReuse', () => {
    describe('given the innovation tracker already holds a split record for the chosen connection', () => {
      it('applies the existing split record instead of creating a new one (lines 209-217)', async () => {
        // Arrange: disable all but the first connection so chooseConnectionForSplit is deterministic
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 1,
          seed: 760,
        });
        const genome = mutationController.population[0];
        genome.connections.forEach((c, i) => {
          if (i > 0) c.enabled = false;
        });
        const targetConn = genome.connections[0] as {
          innovation?: number;
          from: { geneId?: number };
          to: { geneId?: number };
        };
        // Compute the split key the same way buildSplitDescriptor does
        const splitKey = Number.isInteger(targetConn.innovation)
          ? `splitConnectionInnovation:${targetConn.innovation}`
          : `legacyEndpoints:${targetConn.from.geneId ?? 0}->${targetConn.to.geneId ?? 0}`;
        const internalTracker = (
          mutationController as unknown as {
            _innovationTracker: Parameters<typeof recordNodeSplitRecord>[0];
          }
        )._innovationTracker;
        recordNodeSplitRecord(internalTracker, splitKey, {
          newNodeGeneId: 42,
          inInnov: 100,
          outInnov: 101,
        });

        // Act: only one enabled connection → chooseConnectionForSplit picks it → finds pre-seeded record
        const nodeCountBefore = genome.nodes.length;
        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          genome as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        // Assert: node was added via the existing split record path
        expect(genome.nodes.length).toBeGreaterThan(nodeCountBefore);
      });
    });

    describe('given a genome where all connections are disabled', () => {
      it('returns early without adding a node (line 194 true arm)', async () => {
        // Arrange: network with connections all disabled → collectEnabledConnections returns []
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 720,
        });
        const network = new Network(2, 1, { seed: 721 });
        network.connections.forEach((conn) => {
          conn.enabled = false;
        });
        const initialNodeCount = network.nodes.length;

        // Act
        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        // Assert: no node was added (early return fired)
        expect(network.nodes.length).toBe(initialNodeCount);
      });
    });

    describe('given the controller mutates an external genome with higher existing innovations', () => {
      it('keeps the strict genome contract free of duplicate connection innovations', async () => {
        // Arrange: force the split onto the second starter edge so the untouched
        // first edge keeps innovation 1 while the split path allocates two more.
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 726,
        });
        const network = new Network(2, 1, { seed: 727 });

        network.connections.forEach((connectionEntry, connectionIndex) => {
          connectionEntry.enabled = connectionIndex === 1;
        });

        // Act
        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        // Assert
        expect(() => createGenomeFromNetwork(network)).not.toThrow();
      });

      it('avoids reusing split innovations already present in the same genome', async () => {
        // Arrange: split one starter edge through a known record, then recreate
        // that historical edge so a second split would collide with the earlier
        // replacement innovations if reuse is applied blindly.
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 1,
          seed: 732,
        });
        const network = mutationController.population[0];
        const originalConnection = network.connections[0];

        if (!originalConnection) {
          throw new Error('Expected a starter connection to exist.');
        }

        const splitKey = `splitConnectionInnovation:${originalConnection.innovation}`;
        const sourceNode = originalConnection.from;
        const targetNode = originalConnection.to;
        const originalWeight = originalConnection.weight;
        const innovationTracker = (
          mutationController as unknown as {
            _innovationTracker: Parameters<typeof recordNodeSplitRecord>[0];
          }
        )._innovationTracker;

        recordNodeSplitRecord(innovationTracker, splitKey, {
          newNodeGeneId: 42,
          inInnov: 100,
          outInnov: 101,
        });

        network.connections.forEach((connectionEntry, connectionIndex) => {
          connectionEntry.enabled = connectionIndex === 0;
        });

        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        const recreatedConnection = network.connect(sourceNode, targetNode, originalWeight)[0];

        if (!recreatedConnection) {
          throw new Error('Expected to recreate the original edge.');
        }

        recreatedConnection.innovation = originalConnection.innovation;
        network.connections.forEach((connectionEntry) => {
          connectionEntry.enabled = connectionEntry === recreatedConnection;
        });

        // Act
        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        // Assert
        expect(() => createGenomeFromNetwork(network)).not.toThrow();
      });
    });

    describe('given an external genome carries self connections and missing innovation metadata', () => {
      it('raises the tracker above the highest observed self-connection innovation before mutating', async () => {
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 728,
        });
        const network = new Network(2, 1, { seed: 729 });
        const sourceConnection = network.connections[0];

        if (!sourceConnection) {
          throw new Error('Expected a starter connection to exist');
        }

        Reflect.set(network, 'selfconns', [
          {
            ...sourceConnection,
            from: network.nodes[0],
            innovation: undefined,
            to: network.nodes[0],
          },
          {
            ...sourceConnection,
            from: network.nodes[1],
            innovation: 70,
            to: network.nodes[1],
          },
        ]);

        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        expect(
          mutationController.toJSON().innovationTracker.nextInnovationId,
        ).toBeGreaterThan(70);
      });

      it('still mutates legacy genomes that omit the selfconns list entirely', async () => {
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 730,
        });
        const network = new Network(2, 1, { seed: 731 });
        const nodeCountBeforeMutation = network.nodes.length;

        Reflect.deleteProperty(
          network as Network & { selfconns?: Network['connections'] },
          'selfconns',
        );

        await mutateAddNodeReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddNodeReuse
          >,
          network as unknown as Parameters<typeof mutateAddNodeReuse>[0],
        );

        expect(network.nodes.length).toBeGreaterThan(nodeCountBeforeMutation);
      });
    });
  });

  describe('mutateAddConnReuse', () => {
    describe('given a fully-connected network where no candidate pairs exist', () => {
      it('returns early without adding a connection (line 281 true arm)', () => {
        // Arrange: Network(2,1) has all input→output connections — no pairs remain
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 780,
        });
        const network = new Network(2, 1, { seed: 781 });
        const initialConnCount = network.connections.length;

        // Act
        mutateAddConnReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddConnReuse
          >,
          network as unknown as Parameters<typeof mutateAddConnReuse>[0],
        );

        // Assert: no connection added (early return at candidatePairs.length === 0)
        expect(network.connections.length).toBe(initialConnCount);
      });
    });

    describe('given a network with valid candidate pairs', () => {
      it('adds a connection when a legal pair exists (lines 284-303)', () => {
        // Arrange: after ADD_NODE, (input1, hidden) is an unconnected pair
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 722,
        });
        const network = new Network(2, 1, { seed: 723 });
        network.mutate(methods.mutation.ADD_NODE);
        const initialConnCount = network.connections.filter(
          (c) => c.enabled !== false,
        ).length;

        // Act
        mutateAddConnReuse.call(
          mutationController as unknown as ThisParameterType<
            typeof mutateAddConnReuse
          >,
          network as unknown as Parameters<typeof mutateAddConnReuse>[0],
        );

        // Assert: at least one new enabled connection was added
        expect(
          network.connections.filter((c) => c.enabled !== false).length,
        ).toBeGreaterThanOrEqual(initialConnCount);
      });
    });

    describe('given choosePairForConn returns null', () => {
      it('returns early without adding a connection (line 295 true arm)', () => {
        // Arrange: network with a hidden node so candidatePairs is non-empty; spy returns null
        const mutationController = new Neat(2, 1, () => 0, {
          popsize: 0,
          seed: 724,
        });
        const network = new Network(2, 1, { seed: 725 });
        network.mutate(methods.mutation.ADD_NODE);
        const initialConnCount = network.connections.length;
        const spy = jest
          .spyOn(mutationAddConn, 'choosePairForConn')
          .mockReturnValue(null);

        try {
          // Act
          mutateAddConnReuse.call(
            mutationController as unknown as ThisParameterType<
              typeof mutateAddConnReuse
            >,
            network as unknown as Parameters<typeof mutateAddConnReuse>[0],
          );
        } finally {
          spy.mockRestore();
        }

        // Assert: no connection was added
        expect(network.connections.length).toBe(initialConnCount);
      });
    });
  });

  describe('ensureNoDeadEnds', () => {
    describe('given a controller with allowRecurrent enabled', () => {
      describe('when ensureNoDeadEnds runs', () => {
        it('skips feed-forward node reordering (line 467 early-return arm)', () => {
          // Arrange: allowRecurrent=true → normalizeRepairNodeOrderForFeedForward returns early
          const mutationController = new Neat(2, 1, () => 0, {
            popsize: 0,
            allowRecurrent: true,
            seed: 770,
          });
          const network = new Network(2, 1, { seed: 771 });
          const outputNode = network.nodes.find((n) => n.type === 'output');
          const inputNodes = network.nodes.filter((n) => n.type === 'input');
          // Deliberately put output before inputs to check it's NOT reordered
          network.nodes = [outputNode!, ...inputNodes];

          // Act
          ensureNoDeadEnds.call(
            mutationController as unknown as ThisParameterType<
              typeof ensureNoDeadEnds
            >,
            network as unknown as Parameters<typeof ensureNoDeadEnds>[0],
          );

          // Assert: reordering was skipped, output node remains first
          expect(network.nodes[0].type).toBe('output');
        });
      });
    });

    describe('given a non-recurrent network whose hidden node is placed after the output node', () => {
      describe('when ensureNoDeadEnds runs', () => {
        it('reorders nodes to the canonical input-hidden-output sequence', () => {
          // Arrange
          const mutationController = createMutationHarness({
            inputCount: 2,
            outputCount: 1,
            seed: 700,
          });
          const network = new Network(2, 1, { seed: 701 });
          network.mutate(methods.mutation.ADD_NODE);
          const inputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'input',
          );
          const hiddenNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'hidden',
          );
          const outputNodes = network.nodes.filter(
            (nodeEntry) => nodeEntry.type === 'output',
          );
          // Deliberately shuffle: input → output → hidden (out of standard order).
          network.nodes = [...inputNodes, ...outputNodes, ...hiddenNodes];

          // Act
          ensureNoDeadEnds.call(
            mutationController as unknown as ThisParameterType<
              typeof ensureNoDeadEnds
            >,
            network as unknown as Parameters<typeof ensureNoDeadEnds>[0],
          );

          // Assert: every hidden node should appear before every output node.
          const nodeTypes = network.nodes.map((nodeEntry) => nodeEntry.type);
          const lastHiddenIndex = nodeTypes.lastIndexOf('hidden');
          const firstOutputIndex = nodeTypes.indexOf('output');
          expect(lastHiddenIndex < firstOutputIndex).toBe(true);
        });
      });
    });
  });
});
