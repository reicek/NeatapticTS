import type Node from '../../architecture/node';
import Network from '../../architecture/network';
import * as methods from '../../methods/methods';
import Neat from '../../neat';
import { ensureMinHiddenNodes, ensureNoDeadEnds, selectMutationMethod } from './mutation';

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
            mutationController.toJSON().innovationTracker.nodeSplitRecords.length >
              0,
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
  });

  describe('ensureNoDeadEnds', () => {
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
