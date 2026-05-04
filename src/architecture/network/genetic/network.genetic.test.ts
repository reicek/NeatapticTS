import { Architect, Network, methods } from '../../../neataptic';
import { assertValidNativeGenome } from '../../../neat/validate/neat.validate';
import { NeatNativeGenomeValidationError } from '../../../neat/validate/neat.validate.errors';
import { materializeOffspringConnections } from './network.genetic.materialize.utils';
import {
  assignOffspringNodes,
  chooseOffspringConnectionGenes,
  createCrossoverContext,
  createNodeBuildContext,
} from './network.genetic.setup.utils';
import type { ConnectionGene, GeneticNetwork } from '../network.types';
import type { NetworkJSON } from '../network.types';
import Group from '../../group';
import Layer from '../../layer';
import Node from '../../node';

function createCrossOverCallback(
  firstParent: Network | null | undefined,
  secondParent: Network | null | undefined,
  equalFlag?: boolean,
): () => Network {
  if (typeof equalFlag === 'boolean') {
    return () =>
      Network.crossOver(
        firstParent as Network,
        secondParent as Network,
        equalFlag,
      );
  }

  return () =>
    Network.crossOver(firstParent as Network, secondParent as Network);
}

function suppressConsoleWarn(callback: () => void): void {
  const originalWarn = console.warn;
  console.warn = jest.fn();

  try {
    callback();
  } finally {
    console.warn = originalWarn;
  }
}

function buildParentNetworkWithAddedNodes(addedNodeCount: number): Network {
  const network = new Network(2, 1);

  for (
    let nodeMutationIndex = 0;
    nodeMutationIndex < addedNodeCount;
    nodeMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_NODE);
  }

  return network;
}

function buildAcyclicParentNetwork(
  seed: number,
  addedNodeCount: number,
  addedConnectionCount: number,
): Network {
  const network = new Network(2, 2, { seed, enforceAcyclic: true });

  for (
    let nodeMutationIndex = 0;
    nodeMutationIndex < addedNodeCount;
    nodeMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_NODE);
  }

  for (
    let connectionMutationIndex = 0;
    connectionMutationIndex < addedConnectionCount;
    connectionMutationIndex++
  ) {
    network.mutate(methods.mutation.ADD_CONN);
  }

  return network;
}

function setCrossoverRandomSequence(network: Network, samples: number[]): void {
  const randomizedNetwork = network as unknown as {
    _rand?: () => number;
  };
  let sampleIndex = 0;

  randomizedNetwork._rand = () => {
    const selectedSample = samples.at(sampleIndex) ?? samples.at(-1) ?? 0;
    sampleIndex += 1;
    return selectedSample;
  };
}

function cloneNodeForMaterializationTest(sourceNode: Node): Node {
  const clonedNode = new Node(sourceNode.type);
  clonedNode.geneId = sourceNode.geneId;
  clonedNode.bias = sourceNode.bias;
  clonedNode.squash = sourceNode.squash;
  return clonedNode;
}

function reorderParentNodesWithOutputDrift(network: Network): void {
  const inputNodes = network.nodes.filter((node) => node.type === 'input');
  const hiddenNodes = network.nodes.filter((node) => node.type === 'hidden');
  const outputNodes = network.nodes.filter((node) => node.type === 'output');
  const leadingHiddenNode = hiddenNodes.at(0);
  const trailingHiddenNodes = hiddenNodes.slice(1);
  const leadingOutputNode = outputNodes.at(0);
  const trailingOutputNodes = outputNodes.slice(1);

  if (!leadingHiddenNode || !leadingOutputNode) {
    throw new Error(
      'Expected one hidden node and one output node for output-drift coverage.',
    );
  }

  network.nodes = [
    ...inputNodes,
    leadingHiddenNode,
    leadingOutputNode,
    ...trailingHiddenNodes,
    ...trailingOutputNodes,
  ];
}

function createConnectionGeneFromRuntimeConnection(
  connection: Network['connections'][number],
): ConnectionGene {
  return {
    weight: connection.weight,
    innovation: connection.innovation,
    fromGeneId: connection.from.geneId,
    toGeneId: connection.to.geneId,
    gaterGeneId: connection.gater?.geneId ?? null,
    enabled: connection.enabled !== false,
  };
}

function areAllConnectionsFeedForward(network: Network): boolean {
  return network.connections.every((candidateConnection) => {
    const sourceNode = candidateConnection.from as Node | undefined;
    const targetNode = candidateConnection.to as Node | undefined;

    if (!sourceNode || !targetNode) {
      return false;
    }

    const sourceNodeIndex = network.nodes.indexOf(sourceNode);
    const targetNodeIndex = network.nodes.indexOf(targetNode);

    return (
      sourceNodeIndex !== -1 &&
      targetNodeIndex !== -1 &&
      sourceNodeIndex < targetNodeIndex
    );
  });
}

function summarizeTemporalExtensionBag(network: Network): {
  recurrentModuleCount: number;
  gatedBlockCount: number;
  recurrentKinds: string[];
} {
  const serializedJson = network.toJSON() as unknown as NetworkJSON;
  const extensionValues = serializedJson.extensions?.values as
    | {
        recurrentModules?: Array<{ kind?: string }>;
        gatedBlocks?: Array<unknown>;
      }
    | undefined;
  const recurrentModules = Array.isArray(extensionValues?.recurrentModules)
    ? extensionValues.recurrentModules
    : [];
  const gatedBlocks = Array.isArray(extensionValues?.gatedBlocks)
    ? extensionValues.gatedBlocks
    : [];

  return {
    recurrentModuleCount: recurrentModules.length,
    gatedBlockCount: gatedBlocks.length,
    recurrentKinds: recurrentModules
      .map((recurrentModule) => recurrentModule.kind)
      .filter((kind): kind is string => typeof kind === 'string')
      .toSorted(),
  };
}

function createConstructedFeedForwardParent(
  inputNodeLabels: readonly [string, string],
  outputNodeLabel: string,
): Network {
  const leftSensor = new Node('input');
  const rightSensor = new Node('input');
  const hiddenStage = new Group(2);
  const readoutLayer = Layer.dense(1, 'output');
  const readoutNode = readoutLayer.nodes[0];

  leftSensor.describe({ label: inputNodeLabels[0] });
  rightSensor.describe({ label: inputNodeLabels[1] });
  readoutNode.describe({ label: outputNodeLabel });

  leftSensor.connect(hiddenStage);
  rightSensor.connect(hiddenStage);
  hiddenStage.connect(readoutLayer);

  return Network.construct(
    [hiddenStage, rightSensor, readoutLayer, leftSensor],
    {
      inputNodes: [inputNodeLabels[1], inputNodeLabels[0]],
      outputNodes: [outputNodeLabel],
    },
  ).network;
}

describe('network genetic chapter', () => {
  describe('proper-NEAT validator guard', () => {
    describe('given a parent genome with duplicate connection innovations', () => {
      it('fails validation before crossover work begins', () => {
        // Arrange
        const malformedParent = new Network(2, 1, { seed: 412 });
        malformedParent.connections[1].innovation =
          malformedParent.connections[0].innovation;
        const validateParent = () => assertValidNativeGenome(malformedParent);

        // Assert
        expect(validateParent).toThrow(NeatNativeGenomeValidationError);
      });
    });
  });

  describe('Network.crossOver()', () => {
    describe('given both parent networks enforce acyclic topology', () => {
      describe('when crossover materializes the offspring graph', () => {
        it('keeps every offspring connection feed-forward', () => {
          // Arrange
          const parentNetwork1 = buildAcyclicParentNetwork(410, 20, 60);
          const parentNetwork2 = buildAcyclicParentNetwork(411, 40, 20);

          // Act
          const offspringNetwork = Network.crossOver(
            parentNetwork1,
            parentNetwork2,
          );
          const offspringIsFeedForward =
            areAllConnectionsFeedForward(offspringNetwork);

          // Assert
          expect(offspringIsFeedForward).toBe(true);
        });

        it('keeps the offspring topology contract feed-forward', () => {
          // Arrange
          const parentNetwork1 = buildAcyclicParentNetwork(510, 10, 20);
          const parentNetwork2 = buildAcyclicParentNetwork(511, 12, 18);

          // Act
          const offspringNetwork = Network.crossOver(
            parentNetwork1,
            parentNetwork2,
          );

          // Assert
          expect(offspringNetwork.getTopologyIntent()).toBe('feed-forward');
        });
      });
    });

    describe('given both parent networks were built through Network.construct()', () => {
      describe('when crossover materializes the offspring graph', () => {
        it('keeps the explicit input ordering from the first construct parent and stays feed-forward', () => {
          // Arrange
          const firstParent = createConstructedFeedForwardParent(
            ['firstLeftSensor', 'firstRightSensor'],
            'firstReadout',
          );
          const secondParent = createConstructedFeedForwardParent(
            ['secondLeftSensor', 'secondRightSensor'],
            'secondReadout',
          );
          const expectedInputNodeIds = firstParent.inputNodeIds;

          // Act
          const offspringNetwork = Network.crossOver(firstParent, secondParent);
          const actualConstructCrossoverSummary = {
            inputNodeIds: offspringNetwork.inputNodeIds,
            outputNodeCount: offspringNetwork.outputNodeIds.length,
            topologyIntent: offspringNetwork.getTopologyIntent(),
            feedForward: areAllConnectionsFeedForward(offspringNetwork),
          };

          // Assert
          expect(actualConstructCrossoverSummary).toEqual({
            inputNodeIds: expectedInputNodeIds,
            outputNodeCount: 1,
            topologyIntent: 'feed-forward',
            feedForward: true,
          });
        });
      });
    });

    describe('given equal-fitness parents with different hidden-node counts', () => {
      let firstParent: Network;
      let secondParent: Network;
      let offspringNetwork: Network;

      beforeEach(() => {
        // Arrange
        firstParent = buildParentNetworkWithAddedNodes(1);
        secondParent = buildParentNetworkWithAddedNodes(2);
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        offspringNetwork = Network.crossOver(firstParent, secondParent, true);
      });

      describe('when the offspring size is chosen symmetrically', () => {
        it('keeps the node count at or above the smaller parent', () => {
          // Assert
          expect(offspringNetwork.nodes.length).toBeGreaterThanOrEqual(
            Math.min(firstParent.nodes.length, secondParent.nodes.length),
          );
        });

        it('keeps the shared input-output contract intact', () => {
          // Assert
          expect({
            input: offspringNetwork.input,
            output: offspringNetwork.output,
          }).toEqual({
            input: firstParent.input,
            output: firstParent.output,
          });
        });
      });
    });

    describe('given the fitter parent has more hidden nodes', () => {
      let secondParent: Network;
      let offspringNetwork: Network;

      beforeEach(() => {
        // Arrange
        const firstParent = buildParentNetworkWithAddedNodes(1);
        secondParent = buildParentNetworkWithAddedNodes(2);
        firstParent.score = 1;
        secondParent.score = 2;
        setCrossoverRandomSequence(firstParent, [0.99]);

        // Act
        offspringNetwork = Network.crossOver(firstParent, secondParent, false);
      });

      describe('when fitter-parent inheritance decides offspring size', () => {
        it('matches the fitter parent node count', () => {
          // Assert
          expect(offspringNetwork.nodes.length).toBe(secondParent.nodes.length);
        });
      });
    });

    describe('given the parent networks expose incompatible interfaces', () => {
      describe('when the input width differs', () => {
        it('throws', () => {
          // Arrange
          const firstParent = new Network(2, 1);
          const secondParent = new Network(3, 1);
          const crossOverCallback = () =>
            Network.crossOver(firstParent, secondParent);

          // Assert
          expect(crossOverCallback).toThrow();
        });
      });

      describe('when the output width differs', () => {
        it('throws', () => {
          // Arrange
          const firstParent = new Network(2, 1);
          const secondParent = new Network(2, 2);
          const crossOverCallback = () =>
            Network.crossOver(firstParent, secondParent);

          // Assert
          expect(crossOverCallback).toThrow();
        });
      });
    });

    describe('given parent references are missing', () => {
      describe('when both parents are undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            undefined,
            undefined,
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when both parents are null', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(null, null);

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when the first parent is undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            undefined,
            new Network(2, 1),
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });

      describe('when the second parent is undefined', () => {
        it('throws', () => {
          // Arrange
          const crossOverCallback = createCrossOverCallback(
            new Network(2, 1),
            undefined,
          );

          // Act / Assert
          suppressConsoleWarn(() => {
            expect(crossOverCallback).toThrow();
          });
        });
      });
    });

    describe('given the parent graphs are structurally simple', () => {
      describe('when both parents expose no registered connections', () => {
        let firstParent: Network;
        let offspringNetwork: Network;

        beforeEach(() => {
          // Arrange
          firstParent = new Network(2, 1);
          firstParent.connections = [];
          const secondParent = new Network(2, 1);
          secondParent.connections = [];

          // Act
          offspringNetwork = Network.crossOver(firstParent, secondParent, true);
        });

        describe('when the offspring is materialized', () => {
          it('preserves the parent node count', () => {
            // Assert
            expect(offspringNetwork.nodes.length).toBe(
              firstParent.nodes.length,
            );
          });
        });
      });

      describe('when both parents have the same score and size', () => {
        let firstParent: Network;
        let offspringNetwork: Network;

        beforeEach(() => {
          // Arrange
          firstParent = new Network(2, 1);
          const secondParent = new Network(2, 1);
          firstParent.score = 1;
          secondParent.score = 1;

          // Act
          offspringNetwork = Network.crossOver(firstParent, secondParent, true);
        });

        describe('when crossover chooses from equivalent parents', () => {
          it('keeps the shared node count unchanged', () => {
            // Assert
            expect(offspringNetwork.nodes.length).toBe(
              firstParent.nodes.length,
            );
          });
        });
      });
    });

    describe('given both parents are homologous clones', () => {
      it('preserves inherited node gene ids and connection innovations', () => {
        // Arrange
        const firstParent = new Network(2, 1, { seed: 413 });
        firstParent.mutate(methods.mutation.ADD_NODE);
        const secondParent = firstParent.clone();
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        const offspringNetwork = Network.crossOver(
          firstParent,
          secondParent,
          true,
        );

        // Assert
        expect({
          nodeGeneIds: offspringNetwork.nodes.map((node) => node.geneId),
          connectionInnovations: offspringNetwork.connections
            .map((connection) => connection.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        }).toEqual({
          nodeGeneIds: firstParent.nodes.map((node) => node.geneId),
          connectionInnovations: firstParent.connections
            .map((connection) => connection.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
        });
      });

      it('preserves surviving temporal module descriptors on the offspring payload', () => {
        // Arrange
        const firstParent = Architect.lstm(1, 2, 1);
        const secondParent = Network.fromJSON(
          firstParent.toJSON() as unknown as Record<string, unknown>,
        );
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        const offspringNetwork = Network.crossOver(
          firstParent,
          secondParent,
          true,
        );
        const temporalSummary = summarizeTemporalExtensionBag(offspringNetwork);

        // Assert
        expect(temporalSummary).toEqual({
          recurrentModuleCount: 1,
          gatedBlockCount: 1,
          recurrentKinds: ['lstm'],
        });
      });

      it('preserves GRU temporal module descriptors on the offspring payload', () => {
        // Arrange
        const firstParent = Architect.gru(1, 2, 1, { inputToOutput: true });
        const secondParent = Network.fromJSON(
          firstParent.toJSON() as unknown as Record<string, unknown>,
        );
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        const offspringNetwork = Network.crossOver(
          firstParent,
          secondParent,
          true,
        );
        const temporalSummary = summarizeTemporalExtensionBag(offspringNetwork);

        // Assert
        expect(temporalSummary).toEqual({
          recurrentModuleCount: 1,
          gatedBlockCount: 1,
          recurrentKinds: ['gru'],
        });
      });

      it('keeps same-endpoint genes with different innovations in the fitter parent lane', () => {
        // Arrange
        const firstParent = new Network(2, 1, { seed: 414 });
        const secondParent = firstParent.clone();
        firstParent.connections[0].innovation = 101;
        firstParent.connections[0].weight = 0.25;
        secondParent.connections[0].innovation = 202;
        secondParent.connections[0].weight = 0.75;
        firstParent.score = 2;
        secondParent.score = 1;
        setCrossoverRandomSequence(firstParent, [0]);

        // Act
        const offspringNetwork = Network.crossOver(
          firstParent,
          secondParent,
          false,
        );

        // Assert
        expect({
          innovations: offspringNetwork.connections
            .map((connection) => connection.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          hasLowerFitnessInnovation: offspringNetwork.connections.some(
            (connection) => connection.innovation === 202,
          ),
          inheritedWeight:
            offspringNetwork.connections.find(
              (connection) => connection.innovation === 101,
            )?.weight ?? null,
        }).toEqual({
          innovations: firstParent.connections
            .map((connection) => connection.innovation)
            .toSorted(
              (leftInnovation, rightInnovation) =>
                leftInnovation - rightInnovation,
            ),
          hasLowerFitnessInnovation: false,
          inheritedWeight: 0.25,
        });
      });

      it('uses the crossover rng for disabled-gene re-enable decisions', () => {
        // Arrange
        const firstParent = new Network(2, 1, { seed: 415 });
        const secondParent = firstParent.clone();
        firstParent.connections[0].enabled = false;
        secondParent.connections[0].enabled = false;
        firstParent._reenableProb = 0.75;
        secondParent._reenableProb = 0.75;
        setCrossoverRandomSequence(firstParent, [0.75, 0.5]);
        const mathRandomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.99);

        try {
          // Act
          const offspringNetwork = Network.crossOver(
            firstParent,
            secondParent,
            false,
          );

          // Assert
          expect({
            enabled: offspringNetwork.connections[0]?.enabled ?? null,
          }).toEqual({
            enabled: true,
          });
        } finally {
          mathRandomSpy.mockRestore();
        }
      });

      it('keeps output genes in the offspring interface region when one parent runtime order drifts', () => {
        // Arrange
        const firstParent = new Network(3, 2, { seed: 417 });
        firstParent.mutate(methods.mutation.ADD_NODE);
        firstParent.mutate(methods.mutation.ADD_NODE);
        const secondParent = firstParent.clone();
        reorderParentNodesWithOutputDrift(firstParent);
        firstParent.score = 1;
        secondParent.score = 1;

        // Act
        const offspringNetwork = Network.crossOver(
          firstParent,
          secondParent,
          true,
        );

        // Assert
        expect({
          outputNodeCount: offspringNetwork.nodes.filter(
            (node) => node.type === 'output',
          ).length,
          tailNodeTypes: offspringNetwork.nodes
            .slice(-offspringNetwork.output)
            .map((node) => node.type),
          genomeIsValid: (() => {
            try {
              assertValidNativeGenome(offspringNetwork);
              return true;
            } catch {
              return false;
            }
          })(),
        }).toEqual({
          outputNodeCount: 2,
          tailNodeTypes: ['output', 'output'],
          genomeIsValid: true,
        });
      });
    });

    describe('given a chosen gene references historical ids instead of current slot order', () => {
      it('strips runtime node-index hints from the adapter output', () => {
        // Arrange
        const firstParent = new Network(2, 1, { seed: 416 });
        const secondParent = firstParent.clone();
        const crossoverContext = createCrossoverContext(
          firstParent,
          secondParent,
          false,
          () => 0,
        );

        // Act
        const chosenGeneRecord = chooseOffspringConnectionGenes(
          crossoverContext,
        )[0] as unknown as Record<string, unknown>;

        // Assert
        expect({
          hasInnovation: typeof chosenGeneRecord.innovation === 'number',
          hasFromGeneId: typeof chosenGeneRecord.fromGeneId === 'number',
          hasToGeneId: typeof chosenGeneRecord.toGeneId === 'number',
          hasEnabled: typeof chosenGeneRecord.enabled === 'boolean',
          hasFromIndexHint: Reflect.has(chosenGeneRecord, 'from'),
          hasToIndexHint: Reflect.has(chosenGeneRecord, 'to'),
          hasGaterIndexHint: Reflect.has(chosenGeneRecord, 'gater'),
        }).toEqual({
          hasInnovation: true,
          hasFromGeneId: true,
          hasToGeneId: true,
          hasEnabled: true,
          hasFromIndexHint: false,
          hasToIndexHint: false,
          hasGaterIndexHint: false,
        });
      });

      it('materializes endpoints and gaters through gene-id lookup', () => {
        // Arrange
        const offspring = new Network(1, 1) as unknown as GeneticNetwork;
        const gaterNode = new Node('hidden');
        const sourceNode = new Node('input');
        const targetNode = new Node('output');
        gaterNode.geneId = 200;
        sourceNode.geneId = 100;
        targetNode.geneId = 300;
        offspring.nodes = [gaterNode, sourceNode, targetNode];
        offspring.connections = [];
        offspring.selfconns = [];
        offspring.gates = [];
        offspring.nodes.forEach((node, nodeIndex) => {
          node.index = nodeIndex;
        });
        const chosenGene: ConnectionGene = {
          weight: 0.75,
          innovation: 777,
          fromGeneId: 100,
          toGeneId: 300,
          gaterGeneId: 200,
          enabled: true,
        };

        // Act
        materializeOffspringConnections(offspring, [chosenGene]);

        // Assert
        expect({
          fromGeneId: offspring.connections[0].from.geneId,
          toGeneId: offspring.connections[0].to.geneId,
          gaterGeneId: offspring.connections[0].gater?.geneId ?? null,
          innovation: offspring.connections[0].innovation,
        }).toEqual({
          fromGeneId: 100,
          toGeneId: 300,
          gaterGeneId: 200,
          innovation: 777,
        });
      });

      it('rebuilds required hidden nodes from inherited gene identity before materialization', () => {
        // Arrange
        const parentNetwork = new Network(2, 1, { seed: 416 });
        parentNetwork.mutate(methods.mutation.ADD_NODE);
        const hiddenNode = parentNetwork.nodes.find(
          (node) => node.type === 'hidden',
        );
        const inheritedConnection = parentNetwork.connections.find(
          (connection) =>
            connection.from.geneId === hiddenNode?.geneId ||
            connection.to.geneId === hiddenNode?.geneId,
        );

        if (!hiddenNode || !inheritedConnection) {
          throw new Error(
            'Expected a hidden-node split connection for materialization coverage.',
          );
        }

        const offspring = new Network(2, 1) as unknown as GeneticNetwork;
        offspring.nodes = parentNetwork.nodes
          .filter((node) => node.type !== 'hidden')
          .map(cloneNodeForMaterializationTest);
        offspring.connections = [];
        offspring.selfconns = [];
        offspring.gates = [];
        offspring.nodes.forEach((node, nodeIndex) => {
          node.index = nodeIndex;
        });

        const chosenGene =
          createConnectionGeneFromRuntimeConnection(inheritedConnection);

        // Act
        materializeOffspringConnections(
          offspring,
          [chosenGene],
          [parentNetwork as unknown as GeneticNetwork],
        );

        // Assert
        expect({
          hiddenNodeGeneIds: offspring.nodes
            .filter((node) => node.type === 'hidden')
            .map((node) => node.geneId),
          materializedConnectionGeneIds: offspring.connections.map(
            (connection) => [connection.from.geneId, connection.to.geneId],
          ),
        }).toEqual({
          hiddenNodeGeneIds: [hiddenNode.geneId],
          materializedConnectionGeneIds: [
            [chosenGene.fromGeneId, chosenGene.toGeneId],
          ],
        });
      });

      it('keeps inherited backward genes when the offspring topology is unconstrained', () => {
        // Arrange
        const outputNode = new Node('output');
        const inputNode = new Node('input');
        outputNode.geneId = 710;
        inputNode.geneId = 720;
        const offspring = new Network(1, 1, {
          topologyIntent: 'unconstrained',
        }) as unknown as GeneticNetwork;
        offspring.nodes = [inputNode, outputNode];
        offspring.connections = [];
        offspring.selfconns = [];
        offspring.gates = [];
        offspring.nodes.forEach((node, nodeIndex) => {
          node.index = nodeIndex;
        });
        const chosenGene: ConnectionGene = {
          weight: 0.6,
          innovation: 888,
          fromGeneId: outputNode.geneId,
          toGeneId: inputNode.geneId,
          gaterGeneId: null,
          enabled: true,
        };

        // Act
        materializeOffspringConnections(offspring, [chosenGene]);

        // Assert
        expect({
          topologyIntent: offspring.getTopologyIntent(),
          connectionCount: offspring.connections.length,
          connectionGeneIds: offspring.connections.map((connection) => ({
            fromGeneId: connection.from.geneId,
            toGeneId: connection.to.geneId,
            innovation: connection.innovation,
          })),
        }).toEqual({
          topologyIntent: 'unconstrained',
          connectionCount: 1,
          connectionGeneIds: [
            {
              fromGeneId: outputNode.geneId,
              toGeneId: inputNode.geneId,
              innovation: 888,
            },
          ],
        });
      });

      it('prunes inherited backward genes explicitly when the offspring topology is feed-forward', () => {
        // Arrange
        const outputNode = new Node('output');
        const inputNode = new Node('input');
        outputNode.geneId = 810;
        inputNode.geneId = 820;
        const offspring = new Network(1, 1, {
          topologyIntent: 'feed-forward',
        }) as unknown as GeneticNetwork;
        offspring.nodes = [inputNode, outputNode];
        offspring.connections = [];
        offspring.selfconns = [];
        offspring.gates = [];
        offspring.nodes.forEach((node, nodeIndex) => {
          node.index = nodeIndex;
        });
        const chosenGene: ConnectionGene = {
          weight: 0.6,
          innovation: 889,
          fromGeneId: outputNode.geneId,
          toGeneId: inputNode.geneId,
          gaterGeneId: null,
          enabled: true,
        };

        // Act
        materializeOffspringConnections(offspring, [chosenGene]);

        // Assert
        expect({
          topologyIntent: offspring.getTopologyIntent(),
          connectionCount: offspring.connections.length,
          selfConnectionCount: offspring.selfconns.length,
        }).toEqual({
          topologyIntent: 'feed-forward',
          connectionCount: 0,
          selfConnectionCount: 0,
        });
      });

      it('falls back to the available parent output gene when the first parent output partition is short', () => {
        // Arrange
        const firstParent = new Network(2, 2, { seed: 418 });
        const secondParent = firstParent.clone();
        const firstParentOutputNodes = firstParent.nodes.filter(
          (node) => node.type === 'output',
        );
        const secondParentOutputNodes = secondParent.nodes.filter(
          (node) => node.type === 'output',
        );
        const firstParentSecondOutputNode = firstParentOutputNodes.at(1);
        const secondParentSecondOutputNode = secondParentOutputNodes.at(1);

        if (!firstParentSecondOutputNode || !secondParentSecondOutputNode) {
          throw new Error('Expected both parents to expose two output nodes.');
        }

        firstParentSecondOutputNode.type = 'hidden';
        firstParent.score = 1;
        secondParent.score = 1;
        const crossoverContext = createCrossoverContext(
          firstParent,
          secondParent,
          true,
          () => 0,
        );
        const nodeBuildContext = createNodeBuildContext(crossoverContext);

        // Act
        assignOffspringNodes(nodeBuildContext);
        const offspringSecondOutputGeneId = crossoverContext.offspring.nodes
          .filter((node) => node.type === 'output')
          .at(1)?.geneId;

        // Assert
        expect(offspringSecondOutputGeneId).toBe(
          secondParentSecondOutputNode.geneId,
        );
      });

      it('skips unresolved hidden slots when parent hidden partitions do not expose every hidden ordinal', () => {
        // Arrange
        const firstParent = new Network(2, 1, { seed: 419 });
        firstParent.mutate(methods.mutation.ADD_NODE);
        firstParent.mutate(methods.mutation.ADD_NODE);
        const secondParent = new Network(2, 1, { seed: 420 });
        const firstParentHiddenNodes = firstParent.nodes.filter(
          (node) => node.type === 'hidden',
        );
        const firstParentExtraHiddenNode = firstParentHiddenNodes.at(1);

        if (!firstParentExtraHiddenNode) {
          throw new Error(
            'Expected the first parent to expose at least two hidden nodes.',
          );
        }

        firstParentExtraHiddenNode.type = 'output';
        firstParent.score = 2;
        secondParent.score = 1;
        const crossoverContext = createCrossoverContext(
          firstParent,
          secondParent,
          false,
          () => 0,
        );
        const nodeBuildContext = createNodeBuildContext(crossoverContext);

        // Act
        assignOffspringNodes(nodeBuildContext);
        const skippedHiddenSlotCount =
          nodeBuildContext.offspringNodeCount -
          crossoverContext.offspring.nodes.length;

        // Assert
        expect(skippedHiddenSlotCount).toBe(1);
      });
    });

    describe('given parent1 has hidden nodes but parent2 has none and parent1 is the weaker parent', () => {
      describe('when crossover resolves hidden slots in equal mode', () => {
        it('selects each hidden gene from parent1 via the equal-mode fallback even when score1 < score2', () => {
          // Arrange – parent1 weaker (score=1), parent2 stronger (score=2), equal=true
          // → at ordinals where parent2 has no hidden node:
          //   score1 >= score2 = FALSE → || equal = TRUE covers the ||'s second branch (line 486)
          const parent1 = Network.createMLP(2, [3], 1);
          const parent2 = new Network(2, 1);
          (parent1 as unknown as { score: number }).score = 1;
          (parent2 as unknown as { score: number }).score = 2;
          // injected rng=0.9 → offspringNodeCount = max(6) so hidden ordinals 0,1,2 are visited
          const crossoverContext = createCrossoverContext(
            parent1,
            parent2,
            true,
            () => 0.9,
          );
          const nodeBuildContext = createNodeBuildContext(crossoverContext);

          // Act
          assignOffspringNodes(nodeBuildContext);

          // Assert – offspring has all node roles filled (no missing hidden slots skipped)
          expect(
            crossoverContext.offspring.nodes.filter(
              (offspringNode) => offspringNode.type === 'hidden',
            ).length,
          ).toBeGreaterThan(0);
        });
      });
    });

    describe('given parent2 has hidden nodes but parent1 has none and parent2 is the weaker parent', () => {
      describe('when crossover resolves hidden slots in equal mode', () => {
        it('selects each hidden gene from parent2 via the equal-mode fallback even when score2 < score1', () => {
          // Arrange – parent1 stronger (score=2), parent2 weaker (score=1), equal=true
          // → at ordinals where parent1 has no hidden node:
          //   score2 >= score1 = FALSE → || equal = TRUE covers the ||'s second branch (line 490)
          const parent1 = new Network(2, 1);
          const parent2 = Network.createMLP(2, [3], 1);
          (parent1 as unknown as { score: number }).score = 2;
          (parent2 as unknown as { score: number }).score = 1;
          const crossoverContext = createCrossoverContext(
            parent1,
            parent2,
            true,
            () => 0.9,
          );
          const nodeBuildContext = createNodeBuildContext(crossoverContext);

          // Act
          assignOffspringNodes(nodeBuildContext);

          // Assert
          expect(
            crossoverContext.offspring.nodes.filter(
              (offspringNode) => offspringNode.type === 'hidden',
            ).length,
          ).toBeGreaterThan(0);
        });
      });
    });
  });
});
