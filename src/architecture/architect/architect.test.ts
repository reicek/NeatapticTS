import Group from '../group';
import Layer from '../layer';
import Node from '../node';
import Network from '../network';
import Architect from './architect';
import {
  ArchitectInputOutputTypeResolutionError,
  ArchitectInvalidGruConfigurationError,
  ArchitectInvalidGruLayerArgumentsError,
  ArchitectInvalidLstmConfigurationError,
  ArchitectInvalidLstmLayerArgumentsError,
  ArchitectInvalidPerceptronConfigurationError,
  ArchitectInvalidRandomSparseConfigurationError,
  ArchitectZeroInputOutputNodesError,
} from './architect.errors';

type ArchitectLayeredNetwork = Network & {
  layers?: Layer[];
};

function summarizeSparseNetwork(network: Network): {
  biasSignature: string;
  connectionSignature: string;
  gateSignature: string;
  hiddenNodeCount: number;
  inputRoleCount: number;
  outputRoleCount: number;
  selfConnectionSignature: string;
} {
  function resolveConnectionSignature(
    sourceNode: Network['nodes'][number],
    targetNode: Network['nodes'][number],
  ): string {
    return `${network.nodes.indexOf(sourceNode)}->${network.nodes.indexOf(targetNode)}`;
  }

  return {
    biasSignature: network.nodes
      .map((candidateNode) => candidateNode.bias.toFixed(6))
      .join(','),
    connectionSignature: network.connections
      .map(
        (candidateConnection) =>
          `${resolveConnectionSignature(candidateConnection.from, candidateConnection.to)}:${candidateConnection.weight.toFixed(6)}`,
      )
      .join(','),
    gateSignature: network.gates
      .map((candidateConnection) =>
        resolveConnectionSignature(
          candidateConnection.from,
          candidateConnection.to,
        ),
      )
      .join(','),
    hiddenNodeCount: network.nodes.filter(
      (candidateNode) => candidateNode.type === 'hidden',
    ).length,
    inputRoleCount: network.inputNodeIds.length,
    outputRoleCount: network.outputNodeIds.length,
    selfConnectionSignature: network.selfconns
      .map(
        (candidateConnection) =>
          `${resolveConnectionSignature(candidateConnection.from, candidateConnection.to)}:${candidateConnection.weight.toFixed(6)}`,
      )
      .join(','),
  };
}

function createIdentityActivation(): (
  value: number,
  derivative?: boolean,
) => number {
  return (value, derivative = false) => (derivative ? 1 : value);
}

function roundNumericSignature(candidateValue: number): number {
  return Number(candidateValue.toFixed(6));
}

function collectUniqueRoundedValues(candidateValues: number[]): number[] {
  return [...new Set(candidateValues.map(roundNumericSignature))].toSorted(
    (leftValue, rightValue) => leftValue - rightValue,
  );
}

function findNetworkNodeByGeneId(network: Network, geneId: number): Node {
  const resolvedNode = network.nodes.find(
    (candidateNode) => candidateNode.geneId === geneId,
  );

  if (!resolvedNode) {
    throw new Error(`Expected node with gene id ${geneId} to exist.`);
  }

  return resolvedNode;
}

function summarizeNarxDelayLines(network: Network): {
  clearSuggestion: string | null;
  recurrentKinds: string[];
  recurrentModuleCount: number;
  roleSizesPerModule: string[];
  schedulingStateSemantics: string | null;
  uniqueActivationSamples: number[];
  uniqueBiases: number[];
  uniqueCarryWeights: number[];
  uniqueDerivativeSamples: number[];
} {
  const temporalStructure = network.describeTemporalStructure();
  const narxModules = temporalStructure.recurrentModules.filter(
    (candidateModule) => candidateModule.kind === 'narx-memory',
  );
  const moduleNodeIds = narxModules.flatMap((candidateModule) =>
    Object.values(candidateModule.nodeGeneIdsByRole).flatMap(
      (candidateRoleNodeIds) => candidateRoleNodeIds,
    ),
  );
  const moduleNodeIdSet = new Set(moduleNodeIds);
  const moduleNodes = [...moduleNodeIdSet].map((geneId) =>
    findNetworkNodeByGeneId(network, geneId),
  );
  const carryWeights = network.connections
    .filter(
      (candidateConnection) =>
        moduleNodeIdSet.has(candidateConnection.from.geneId) &&
        moduleNodeIdSet.has(candidateConnection.to.geneId),
    )
    .map((candidateConnection) => candidateConnection.weight);
  const schedulingDiagnostics = network.getActivationSchedulingDiagnostics();

  return {
    clearSuggestion: schedulingDiagnostics.suggestions.at(0) ?? null,
    recurrentKinds: narxModules
      .map((candidateModule) => candidateModule.kind)
      .toSorted(),
    recurrentModuleCount: narxModules.length,
    roleSizesPerModule: narxModules
      .map((candidateModule) =>
        Object.values(candidateModule.nodeGeneIdsByRole)
          .map((candidateRoleNodeIds) => candidateRoleNodeIds.length)
          .join(','),
      )
      .toSorted(),
    schedulingStateSemantics: schedulingDiagnostics.stateSemantics,
    uniqueActivationSamples: collectUniqueRoundedValues(
      moduleNodes.map((candidateNode) => candidateNode.squash(0.25)),
    ),
    uniqueBiases: collectUniqueRoundedValues(
      moduleNodes.map((candidateNode) => candidateNode.bias),
    ),
    uniqueCarryWeights: collectUniqueRoundedValues(carryWeights),
    uniqueDerivativeSamples: collectUniqueRoundedValues(
      moduleNodes.map((candidateNode) => candidateNode.squash(0.25, true)),
    ),
  };
}

function configureNarxClearStateFixture(network: Network): void {
  network.nodes.forEach((candidateNode) => {
    if (candidateNode.type === 'input') {
      return;
    }

    candidateNode.bias = 0;
    candidateNode.squash = createIdentityActivation();
  });

  network.connections.forEach((candidateConnection) => {
    candidateConnection.weight = 1;
  });
}

function resolveNarxBoundaryNodes(network: Network): {
  delayedInputMemoryNode?: Node;
  inputMemoryNode: Node;
  inputNode: Node;
  outputMemoryNode: Node;
  outputNode: Node;
} {
  const inputNode = network.nodes.find(
    (candidateNode) => candidateNode.type === 'input',
  );
  const outputNode = network.nodes.find(
    (candidateNode) => candidateNode.type === 'output',
  );

  if (!inputNode || !outputNode) {
    throw new Error(
      'Expected NARX fixture to expose one input and one output node.',
    );
  }

  const inputMemoryNode = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === inputNode &&
      candidateConnection.to.type === 'variant',
  )?.to;
  const outputMemoryNode = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === outputNode &&
      candidateConnection.to.type === 'variant',
  )?.to;

  if (!inputMemoryNode || !outputMemoryNode) {
    throw new Error(
      'Expected NARX fixture to expose both input and output delay lines.',
    );
  }

  const delayedInputMemoryNode = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === inputMemoryNode &&
      candidateConnection.to.type === 'variant',
  )?.to;

  return {
    delayedInputMemoryNode,
    inputMemoryNode,
    inputNode,
    outputMemoryNode,
    outputNode,
  };
}

function configureNarxRunningTotalPredictor(network: Network): void {
  const { inputNode, outputMemoryNode, outputNode } =
    resolveNarxBoundaryNodes(network);

  network.nodes.forEach((candidateNode) => {
    if (candidateNode.type === 'input') {
      return;
    }

    candidateNode.bias = 0;
    candidateNode.squash = createIdentityActivation();
  });

  network.connections.forEach((candidateConnection) => {
    candidateConnection.weight = 0;
  });

  const inputToMemoryConnection = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === inputNode &&
      candidateConnection.to === outputNode,
  );
  const outputToMemoryConnection = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === outputNode &&
      candidateConnection.to === outputMemoryNode,
  );
  const outputMemoryToOutputConnection = network.connections.find(
    (candidateConnection) =>
      candidateConnection.from === outputMemoryNode &&
      candidateConnection.to === outputNode,
  );

  if (
    !inputToMemoryConnection ||
    !outputToMemoryConnection ||
    !outputMemoryToOutputConnection
  ) {
    throw new Error(
      'Expected NARX fixture connections needed for the sequence predictor.',
    );
  }

  inputToMemoryConnection.weight = 1;
  outputToMemoryConnection.weight = 1;
  outputMemoryToOutputConnection.weight = 1;
}

function captureSequenceOutputs(
  network: Network,
  inputSequence: number[],
): number[] {
  return inputSequence.map((inputValue) => network.activate([inputValue])[0]);
}

function roundSequenceOutputs(outputSequence: number[]): number[] {
  return outputSequence.map(roundNumericSignature);
}

function countDirectInputToOutputConnections(network: Network): number {
  return network.connections.filter(
    (candidateConnection) =>
      candidateConnection.from.type === 'input' &&
      candidateConnection.to.type === 'output',
  ).length;
}

function summarizeHydratedRecurrentExtensions(network: Network): {
  gatedBlockCount: number;
  recurrentKinds: string[];
  recurrentModuleCount: number;
} {
  const temporalStructure = network.describeTemporalStructure();

  return {
    gatedBlockCount: temporalStructure.gatedBlocks.length,
    recurrentKinds: temporalStructure.recurrentModules
      .map((candidateModule) => candidateModule.kind)
      .toSorted(),
    recurrentModuleCount: temporalStructure.recurrentModules.length,
  };
}

function summarizeRecurrentArchitectureBoundary(network: Network): {
  clearSuggestion: string | null;
  directInputToOutputConnections: number;
  gatedBlockCount: number;
  hasCycles: boolean;
  recurrentKinds: string[];
  recurrentModuleCount: number;
  stateSemantics: string | null;
  topologyIntent: string;
} {
  network.activate(new Array(network.input).fill(0));
  const schedulingDiagnostics = network.getActivationSchedulingDiagnostics();
  const hydratedExtensions = summarizeHydratedRecurrentExtensions(network);

  return {
    clearSuggestion: schedulingDiagnostics.suggestions.at(0) ?? null,
    directInputToOutputConnections:
      countDirectInputToOutputConnections(network),
    gatedBlockCount: hydratedExtensions.gatedBlockCount,
    hasCycles: network.describeArchitecture().hasCycles,
    recurrentKinds: hydratedExtensions.recurrentKinds,
    recurrentModuleCount: hydratedExtensions.recurrentModuleCount,
    stateSemantics: schedulingDiagnostics.stateSemantics,
    topologyIntent: network.getTopologyIntent(),
  };
}

function createConstructedPerceptronEquivalent(): Network {
  const leftSensor = new Node('input');
  const rightSensor = new Node('input');
  const hiddenStage = new Group(3);
  const readoutLayer = Layer.dense(1, 'output');
  const readoutNode = readoutLayer.nodes[0];

  leftSensor.describe({ label: 'leftSensor' });
  rightSensor.describe({ label: 'rightSensor' });
  readoutNode.describe({ label: 'readout' });

  leftSensor.connect(hiddenStage);
  rightSensor.connect(hiddenStage);
  hiddenStage.connect(readoutLayer);

  return Network.construct(
    [hiddenStage, rightSensor, readoutLayer, leftSensor],
    {
      inputNodes: ['leftSensor', 'rightSensor'],
      outputNodes: ['readout'],
    },
  ).network;
}

describe('Architect', () => {
  describe('construct()', () => {
    describe('given one mixed primitive list includes grouped nodes, direct nodes, gates, and a self connection', () => {
      describe('when constructing the network directly from those primitives', () => {
        it('collects the unique nodes plus the forward, gated, and self connections', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenGroup = new Group(1);
          const hiddenNode = hiddenGroup.nodes[0];
          const outputNode = new Node('output');

          inputNode.connect(hiddenGroup);
          const hiddenToOutputConnection = hiddenNode.connect(outputNode)[0];
          hiddenNode.connections.gated.push(hiddenToOutputConnection);
          hiddenToOutputConnection.gater = hiddenNode;
          hiddenNode.connect(hiddenNode)[0].weight = 0.5;

          // Act
          const network = Architect.construct([
            hiddenGroup,
            inputNode,
            outputNode,
          ]);

          // Assert
          expect({
            connections: network.connections.length,
            gates: network.gates.length,
            input: network.input,
            nodes: network.nodes.length,
            output: network.output,
            selfconns: network.selfconns.length,
          }).toStrictEqual({
            connections: 2,
            gates: 1,
            input: 1,
            nodes: 3,
            output: 1,
            selfconns: 1,
          });
        });
      });
    });

    describe('given one construct request contains malformed connection buckets', () => {
      describe('when collecting connections from those nodes', () => {
        it('ignores undefined, non-array, and non-Connection entries', () => {
          // Arrange
          const inputNode = new Node('input');
          const hiddenNode = new Node('hidden');
          const outputNode = new Node('output');

          (
            inputNode as unknown as {
              connections?: unknown;
            }
          ).connections = undefined;
          (
            hiddenNode as unknown as {
              connections: unknown;
            }
          ).connections = {
            gated: [{}],
            in: [],
            out: [{}],
            self: [],
          };
          (
            outputNode as unknown as {
              connections: unknown;
            }
          ).connections = {
            gated: undefined,
            in: [],
            out: undefined,
            self: [],
          };

          // Act
          const network = Architect.construct([
            inputNode,
            hiddenNode,
            outputNode,
          ]);

          // Assert
          expect({
            connections: network.connections.length,
            gates: network.gates.length,
            input: network.input,
            output: network.output,
            selfconns: network.selfconns.length,
          }).toStrictEqual({
            connections: 0,
            gates: 0,
            input: 1,
            output: 1,
            selfconns: 0,
          });
        });
      });
    });

    describe('given one layer contains an entry that is neither a node nor a group', () => {
      describe('when building the network from that layer plus direct input and output nodes', () => {
        it('ignores the malformed layer entry', () => {
          // Arrange
          const malformedLayer = new Layer();
          const inputNode = new Node('input');
          const outputNode = new Node('output');

          (malformedLayer as unknown as { nodes: unknown[] }).nodes = [{}];

          // Act
          const network = Architect.construct([
            malformedLayer,
            inputNode,
            outputNode,
          ]);

          // Assert
          expect({
            input: network.input,
            nodes: network.nodes.length,
            output: network.output,
          }).toStrictEqual({
            input: 1,
            nodes: 2,
            output: 1,
          });
        });
      });
    });

    describe('given one construct list item is not a group, layer, or node', () => {
      describe('when building the network from that list plus direct input and output nodes', () => {
        it('ignores the unsupported list item', () => {
          // Arrange
          const inputNode = new Node('input');
          const outputNode = new Node('output');

          // Act
          const network = Architect.construct([
            {} as never,
            inputNode,
            outputNode,
          ]);

          // Assert
          expect({
            input: network.input,
            nodes: network.nodes.length,
            output: network.output,
          }).toStrictEqual({
            input: 1,
            nodes: 2,
            output: 1,
          });
        });
      });
    });

    describe('given one direct node appears twice in the construct list', () => {
      describe('when building the network from those primitives', () => {
        it('deduplicates the repeated node', () => {
          // Arrange
          const inputNode = new Node('input');
          const outputNode = new Node('output');

          // Act
          const network = Architect.construct([
            inputNode,
            inputNode,
            outputNode,
          ]);

          // Assert
          expect({
            input: network.input,
            nodes: network.nodes.length,
            output: network.output,
          }).toStrictEqual({
            input: 1,
            nodes: 2,
            output: 1,
          });
        });
      });
    });

    describe('given construction cannot infer any input or output node roles', () => {
      describe('when building the network from those primitives', () => {
        it('throws the input-output type resolution error', () => {
          // Arrange
          const createNetwork = () => Architect.construct([new Node('hidden')]);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInputOutputTypeResolutionError,
          );
        });
      });
    });

    describe('given construction resolves only input nodes and no output nodes', () => {
      describe('when building the network from those primitives', () => {
        it('throws the input-output type resolution error', () => {
          // Arrange
          const createNetwork = () => Architect.construct([new Node('input')]);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInputOutputTypeResolutionError,
          );
        });
      });
    });

    describe('given construction resolves only output nodes and no input nodes', () => {
      describe('when building the network from those primitives', () => {
        it('throws the input-output type resolution error', () => {
          // Arrange
          const createNetwork = () => Architect.construct([new Node('output')]);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInputOutputTypeResolutionError,
          );
        });
      });
    });

    describe('given explicit input and output node types are cleared during role refresh', () => {
      describe('when building the network from those primitives', () => {
        it('throws the zero-interface construction error', () => {
          // Arrange
          const refreshExplicitIORolesSpy = jest
            .spyOn(Network.prototype, 'refreshExplicitIORoles')
            .mockImplementation(function refreshWithoutRoles(this: Network) {
              this.input = 0;
              this.output = 0;
            });

          try {
            const createNetwork = () =>
              Architect.construct([new Node('input'), new Node('output')]);

            // Assert
            expect(createNetwork).toThrow(ArchitectZeroInputOutputNodesError);
          } finally {
            refreshExplicitIORolesSpy.mockRestore();
          }
        });
      });
    });
  });

  describe('perceptron()', () => {
    describe('given fewer than three layer sizes', () => {
      describe('when constructing the network', () => {
        it('throws the perceptron configuration error', () => {
          // Arrange
          const createNetwork = () => Architect.perceptron(2, 1);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidPerceptronConfigurationError,
          );
        });
      });
    });

    describe('given input, hidden, and output sizes', () => {
      describe('when constructing the network', () => {
        it('preserves explicit input and output role counts', () => {
          // Arrange
          const network = Architect.perceptron(2, 3, 1);

          // Act
          const roleCounts = {
            inputNodeIds: network.inputNodeIds.length,
            outputNodeIds: network.outputNodeIds.length,
            inputNodes: network.nodes.filter((node) => node.type === 'input')
              .length,
            outputNodes: network.nodes.filter((node) => node.type === 'output')
              .length,
          };

          // Assert
          expect(roleCounts).toStrictEqual({
            inputNodeIds: 2,
            outputNodeIds: 1,
            inputNodes: 2,
            outputNodes: 1,
          });
        });
      });
    });

    describe('given one requested hidden layer is smaller than the minimum supported size', () => {
      describe('when constructing the network', () => {
        it('grows that hidden layer to the minimum width', () => {
          // Arrange
          const network = Architect.perceptron(
            4,
            1,
            2,
          ) as ArchitectLayeredNetwork;

          // Act
          const hiddenLayerSizes =
            network.describeArchitecture().hiddenLayerSizes;

          // Assert
          expect(hiddenLayerSizes).toStrictEqual([3]);
        });
      });
    });

    describe('given one construct-built feed-forward graph matches the builder width', () => {
      describe('when comparing the public architecture contract', () => {
        it('stays interoperable with the preconfigured builder surface', () => {
          // Arrange
          const builderNetwork = Architect.perceptron(2, 3, 1);
          const constructedNetwork = createConstructedPerceptronEquivalent();

          // Act
          const actualArchitectureBoundarySummary = {
            builder: {
              topologyIntent: builderNetwork.getTopologyIntent(),
              inputNodeIds: builderNetwork.inputNodeIds.length,
              outputNodeIds: builderNetwork.outputNodeIds.length,
              hiddenLayerSizes:
                builderNetwork.describeArchitecture().hiddenLayerSizes,
              hasCycles: builderNetwork.describeArchitecture().hasCycles,
            },
            constructed: {
              topologyIntent: constructedNetwork.getTopologyIntent(),
              inputNodeIds: constructedNetwork.inputNodeIds.length,
              outputNodeIds: constructedNetwork.outputNodeIds.length,
              hiddenLayerSizes:
                constructedNetwork.describeArchitecture().hiddenLayerSizes,
              hasCycles: constructedNetwork.describeArchitecture().hasCycles,
            },
          };

          // Assert
          expect(actualArchitectureBoundarySummary).toStrictEqual({
            builder: {
              topologyIntent: 'feed-forward',
              inputNodeIds: 2,
              outputNodeIds: 1,
              hiddenLayerSizes: [3],
              hasCycles: false,
            },
            constructed: {
              topologyIntent: 'feed-forward',
              inputNodeIds: 2,
              outputNodeIds: 1,
              hiddenLayerSizes: [3],
              hasCycles: false,
            },
          });
        });
      });
    });
  });

  describe('randomSparse()', () => {
    describe('given the sparse builder seed is not finite', () => {
      describe('when constructing the network', () => {
        it('throws the sparse-builder configuration error', () => {
          // Arrange
          const createNetwork = () =>
            Architect.randomSparse(1, 1, 1, { seed: Number.POSITIVE_INFINITY });

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidRandomSparseConfigurationError,
          );
        });
      });
    });

    describe('given the legacy random wrapper omits the options object', () => {
      describe('when constructing the network', () => {
        it('uses the default sparse-profile options', () => {
          // Arrange
          const network = Architect.random(2, 3, 1);

          // Act
          const roleCounts = {
            hiddenNodeCount: network.nodes.filter(
              (candidateNode) => candidateNode.type === 'hidden',
            ).length,
            inputRoleCount: network.inputNodeIds.length,
            outputRoleCount: network.outputNodeIds.length,
          };

          // Assert
          expect(roleCounts).toStrictEqual({
            hiddenNodeCount: 3,
            inputRoleCount: 2,
            outputRoleCount: 1,
          });
        });
      });
    });

    describe('given the sparse builder input size is not positive', () => {
      describe('when constructing the network', () => {
        it('throws the positive-dimension configuration error', () => {
          // Arrange
          const createNetwork = () => Architect.randomSparse(0, 1, 1);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidRandomSparseConfigurationError,
          );
        });
      });
    });

    describe('given the sparse builder hidden size is negative', () => {
      describe('when constructing the network', () => {
        it('throws the non-negative-dimension configuration error', () => {
          // Arrange
          const createNetwork = () => Architect.randomSparse(1, -1, 1);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidRandomSparseConfigurationError,
          );
        });
      });
    });

    describe('given valid sparse builder sizes', () => {
      describe('when constructing the network', () => {
        it('preserves explicit input and output role counts', () => {
          // Arrange
          const network = Architect.randomSparse(2, 3, 1);

          // Act
          const roleCounts = {
            inputNodeIds: network.inputNodeIds.length,
            outputNodeIds: network.outputNodeIds.length,
            inputNodes: network.nodes.filter((node) => node.type === 'input')
              .length,
            outputNodes: network.nodes.filter((node) => node.type === 'output')
              .length,
          };

          // Assert
          expect(roleCounts).toStrictEqual({
            inputNodeIds: 2,
            outputNodeIds: 1,
            inputNodes: 2,
            outputNodes: 1,
          });
        });
      });
    });

    describe('given two sparse builder requests share the same seed', () => {
      describe('when constructing both networks', () => {
        it('replays the same sparse topology and parameter signatures', () => {
          // Arrange
          const firstNetwork = Architect.randomSparse(2, 4, 1, {
            connections: 8,
            backConnections: 1,
            selfConnections: 1,
            gates: 1,
            seed: 31415,
          });
          const secondNetwork = Architect.randomSparse(2, 4, 1, {
            connections: 8,
            backConnections: 1,
            selfConnections: 1,
            gates: 1,
            seed: 31415,
          });

          // Act
          const sparseSignatures = {
            first: summarizeSparseNetwork(firstNetwork),
            second: summarizeSparseNetwork(secondNetwork),
          };

          // Assert
          expect(sparseSignatures.first).toStrictEqual(sparseSignatures.second);
        });
      });
    });

    describe('given the legacy random wrapper uses the same sparse request and seed', () => {
      describe('when constructing both compatibility surfaces', () => {
        it('preserves the seeded sparse builder contract', () => {
          // Arrange
          const legacyNetwork = Architect.random(2, 4, 1, {
            connections: 8,
            backconnections: 1,
            selfconnections: 1,
            gates: 1,
            seed: 27182,
          });
          const sparseNetwork = Architect.randomSparse(2, 4, 1, {
            connections: 8,
            backConnections: 1,
            selfConnections: 1,
            gates: 1,
            seed: 27182,
          });

          // Act
          const contractSummary = {
            legacy: summarizeSparseNetwork(legacyNetwork),
            sparse: summarizeSparseNetwork(sparseNetwork),
          };

          // Assert
          expect(contractSummary.legacy).toStrictEqual(contractSummary.sparse);
        });
      });
    });

    describe('given an impossible forward-connection request', () => {
      describe('when constructing the network', () => {
        it('throws a clear sparse-builder configuration error', () => {
          // Arrange
          const createNetwork = () =>
            Architect.randomSparse(1, 0, 1, { connections: 2 });

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidRandomSparseConfigurationError,
          );
        });
      });
    });
  });

  describe('narx()', () => {
    describe('given the hidden-layer argument is already an array with two stages', () => {
      describe('when constructing the network', () => {
        it('preserves both requested hidden stages alongside the delay modules', () => {
          // Arrange
          const network = Architect.narx(1, [2, 2], 1, 1, 1);

          // Act
          const narxSummary = {
            hiddenNodeCount: network.nodes.filter(
              (candidateNode) => candidateNode.type === 'hidden',
            ).length,
            recurrentModuleCount:
              summarizeNarxDelayLines(network).recurrentModuleCount,
          };

          // Assert
          expect(narxSummary).toStrictEqual({
            hiddenNodeCount: 4,
            recurrentModuleCount: 2,
          });
        });
      });
    });

    describe('given explicit input and output delay lines', () => {
      describe('when constructing the network', () => {
        it('hydrates identity delay blocks with unit carry links and clear-state guidance', () => {
          // Arrange
          const network = Architect.narx(1, 2, 1, 2, 2);

          // Act
          network.activate([0]);
          const delayLineSummary = summarizeNarxDelayLines(network);

          // Assert
          expect(delayLineSummary).toStrictEqual({
            clearSuggestion:
              'Call clear() before a new independent sequence when carried recurrent state should reset.',
            recurrentKinds: ['narx-memory', 'narx-memory'],
            recurrentModuleCount: 2,
            roleSizesPerModule: ['1,1', '1,1'],
            schedulingStateSemantics: 'carry',
            uniqueActivationSamples: [0.25],
            uniqueBiases: [0],
            uniqueCarryWeights: [1],
            uniqueDerivativeSamples: [1],
          });
        });
      });
    });

    describe('given one NARX delay line is configured as a running-total predictor', () => {
      describe('when one short independent sequence is activated after clear()', () => {
        it('predicts the carried running total across the sequence', () => {
          // Arrange
          const network = Architect.narx(1, 0, 1, 1, 1);
          const inputSequence = [0.2, 0.7, 0.4];

          configureNarxRunningTotalPredictor(network);

          // Act
          network.clear();
          const predictedSequence = roundSequenceOutputs(
            captureSequenceOutputs(network, inputSequence),
          );

          // Assert
          expect(predictedSequence).toStrictEqual([0.2, 0.9, 1.3]);
        });
      });
    });

    describe('given one NARX runtime replays the same sequence twice', () => {
      describe('when clear() is only called before the third replay', () => {
        it('resets the carried delay-line state before the next independent sequence starts', () => {
          // Arrange
          const network = Architect.narx(1, 0, 1, 1, 1);
          const inputSequence = [1, 0, 0];

          configureNarxClearStateFixture(network);

          const firstReplay = roundSequenceOutputs(
            captureSequenceOutputs(network, inputSequence),
          );
          const carriedReplay = roundSequenceOutputs(
            captureSequenceOutputs(network, inputSequence),
          );

          // Act
          network.clear();
          const clearedReplay = roundSequenceOutputs(
            captureSequenceOutputs(network, inputSequence),
          );

          // Assert
          expect({
            carriedFirstOutputChanged: carriedReplay[0] !== firstReplay[0],
            clearedReplayMatchesInitial:
              clearedReplay.join(',') === firstReplay.join(','),
          }).toStrictEqual({
            carriedFirstOutputChanged: true,
            clearedReplayMatchesInitial: true,
          });
        });
      });
    });
  });

  describe('lstm()', () => {
    describe('given one LSTM layer size is not a positive finite number', () => {
      describe('when constructing the network', () => {
        it('throws the LSTM layer-arguments error', () => {
          // Arrange
          const createNetwork = () => Architect.lstm(2, Number.NaN, 1);

          // Assert
          expect(createNetwork).toThrow(
            ArchitectInvalidLstmLayerArgumentsError,
          );
        });
      });
    });

    describe('given fewer than three LSTM layer sizes', () => {
      describe('when constructing the network', () => {
        it('throws the LSTM configuration error', () => {
          // Arrange
          const createNetwork = () => Architect.lstm(2, 1);

          // Assert
          expect(createNetwork).toThrow(ArchitectInvalidLstmConfigurationError);
        });
      });
    });

    describe('given the direct input-to-output shortcut is disabled', () => {
      describe('when constructing the network', () => {
        it('keeps the recurrent descriptor boundary without the shortcut edge', () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1, { inputToOutput: false });

          // Act
          const recurrentSummary =
            summarizeRecurrentArchitectureBoundary(network);

          // Assert
          expect(recurrentSummary).toStrictEqual({
            clearSuggestion:
              'Call clear() before a new independent sequence when carried recurrent state should reset.',
            directInputToOutputConnections: 0,
            gatedBlockCount: 1,
            hasCycles: false,
            recurrentKinds: ['lstm'],
            recurrentModuleCount: 1,
            stateSemantics: 'carry',
            topologyIntent: 'unconstrained',
          });
        });
      });
    });

    describe('given the LSTM shortcut options are omitted', () => {
      describe('when constructing the network', () => {
        it('keeps the default direct input-to-output shortcut', () => {
          // Arrange
          const network = Architect.lstm(2, 3, 1);

          // Act
          const recurrentSummary =
            summarizeRecurrentArchitectureBoundary(network);

          // Assert
          expect({
            directInputToOutputConnections:
              recurrentSummary.directInputToOutputConnections,
            recurrentKinds: recurrentSummary.recurrentKinds,
          }).toStrictEqual({
            directInputToOutputConnections: 2,
            recurrentKinds: ['lstm'],
          });
        });
      });
    });
  });

  describe('gru()', () => {
    describe('given one GRU layer size is not a positive finite number', () => {
      describe('when constructing the network', () => {
        it('throws the GRU layer-arguments error', () => {
          // Arrange
          const createNetwork = () => Architect.gru(2, Number.NaN, 1);

          // Assert
          expect(createNetwork).toThrow(ArchitectInvalidGruLayerArgumentsError);
        });
      });
    });

    describe('given fewer than three GRU layer sizes', () => {
      describe('when constructing the network', () => {
        it('throws the GRU configuration error', () => {
          // Arrange
          const createNetwork = () => Architect.gru(2, 1);

          // Assert
          expect(createNetwork).toThrow(ArchitectInvalidGruConfigurationError);
        });
      });
    });

    describe('given the direct input-to-output shortcut is toggled', () => {
      describe('when constructing the network', () => {
        it('adds the shortcut only for the enabled builder request', () => {
          // Arrange
          const disabledNetwork = Architect.gru(2, 3, 1, {
            inputToOutput: false,
          });
          const enabledNetwork = Architect.gru(2, 3, 1, {
            inputToOutput: true,
          });

          // Act
          const recurrentBoundarySummary = {
            disabled: summarizeRecurrentArchitectureBoundary(disabledNetwork),
            enabled: summarizeRecurrentArchitectureBoundary(enabledNetwork),
          };

          // Assert
          expect(recurrentBoundarySummary).toStrictEqual({
            disabled: {
              clearSuggestion:
                'Call clear() before a new independent sequence when carried recurrent state should reset.',
              directInputToOutputConnections: 0,
              gatedBlockCount: 1,
              hasCycles: true,
              recurrentKinds: ['gru'],
              recurrentModuleCount: 1,
              stateSemantics: 'carry',
              topologyIntent: 'unconstrained',
            },
            enabled: {
              clearSuggestion:
                'Call clear() before a new independent sequence when carried recurrent state should reset.',
              directInputToOutputConnections: 2,
              gatedBlockCount: 1,
              hasCycles: true,
              recurrentKinds: ['gru'],
              recurrentModuleCount: 1,
              stateSemantics: 'carry',
              topologyIntent: 'unconstrained',
            },
          });
        });
      });
    });

    describe('given the GRU shortcut options are omitted', () => {
      describe('when constructing the network', () => {
        it('keeps the default no-shortcut topology', () => {
          // Arrange
          const network = Architect.gru(2, 3, 1);

          // Act
          const recurrentSummary =
            summarizeRecurrentArchitectureBoundary(network);

          // Assert
          expect({
            directInputToOutputConnections:
              recurrentSummary.directInputToOutputConnections,
            recurrentKinds: recurrentSummary.recurrentKinds,
          }).toStrictEqual({
            directInputToOutputConnections: 0,
            recurrentKinds: ['gru'],
          });
        });
      });
    });
  });

  describe('hopfield()', () => {
    describe('given one Hopfield size request', () => {
      describe('when constructing the network', () => {
        it('creates matched input and output roles with step-activated outputs', () => {
          // Arrange
          const network = Architect.hopfield(2);

          // Act
          const hopfieldSummary = {
            inputRoleCount: network.inputNodeIds.length,
            outputActivationSamples: network.nodes
              .filter((candidateNode) => candidateNode.type === 'output')
              .map((candidateNode) => candidateNode.squash(0.25)),
            outputRoleCount: network.outputNodeIds.length,
          };

          // Assert
          expect(hopfieldSummary).toStrictEqual({
            inputRoleCount: 2,
            outputActivationSamples: [1, 1],
            outputRoleCount: 2,
          });
        });
      });
    });
  });

  describe('enforceMinimumHiddenLayerSizes()', () => {
    describe('given the network exposes no architect layer metadata', () => {
      describe('when normalizing hidden layer sizes', () => {
        it('returns the same network unchanged', () => {
          // Arrange
          const network = new Network(2, 1);

          // Act
          const normalizedNetwork =
            Architect.enforceMinimumHiddenLayerSizes(network);

          // Assert
          expect(normalizedNetwork).toBe(network);
        });
      });
    });

    describe('given the hidden layers already satisfy the minimum size rule', () => {
      describe('when normalizing hidden layer sizes', () => {
        it('keeps the existing node and connection counts unchanged', () => {
          // Arrange
          const network = Architect.perceptron(
            2,
            3,
            1,
          ) as ArchitectLayeredNetwork;
          const beforeNormalization = {
            connections: network.connections.length,
            hiddenNodes: network.layers?.[1]?.nodes.length,
            nodes: network.nodes.length,
          };

          // Act
          Architect.enforceMinimumHiddenLayerSizes(network);

          // Assert
          expect({
            connections: network.connections.length,
            hiddenNodes: network.layers?.[1]?.nodes.length,
            nodes: network.nodes.length,
          }).toStrictEqual(beforeNormalization);
        });
      });
    });

    describe('given one hidden layer has been shrunk below the minimum size', () => {
      describe('when normalizing hidden layer sizes', () => {
        it('recreates the missing hidden nodes and reconnects both adjacent layers', () => {
          // Arrange
          const network = Architect.perceptron(
            3,
            3,
            2,
          ) as ArchitectLayeredNetwork;
          const inputLayer = network.layers?.[0];
          const hiddenLayer = network.layers?.[1];
          const outputLayer = network.layers?.[2];

          if (
            !inputLayer ||
            !hiddenLayer ||
            !outputLayer ||
            !hiddenLayer.output
          ) {
            throw new Error(
              'Expected the perceptron fixture to expose three layers.',
            );
          }

          const retainedHiddenNode = hiddenLayer.nodes[0];
          hiddenLayer.nodes = [retainedHiddenNode];
          hiddenLayer.output.nodes = [retainedHiddenNode];
          network.nodes = [
            ...inputLayer.nodes,
            retainedHiddenNode,
            ...outputLayer.nodes,
          ];
          network.connections = network.connections.filter(
            (candidateConnection) =>
              network.nodes.includes(candidateConnection.from) &&
              network.nodes.includes(candidateConnection.to),
          );

          // Act
          Architect.enforceMinimumHiddenLayerSizes(network);

          // Assert
          expect({
            connectionCount: network.connections.length,
            hiddenLayerOutputNodes: hiddenLayer.output.nodes.length,
            hiddenNodes: hiddenLayer.nodes.length,
            totalHiddenNodes: network.nodes.filter(
              (candidateNode) => candidateNode.type === 'hidden',
            ).length,
          }).toStrictEqual({
            connectionCount: 15,
            hiddenLayerOutputNodes: 3,
            hiddenNodes: 3,
            totalHiddenNodes: 3,
          });
        });
      });
    });

    describe('given neighboring layer outputs are unavailable during hidden-layer growth', () => {
      describe('when normalizing hidden layer sizes', () => {
        it('grows the hidden nodes without adding bridge connections', () => {
          // Arrange
          const existingHiddenNode = new Node('hidden');
          const network = {
            connections: [],
            input: 3,
            layers: [
              { output: null },
              { nodes: [existingHiddenNode], output: null },
              { output: null },
            ],
            nodes: [existingHiddenNode],
            output: 2,
          } as unknown as ArchitectLayeredNetwork;

          // Act
          Architect.enforceMinimumHiddenLayerSizes(network);

          // Assert
          expect({
            connections: network.connections.length,
            hiddenNodes: network.layers?.[1]?.nodes.length,
            totalNodes: network.nodes.length,
          }).toStrictEqual({
            connections: 0,
            hiddenNodes: 3,
            totalNodes: 3,
          });
        });
      });
    });
  });
});
