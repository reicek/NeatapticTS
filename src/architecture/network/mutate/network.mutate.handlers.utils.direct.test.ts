import { config } from '../../../config';
import mutation from '../../../methods/mutation/mutation';
import Layer from '../../layer/layer';
import Node from '../../node';
import Network from '../network';
import type { NetworkJSON } from '../network.types';
import { NetworkMutateRecurrentLayerOutputInitializationError } from './network.mutate.errors';
import {
  addBackConn,
  addConn,
  addGate,
  addGRUNode,
  addLSTMNode,
  addNode,
  modActivation,
  modBias,
  modWeight,
  reinitWeight,
  subConn,
  subGate,
  subBackConn,
  subNode,
  swapNodes,
} from './network.mutate.handlers.utils';
import * as temporalExtensions from '../network.temporal.extensions.utils';
import { SUB_NODE_STABILITY_WEIGHT_DELTA } from './network.mutate.utils.types';

function countHiddenNodes(network: Network): number {
  return network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  ).length;
}

function createSingleConnectionNetwork(seed: number): Network {
  return new Network(1, 1, { seed, enforceAcyclic: false });
}

function createAcyclicSingleConnectionNetwork(seed: number): Network {
  return new Network(1, 1, { seed, enforceAcyclic: true });
}

function createSequentialRandom(values: number[]): () => number {
  let currentIndex = 0;
  const finalValue = values.at(-1) ?? 0;

  return () => {
    const nextValue = values[currentIndex] ?? finalValue;
    currentIndex += 1;
    return nextValue;
  };
}

function disconnectAllConnections(network: Network): void {
  const removableConnections = [...network.connections, ...network.selfconns];

  removableConnections.forEach((candidateConnection) => {
    network.disconnect(candidateConnection.from, candidateConnection.to);
  });
}

function readBiasSignature(network: Network): string {
  return network.nodes
    .map((candidateNode) => String(candidateNode.bias))
    .join(',');
}

function readSquashSignature(network: Network): string {
  return network.nodes
    .map((candidateNode) => candidateNode.squash.name)
    .join(',');
}

function readWeightSignature(network: Network): string {
  return network.connections
    .map((candidateConnection) => candidateConnection.weight.toFixed(6))
    .join(',');
}

function countTemporalDescriptors(network: Network): number {
  const serializedJson = network.toJSON() as unknown as NetworkJSON;
  const extensionValues = serializedJson.extensions?.values as
    | {
        recurrentModules?: Array<unknown>;
      }
    | undefined;
  const recurrentModules = Array.isArray(extensionValues?.recurrentModules)
    ? extensionValues.recurrentModules
    : [];

  return recurrentModules.length;
}

describe('network mutate handler utility chapter', () => {
  const originalDeterministicChainMode = config.deterministicChainMode ?? false;
  const originalWarnings = config.warnings ?? false;

  afterEach(() => {
    config.deterministicChainMode = originalDeterministicChainMode;
    config.warnings = originalWarnings;
    jest.restoreAllMocks();
  });

  describe('addNode', () => {
    describe('given deterministic chain mode is enabled without an output endpoint', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = createSingleConnectionNetwork(10_201);
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        network.nodes = network.nodes.filter(
          (candidateNode) => candidateNode.type !== 'output',
        );

        // Act
        addNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the forward graph is empty before a random split mutation', () => {
      it('seeds one forward edge and then inserts one hidden split node', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_202);
        disconnectAllConnections(network);

        // Act
        addNode.call(network);

        // Assert
        expect({
          connectionCount: network.connections.length,
          hiddenCount: countHiddenNodes(network),
        }).toEqual({
          connectionCount: 2,
          hiddenCount: 1,
        });
      });
    });

    describe('given the split connection was previously gated', () => {
      it('reassigns the previous gater onto the selected split leg', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_203);
        const outputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!outputNode) {
          throw new Error('Expected one output node for gate reassignment test.');
        }

        network.gate(outputNode, network.connections[0]);
        Reflect.set(network, '_rand', createSequentialRandom([0, 1]));

        // Act
        addNode.call(network);

        // Assert
        expect({
          fromType: network.gates[0]?.from.type,
          gaterType: network.gates[0]?.gater?.type,
          toType: network.gates[0]?.to.type,
        }).toEqual({
          fromType: 'input',
          gaterType: 'output',
          toType: 'hidden',
        });
      });
    });

    describe('given the gated split keeps the hidden-to-target leg', () => {
      it('reassigns the previous gater onto the hidden-to-output leg', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_240);
        const outputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!outputNode) {
          throw new Error('Expected one output node for alternate gate reassignment test.');
        }

        network.gate(outputNode, network.connections[0]);
        Reflect.set(network, '_rand', createSequentialRandom([0, 0]));

        // Act
        addNode.call(network);

        // Assert
        expect({
          fromType: network.gates[0]?.from.type,
          gaterType: network.gates[0]?.gater?.type,
          toType: network.gates[0]?.to.type,
        }).toEqual({
          fromType: 'hidden',
          gaterType: 'output',
          toType: 'output',
        });
      });
    });

    describe('given deterministic chain state is already initialized with one stray side edge', () => {
      it('prunes the stray side edge while adding one hidden chain node', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = new Network(1, 2, {
          seed: 10_224,
          enforceAcyclic: false,
        });
        const inputNode = network.nodes[0];
        const secondOutputNode = network.nodes[2];
        Reflect.set(network, '_detChain', [inputNode]);

        // Act
        addNode.call(network);

        // Assert
        expect({
          hasStrayConnection: inputNode.isProjectingTo(secondOutputNode),
          hiddenCount: countHiddenNodes(network),
        }).toEqual({
          hasStrayConnection: false,
          hiddenCount: 1,
        });
      });
    });

    describe('given disconnecting one deterministic stray edge throws', () => {
      it('suppresses the disconnect failure and keeps the stray edge intact', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = new Network(1, 2, {
          seed: 10_225,
          enforceAcyclic: false,
        });
        const inputNode = network.nodes[0];
        const secondOutputNode = network.nodes[2];
        const originalDisconnect = network.disconnect.bind(network);
        Reflect.set(network, '_detChain', [inputNode]);
        jest.spyOn(network, 'disconnect').mockImplementation((from, to) => {
          if (from === inputNode && to === secondOutputNode) {
            throw new Error('Expected deterministic stray-edge disconnect failure.');
          }

          return originalDisconnect(from, to);
        });

        // Act
        addNode.call(network);

        // Assert
        expect(inputNode.isProjectingTo(secondOutputNode)).toBe(true);
      });
    });

    describe('given deterministic chain state is present but empty', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = createSingleConnectionNetwork(10_226);
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        Reflect.set(network, '_detChain', []);

        // Act
        addNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the deterministic chain tail is missing', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = createSingleConnectionNetwork(10_227);
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        Reflect.set(network, '_detChain', [undefined]);

        // Act
        addNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the deterministic terminal connection cannot be materialized', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        config.deterministicChainMode = true;
        const network = createSingleConnectionNetwork(10_228);
        const inputNode = network.nodes[0];
        const outputNode = network.nodes[1];
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        network.disconnect(inputNode, outputNode);
        Reflect.set(network, '_detChain', [inputNode]);
        jest.spyOn(network, 'connect').mockReturnValue([]);

        // Act
        addNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the random split selector points past the available connection list', () => {
      it('returns without adding a hidden node', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_229);
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        Reflect.set(network, '_rand', () => 1);

        // Act
        addNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the empty graph also lacks input and output endpoints', () => {
      it('returns without adding any nodes', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_230);
        disconnectAllConnections(network);
        network.nodes = [];

        // Act
        addNode.call(network);

        // Assert
        expect(network.nodes.length).toBe(0);
      });
    });
  });

  describe('addConn', () => {
    describe('given acyclic mode adds the only missing forward shortcut', () => {
      it('marks the topology dirty and increases the forward connection count', () => {
        // Arrange
        const network = createAcyclicSingleConnectionNetwork(10_215);
        addNode.call(network);
        Reflect.set(network, '_topoDirty', false);
        const connectionCountBeforeMutation = network.connections.length;

        // Act
        addConn.call(network);

        // Assert
        expect({
          connectionCount: network.connections.length,
          topoDirty: Reflect.get(network, '_topoDirty'),
        }).toEqual({
          connectionCount: connectionCountBeforeMutation + 1,
          topoDirty: true,
        });
      });
    });
  });

  describe('subNode', () => {
    describe('given the hidden-node selector points past the available candidate list', () => {
      it('returns without removing a hidden node', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_204);
        addNode.call(network);
        const hiddenCountBeforeMutation = countHiddenNodes(network);
        Reflect.set(network, '_rand', () => 1);

        // Act
        subNode.call(network);
        const hiddenCountAfterMutation = countHiddenNodes(network);

        // Assert
        expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
      });
    });

    describe('given the selected hidden node is isolated and no connections remain after removal', () => {
      it('removes the hidden node and safely skips the stability nudge', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_205);
        addNode.call(network);
        disconnectAllConnections(network);
        Reflect.set(network, '_rand', () => 0);

        // Act
        subNode.call(network);

        // Assert
        expect({
          connectionCount: network.connections.length,
          hiddenCount: countHiddenNodes(network),
        }).toEqual({
          connectionCount: 0,
          hiddenCount: 0,
        });
      });
    });

    describe('given one downstream connection remains after hidden-node removal', () => {
      it('applies the tiny stability nudge to the first remaining connection', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 10_231,
          enforceAcyclic: false,
        });
        Reflect.set(network, '_rand', () => 0);
        addNode.call(network);
        const preservedConnection = network.connections.find(
          (candidateConnection) => candidateConnection.from === network.nodes[1],
        );

        if (!preservedConnection) {
          throw new Error('Expected one preserved downstream connection for stability nudge test.');
        }

        const preservedConnectionWeightBeforeMutation = preservedConnection.weight;

        // Act
        subNode.call(network);

        // Assert
        expect(preservedConnection.weight).toBeCloseTo(
          preservedConnectionWeightBeforeMutation + SUB_NODE_STABILITY_WEIGHT_DELTA,
          12,
        );
      });
    });

    describe('given warnings are enabled and no hidden nodes exist', () => {
      it('emits one warning and leaves the hidden-node count unchanged', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createSingleConnectionNetwork(10_206);

        // Act
        subNode.call(network);

        // Assert
        expect({
          hiddenCount: countHiddenNodes(network),
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          hiddenCount: 0,
          warningCount: 1,
        });
      });
    });
  });

  describe('subConn', () => {
    describe('given one fully redundant forward edge is available', () => {
      it('disconnects exactly one removable forward connection', () => {
        // Arrange
        const network = new Network(2, 2, {
          seed: 10_216,
          enforceAcyclic: false,
        });
        const connectionCountBeforeMutation = network.connections.length;
        Reflect.set(network, '_rand', () => 0);

        // Act
        subConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation - 1);
      });
    });

    describe('given removing a forward edge would isolate its target peer group from the source', () => {
      it('returns without disconnecting the guarded connection', () => {
        // Arrange
        const network = new Network(2, 1, {
          seed: 10_217,
          enforceAcyclic: false,
        });
        const hiddenNode = new Node('hidden', undefined, () => 0.5);
        network.nodes.splice(network.nodes.length - network.output, 0, hiddenNode);
        network.connect(network.nodes[0], hiddenNode);
        const connectionCountBeforeMutation = network.connections.length;
        Reflect.set(network, '_rand', () => 0);

        // Act
        subConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation);
      });
    });

    describe('given malformed topology metadata makes the peer set empty', () => {
      it('still disconnects one otherwise removable forward connection', () => {
        // Arrange
        const network = new Network(2, 2, {
          seed: 10_241,
          enforceAcyclic: false,
        });
        const connectionCountBeforeMutation = network.connections.length;
        network.input = 0;
        network.output = 0;
        Reflect.set(network, '_rand', () => 0);

        // Act
        subConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation - 1);
      });
    });
  });

  describe('modWeight', () => {
    describe('given no normal or self connections remain', () => {
      it('returns without changing the empty weight signature', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_207);
        disconnectAllConnections(network);

        // Act
        modWeight.call(network, mutation.MOD_WEIGHT);

        // Assert
        expect(readWeightSignature(network)).toBe('');
      });
    });

    describe('given exact min and max overrides are provided', () => {
      it('applies the exact configured weight delta', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_208);
        const weightBeforeMutation = network.connections[0].weight;

        // Act
        modWeight.call(network, {
          max: 0.25,
          min: 0.25,
        } as Parameters<typeof modWeight>[0]);

        // Assert
        expect(network.connections[0].weight - weightBeforeMutation).toBeCloseTo(
          0.25,
          12,
        );
      });
    });

    describe('given no explicit delta range overrides are provided', () => {
      it('falls back to the default symmetric delta range', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_232);
        const weightBeforeMutation = network.connections[0].weight;
        Reflect.set(network, '_rand', createSequentialRandom([0, 0.5]));

        // Act
        modWeight.call(network);

        // Assert
        expect(network.connections[0].weight).toBeCloseTo(weightBeforeMutation, 12);
      });
    });
  });

  describe('modBias', () => {
    describe('given the random selector points past the mutable non-input nodes', () => {
      it('returns without changing the node bias signature', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_209);
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', () => 1);

        // Act
        modBias.call(network, mutation.MOD_BIAS);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given one mutable non-input node is selected', () => {
      it('forwards the bias mutation call onto the selected node', () => {
        // Arrange
        const randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.5);
        const network = createSingleConnectionNetwork(10_233);
        const biasSignatureBeforeMutation = readBiasSignature(network);

        // Act
        modBias.call(network, mutation.MOD_BIAS);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect({
          biasSignatureAfterMutation,
          randomCallCount: randomSpy.mock.calls.length,
        }).toEqual({
          biasSignatureAfterMutation: biasSignatureBeforeMutation,
          randomCallCount: 1,
        });
      });
    });
  });

  describe('modActivation', () => {
    describe('given the default mutation policy still allows output nodes', () => {
      it('changes the output squash signature on a one-output network', () => {
        // Arrange
        const randomSpy = jest.spyOn(Math, 'random').mockReturnValue(0);
        const network = createSingleConnectionNetwork(10_210);
        const squashSignatureBeforeMutation = readSquashSignature(network);

        // Act
        modActivation.call(network, mutation.MOD_ACTIVATION);
        const squashSignatureAfterMutation = readSquashSignature(network);

        // Assert
        expect({
          randomCallCount: randomSpy.mock.calls.length,
          squashChanged: squashSignatureAfterMutation !== squashSignatureBeforeMutation,
        }).toEqual({
          randomCallCount: 1,
          squashChanged: true,
        });
      });
    });

    describe('given no explicit output-mutation override is provided', () => {
      it('still allows mutating the output-node squash function', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_234);
        network.nodes = network.nodes.filter(
          (candidateNode) => candidateNode.type === 'input',
        );
        const nodeCountBeforeMutation = network.nodes.length;

        // Act
        modActivation.call(network);

        // Assert
        expect(network.nodes.length).toBe(nodeCountBeforeMutation);
      });
    });
  });

  describe('addGate', () => {
    describe('given every connection is already gated', () => {
      it('emits one warning and keeps the gate count unchanged', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createSingleConnectionNetwork(10_211);
        const outputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!outputNode) {
          throw new Error('Expected one output node for add-gate saturation test.');
        }

        network.gate(outputNode, network.connections[0]);
        const gateCountBeforeMutation = network.gates.length;

        // Act
        addGate.call(network);

        // Assert
        expect({
          gateCount: network.gates.length,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          gateCount: gateCountBeforeMutation,
          warningCount: 1,
        });
      });
    });
  });

  describe('subGate', () => {
    describe('given no gated connections exist', () => {
      it('emits one warning and keeps the gate shelf empty', () => {
        // Arrange
        config.warnings = true;
        const warnSpy = jest
          .spyOn(console, 'warn')
          .mockImplementation(() => undefined);
        const network = createSingleConnectionNetwork(10_212);

        // Act
        subGate.call(network);

        // Assert
        expect({
          gateCount: network.gates.length,
          warningCount: warnSpy.mock.calls.length,
        }).toEqual({
          gateCount: 0,
          warningCount: 1,
        });
      });
    });
  });

  describe('addBackConn', () => {
    describe('given no backward candidate pairs exist', () => {
      it('returns without changing the forward connection count', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_235);
        const connectionCountBeforeMutation = network.connections.length;

        // Act
        addBackConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation);
      });
    });

    describe('given one later output can still connect backward to an earlier output', () => {
      it('adds one backward connection', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_236,
          enforceAcyclic: false,
        });
        const connectionCountBeforeMutation = network.connections.length;
        Reflect.set(network, '_rand', () => 0);

        // Act
        addBackConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation + 1);
      });
    });

    describe('given the only backward candidate pair is already projected', () => {
      it('returns without changing the forward connection count', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_242,
          enforceAcyclic: false,
        });
        Reflect.set(network, '_rand', () => 0);
        addBackConn.call(network);
        const connectionCountBeforeMutation = network.connections.length;

        // Act
        addBackConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation);
      });
    });
  });

  describe('subBackConn', () => {
    describe('given no backward connections satisfy the removal constraints', () => {
      it('returns without changing the forward connection count', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_218);
        const connectionCountBeforeMutation = network.connections.length;

        // Act
        subBackConn.call(network);

        // Assert
        expect(network.connections.length).toBe(connectionCountBeforeMutation);
      });
    });
  });

  describe('swapNodes', () => {
    describe('given fewer than two swappable nodes exist', () => {
      it('returns without changing the node bias signature', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_243);
        const biasSignatureBeforeMutation = readBiasSignature(network);

        // Act
        swapNodes.call(network, mutation.SWAP_NODES);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given no explicit output-swap override is provided', () => {
      it('still swaps the output-node bias signature', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_248,
          enforceAcyclic: false,
        });
        network.nodes[1].bias = 4;
        network.nodes[2].bias = 5;
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', createSequentialRandom([0, 0]));

        // Act
        swapNodes.call(network);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).not.toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given the default policy still allows swapping output nodes', () => {
      it('swaps the output-node bias signature', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_219,
          enforceAcyclic: false,
        });
        network.nodes[1].bias = 1;
        network.nodes[2].bias = 2;
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', createSequentialRandom([0, 0]));

        // Act
        swapNodes.call(network, mutation.SWAP_NODES);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).not.toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given the distinct second-node selector points past its only candidate', () => {
      it('returns without changing the output-node bias signature', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_220,
          enforceAcyclic: false,
        });
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', createSequentialRandom([0, 1]));

        // Act
        swapNodes.call(network, mutation.SWAP_NODES);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given the first-node selector points past the swap candidate list', () => {
      it('returns without changing the output-node bias signature', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_237,
          enforceAcyclic: false,
        });
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', () => 1);

        // Act
        swapNodes.call(network, mutation.SWAP_NODES);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
      });
    });

    describe('given the candidate list contains the same node reference twice', () => {
      it('returns without changing the duplicated bias signature', () => {
        // Arrange
        const network = new Network(1, 2, {
          seed: 10_238,
          enforceAcyclic: false,
        });
        network.nodes[1].bias = 3;
        network.nodes[2] = network.nodes[1];
        const biasSignatureBeforeMutation = readBiasSignature(network);
        Reflect.set(network, '_rand', () => 0);

        // Act
        swapNodes.call(network, mutation.SWAP_NODES);
        const biasSignatureAfterMutation = readBiasSignature(network);

        // Assert
        expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
      });
    });
  });

  describe('addLSTMNode', () => {
    describe('given no forward connections remain', () => {
      it('returns without changing the node count', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_213);
        const nodeCountBeforeMutation = network.nodes.length;
        disconnectAllConnections(network);

        // Act
        addLSTMNode.call(network);
        const nodeCountAfterMutation = network.nodes.length;

        // Assert
        expect(nodeCountAfterMutation).toBe(nodeCountBeforeMutation);
      });
    });

    describe('given the recurrent role split cannot be derived', () => {
      it('skips appending temporal descriptors', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_244);
        jest
          .spyOn(temporalExtensions, 'splitLstmLayerNodes')
          .mockReturnValue(undefined);

        // Act
        addLSTMNode.call(network);

        // Assert
        expect(countTemporalDescriptors(network)).toBe(0);
      });
    });
  });

  describe('addGRUNode', () => {
    describe('given the expanded connection was previously gated', () => {
      it('preserves the previous gater on the latest reconnection edge', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_221);
        const outputNode = network.nodes.find(
          (candidateNode) => candidateNode.type === 'output',
        );

        if (!outputNode) {
          throw new Error('Expected one output node for recurrent gate preservation test.');
        }

        network.gate(outputNode, network.connections[0]);

        // Act
        addGRUNode.call(network);

        // Assert
        expect(network.connections.at(-1)?.gater).toBe(outputNode);
      });
    });

    describe('given no forward connections remain', () => {
      it('returns without changing the node count', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_239);
        const nodeCountBeforeMutation = network.nodes.length;
        disconnectAllConnections(network);

        // Act
        addGRUNode.call(network);

        // Assert
        expect(network.nodes.length).toBe(nodeCountBeforeMutation);
      });
    });

    describe('given the connection selector points past the available forward edge', () => {
      it('returns without changing the node count', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_245);
        const nodeCountBeforeMutation = network.nodes.length;
        Reflect.set(network, '_rand', () => 1);

        // Act
        addGRUNode.call(network);

        // Assert
        expect(network.nodes.length).toBe(nodeCountBeforeMutation);
      });
    });

    describe('given the recurrent layer output is malformed', () => {
      it('throws the recurrent-layer output initialization error', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_246);
        jest.spyOn(Layer, 'gru').mockReturnValue({
          nodes: [],
        } as never);

        // Act and Assert
        expect(() => addGRUNode.call(network)).toThrow(
          NetworkMutateRecurrentLayerOutputInitializationError,
        );
      });
    });

    describe('given the recurrent self-connection is duplicated across registration shelves', () => {
      it('keeps the duplicate self-connection registered only once and skips temporal descriptors', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_247);
        const recurrentNode = new Node('hidden', undefined, () => 0.5);
        const sharedSelfConnection = recurrentNode.connect(recurrentNode)[0];
        recurrentNode.connections.out.push(sharedSelfConnection);
        network.selfconns.push(sharedSelfConnection);
        jest.spyOn(Layer, 'gru').mockReturnValue({
          nodes: [recurrentNode],
          output: { nodes: [recurrentNode] },
        } as never);
        jest
          .spyOn(temporalExtensions, 'splitGruLayerNodes')
          .mockReturnValue(undefined);

        // Act
        addGRUNode.call(network);

        // Assert
        expect({
          duplicateSelfConnectionCount: network.selfconns.filter(
            (candidateConnection) => candidateConnection === sharedSelfConnection,
          ).length,
          temporalDescriptorCount: countTemporalDescriptors(network),
        }).toEqual({
          duplicateSelfConnectionCount: 1,
          temporalDescriptorCount: 0,
        });
      });
    });
  });

  describe('reinitWeight', () => {
    describe('given exact min and max overrides are provided', () => {
      it('reinitializes the selected connection group to the exact configured weight', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_214);

        // Act
        reinitWeight.call(network, {
          max: 0.125,
          min: 0.125,
        } as Parameters<typeof reinitWeight>[0]);

        // Assert
        expect(network.connections[0].weight).toBeCloseTo(0.125, 12);
      });
    });

    describe('given the random selector points past the mutable non-input nodes', () => {
      it('returns without changing the weight signature', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_222);
        const weightSignatureBeforeMutation = readWeightSignature(network);
        Reflect.set(network, '_rand', () => 1);

        // Act
        reinitWeight.call(network, mutation.REINIT_WEIGHT);
        const weightSignatureAfterMutation = readWeightSignature(network);

        // Assert
        expect(weightSignatureAfterMutation).toBe(weightSignatureBeforeMutation);
      });
    });

    describe('given no explicit min and max overrides are provided', () => {
      it('falls back to the default symmetric sampling range', () => {
        // Arrange
        const network = createSingleConnectionNetwork(10_223);
        Reflect.set(network, '_rand', createSequentialRandom([0, 0.5]));

        // Act
        reinitWeight.call(network);

        // Assert
        expect(network.connections[0].weight).toBeCloseTo(0, 12);
      });
    });
  });
});