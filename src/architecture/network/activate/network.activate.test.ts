import Network from '../network';
import Node from '../../node';
import {
  activateBatch as activateBatchUtils,
  activateRaw as activateRawUtils,
} from './network.activate.utils';

type FastSlabActivate = (input: number[]) => number[];
type CanUseFastSlab = () => boolean;
type ActivateBatch = (inputs: unknown) => unknown;

function setFastSlabHooks(
  network: Network,
  hooks: {
    canUseFastSlab?: CanUseFastSlab;
    fastSlabActivate?: FastSlabActivate;
  },
): void {
  if (hooks.canUseFastSlab) {
    Reflect.set(network, '_canUseFastSlab', hooks.canUseFastSlab);
  }

  if (hooks.fastSlabActivate) {
    Reflect.set(network, '_fastSlabActivate', hooks.fastSlabActivate);
  }
}

function invokeActivateBatch(network: Network, inputs: unknown): unknown {
  const activateBatch = Reflect.get(network, 'activateBatch') as ActivateBatch;
  return activateBatch.call(network, inputs);
}

function setTopoDirty(network: Network, isDirty: boolean): void {
  Reflect.set(network, '_topoDirty', isDirty);
}

function configureDeterministicNode(node: Node): void {
  node.bias = 0;
  node.squash = createIdentityActivation();
}

function createIdentityActivation(): (
  value: number,
  derivate?: boolean,
) => number {
  return (value, derivate = false) => (derivate ? 1 : value);
}

function runActivationMode(
  network: Network,
  inputVector: number[],
  activationMode: 'activate' | 'noTrace',
): number[] {
  return activationMode === 'activate'
    ? network.activate(inputVector)
    : network.noTraceActivate(inputVector);
}

function createAcyclicScheduleAdoptionScenario(
  activationMode: 'activate' | 'noTrace',
): { network: Network; inputVector: number[]; expectedOutput: number[] } {
  const network = new Network(2, 2, {
    seed: 210,
    enforceAcyclic: true,
  });
  const inputNodes = network.nodes.filter((nodeEntry) => nodeEntry.type === 'input');
  const outputNodes = network.nodes.filter((nodeEntry) => nodeEntry.type === 'output');

  if (inputNodes.length !== 2 || outputNodes.length !== 2) {
    throw new Error('Expected two input nodes and two output nodes');
  }

  setFastSlabHooks(network, {
    canUseFastSlab: () => false,
  });

  const hiddenNode = new Node('hidden');
  configureDeterministicNode(hiddenNode);
  outputNodes.forEach(configureDeterministicNode);

  network.connections.slice().forEach((connection) => {
    network.disconnect(connection.from, connection.to);
  });

  network.nodes = [
    inputNodes[0],
    inputNodes[1],
    hiddenNode,
    outputNodes[0],
    outputNodes[1],
  ];

  network.connect(inputNodes[0], hiddenNode)[0].weight = 2;
  network.connect(inputNodes[1], hiddenNode)[0].weight = 3;
  network.connect(hiddenNode, outputNodes[0])[0].weight = 4;
  network.connect(hiddenNode, outputNodes[1])[0].weight = -1;

  const inputVector = [2, 1];
  network.clear();
  const expectedOutput = [...runActivationMode(network, inputVector, activationMode)];

  network.clear();
  network.nodes = [
    outputNodes[1],
    inputNodes[1],
    hiddenNode,
    outputNodes[0],
    inputNodes[0],
  ];
  setTopoDirty(network, true);

  return {
    network,
    inputVector,
    expectedOutput,
  };
}

function createRecurrentScheduleAdoptionScenario(
  activationMode: 'activate' | 'noTrace',
): { network: Network; inputVector: number[]; expectedOutput: number[] } {
  const network = new Network(1, 1, {
    seed: 211,
    enforceAcyclic: false,
  });
  const inputNode = network.nodes.find((nodeEntry) => nodeEntry.type === 'input');
  const outputNode = network.nodes.find((nodeEntry) => nodeEntry.type === 'output');

  if (!inputNode || !outputNode) {
    throw new Error('Expected one input node and one output node');
  }

  setFastSlabHooks(network, {
    canUseFastSlab: () => false,
  });

  const hiddenNode = new Node('hidden');
  configureDeterministicNode(hiddenNode);
  configureDeterministicNode(outputNode);

  network.connections.slice().forEach((connection) => {
    network.disconnect(connection.from, connection.to);
  });

  network.nodes = [inputNode, hiddenNode, outputNode];

  network.connect(inputNode, hiddenNode)[0].weight = 2;
  network.connect(hiddenNode, hiddenNode)[0].weight = 1;
  network.connect(hiddenNode, outputNode)[0].weight = 3;

  const inputVector = [2];
  network.clear();
  const expectedOutput = [...runActivationMode(network, inputVector, activationMode)];

  network.clear();
  network.nodes = [inputNode, outputNode, hiddenNode];
  setTopoDirty(network, true);

  return {
    network,
    inputVector,
    expectedOutput,
  };
}

describe('network activate chapter', () => {
  describe('activate()', () => {
    describe('given the input width is too small', () => {
      describe('when activation starts', () => {
        it('throws an input size mismatch error', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 10 });

          // Act
          const activateWithInvalidInput = () => network.activate([1]);

          // Assert
          expect(activateWithInvalidInput).toThrow(/Input size mismatch/);
        });
      });
    });

    describe('given the input width is too large', () => {
      describe('when activation starts', () => {
        it('throws an input size mismatch error', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 10 });

          // Act
          const activateWithInvalidInput = () => network.activate([1, 2, 3]);

          // Assert
          expect(activateWithInvalidInput).toThrow(/Input size mismatch/);
        });
      });
    });

    describe('given the input width matches the network', () => {
      describe('when activation completes', () => {
        it('returns the configured output width', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 10 });
          const expectedLength = 1;

          // Act
          const output = network.activate([1, 2]);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });

      describe('when node storage order drifts away from the compiled acyclic schedule', () => {
        it('still follows explicit IO roles and scheduled output order', () => {
          // Arrange
          const activationScenario = createAcyclicScheduleAdoptionScenario(
            'activate',
          );

          // Act
          const output = activationScenario.network.activate(
            activationScenario.inputVector,
          );

          // Assert
          expect(output).toEqual(activationScenario.expectedOutput);
        });
      });

      describe('when recurrent node storage order drifts away from the compiled schedule', () => {
        it('still follows the recurrent component order', () => {
          // Arrange
          const activationScenario = createRecurrentScheduleAdoptionScenario(
            'activate',
          );

          // Act
          const output = activationScenario.network.activate(
            activationScenario.inputVector,
          );

          // Assert
          expect(output).toEqual(activationScenario.expectedOutput);
        });
      });
    });

    describe('given the graph contains a cycle', () => {
      describe('when an output node gates itself directly', () => {
        it('does not throw during activation', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 19 });
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );
          if (!outputNode) {
            throw new Error('Expected an output node to exist');
          }

          network.connect(outputNode, outputNode);
          const activateWithSelfCycle = () => network.activate([0.5]);

          // Assert
          expect(activateWithSelfCycle).not.toThrow();
        });
      });

      describe('when a hidden node and the output node form a two-node cycle', () => {
        it('does not throw during activation', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 20 });
          network.mutate('ADD_NODE');
          const hiddenNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'hidden',
          );
          const outputNode = network.nodes.find(
            (nodeEntry) => nodeEntry.type === 'output',
          );

          if (!hiddenNode || !outputNode) {
            throw new Error('Expected hidden and output nodes to exist');
          }

          if (!hiddenNode.isProjectingTo(outputNode)) {
            network.connect(hiddenNode, outputNode);
          }

          network.connect(outputNode, hiddenNode);
          const activateWithTwoNodeCycle = () => network.activate([0.1, 0.2]);

          // Assert
          expect(activateWithTwoNodeCycle).not.toThrow();
        });
      });
    });
  });

  describe('noTraceActivate()', () => {
    describe('given fast slab execution is available', () => {
      describe('when the fast slab path succeeds', () => {
        it('returns the fast slab output value', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 11,
            enforceAcyclic: true,
          });
          const expectedValue = 1;

          setFastSlabHooks(network, {
            canUseFastSlab: () => true,
            fastSlabActivate: (input) => [input[0] + input[1]],
          });

          // Act
          const output = network.noTraceActivate([0.25, 0.75]);

          // Assert
          expect(output[0]).toBe(expectedValue);
        });
      });

      describe('when the fast slab path throws', () => {
        it('falls back to the standard loop output width', () => {
          // Arrange
          const network = new Network(3, 2, {
            seed: 12,
            enforceAcyclic: true,
          });
          const expectedLength = 2;

          setFastSlabHooks(network, {
            canUseFastSlab: () => true,
            fastSlabActivate: () => {
              throw new Error('forced');
            },
          });

          // Act
          const output = network.noTraceActivate([0.1, 0.2, 0.3]);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });
    });

    describe('given the input width is invalid', () => {
      describe('when activation starts', () => {
        it('throws an input size mismatch error', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 13 });

          // Act
          const activateWithInvalidInput = () =>
            network.noTraceActivate([1, 2, 3]);

          // Assert
          expect(activateWithInvalidInput).toThrow(/Input size mismatch/);
        });
      });
    });

    describe('given the input width matches the network', () => {
      describe('when activation completes', () => {
        it('returns the configured output width', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 13 });
          const expectedLength = 1;

          // Act
          const output = network.noTraceActivate([1, 2]);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });

      describe('when node storage order drifts away from the compiled acyclic schedule', () => {
        it('still follows explicit IO roles and scheduled output order', () => {
          // Arrange
          const activationScenario = createAcyclicScheduleAdoptionScenario(
            'noTrace',
          );

          // Act
          const output = activationScenario.network.noTraceActivate(
            activationScenario.inputVector,
          );

          // Assert
          expect(output).toEqual(activationScenario.expectedOutput);
        });
      });

      describe('when recurrent node storage order drifts away from the compiled schedule', () => {
        it('still follows the recurrent component order', () => {
          // Arrange
          const activationScenario = createRecurrentScheduleAdoptionScenario(
            'noTrace',
          );

          // Act
          const output = activationScenario.network.noTraceActivate(
            activationScenario.inputVector,
          );

          // Assert
          expect(output).toEqual(activationScenario.expectedOutput);
        });
      });
    });
  });

  describe('activateRaw()', () => {
    describe('given activation-array reuse is disabled', () => {
      describe('when raw activation runs', () => {
        it('returns the configured output width', () => {
          // Arrange
          const network = new Network(2, 2, { seed: 14 });
          const expectedLength = 2;

          // Act
          const output = network.activateRaw([0.4, 0.6]);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });
    });

    describe('given activation-array reuse is enabled', () => {
      describe('when raw activation runs', () => {
        it('still returns the configured output width', () => {
          // Arrange
          const network = new Network(2, 3, {
            seed: 15,
            reuseActivationArrays: true,
            returnTypedActivations: true,
          });
          const expectedLength = 3;

          // Act
          const output = network.activateRaw([0.5, 0.5]);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });
    });
  });

  describe('activateBatch()', () => {
    describe('given a valid input matrix', () => {
      describe('when multiple rows are activated', () => {
        it('returns one output row per input row', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 16 });
          const batch = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
          ];
          const expectedLength = batch.length;

          // Act
          const output = network.activateBatch(batch);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });

      describe('when a single row is activated', () => {
        it('matches the non-batch output width', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 42,
            enforceAcyclic: true,
          });
          const inputVector = [0.1, -0.2];
          const expectedLength = network.activate(inputVector).length;

          // Act
          const batchOutput = network.activateBatch([inputVector]);

          // Assert
          expect(batchOutput[0].length).toBe(expectedLength);
        });
      });

      describe('when a single row is activated for value parity', () => {
        it('matches the non-batch first output value', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 42,
            enforceAcyclic: true,
          });
          const inputVector = [0.1, -0.2];
          const expectedValue = network.activate(inputVector)[0];

          // Act
          const batchOutput = network.activateBatch([inputVector]);

          // Assert
          expect(batchOutput[0][0]).toBe(expectedValue);
        });
      });

      describe('when several rows are activated', () => {
        it('preserves the batch row count', () => {
          // Arrange
          const network = new Network(3, 2, {
            seed: 7,
            enforceAcyclic: true,
          });
          const batch = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
          ];
          const expectedLength = batch.length;

          // Act
          const output = network.activateBatch(batch);

          // Assert
          expect(output.length).toBe(expectedLength);
        });
      });
    });

    describe('given one row has the wrong width', () => {
      describe('when batch activation starts', () => {
        it('throws an error that names the failing row index', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 17 });

          // Act
          const activateWithInvalidRow = () =>
            network.activateBatch([[1, 2], [3]]);

          // Assert
          expect(activateWithInvalidRow).toThrow(/Input\[1\] size mismatch/);
        });
      });
    });

    describe('given the top-level input is not an array', () => {
      describe('when batch activation starts', () => {
        it('throws the collection-shape error', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 18 });

          // Act
          const activateWithInvalidCollection = () =>
            invokeActivateBatch(network, 'nope');

          // Assert
          expect(activateWithInvalidCollection).toThrow(
            /inputs must be an array/,
          );
        });
      });
    });
  });

  describe('activation utility defaults', () => {
    describe('given raw activation omits its optional flags', () => {
      describe('when the helper is called directly', () => {
        it('uses the default training and recursion-depth values', () => {
          // Arrange
          const network = Network.createMLP(2, [2], 1);

          // Act
          const outputValues = activateRawUtils.call(network, [0.1, 0.9]);

          // Assert
          expect(outputValues.length).toBe(1);
        });
      });
    });

    describe('given batch activation omits its optional training flag', () => {
      describe('when the helper is called directly', () => {
        it('uses the default non-training batch path', () => {
          // Arrange
          const network = Network.createMLP(2, [2], 1);

          // Act
          const batchOutput = activateBatchUtils.call(network, [[0.1, 0.9]]);

          // Assert
          expect(batchOutput.length).toBe(1);
        });
      });
    });
  });
});
