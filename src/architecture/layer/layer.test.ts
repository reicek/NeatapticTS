import Connection from '../connection/connection';
import Group from '../group/group';
import * as methods from '../../methods/methods';
import Node from '../node/node';
import Layer from './layer';

describe('Layer', () => {
  describe('constructor', () => {
    describe('given a fresh layer', () => {
      describe('when reading the default output group', () => {
        it('starts as null', () => {
          // Arrange
          const layer = new Layer();

          // Act
          const actualOutput = layer.output;

          // Assert
          expect(actualOutput).toBeNull();
        });
      });

      describe('when reading the node list', () => {
        it('starts empty', () => {
          // Arrange
          const layer = new Layer();

          // Act
          const actualNodes = layer.nodes;

          // Assert
          expect(actualNodes).toStrictEqual([]);
        });
      });
    });
  });

  describe('activate()', () => {
    describe('given a hidden layer with three nodes', () => {
      describe('when activating it with explicit values', () => {
        it('returns one activation per node', () => {
          // Arrange
          const layer = createTestLayer(3, 'hidden');

          // Act
          const activations = layer.activate([0.5, -0.2, 0.9]);

          // Assert
          expect(activations).toHaveLength(3);
        });
      });
    });
  });

  describe('dense()', () => {
    describe('given an explicit primitive role', () => {
      describe('when creating a dense layer', () => {
        it('allocates nodes with that role', () => {
          // Arrange
          const requestedRole = 'output';

          // Act
          const layer = Layer.dense(3, requestedRole);

          // Assert
          expect(layer.nodes.map((node) => node.type)).toStrictEqual([
            requestedRole,
            requestedRole,
            requestedRole,
          ]);
        });

        it('describes the dense block with role-aware intent metadata', () => {
          // Arrange
          const requestedRole = 'output';

          // Act
          const layer = Layer.dense(3, requestedRole);

          // Assert
          expect({
            intent: layer.intent,
            metadata: layer.metadata,
          }).toStrictEqual({
            intent: requestedRole,
            metadata: { family: 'dense', size: 3 },
          });
        });
      });
    });
  });

  describe('memory()', () => {
    describe('given a fixed memory shape', () => {
      describe('when creating the layer', () => {
        it('describes the delay-line boundary with metadata', () => {
          // Arrange
          const blockSize = 2;
          const memorySteps = 3;

          // Act
          const layer = Layer.memory(blockSize, memorySteps);

          // Assert
          expect({
            intent: layer.intent,
            metadata: layer.metadata,
          }).toStrictEqual({
            intent: 'memory',
            metadata: { family: 'memory', memorySteps, size: blockSize },
          });
        });
      });
    });
  });

  describe('propagate()', () => {
    describe('given a mismatched target array', () => {
      describe('when propagating through the layer', () => {
        it('throws the size-mismatch error', () => {
          // Arrange
          const layer = createTestLayer(3, 'hidden');
          const runPropagation = () => layer.propagate(0.1, 0.9, [0.1, 0.2]);

          // Act
          const thrownMessage = captureErrorMessage(runPropagation);

          // Assert
          expect(thrownMessage).toBe(
            'Array with values should be same as the amount of nodes!',
          );
        });
      });
    });
  });

  describe('connect()', () => {
    describe('given a layer with no output group', () => {
      describe('when connecting to a group', () => {
        it('throws the missing-output error', () => {
          // Arrange
          const sourceLayer = new Layer();
          const targetGroup = new Group(2);
          const runConnection = () => sourceLayer.connect(targetGroup);

          // Act
          const thrownMessage = captureErrorMessage(runConnection);

          // Assert
          expect(thrownMessage).toBe(
            'Layer output is not defined. Cannot connect from this layer.',
          );
        });
      });
    });

    describe('given a source layer with an output group', () => {
      describe('when connecting to a group target', () => {
        it('returns the connections created by the output group', () => {
          // Arrange
          const sourceLayer = createTestLayer(3, 'hidden');
          const targetGroup = new Group(2);

          // Act
          const createdConnections = sourceLayer.connect(
            targetGroup,
            methods.groupConnection.ALL_TO_ALL,
          );

          // Assert
          expect(createdConnections).toHaveLength(6);
        });
      });

      describe('when connecting to another layer', () => {
        it('delegates through the target layer input surface', () => {
          // Arrange
          const sourceLayer = createTestLayer(2, 'hidden');
          const targetLayer = createTestLayer(2, 'hidden');
          const inputSpy = jest.spyOn(targetLayer, 'input');

          // Act
          sourceLayer.connect(targetLayer, methods.groupConnection.ONE_TO_ONE);

          // Assert
          expect(inputSpy).toHaveBeenCalledTimes(1);
          inputSpy.mockRestore();
        });
      });
    });
  });

  describe('gate()', () => {
    describe('given a layer with no output group', () => {
      describe('when gating connections', () => {
        it('throws the missing-output error', () => {
          // Arrange
          const layer = new Layer();
          const connections = [
            new Connection(new Node('hidden'), new Node('hidden')),
          ];
          const runGate = () => layer.gate(connections, methods.gating.INPUT);

          // Act
          const thrownMessage = captureErrorMessage(runGate);

          // Assert
          expect(thrownMessage).toBe(
            'Layer output is not defined. Cannot gate from this layer.',
          );
        });
      });
    });
  });

  describe('set()', () => {
    describe('given a layer with ordinary nodes', () => {
      describe('when applying a shared bias', () => {
        it('updates every node in the layer', () => {
          // Arrange
          const layer = createTestLayer(3, 'hidden');
          layer.set({ bias: 0.7 });

          // Act
          const everyNodeMatches = layer.nodes.every(
            (node) => node.bias === 0.7,
          );

          // Assert
          expect(everyNodeMatches).toBe(true);
        });
      });
    });
  });
});

function createTestLayer(size: number, type: string): Layer {
  const layer = new Layer();

  for (let nodeIndex = 0; nodeIndex < size; nodeIndex++) {
    layer.nodes.push(new Node(type));
  }

  layer.output = new Group(size);
  return layer;
}

function captureErrorMessage(runAction: () => unknown): string | undefined {
  try {
    runAction();
  } catch (error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }

  return undefined;
}
