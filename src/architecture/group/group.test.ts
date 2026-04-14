import Layer from '../layer/layer';
import Node from '../node/node';
import * as methods from '../../methods/methods';
import Group from './group';

interface GroupJsonShape {
  size: number;
  nodeIndices: Array<number | undefined>;
  connections: {
    in: number;
    out: number;
    self: number;
  };
}

describe('Group', () => {
  describe('constructor', () => {
    describe('given a requested size', () => {
      describe('when creating the group', () => {
        it('creates that many nodes', () => {
          // Arrange
          const requestedSize = 5;

          // Act
          const group = new Group(requestedSize);

          // Assert
          expect(group.nodes).toHaveLength(requestedSize);
        });

        it('starts with empty outgoing group connections', () => {
          // Arrange
          const requestedSize = 5;

          // Act
          const group = new Group(requestedSize);

          // Assert
          expect(group.connections.out).toStrictEqual([]);
        });
      });
    });

    describe('given an explicit node role', () => {
      describe('when creating the group', () => {
        it('allocates nodes with that role', () => {
          // Arrange
          const requestedRole = 'input';

          // Act
          const group = new Group(3, requestedRole);

          // Assert
          expect(group.nodes.map((node) => node.type)).toStrictEqual([
            requestedRole,
            requestedRole,
            requestedRole,
          ]);
        });
      });
    });
  });

  describe('describe()', () => {
    describe('given a role-aware group', () => {
      describe('when applying label and scalar metadata', () => {
        it('stores additive boundary metadata on the group', () => {
          // Arrange
          const group = new Group(3, 'input');

          // Act
          group.describe({
            label: 'sensorBlock',
            metadata: { stage: 1, reusable: true },
          });

          // Assert
          expect({
            label: group.label,
            intent: group.intent,
            metadata: group.metadata,
          }).toStrictEqual({
            label: 'sensorBlock',
            intent: 'input',
            metadata: { size: 3, stage: 1, reusable: true },
          });
        });
      });
    });
  });

  describe('activate()', () => {
    describe('given an input-sized value array', () => {
      describe('when activating the group', () => {
        it('returns one activation per node', () => {
          // Arrange
          const group = new Group(3);
          const inputValues = [0.5, -0.2, 0.9];
          group.nodes.forEach((node) => {
            node.type = 'input';
          });

          // Act
          const activations = group.activate(inputValues);

          // Assert
          expect(activations).toHaveLength(3);
        });
      });
    });

    describe('given a mismatched value array', () => {
      describe('when activating the group', () => {
        it('throws the size-mismatch error', () => {
          // Arrange
          const group = new Group(3);
          const runActivation = () => group.activate([1, 2]);

          // Act
          const thrownMessage = captureErrorMessage(runActivation);

          // Assert
          expect(thrownMessage).toBe(
            'Array with values should be same as the amount of nodes!',
          );
        });
      });
    });
  });

  describe('propagate()', () => {
    describe('given a mismatched target array', () => {
      describe('when propagating through the group', () => {
        it('throws the size-mismatch error', () => {
          // Arrange
          const group = new Group(3);
          const runPropagation = () => group.propagate(0.1, 0.9, [0.1, 0.2]);

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
    describe('given another same-sized group', () => {
      describe('when using ONE_TO_ONE wiring', () => {
        it('creates one connection per source node', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(2);

          // Act
          const createdConnections = sourceGroup.connect(
            targetGroup,
            methods.groupConnection.ONE_TO_ONE,
          );

          // Assert
          expect(createdConnections).toHaveLength(2);
        });
      });
    });

    describe('given a different-sized group', () => {
      describe('when using ONE_TO_ONE wiring', () => {
        it('throws the one-to-one size mismatch error', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(3);
          const runConnection = () =>
            sourceGroup.connect(
              targetGroup,
              methods.groupConnection.ONE_TO_ONE,
            );

          // Act
          const thrownMessage = captureErrorMessage(runConnection);

          // Assert
          expect(thrownMessage).toBe(
            'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.',
          );
        });
      });
    });

    describe('given a node target', () => {
      describe('when connecting the group to that node', () => {
        it('creates one connection per group node', () => {
          // Arrange
          const sourceGroup = new Group(3);
          const targetNode = new Node('hidden');

          // Act
          const createdConnections = sourceGroup.connect(targetNode);

          // Assert
          expect(createdConnections).toHaveLength(3);
        });
      });
    });

    describe('given a layer target', () => {
      describe('when connecting into that layer', () => {
        it('delegates to the layer input surface', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetLayer = createTestLayer(2);
          const inputSpy = jest.spyOn(targetLayer, 'input');

          // Act
          sourceGroup.connect(targetLayer);

          // Assert
          expect(inputSpy).toHaveBeenCalledTimes(1);
          inputSpy.mockRestore();
        });
      });
    });
  });

  describe('gate()', () => {
    describe('given no gating method', () => {
      describe('when gating a connection', () => {
        it('throws the gating-method-required error', () => {
          // Arrange
          const gatingGroup = new Group(2);
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const connection = sourceNode.connect(targetNode)[0];
          const runGate = () => gatingGroup.gate(connection, undefined);

          // Act
          const thrownMessage = captureErrorMessage(runGate);

          // Assert
          expect(thrownMessage).toBe(
            'Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF',
          );
        });
      });
    });

    describe('given two ordinary connections', () => {
      describe('when using INPUT gating', () => {
        it('cycles the group nodes as gaters', () => {
          // Arrange
          const gatingGroup = new Group(2);
          const sourceNodeOne = new Node('hidden');
          const sourceNodeTwo = new Node('hidden');
          const targetNodeOne = new Node('hidden');
          const targetNodeTwo = new Node('hidden');
          const connectionOne = sourceNodeOne.connect(targetNodeOne)[0];
          const connectionTwo = sourceNodeTwo.connect(targetNodeTwo)[0];
          gatingGroup.gate(
            [connectionOne, connectionTwo],
            methods.gating.INPUT,
          );

          // Act
          const actualGaters = [connectionOne.gater, connectionTwo.gater];

          // Assert
          expect(actualGaters).toStrictEqual([
            gatingGroup.nodes[0],
            gatingGroup.nodes[1],
          ]);
        });
      });
    });
  });

  describe('set()', () => {
    describe('given shared bias, squash, and type values', () => {
      describe('when applying them to the group', () => {
        it('updates every node in the group', () => {
          // Arrange
          const group = new Group(3);
          group.set({
            bias: 0.5,
            squash: methods.Activation.relu,
            type: 'output',
          });

          // Act
          const everyNodeMatches = group.nodes.every(
            (node) =>
              node.bias === 0.5 &&
              node.squash === methods.Activation.relu &&
              node.type === 'output',
          );

          // Assert
          expect(everyNodeMatches).toBe(true);
        });
      });
    });
  });

  describe('toJSON()', () => {
    describe('given a group with one outgoing connection', () => {
      describe('when serializing the group', () => {
        it('reports the current size, node indices, and connection counts', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetNode = new Node('hidden');
          sourceGroup.connect(targetNode);
          const expectedJson: GroupJsonShape = {
            size: 2,
            nodeIndices: sourceGroup.nodes.map((node) => node.index),
            connections: {
              in: 0,
              out: 2,
              self: 0,
            },
          };

          // Act
          const actualJson = sourceGroup.toJSON();

          // Assert
          expect(actualJson).toStrictEqual(expectedJson);
        });
      });
    });
  });
});

function createTestLayer(size: number): Layer {
  const layer = new Layer();

  for (let nodeIndex = 0; nodeIndex < size; nodeIndex++) {
    layer.nodes.push(new Node('hidden'));
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
