import Layer from '../layer/layer';
import Node from '../node/node';
import { config } from '../../config';
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

      describe('when applying only an explicit intent', () => {
        it('keeps the existing metadata while replacing the boundary intent', () => {
          // Arrange
          const group = new Group(3, 'input');
          group.describe({ metadata: { stage: 'encoder' } });

          // Act
          group.describe({ intent: 'gate' });

          // Assert
          expect({
            label: group.label,
            intent: group.intent,
            metadata: group.metadata,
          }).toStrictEqual({
            label: null,
            intent: 'gate',
            metadata: { size: 3, stage: 'encoder' },
          });
        });
      });
    });
  });

  describe('activate()', () => {
    describe('given no explicit value array', () => {
      describe('when activating the group', () => {
        it('delegates to each node without passing an input value', () => {
          // Arrange
          const group = new Group(2);
          jest.spyOn(group.nodes[0], 'activate').mockReturnValue(0.25);
          jest.spyOn(group.nodes[1], 'activate').mockReturnValue(0.75);

          // Act
          const activations = group.activate();

          // Assert
          expect(activations).toStrictEqual([0.25, 0.75]);
        });
      });
    });

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
    describe('given no explicit target array', () => {
      describe('when propagating through the group', () => {
        it('delegates the default propagation signature to every node', () => {
          // Arrange
          const group = new Group(2);
          const propagationSpies = group.nodes.map((node) =>
            jest.spyOn(node, 'propagate').mockImplementation(() => undefined),
          );

          // Act
          group.propagate(0.1, 0.9);

          // Assert
          expect(propagationSpies.map((spy) => spy.mock.calls)).toStrictEqual([
            [[0.1, 0.9, true, 0]],
            [[0.1, 0.9, true, 0]],
          ]);
        });
      });
    });

    describe('given a target value for each node', () => {
      describe('when propagating through the group', () => {
        it('passes the aligned target value into every node propagation call', () => {
          // Arrange
          const group = new Group(2);
          const propagationSpies = group.nodes.map((node) =>
            jest.spyOn(node, 'propagate').mockImplementation(() => undefined),
          );

          // Act
          group.propagate(0.1, 0.9, [0.3, 0.7]);

          // Assert
          expect(propagationSpies.map((spy) => spy.mock.calls)).toStrictEqual([
            [[0.1, 0.9, true, 0, 0.3]],
            [[0.1, 0.9, true, 0, 0.7]],
          ]);
        });
      });
    });

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
    describe('given another group and no explicit wiring pattern', () => {
      describe('when warnings are disabled', () => {
        it('defaults to all-to-all wiring without emitting a warning', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(2);
          const warningSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => undefined);

          // Act
          const createdConnections = runWithWarningsSetting(false, () =>
            sourceGroup.connect(targetGroup),
          );

          // Assert
          expect({
            created: createdConnections.length,
            sourceOut: sourceGroup.connections.out.length,
            targetIn: targetGroup.connections.in.length,
            warnings: warningSpy.mock.calls.length,
          }).toStrictEqual({
            created: 4,
            sourceOut: 4,
            targetIn: 4,
            warnings: 0,
          });

          warningSpy.mockRestore();
        });
      });

      describe('when warnings are enabled', () => {
        it('announces the default all-to-all fallback', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(2);
          const warningSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => undefined);

          // Act
          const createdConnections = runWithWarningsSetting(true, () =>
            sourceGroup.connect(targetGroup),
          );

          // Assert
          expect({
            created: createdConnections.length,
            warningMessages: warningSpy.mock.calls.map(([message]) => message),
          }).toStrictEqual({
            created: 4,
            warningMessages: [
              'No group connection specified, using ALL_TO_ALL by default.',
            ],
          });

          warningSpy.mockRestore();
        });
      });
    });

    describe('given the same group and no explicit wiring pattern', () => {
      describe('when warnings are disabled', () => {
        it('defaults to self one-to-one wiring without emitting a warning', () => {
          // Arrange
          const group = new Group(2);
          const warningSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => undefined);

          // Act
          const createdConnections = runWithWarningsSetting(false, () =>
            group.connect(group),
          );

          // Assert
          expect({
            created: createdConnections.length,
            selfConnections: group.connections.self.length,
            warnings: warningSpy.mock.calls.length,
          }).toStrictEqual({
            created: 2,
            selfConnections: 2,
            warnings: 0,
          });

          warningSpy.mockRestore();
        });
      });

      describe('when warnings are enabled', () => {
        it('announces the default self one-to-one fallback', () => {
          // Arrange
          const group = new Group(2);
          const warningSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => undefined);

          // Act
          const createdConnections = runWithWarningsSetting(true, () =>
            group.connect(group),
          );

          // Assert
          expect({
            created: createdConnections.length,
            warningMessages: warningSpy.mock.calls.map(([message]) => message),
          }).toStrictEqual({
            created: 2,
            warningMessages: [
              'Connecting group to itself, using ONE_TO_ONE by default.',
            ],
          });

          warningSpy.mockRestore();
        });
      });
    });

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

    describe('given the same group', () => {
      describe('when using ALL_TO_ELSE wiring', () => {
        it('skips self-connections while wiring every other node pair', () => {
          // Arrange
          const group = new Group(3);

          // Act
          const createdConnections = group.connect(
            group,
            methods.groupConnection.ALL_TO_ELSE,
          );

          // Assert
          expect({
            created: createdConnections.length,
            incoming: group.connections.in.length,
            outgoing: group.connections.out.length,
            self: group.connections.self.length,
          }).toStrictEqual({
            created: 6,
            incoming: 6,
            outgoing: 6,
            self: 0,
          });
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

    describe('given another group and an unsupported wiring pattern', () => {
      describe('when connecting the groups', () => {
        it('returns no created connections', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(2);

          // Act
          const createdConnections = sourceGroup.connect(
            targetGroup,
            'unsupported-pattern',
          );

          // Assert
          expect(createdConnections).toStrictEqual([]);
        });
      });
    });

    describe('given an unsupported runtime target', () => {
      describe('when connecting the group', () => {
        it('returns no created connections', () => {
          // Arrange
          const sourceGroup = new Group(2);

          // Act
          const createdConnections = sourceGroup.connect({} as unknown as Node);

          // Assert
          expect(createdConnections).toStrictEqual([]);
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

    describe('given one ordinary connection', () => {
      describe('when using INPUT gating', () => {
        it('accepts a single connection without wrapping it in an array first', () => {
          // Arrange
          const gatingGroup = new Group(2);
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const connection = sourceNode.connect(targetNode)[0];

          // Act
          gatingGroup.gate(connection, methods.gating.INPUT);

          // Assert
          expect(connection.gater).toBe(gatingGroup.nodes[0]);
        });
      });
    });

    describe('given two selected outgoing connections from one source node', () => {
      describe('when using OUTPUT gating', () => {
        it('gates only the selected outgoing connections for that source node', () => {
          // Arrange
          const gatingGroup = new Group(1);
          const sourceNode = new Node('hidden');
          const targetNodeOne = new Node('hidden');
          const targetNodeTwo = new Node('hidden');
          const targetNodeThree = new Node('hidden');
          const connectionOne = sourceNode.connect(targetNodeOne)[0];
          const connectionTwo = sourceNode.connect(targetNodeTwo)[0];
          const connectionThree = sourceNode.connect(targetNodeThree)[0];

          // Act
          gatingGroup.gate(
            [connectionOne, connectionTwo],
            methods.gating.OUTPUT,
          );

          // Assert
          expect([
            connectionOne.gater,
            connectionTwo.gater,
            connectionThree.gater,
          ]).toStrictEqual([gatingGroup.nodes[0], gatingGroup.nodes[0], null]);
        });
      });
    });

    describe('given self-connections from two source nodes', () => {
      describe('when using SELF gating', () => {
        it('assigns one group node to each gated self-connection', () => {
          // Arrange
          const gatingGroup = new Group(2);
          const sourceNodeOne = new Node('hidden');
          const sourceNodeTwo = new Node('hidden');
          const selfConnectionOne = sourceNodeOne.connect(sourceNodeOne)[0];
          const selfConnectionTwo = sourceNodeTwo.connect(sourceNodeTwo)[0];

          // Act
          gatingGroup.gate(
            [selfConnectionOne, selfConnectionTwo],
            methods.gating.SELF,
          );

          // Assert
          expect([
            selfConnectionOne.gater,
            selfConnectionTwo.gater,
          ]).toStrictEqual([gatingGroup.nodes[0], gatingGroup.nodes[1]]);
        });
      });
    });

    describe('given a legacy single self-connection shape', () => {
      describe('when using SELF gating', () => {
        it('accepts the non-array self-connection form', () => {
          // Arrange
          const gatingGroup = new Group(1);
          const sourceNode = new Node('hidden');
          const selfConnection = sourceNode.connect(sourceNode)[0];
          (sourceNode.connections as { self: unknown }).self = selfConnection;

          // Act
          gatingGroup.gate(selfConnection, methods.gating.SELF);

          // Assert
          expect(selfConnection.gater).toBe(gatingGroup.nodes[0]);
        });
      });
    });

    describe('given one ordinary connection from a node that also has a self-connection', () => {
      describe('when using SELF gating', () => {
        it('leaves the self-connection ungated when it is not part of the selected set', () => {
          // Arrange
          const gatingGroup = new Group(1);
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const selfConnection = sourceNode.connect(sourceNode)[0];
          const outgoingConnection = sourceNode.connect(targetNode)[0];

          // Act
          gatingGroup.gate(outgoingConnection, methods.gating.SELF);

          // Assert
          expect(selfConnection.gater).toBeNull();
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

    describe('given no shared property overrides', () => {
      describe('when applying them to the group', () => {
        it('leaves node configuration and group intent unchanged', () => {
          // Arrange
          const group = new Group(2, 'output');
          const baselineNodeState = group.nodes.map((node) => ({
            bias: node.bias,
            squash: node.squash,
            type: node.type,
          }));

          // Act
          group.set({});

          // Assert
          expect({
            intent: group.intent,
            nodeState: group.nodes.map((node) => ({
              bias: node.bias,
              squash: node.squash,
              type: node.type,
            })),
          }).toStrictEqual({
            intent: 'output',
            nodeState: baselineNodeState,
          });
        });
      });
    });
  });

  describe('disconnect()', () => {
    describe('given an outgoing connection from one group to another', () => {
      describe('when disconnecting the target group one-sided', () => {
        it('removes the shared group-level connection bookkeeping from both sides', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetGroup = new Group(2);
          sourceGroup.connect(targetGroup, methods.groupConnection.ALL_TO_ALL);

          // Act
          sourceGroup.disconnect(targetGroup);

          // Assert
          expect({
            sourceOut: sourceGroup.connections.out.length,
            targetIn: targetGroup.connections.in.length,
          }).toStrictEqual({
            sourceOut: 0,
            targetIn: 0,
          });
        });
      });
    });

    describe('given reciprocal one-to-one group connections', () => {
      describe('when disconnecting the target group two-sided', () => {
        it('removes incoming and outgoing group bookkeeping on both sides', () => {
          // Arrange
          const sourceGroup = new Group(1);
          const targetGroup = new Group(1);
          sourceGroup.connect(targetGroup, methods.groupConnection.ONE_TO_ONE);
          targetGroup.connect(sourceGroup, methods.groupConnection.ONE_TO_ONE);

          // Act
          sourceGroup.disconnect(targetGroup, true);

          // Assert
          expect({
            sourceIn: sourceGroup.connections.in.length,
            sourceOut: sourceGroup.connections.out.length,
            targetIn: targetGroup.connections.in.length,
            targetOut: targetGroup.connections.out.length,
          }).toStrictEqual({
            sourceIn: 0,
            sourceOut: 0,
            targetIn: 0,
            targetOut: 0,
          });
        });
      });
    });

    describe('given reciprocal one-to-one group connections plus unrelated bookkeeping entries', () => {
      describe('when disconnecting the target group two-sided', () => {
        it('preserves unrelated bookkeeping entries while removing the matched reciprocal links', () => {
          // Arrange
          const sourceGroup = new Group(1);
          const targetGroup = new Group(1);
          sourceGroup.connect(targetGroup, methods.groupConnection.ONE_TO_ONE);
          targetGroup.connect(sourceGroup, methods.groupConnection.ONE_TO_ONE);

          const extraInboundSource = new Node('hidden');
          const extraInboundConnection = extraInboundSource.connect(
            sourceGroup.nodes[0],
          )[0];
          sourceGroup.connections.in.push(extraInboundConnection);

          const extraTargetOutputConnection = targetGroup.nodes[0].connect(
            new Node('hidden'),
          )[0];
          targetGroup.connections.out.push(extraTargetOutputConnection);

          const extraTargetInputSource = new Node('hidden');
          const extraTargetInputConnection = extraTargetInputSource.connect(
            targetGroup.nodes[0],
          )[0];
          targetGroup.connections.in.push(extraTargetInputConnection);

          // Act
          sourceGroup.disconnect(targetGroup, true);

          // Assert
          expect({
            sourceIn: sourceGroup.connections.in,
            sourceOut: sourceGroup.connections.out,
            targetIn: targetGroup.connections.in,
            targetOut: targetGroup.connections.out,
          }).toStrictEqual({
            sourceIn: [extraInboundConnection],
            sourceOut: [],
            targetIn: [extraTargetInputConnection],
            targetOut: [extraTargetOutputConnection],
          });
        });
      });
    });

    describe('given an outgoing connection from the group to one node', () => {
      describe('when disconnecting that node one-sided', () => {
        it('removes the outgoing group bookkeeping', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetNode = new Node('hidden');
          sourceGroup.connect(targetNode);

          // Act
          sourceGroup.disconnect(targetNode);

          // Assert
          expect(sourceGroup.connections.out).toStrictEqual([]);
        });
      });
    });

    describe('given reciprocal connections between one node and the group', () => {
      describe('when disconnecting that node two-sided', () => {
        it('removes the incoming and outgoing group bookkeeping', () => {
          // Arrange
          const sourceGroup = new Group(2);
          const targetNode = new Node('hidden');
          sourceGroup.connect(targetNode);
          const inboundConnections = targetNode.connect(sourceGroup);
          sourceGroup.connections.in.push(...inboundConnections);

          // Act
          sourceGroup.disconnect(targetNode, true);

          // Assert
          expect({
            incoming: sourceGroup.connections.in.length,
            outgoing: sourceGroup.connections.out.length,
            targetOutgoing: targetNode.connections.out.length,
          }).toStrictEqual({
            incoming: 0,
            outgoing: 0,
            targetOutgoing: 0,
          });
        });
      });
    });

    describe('given an unsupported runtime target', () => {
      describe('when disconnecting the group', () => {
        it('leaves the existing bookkeeping unchanged', () => {
          // Arrange
          const sourceGroup = new Group(1);
          const targetNode = new Node('hidden');
          sourceGroup.connect(targetNode);

          // Act
          sourceGroup.disconnect({} as unknown as Node);

          // Assert
          expect(sourceGroup.connections.out).toHaveLength(1);
        });
      });
    });
  });

  describe('clear()', () => {
    describe('given a node with dynamic activation, trace, and gating state', () => {
      describe('when clearing the group', () => {
        it('resets the tracked runtime state on every node', () => {
          // Arrange
          const group = new Group(1);
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const inboundConnection = sourceNode.connect(group.nodes[0])[0];
          const selfConnection = group.nodes[0].connect(group.nodes[0])[0];
          const gatedConnection = group.nodes[0].connect(targetNode)[0];

          inboundConnection.eligibility = 0.4;
          selfConnection.eligibility = 0.6;
          group.nodes[0].gate(gatedConnection);
          group.nodes[0].activation = 1;
          group.nodes[0].state = 2;
          group.nodes[0].old = 3;
          group.nodes[0].error = { gated: 1, projected: 1, responsibility: 1 };

          // Act
          group.clear();

          // Assert
          expect({
            activation: group.nodes[0].activation,
            gatedGain: gatedConnection.gain,
            inboundEligibility: inboundConnection.eligibility,
            old: group.nodes[0].old,
            projectedError: group.nodes[0].error.projected,
            responsibilityError: group.nodes[0].error.responsibility,
            selfEligibility: selfConnection.eligibility,
            state: group.nodes[0].state,
          }).toStrictEqual({
            activation: 0,
            gatedGain: 0,
            inboundEligibility: 0,
            old: 0,
            projectedError: 0,
            responsibilityError: 0,
            selfEligibility: 0,
            state: 0,
          });
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

function runWithWarningsSetting<T>(warnings: boolean, action: () => T): T {
  const previousWarnings = config.warnings;
  config.warnings = warnings;

  try {
    return action();
  } finally {
    config.warnings = previousWarnings;
  }
}

function captureErrorMessage(runAction: () => unknown): string | undefined {
  try {
    runAction();
  } catch (error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }

  return undefined;
}
