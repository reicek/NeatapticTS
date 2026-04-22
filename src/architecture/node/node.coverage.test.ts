import Connection from '../connection/connection';
import { config } from '../../config';
import Activation from '../../methods/activation/activation';
import { methods } from '../../neataptic';
import {
  NodeInvalidConnectionTargetTypeError,
  NodeUnsupportedMutationMethodError,
} from './node.errors';
import Node from './node';

function withMockedMathRandom<T>(randomValue: number, run: () => T): T {
  const mathRandomSpy = jest.spyOn(Math, 'random').mockReturnValue(randomValue);

  try {
    return run();
  } finally {
    mathRandomSpy.mockRestore();
  }
}

function withSuppressedWarnings<T>(run: (warningSpy: jest.SpyInstance) => T): T {
  const warningSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

  try {
    return run(warningSpy);
  } finally {
    warningSpy.mockRestore();
  }
}

describe('Node', () => {
  describe('coverage helpers', () => {
    describe('given the gene id counter receives a non-finite value', () => {
      describe('when syncing the counter', () => {
        it('keeps the next created gene id on the normal increment path', () => {
          // Arrange
          const baselineNode = new Node('hidden');

          // Act
          Node.syncGeneIdCounter(Number.POSITIVE_INFINITY);
          const nextNode = new Node('hidden');

          // Assert
          expect(nextNode.geneId - baselineNode.geneId).toBe(1);
        });
      });
    });

    describe('given an output node with a custom intent override', () => {
      describe('when applying the descriptor update', () => {
        it('stores the provided intent without disturbing the other descriptor fields', () => {
          // Arrange
          const node = new Node('output');

          // Act
          node.describe({ intent: 'memory' });

          // Assert
          expect({
            intent: node.intent,
            label: node.label,
            metadata: node.metadata,
          }).toStrictEqual({
            intent: 'memory',
            label: null,
            metadata: {},
          });
        });
      });
    });

    describe('given a configured hidden node', () => {
      describe('when serializing the node to JSON', () => {
        it('returns the runtime-facing primitive fields', () => {
          // Arrange
          const node = new Node('hidden');
          node.bias = 0.25;
          node.response = 1.5;
          node.squash = Activation.tanh;
          node.mask = 0.75;

          // Act
          const serializedNode = node.toJSON();

          // Assert
          expect(serializedNode).toStrictEqual({
            bias: 0.25,
            index: node.index,
            mask: 0.75,
            response: 1.5,
            squash: Activation.tanh.name,
            type: 'hidden',
          });
        });
      });
    });

    describe('given a serialized node with an unknown squash name', () => {
      describe('when restoring the node from JSON', () => {
        it('falls back to identity and the neutral response', () => {
          // Arrange
          let restoredNode: Node | null = null;

          // Act
          const restorationResult = withSuppressedWarnings((warningSpy) => {
            restoredNode = Node.fromJSON({
              bias: 0.5,
              mask: 0.25,
              squash: 'unknownSquash',
              type: 'output',
            });

            return {
              response: restoredNode.response,
              squash: restoredNode.squash,
              warningCount: warningSpy.mock.calls.length,
            };
          });

          // Assert
          expect(restorationResult).toStrictEqual({
            response: 1,
            squash: Activation.identity,
            warningCount: 1,
          });
        });
      });
    });

    describe('given a direct projection between two nodes', () => {
      describe('when checking the forward and reverse connectivity queries', () => {
        it('reports the projection from both perspectives', () => {
          // Arrange
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          sourceNode.connect(targetNode);

          // Act
          const connectivityResult = {
            isConnectedTo: sourceNode.isConnectedTo(targetNode),
            isProjectedBy: targetNode.isProjectedBy(sourceNode),
          };

          // Assert
          expect(connectivityResult).toStrictEqual({
            isConnectedTo: true,
            isProjectedBy: true,
          });
        });
      });
    });

    describe('given a node with a self-connection', () => {
      describe('when checking reverse projection from itself', () => {
        it('treats the self-loop as an incoming projection', () => {
          // Arrange
          const node = new Node('hidden');
          node.connect(node);

          // Act
          const isProjectedBySelf = node.isProjectedBy(node);

          // Assert
          expect(isProjectedBySelf).toBe(true);
        });
      });
    });

    describe('given a mutation with multiple allowed activation functions', () => {
      describe('when mutating the node activation', () => {
        it('switches the squash function to a different allowed activation', () => {
          // Arrange
          const node = new Node('hidden');
          const allowedActivations =
            methods.mutation.MOD_ACTIVATION.allowed ?? [];

          // Act
          const mutationResult = withMockedMathRandom(0, () => {
            node.mutate(methods.mutation.MOD_ACTIVATION);

            return {
              changed: node.squash !== Activation.logistic,
              remainsAllowed: allowedActivations.includes(node.squash),
            };
          });

          // Assert
          expect(mutationResult).toStrictEqual({
            changed: true,
            remainsAllowed: true,
          });
        });
      });
    });

    describe('given a node with incoming, outgoing, and self weights', () => {
      describe('when reinitializing the node weights', () => {
        it('replaces each weight inside the configured mutation range', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const incomingConnection = sourceNode.connect(node, 0)[0];
          const outgoingConnection = node.connect(targetNode, 0)[0];
          const selfConnection = node.connect(node, 0)[0];
          const reinitializeWeightsMutation = methods.mutation.REINIT_WEIGHT;
          const expectedWeight =
            0.75 *
              ((reinitializeWeightsMutation.max ?? 1) -
                (reinitializeWeightsMutation.min ?? -1)) +
            (reinitializeWeightsMutation.min ?? -1);

          // Act
          const reinitializedWeights = withMockedMathRandom(0.75, () => {
            node.mutate(reinitializeWeightsMutation);

            return {
              incoming: incomingConnection.weight,
              outgoing: outgoingConnection.weight,
              self: selfConnection.weight,
            };
          });

          // Assert
          expect(reinitializedWeights).toStrictEqual({
            incoming: expectedWeight,
            outgoing: expectedWeight,
            self: expectedWeight,
          });
        });
      });
    });

    describe('given the batch-normalization mutation', () => {
      describe('when mutating the node', () => {
        it('marks the node with the batchNorm flag', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          node.mutate(methods.mutation.BATCH_NORM);

          // Assert
          expect(
            Reflect.get(node as unknown as { batchNorm?: boolean }, 'batchNorm'),
          ).toBe(true);
        });
      });
    });

    describe('given a method-shaped object that matches a known mutation name only by metadata', () => {
      describe('when mutating the node', () => {
        it('throws the unsupported mutation error', () => {
          // Arrange
          const node = new Node('hidden');
          const unsupportedMutation = { name: 'MOD_BIAS' };

          // Act
          const applyUnsupportedMutation = () => {
            node.mutate(unsupportedMutation);
          };

          // Assert
          expect(applyUnsupportedMutation).toThrow(
            NodeUnsupportedMutationMethodError,
          );
        });
      });
    });

    describe('given a node with an invalid squash function and mask type', () => {
      describe('when activating without an explicit input', () => {
        it('falls back to identity and restores the numeric mask', () => {
          // Arrange
          const node = new Node('hidden');
          const originalWarnings = config.warnings;
          node.bias = 2;
          node.mask = 'invalid-mask' as unknown as number;
          node.squash = null as unknown as (x: number, derivate?: boolean) => number;

          // Act
          const activationResult = withSuppressedWarnings(() => {
            config.warnings = true;

            try {
              const activationValue = node.activate();

              return {
                activation: activationValue,
                derivative: node.derivative,
                mask: node.mask,
                squash: node.squash,
              };
            } finally {
              config.warnings = originalWarnings;
            }
          });

          // Assert
          expect(activationResult).toStrictEqual({
            activation: 2,
            derivative: 1,
            mask: 1,
            squash: Activation.identity,
          });
        });
      });
    });

    describe('given skipped self and incoming connections during activation', () => {
      describe('when the node activates from its stored state', () => {
        it('ignores disabled or drop-connected contributions', () => {
          // Arrange
          const node = new Node('hidden');
          const disabledSourceNode = new Node('input');
          const dropConnectedSourceNode = new Node('input');
          const disabledIncomingConnection = disabledSourceNode.connect(node, 5)[0];
          const dropConnectedIncomingConnection = dropConnectedSourceNode.connect(
            node,
            7,
          )[0];
          const selfConnection = node.connect(node, 3)[0];

          node.squash = Activation.identity;
          node.bias = 0.1;
          node.state = 4;
          disabledSourceNode.activation = 2;
          dropConnectedSourceNode.activation = 3;
          disabledIncomingConnection.enabled = false;
          dropConnectedIncomingConnection.dcMask = 0;
          selfConnection.dcMask = 0;

          // Act
          const activationValue = node.activate();

          // Assert
          expect(activationValue).toBeCloseTo(0.1, 12);
        });
      });
    });

    describe('given an invalid connection target object', () => {
      describe('when creating a connection', () => {
        it('throws the invalid-target-type error', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const connectInvalidTarget = () => {
            node.connect({} as unknown as { nodes: Node[] });
          };

          // Assert
          expect(connectInvalidTarget).toThrow(
            NodeInvalidConnectionTargetTypeError,
          );
        });
      });
    });

    describe('given reciprocal projections where one connection is gated', () => {
      describe('when disconnecting both sides at once', () => {
        it('removes both projections and clears the gating reference', () => {
          // Arrange
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const gaterNode = new Node('hidden');
          const forwardConnection = sourceNode.connect(targetNode)[0];
          targetNode.connect(sourceNode);
          gaterNode.gate(forwardConnection);

          // Act
          sourceNode.disconnect(targetNode, true);

          // Assert
          expect({
            forwardProjection: sourceNode.isConnectedTo(targetNode),
            reverseProjection: targetNode.isConnectedTo(sourceNode),
            gatedConnectionCount: gaterNode.connections.gated.length,
            gater: forwardConnection.gater,
          }).toStrictEqual({
            forwardProjection: false,
            reverseProjection: false,
            gatedConnectionCount: 0,
            gater: null,
          });
        });
      });
    });

    describe('given an incomplete connection object', () => {
      describe('when trying to gate it', () => {
        it('warns and leaves the gated connection list unchanged', () => {
          // Arrange
          const gaterNode = new Node('hidden');
          const incompleteConnection = { from: null, to: null } as unknown as Connection;

          // Act
          const gateResult = withSuppressedWarnings((warningSpy) => {
            gaterNode.gate(incompleteConnection);

            return {
              gatedConnectionCount: gaterNode.connections.gated.length,
              warningCount: warningSpy.mock.calls.length,
            };
          });

          // Assert
          expect(gateResult).toStrictEqual({
            gatedConnectionCount: 0,
            warningCount: 1,
          });
        });
      });
    });

    describe('given a connection already gated by the same node', () => {
      describe('when gating it again', () => {
        it('warns without duplicating the gated entry', () => {
          // Arrange
          const gaterNode = new Node('hidden');
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const connection = sourceNode.connect(targetNode)[0];
          gaterNode.gate(connection);

          // Act
          const gateResult = withSuppressedWarnings((warningSpy) => {
            gaterNode.gate(connection);

            return {
              gatedConnectionCount: gaterNode.connections.gated.length,
              warningCount: warningSpy.mock.calls.length,
            };
          });

          // Assert
          expect(gateResult).toStrictEqual({
            gatedConnectionCount: 1,
            warningCount: 1,
          });
        });
      });
    });

    describe('given a connection already gated by another node', () => {
      describe('when a second node tries to gate it', () => {
        it('warns and preserves the original gater', () => {
          // Arrange
          const firstGaterNode = new Node('hidden');
          const secondGaterNode = new Node('hidden');
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const connection = sourceNode.connect(targetNode)[0];
          firstGaterNode.gate(connection);

          // Act
          const gateResult = withSuppressedWarnings((warningSpy) => {
            secondGaterNode.gate(connection);

            return {
              gater: connection.gater,
              secondGaterCount: secondGaterNode.connections.gated.length,
              warningCount: warningSpy.mock.calls.length,
            };
          });

          // Assert
          expect(gateResult).toStrictEqual({
            gater: firstGaterNode,
            secondGaterCount: 0,
            warningCount: 1,
          });
        });
      });
    });
  });
});