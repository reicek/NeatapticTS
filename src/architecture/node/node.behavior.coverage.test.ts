import Connection from '../connection/connection';
import { config } from '../../config';
import Activation from '../../methods/activation/activation';
import * as rawMethods from '../../methods/methods';
import {
  NodeMutationMethodRequiredError,
  NodeUndefinedConnectionTargetError,
  NodeUnknownMutationMethodError,
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

function withTemporaryProperties<T>(
  targetObject: Record<string, unknown>,
  temporaryProperties: Record<string, unknown>,
  run: () => T,
): T {
  const originalProperties = Object.fromEntries(
    Object.keys(temporaryProperties).map((propertyName) => [
      propertyName,
      targetObject[propertyName],
    ]),
  );

  try {
    for (const [propertyName, propertyValue] of Object.entries(
      temporaryProperties,
    )) {
      targetObject[propertyName] = propertyValue;
    }

    return run();
  } finally {
    for (const [propertyName, propertyValue] of Object.entries(
      originalProperties,
    )) {
      targetObject[propertyName] = propertyValue;
    }
  }
}

describe('Node', () => {
  describe('behavior coverage helpers', () => {
    describe('given a node created with a custom activation', () => {
      describe('when creating the node', () => {
        it('keeps the provided squash function and a concrete index', () => {
          // Arrange
          const node = new Node('hidden', Activation.relu);

          // Act
          const creationResult = {
            hasIndex: typeof node.index === 'number',
            squash: node.squash,
          };

          // Assert
          expect(creationResult).toStrictEqual({
            hasIndex: true,
            squash: Activation.relu,
          });
        });
      });
    });

    describe('given the logistic activation is temporarily unavailable', () => {
      describe('when creating a node without a custom activation', () => {
        it('falls back to the inline identity function', () => {
          // Arrange
          const activationTable = rawMethods.Activation as unknown as Record<
            string,
            unknown
          >;

          // Act
          const fallbackActivation = withTemporaryProperties(
            activationTable,
            { logistic: undefined },
            () => {
              const node = new Node('hidden');
              return node.squash(4);
            },
          );

          // Assert
          expect(fallbackActivation).toBe(4);
        });
      });
    });

    describe('given a finite restored gene id', () => {
      describe('when syncing the counter', () => {
        it('advances the next created gene id beyond the restored maximum', () => {
          // Arrange
          const baselineNode = new Node('hidden');
          const restoredMaximumGeneId = baselineNode.geneId + 10;

          // Act
          Node.syncGeneIdCounter(restoredMaximumGeneId);
          const nextNode = new Node('hidden');

          // Assert
          expect(nextNode.geneId).toBe(restoredMaximumGeneId + 1);
        });
      });
    });

    describe('given a node without a squash function', () => {
      describe('when serializing to JSON', () => {
        it('stores a null squash name', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = null as unknown as (
            inputValue: number,
            shouldDerivative?: boolean,
          ) => number;

          // Act
          const serializedNode = node.toJSON();

          // Assert
          expect(serializedNode.squash).toBeNull();
        });
      });
    });

    describe('given a serialized node with an empty squash name', () => {
      describe('when restoring the node from JSON', () => {
        it('keeps the constructor squash and explicit response', () => {
          // Arrange
          const restorationPayload = {
            bias: 0.25,
            mask: 1,
            response: 1.25,
            squash: '',
            type: 'output',
          };

          // Act
          const restoredNode = Node.fromJSON(restorationPayload);

          // Assert
          expect({
            response: restoredNode.response,
            squash: restoredNode.squash,
          }).toStrictEqual({
            response: 1.25,
            squash: Activation.logistic,
          });
        });
      });
    });

    describe('given no mutation method', () => {
      describe('when mutating the node', () => {
        it('throws the required-method error', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const applyMissingMutation = () => {
            node.mutate(undefined);
          };

          // Assert
          expect(applyMissingMutation).toThrow(
            NodeMutationMethodRequiredError,
          );
        });
      });
    });

    describe('given an unknown mutation name', () => {
      describe('when mutating the node', () => {
        it('throws the unknown-mutation error', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const applyUnknownMutation = () => {
            node.mutate({ name: 'NOT_A_REAL_MUTATION' });
          };

          // Assert
          expect(applyUnknownMutation).toThrow(NodeUnknownMutationMethodError);
        });
      });
    });

    describe('given an unknown mutation without a name field', () => {
      describe('when mutating the node', () => {
        it('uses the undefined-name fallback in the error message', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const applyUnnamedMutation = () => {
            node.mutate({});
          };

          // Assert
          expect(applyUnnamedMutation).toThrow('Unknown mutation method: undefined');
        });
      });
    });

    describe('given activation mutation without allowed functions', () => {
      describe('when mutating the node', () => {
        it('warns and leaves the current squash unchanged', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.tanh;
          const activationMutation = rawMethods.mutation.MOD_ACTIVATION as
            Record<string, unknown>;

          // Act
          const mutationResult = withSuppressedWarnings((warningSpy) =>
            withTemporaryProperties(
              activationMutation,
              { allowed: [] },
              () => {
                node.mutate(rawMethods.mutation.MOD_ACTIVATION);

                return {
                  squash: node.squash,
                  warningCount: warningSpy.mock.calls.length,
                };
              },
            ),
          );

          // Assert
          expect(mutationResult).toStrictEqual({
            squash: Activation.tanh,
            warningCount: 1,
          });
        });
      });
    });

    describe('given activation mutation with one allowed function', () => {
      describe('when mutating the node', () => {
        it('keeps the only allowed activation', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.tanh;
          const activationMutation = rawMethods.mutation.MOD_ACTIVATION as
            Record<string, unknown>;

          // Act
          const resultingSquash = withTemporaryProperties(
            activationMutation,
            { allowed: [Activation.tanh] },
            () =>
              withMockedMathRandom(0.6, () => {
                node.mutate(rawMethods.mutation.MOD_ACTIVATION);
                return node.squash;
              }),
          );

          // Assert
          expect(resultingSquash).toBe(Activation.tanh);
        });
      });
    });

    describe('given a method object whose name changes after validation', () => {
      describe('when mutating the node', () => {
        it('uses the undefined-name fallback in the unsupported-method error', () => {
          // Arrange
          const node = new Node('hidden');
          let readCount = 0;
          const unstableMutationMethod = {
            get name() {
              readCount += 1;
              return readCount <= 2 ? 'MOD_BIAS' : undefined;
            },
          };

          // Act
          const applyUnstableMutation = () => {
            node.mutate(unstableMutationMethod);
          };

          // Assert
          expect(applyUnstableMutation).toThrow(
            'Unsupported mutation method: undefined',
          );
        });
      });
    });

    describe('given bias mutation without an explicit range', () => {
      describe('when mutating the node', () => {
        it('uses the default min and max values', () => {
          // Arrange
          const node = new Node('hidden');
          node.bias = 0;
          const biasMutation = rawMethods.mutation.MOD_BIAS as Record<
            string,
            unknown
          >;

          // Act
          const mutatedBias = withTemporaryProperties(
            biasMutation,
            { min: undefined, max: undefined },
            () =>
              withMockedMathRandom(0.75, () => {
                node.mutate(rawMethods.mutation.MOD_BIAS);
                return node.bias;
              }),
          );

          // Assert
          expect(mutatedBias).toBeCloseTo(0.5, 12);
        });
      });
    });

    describe('given weight reinitialization without an explicit range', () => {
      describe('when mutating the node', () => {
        it('uses the default range for incoming, outgoing, and self weights', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const incomingConnection = sourceNode.connect(node, 0)[0];
          const outgoingConnection = node.connect(targetNode, 0)[0];
          const selfConnection = node.connect(node, 0)[0];
          const reinitializeWeightsMutation = rawMethods.mutation.REINIT_WEIGHT as Record<
            string,
            unknown
          >;

          // Act
          const reinitializedWeights = withTemporaryProperties(
            reinitializeWeightsMutation,
            { min: undefined, max: undefined },
            () =>
              withMockedMathRandom(0.75, () => {
                node.mutate(rawMethods.mutation.REINIT_WEIGHT);

                return {
                  incoming: incomingConnection.weight,
                  outgoing: outgoingConnection.weight,
                  self: selfConnection.weight,
                };
              }),
          );

          // Assert
          expect(reinitializedWeights).toStrictEqual({
            incoming: 0.5,
            outgoing: 0.5,
            self: 0.5,
          });
        });
      });
    });

    describe('given an undefined connection target', () => {
      describe('when creating a connection', () => {
        it('throws the undefined-target error', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const connectUndefinedTarget = () => {
            node.connect(undefined as unknown as Node);
          };

          // Assert
          expect(connectUndefinedTarget).toThrow(
            NodeUndefinedConnectionTargetError,
          );
        });
      });
    });

    describe('given an existing self connection', () => {
      describe('when connecting the node to itself again', () => {
        it('preserves the single self loop', () => {
          // Arrange
          const node = new Node('hidden');
          node.connect(node, 0.25);

          // Act
          const duplicateSelfConnections = node.connect(node, 0.5);

          // Assert
          expect({
            returnedConnectionCount: duplicateSelfConnections.length,
            selfConnectionCount: node.connections.self.length,
            weight: node.connections.self[0].weight,
          }).toStrictEqual({
            returnedConnectionCount: 0,
            selfConnectionCount: 1,
            weight: 0.25,
          });
        });
      });
    });

    describe('given a node with a self loop', () => {
      describe('when disconnecting from itself', () => {
        it('removes every self connection', () => {
          // Arrange
          const node = new Node('hidden');
          node.connect(node, 0.25);

          // Act
          node.disconnect(node);

          // Assert
          expect(node.connections.self).toStrictEqual([]);
        });
      });
    });

    describe('given a node with a self loop', () => {
      describe('when checking forward self projection', () => {
        it('treats the self loop as an outgoing projection', () => {
          // Arrange
          const node = new Node('hidden');
          node.connect(node, 0.25);

          // Act
          const isProjectingToSelf = node.isProjectingTo(node);

          // Assert
          expect(isProjectingToSelf).toBe(true);
        });
      });
    });

    describe('given one kept projection and one removed projection', () => {
      describe('when disconnecting the removed target', () => {
        it('preserves the unrelated outgoing connection', () => {
          // Arrange
          const sourceNode = new Node('hidden');
          const keptTargetNode = new Node('hidden');
          const removedTargetNode = new Node('hidden');
          sourceNode.connect(keptTargetNode, 0.5);
          sourceNode.connect(removedTargetNode, 0.25);

          // Act
          sourceNode.disconnect(removedTargetNode);

          // Assert
          expect(sourceNode.connections.out.map((connection) => connection.to)).toStrictEqual([
            keptTargetNode,
          ]);
        });
      });
    });

    describe('given an array of valid connections', () => {
      describe('when gating them together', () => {
        it('stores this node as the gater for each connection', () => {
          // Arrange
          const gaterNode = new Node('hidden');
          const sourceNode = new Node('hidden');
          const firstTargetNode = new Node('hidden');
          const secondTargetNode = new Node('hidden');
          const firstConnection = sourceNode.connect(firstTargetNode)[0];
          const secondConnection = sourceNode.connect(secondTargetNode)[0];

          // Act
          gaterNode.gate([firstConnection, secondConnection]);

          // Assert
          expect([firstConnection.gater, secondConnection.gater]).toStrictEqual([
            gaterNode,
            gaterNode,
          ]);
        });
      });
    });

    describe('given owned, missing, and foreign gated connections', () => {
      describe('when ungating them as one array', () => {
        it('clears only the owned gated connection', () => {
          // Arrange
          const gaterNode = new Node('hidden');
          const otherGaterNode = new Node('hidden');
          const sourceNode = new Node('hidden');
          const ownedTargetNode = new Node('hidden');
          const foreignTargetNode = new Node('hidden');
          const ownedConnection = sourceNode.connect(ownedTargetNode)[0];
          const foreignConnection = sourceNode.connect(foreignTargetNode)[0];
          gaterNode.gate(ownedConnection);
          otherGaterNode.gate(foreignConnection);

          // Act
          gaterNode.ungate([
            ownedConnection,
            null as unknown as Connection,
            foreignConnection,
          ]);

          // Assert
          expect({
            foreignGater: foreignConnection.gater,
            ownedGain: ownedConnection.gain,
            ownedGater: ownedConnection.gater,
            remainingGatedConnections: gaterNode.connections.gated.length,
          }).toStrictEqual({
            foreignGater: otherGaterNode,
            ownedGain: 1,
            ownedGater: null,
            remainingGatedConnections: 0,
          });
        });
      });
    });

    describe('given a hidden node with explicit input and traces enabled', () => {
      describe('when activating the node', () => {
        it('updates both gated gain and incoming eligibility', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const incomingConnection = sourceNode.connect(node, 1)[0];
          const gatedConnection = sourceNode.connect(targetNode, 1)[0];
          sourceNode.activation = 0.75;
          node.squash = Activation.identity;
          node.response = 2;
          node.gate(gatedConnection);

          // Act
          const activationValue = node.activate(0.5);

          // Assert
          expect({
            activation: activationValue,
            derivative: node.derivative,
            gain: gatedConnection.gain,
            eligibility: incomingConnection.eligibility,
          }).toStrictEqual({
            activation: 1,
            derivative: 2,
            gain: 1,
            eligibility: 0.75,
          });
        });
      });
    });

    describe('given an input node with an explicit input value', () => {
      describe('when activating the node', () => {
        it('stores that input directly as the activation', () => {
          // Arrange
          const node = new Node('input');

          // Act
          const activationValue = node.activate(0.75);

          // Assert
          expect(activationValue).toBe(0.75);
        });
      });
    });

    describe('given a hidden node with explicit input and traces disabled', () => {
      describe('when activating the node without traces', () => {
        it('updates gated gain without touching incoming eligibility', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const incomingConnection = sourceNode.connect(node, 1)[0];
          const gatedConnection = sourceNode.connect(targetNode, 1)[0];
          sourceNode.activation = 0.75;
          node.squash = Activation.identity;
          node.response = 2;
          node.gate(gatedConnection);

          // Act
          const activationValue = node.noTraceActivate(0.5);

          // Assert
          expect({
            activation: activationValue,
            derivative: node.derivative,
            gain: gatedConnection.gain,
            eligibility: incomingConnection.eligibility,
          }).toStrictEqual({
            activation: 1,
            derivative: 2,
            gain: 1,
            eligibility: 0,
          });
        });
      });
    });

    describe('given a hidden node with stored-state activation and no traces', () => {
      describe('when activating without an explicit input', () => {
        it('leaves incoming eligibility unchanged on the no-trace path', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const incomingConnection = sourceNode.connect(node, 1)[0];
          node.squash = Activation.identity;
          node.bias = 2;
          sourceNode.activation = 0.75;

          // Act
          const activationValue = node.noTraceActivate();

          // Assert
          expect({
            activation: activationValue,
            eligibility: incomingConnection.eligibility,
          }).toStrictEqual({
            activation: 2.75,
            eligibility: 0,
          });
        });
      });
    });

    describe('given warnings are disabled and the squash function is invalid', () => {
      describe('when activating the node', () => {
        it('repairs the squash function without logging a warning', () => {
          // Arrange
          const node = new Node('hidden');
          const originalWarnings = config.warnings;
          node.bias = 2;
          node.mask = 'invalid-mask' as unknown as number;
          node.squash = null as unknown as (
            inputValue: number,
            shouldDerivative?: boolean,
          ) => number;

          // Act
          const activationResult = withSuppressedWarnings((warningSpy) => {
            config.warnings = false;

            try {
              node.activate();

              return {
                activation: node.activation,
                repairedSquashValue: node.squash(3),
                warningCount: warningSpy.mock.calls.length,
              };
            } finally {
              config.warnings = originalWarnings;
            }
          });

          // Assert
          expect(activationResult).toStrictEqual({
            activation: 2,
            repairedSquashValue: 3,
            warningCount: 0,
          });
        });
      });
    });

    describe('given a hidden node with gated outgoing connections', () => {
      describe('when activating from stored state', () => {
        it('updates the gated connection gain to the new activation', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const gatedConnection = sourceNode.connect(targetNode, 1)[0];
          node.squash = Activation.identity;
          node.bias = 2;
          node.gate(gatedConnection);

          // Act
          const activationValue = node.activate();

          // Assert
          expect({
            activation: activationValue,
            gain: gatedConnection.gain,
          }).toStrictEqual({
            activation: 2,
            gain: 2,
          });
        });
      });
    });

    describe('given a node with dynamic state and traces', () => {
      describe('when clearing the node', () => {
        it('resets traces, gains, errors, and stateful activations', () => {
          // Arrange
          const node = new Node('hidden');
          const sourceNode = new Node('input');
          const targetNode = new Node('output');
          const incomingConnection = sourceNode.connect(node, 0.5)[0];
          const gatedConnection = node.connect(targetNode, 0.25)[0];
          const selfConnection = node.connect(node, 0.75)[0];
          incomingConnection.eligibility = 3;
          incomingConnection.xtrace = { nodes: [sourceNode], values: [4] };
          selfConnection.eligibility = 5;
          selfConnection.xtrace = { nodes: [node], values: [6] };
          gatedConnection.gain = 2;
          node.connections.gated.push(gatedConnection);
          node.error = { responsibility: 7, projected: 8, gated: 9 };
          node.old = 10;
          node.state = 11;
          node.activation = 12;

          // Act
          node.clear();

          // Assert
          expect({
            activation: node.activation,
            error: node.error,
            gatedGain: gatedConnection.gain,
            incomingEligibility: incomingConnection.eligibility,
            incomingXtraceLength: incomingConnection.xtrace.nodes.length,
            old: node.old,
            selfEligibility: selfConnection.eligibility,
            selfXtraceLength: selfConnection.xtrace.nodes.length,
            state: node.state,
          }).toStrictEqual({
            activation: 0,
            error: { responsibility: 0, projected: 0, gated: 0 },
            gatedGain: 0,
            incomingEligibility: 0,
            incomingXtraceLength: 0,
            old: 0,
            selfEligibility: 0,
            selfXtraceLength: 0,
            state: 0,
          });
        });
      });
    });
  });
});