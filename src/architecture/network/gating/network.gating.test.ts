import { Architect } from '../../../neataptic';
import { config } from '../../../config';
import mutation from '../../../methods/mutation/mutation';
import Node from '../../node';
import Network from '../network';
import type { NetworkJSON } from '../network.types';
import { gatingRemoveNode } from '../network.utils';
import { NetworkGatingNodeMembershipError } from './network.gating.errors';

function createAcyclicNetwork(seed: number): Network {
  return new Network(1, 1, { seed, enforceAcyclic: true });
}

function getNetworkRandomGenerator(network: Network): () => number {
  return Reflect.get(network, '_rand') as () => number;
}

function summarizeHydratedTemporalExtensionBag(network: Network): {
  recurrentModuleCount: number;
  gatedBlockCount: number;
  recurrentKinds: string[];
} {
  const hydratedExtensions = Reflect.get(network, '_serializedExtensions') as
    | NetworkJSON['extensions']
    | undefined;
  const extensionValues = hydratedExtensions?.values as
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

describe('network gating chapter', () => {
  describe('gate()', () => {
    describe('given the gater belongs to a different network', () => {
      describe('when gate() is called', () => {
        it('throws the node-membership error', () => {
          // Arrange
          const network = createAcyclicNetwork(340);
          const foreignNetwork = createAcyclicNetwork(341);
          const connection = network.connections[0];

          // Act
          const gateWithForeignNode = () => {
            network.gate(foreignNetwork.nodes[1], connection);
          };

          // Assert
          expect(gateWithForeignNode).toThrow(NetworkGatingNodeMembershipError);
        });
      });
    });

    describe('given the connection is already gated', () => {
      describe('when gate() is called again', () => {
        it('does not duplicate the tracked gate entry', () => {
          // Arrange
          const network = createAcyclicNetwork(342);
          const connection = network.connections[0];
          network.gate(network.nodes[0], connection);

          // Act
          network.gate(network.nodes[0], connection);
          const trackedGateCount = network.gates.length;

          // Assert
          expect(trackedGateCount).toBe(1);
        });
      });

      describe('when warnings are enabled and gate() is called again', () => {
        it('warns that the connection is already gated', () => {
          // Arrange
          const network = createAcyclicNetwork(343);
          const connection = network.connections[0];
          network.gate(network.nodes[0], connection);
          const originalWarnings = config.warnings;
          config.warnings = true;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            network.gate(network.nodes[0], connection);

            // Assert
            expect(warnSpy).toHaveBeenCalledWith(
              'Connection is already gated. Skipping.',
            );
          } finally {
            warnSpy.mockRestore();
            config.warnings = originalWarnings;
          }
        });
      });
    });
  });

  describe('ungate()', () => {
    describe('given the connection is not in the gates list', () => {
      describe('when ungate() is called', () => {
        it('keeps the tracked gate count unchanged', () => {
          // Arrange
          const network = createAcyclicNetwork(344);
          const connection = network.connections[0];
          const trackedGateCountBeforeUngate = network.gates.length;

          // Act
          network.ungate(connection);
          const trackedGateCountAfterUngate = network.gates.length;

          // Assert
          expect(trackedGateCountAfterUngate).toBe(
            trackedGateCountBeforeUngate,
          );
        });
      });

      describe('when warnings are enabled and ungate() is called', () => {
        it('warns that the connection is missing from the gates list', () => {
          // Arrange
          const network = createAcyclicNetwork(345);
          const connection = network.connections[0];
          const originalWarnings = config.warnings;
          config.warnings = true;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            network.ungate(connection);

            // Assert
            expect(warnSpy).toHaveBeenCalledWith(
              'Attempted to ungate a connection not in the gates list.',
            );
          } finally {
            warnSpy.mockRestore();
            config.warnings = originalWarnings;
          }
        });
      });
    });

    describe('given the connection argument is invalid', () => {
      describe('when ungate() is called', () => {
        it('does not throw', () => {
          // Arrange
          const network = createAcyclicNetwork(346);

          // Act
          const ungateInvalidConnection = () =>
            (network.ungate as (connection: unknown) => void)(undefined);

          // Assert
          expect(ungateInvalidConnection).not.toThrow();
        });
      });

      describe('when warnings are enabled and ungate() is called', () => {
        it('warns that the connection is missing from the gates list', () => {
          // Arrange
          const network = createAcyclicNetwork(347);
          const originalWarnings = config.warnings;
          config.warnings = true;
          const warnSpy = jest
            .spyOn(console, 'warn')
            .mockImplementation(() => {});

          try {
            // Act
            (network.ungate as (connection: unknown) => void)(undefined);

            // Assert
            expect(warnSpy).toHaveBeenCalledWith(
              'Attempted to ungate a connection not in the gates list.',
            );
          } finally {
            warnSpy.mockRestore();
            config.warnings = originalWarnings;
          }
        });
      });
    });

    describe('given the connection is in the gates list', () => {
      describe('when ungate() is called', () => {
        it('removes the tracked gate entry', () => {
          // Arrange
          const network = createAcyclicNetwork(348);
          const connection = network.connections[0];
          network.gate(network.nodes[0], connection);

          // Act
          network.ungate(connection);
          const trackedGateCount = network.gates.length;

          // Assert
          expect(trackedGateCount).toBe(0);
        });

        it('retires the hydrated gated-block descriptor immediately', () => {
          // Arrange
          const network = Architect.lstm(1, 1, 1);
          const gatedConnection = network.gates[0];

          if (!gatedConnection) {
            throw new Error('Expected an LSTM fixture gate to remove.');
          }

          const summaryBeforeUngate =
            summarizeHydratedTemporalExtensionBag(network);

          // Act
          network.ungate(gatedConnection);
          const summaryAfterUngate =
            summarizeHydratedTemporalExtensionBag(network);

          // Assert
          expect({
            summaryBeforeUngate,
            summaryAfterUngate,
          }).toEqual({
            summaryBeforeUngate: {
              recurrentModuleCount: 1,
              gatedBlockCount: 1,
              recurrentKinds: ['lstm'],
            },
            summaryAfterUngate: {
              recurrentModuleCount: 1,
              gatedBlockCount: 0,
              recurrentKinds: ['lstm'],
            },
          });
        });
      });
    });
  });

  describe('gatingRemoveNode()', () => {
    describe('given keep_gates is enabled for gate-aware removal', () => {
      describe('when a gated hidden node is removed', () => {
        it('retains at least one gated bridge connection', () => {
          // Arrange
          const originalKeepGates = mutation.SUB_NODE.keep_gates;
          mutation.SUB_NODE.keep_gates = true;

          try {
            const network = createAcyclicNetwork(349);
            const outputNode = network.nodes.at(-1);
            if (!outputNode) {
              throw new Error('Output node should exist');
            }

            network.disconnect(network.nodes[0], outputNode);
            const hiddenNode = new Node(
              'hidden',
              undefined,
              getNetworkRandomGenerator(network),
            );
            network.nodes.splice(1, 0, hiddenNode);
            const inboundConnection = network.connect(
              network.nodes[0],
              hiddenNode,
            )[0];
            const outboundConnection = network.connect(
              hiddenNode,
              outputNode,
            )[0];
            network.gate(network.nodes[0], inboundConnection);
            network.gate(outputNode, outboundConnection);

            // Act
            gatingRemoveNode.call(network, hiddenNode);
            const trackedGateCount = network.gates.length;

            // Assert
            expect(trackedGateCount > 0).toBe(true);
          } finally {
            mutation.SUB_NODE.keep_gates = originalKeepGates;
          }
        });
      });
    });
  });
});
