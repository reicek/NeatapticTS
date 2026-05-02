import { config } from '../../../config';
import mutation from '../../../methods/mutation/mutation';
import { createGenomeFromNetwork } from '../../../neat/genome/genome';
import Network from '../network';
import type { NetworkJSON } from '../network.types';

function countHiddenNodes(network: Network): number {
  return network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  ).length;
}

function countBackwardConnections(network: Network): number {
  return network.connections.filter(
    (candidateConnection) =>
      network.nodes.indexOf(candidateConnection.from) >
      network.nodes.indexOf(candidateConnection.to),
  ).length;
}

function readSquashSignature(network: Network): string {
  return network.nodes
    .map((candidateNode) => candidateNode.squash.name)
    .join(',');
}

function readBiasSignature(network: Network): string {
  return network.nodes
    .map((candidateNode) => String(candidateNode.bias))
    .join(',');
}

function readWeightSignature(network: Network): string {
  return network.connections
    .map((candidateConnection) => candidateConnection.weight.toFixed(6))
    .join(',');
}

function hasHiddenBatchNormTag(network: Network): boolean {
  return network.nodes.some(
    (candidateNode) =>
      candidateNode.type === 'hidden' &&
      Reflect.get(candidateNode, '_batchNorm') === true,
  );
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

function createBackwardConnectionRemovalNetwork(): Network {
  const network = new Network(1, 2, { seed: 423, enforceAcyclic: false });

  network.mutate(mutation.ADD_NODE);
  network.mutate(mutation.ADD_NODE);

  const hiddenNodes = network.nodes.filter(
    (candidateNode) => candidateNode.type === 'hidden',
  );
  const outputNodes = network.nodes.filter(
    (candidateNode) => candidateNode.type === 'output',
  );
  const inputNode = network.nodes[0];
  const firstHiddenNode = hiddenNodes[0];

  if (!firstHiddenNode) {
    throw new Error('Expected at least one hidden node');
  }

  outputNodes.forEach((outputNode) => {
    if (!firstHiddenNode.isProjectingTo(outputNode)) {
      network.connect(firstHiddenNode, outputNode);
    }
  });

  outputNodes.slice(0, 2).forEach((outputNode) => {
    if (!outputNode.isProjectingTo(inputNode)) {
      network.connect(outputNode, inputNode);
    }
  });

  if (!firstHiddenNode.isProjectingTo(inputNode)) {
    network.connect(firstHiddenNode, inputNode);
  }

  return network;
}

describe('network mutate chapter', () => {
  describe('Network.mutate()', () => {
    describe('given the mutation method is unknown', () => {
      describe('when mutate() is called', () => {
        it('leaves the connection count unchanged', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 420 });
          const connectionCountBeforeMutation = network.connections.length;

          // Act
          (network.mutate as (method: unknown) => void)('NON_EXISTENT_MUT');
          const connectionCountAfterMutation = network.connections.length;

          // Assert
          expect(connectionCountAfterMutation).toBe(
            connectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given deterministic chain mode is enabled', () => {
      describe('when ADD_NODE runs', () => {
        it('adds exactly one hidden node to the chain', () => {
          // Arrange
          const originalDeterministicChainMode =
            config.deterministicChainMode ?? false;
          config.deterministicChainMode = true;

          try {
            const network = new Network(1, 1, { seed: 421 });
            const hiddenCountBeforeMutation = countHiddenNodes(network);

            // Act
            network.mutate(mutation.ADD_NODE);
            const hiddenCountAfterMutation = countHiddenNodes(network);

            // Assert
            expect(hiddenCountAfterMutation - hiddenCountBeforeMutation).toBe(
              1,
            );
          } finally {
            config.deterministicChainMode = originalDeterministicChainMode;
          }
        });
      });
    });

    describe('given ADD_SELF_CONN runs under acyclic enforcement', () => {
      describe('when mutate() is called', () => {
        it('does not add a self connection', () => {
          // Arrange
          const network = new Network(2, 1, {
            seed: 422,
            enforceAcyclic: true,
          });
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          network.mutate(mutation.ADD_SELF_CONN);
          const selfConnectionCountAfterMutation = network.selfconns.length;

          // Assert
          expect(selfConnectionCountAfterMutation).toBe(
            selfConnectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given ADD_BACK_CONN runs under acyclic enforcement', () => {
      describe('when mutate() is called', () => {
        it('does not add a backward connection', () => {
          // Arrange
          const network = new Network(2, 2, {
            seed: 424,
            enforceAcyclic: true,
          });
          const connectionCountBeforeMutation = network.connections.length;

          // Act
          network.mutate(mutation.ADD_BACK_CONN);
          const connectionCountAfterMutation = network.connections.length;

          // Assert
          expect(connectionCountAfterMutation).toBe(
            connectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given ADD_GATE is followed by SUB_GATE', () => {
      describe('when both mutations run', () => {
        it('reduces the tracked gate count after the removal step', () => {
          // Arrange
          const network = new Network(2, 2, { seed: 425 });
          network.mutate(mutation.ADD_GATE);
          const trackedGateCountAfterAdd = network.gates.length;

          // Act
          network.mutate(mutation.SUB_GATE);
          const trackedGateCountAfterRemove = network.gates.length;

          // Assert
          expect(trackedGateCountAfterAdd > trackedGateCountAfterRemove).toBe(
            true,
          );
        });
      });
    });

    describe('given MOD_ACTIVATION excludes output nodes', () => {
      describe('when the network has no hidden nodes', () => {
        it('leaves the squash signature unchanged', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 426 });
          const squashSignatureBeforeMutation = readSquashSignature(network);

          // Act
          network.mutate({ ...mutation.MOD_ACTIVATION, mutateOutput: false });
          const squashSignatureAfterMutation = readSquashSignature(network);

          // Assert
          expect(squashSignatureAfterMutation).toBe(
            squashSignatureBeforeMutation,
          );
        });
      });
    });

    describe('given SWAP_NODES excludes output nodes', () => {
      describe('when fewer than two swappable nodes exist', () => {
        it('leaves the node bias signature unchanged', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 427 });
          const biasSignatureBeforeMutation = readBiasSignature(network);

          // Act
          network.mutate({ ...mutation.SWAP_NODES, mutateOutput: false });
          const biasSignatureAfterMutation = readBiasSignature(network);

          // Assert
          expect(biasSignatureAfterMutation).toBe(biasSignatureBeforeMutation);
        });
      });
    });

    describe('given SUB_NODE runs without hidden nodes', () => {
      describe('when mutate() is called', () => {
        it('leaves the hidden-node count unchanged', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 428 });
          const hiddenCountBeforeMutation = countHiddenNodes(network);

          // Act
          network.mutate(mutation.SUB_NODE);
          const hiddenCountAfterMutation = countHiddenNodes(network);

          // Assert
          expect(hiddenCountAfterMutation).toBe(hiddenCountBeforeMutation);
        });

        it('keeps strict genome conversion valid after removing one previously added hidden node', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 4281 });
          network.mutate(mutation.ADD_NODE);

          // Act
          network.mutate(mutation.SUB_NODE);

          // Assert
          expect(() => createGenomeFromNetwork(network)).not.toThrow();
        });
      });
    });

    describe('given ADD_CONN has no unused forward pairs available', () => {
      describe('when mutate() is called', () => {
        it('leaves the connection count unchanged', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 429 });
          const connectionCountBeforeMutation = network.connections.length;

          // Act
          network.mutate(mutation.ADD_CONN);
          const connectionCountAfterMutation = network.connections.length;

          // Assert
          expect(connectionCountAfterMutation).toBe(
            connectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given SUB_SELF_CONN runs without self connections', () => {
      describe('when mutate() is called', () => {
        it('keeps the self-connection count at zero', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 430 });
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          network.mutate(mutation.SUB_SELF_CONN);
          const selfConnectionCountAfterMutation = network.selfconns.length;

          // Assert
          expect(selfConnectionCountAfterMutation).toBe(
            selfConnectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given every eligible node already has a self loop', () => {
      describe('when ADD_SELF_CONN runs', () => {
        it('does not increase the self-connection count', () => {
          // Arrange
          const network = new Network(1, 2, {
            seed: 431,
            enforceAcyclic: false,
          });
          network.nodes.forEach((candidateNode, candidateIndex) => {
            if (candidateIndex >= network.input) {
              network.connect(candidateNode, candidateNode);
            }
          });
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          network.mutate(mutation.ADD_SELF_CONN);
          const selfConnectionCountAfterMutation = network.selfconns.length;

          // Assert
          expect(selfConnectionCountAfterMutation).toBe(
            selfConnectionCountBeforeMutation,
          );
        });
      });
    });

    describe('given ADD_LSTM_NODE runs under acyclic enforcement', () => {
      describe('when mutate() is called', () => {
        it('leaves the node count unchanged', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 432,
            enforceAcyclic: true,
          });
          const nodeCountBeforeMutation = network.nodes.length;

          // Act
          network.mutate(mutation.ADD_LSTM_NODE);
          const nodeCountAfterMutation = network.nodes.length;

          // Assert
          expect(nodeCountAfterMutation).toBe(nodeCountBeforeMutation);
        });
      });
    });

    describe('given ADD_GRU_NODE runs under acyclic enforcement', () => {
      describe('when mutate() is called', () => {
        it('leaves the node count unchanged', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 433,
            enforceAcyclic: true,
          });
          const nodeCountBeforeMutation = network.nodes.length;

          // Act
          network.mutate(mutation.ADD_GRU_NODE);
          const nodeCountAfterMutation = network.nodes.length;

          // Assert
          expect(nodeCountAfterMutation).toBe(nodeCountBeforeMutation);
        });
      });
    });

    describe('given BATCH_NORM runs without hidden nodes', () => {
      describe('when mutate() is called', () => {
        it('does not set a hidden-node batch-norm tag', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 434 });

          // Act
          network.mutate(mutation.BATCH_NORM);
          const hiddenBatchNormWasSet = hasHiddenBatchNormTag(network);

          // Assert
          expect(hiddenBatchNormWasSet).toBe(false);
        });
      });
    });

    describe('given ADD_SELF_CONN runs with acyclicity disabled', () => {
      describe('when mutate() is called', () => {
        it('adds a self connection', () => {
          // Arrange
          const network = new Network(2, 2, {
            seed: 435,
            enforceAcyclic: false,
          });
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          network.mutate(mutation.ADD_SELF_CONN);
          const selfConnectionCountAfterMutation = network.selfconns.length;

          // Assert
          expect(
            selfConnectionCountAfterMutation >
              selfConnectionCountBeforeMutation,
          ).toBe(true);
        });
      });
    });

    describe('given SUB_SELF_CONN runs after a self loop exists', () => {
      describe('when mutate() is called', () => {
        it('removes exactly one self connection', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 436,
            enforceAcyclic: false,
          });
          network.mutate(mutation.ADD_SELF_CONN);
          const selfConnectionCountBeforeMutation = network.selfconns.length;

          // Act
          network.mutate(mutation.SUB_SELF_CONN);
          const selfConnectionCountAfterMutation = network.selfconns.length;

          // Assert
          expect(
            selfConnectionCountBeforeMutation -
              selfConnectionCountAfterMutation,
          ).toBe(1);
        });
      });
    });

    describe('given ADD_BACK_CONN runs with acyclicity disabled', () => {
      describe('when mutate() is called', () => {
        it('increases the backward-connection count', () => {
          // Arrange
          const network = new Network(2, 2, {
            seed: 437,
            enforceAcyclic: false,
          });
          const backwardConnectionCountBeforeMutation =
            countBackwardConnections(network);

          // Act
          network.mutate(mutation.ADD_BACK_CONN);
          const backwardConnectionCountAfterMutation =
            countBackwardConnections(network);

          // Assert
          expect(
            backwardConnectionCountAfterMutation >
              backwardConnectionCountBeforeMutation,
          ).toBe(true);
        });
      });
    });

    describe('given SUB_BACK_CONN runs after redundant backward edges exist', () => {
      describe('when mutate() is called', () => {
        it('reduces the backward-connection count', () => {
          // Arrange
          const network = createBackwardConnectionRemovalNetwork();
          const backwardConnectionCountBeforeMutation =
            countBackwardConnections(network);

          // Act
          network.mutate(mutation.SUB_BACK_CONN);
          const backwardConnectionCountAfterMutation =
            countBackwardConnections(network);

          // Assert
          expect(
            backwardConnectionCountAfterMutation <
              backwardConnectionCountBeforeMutation,
          ).toBe(true);
        });
      });
    });

    describe('given REINIT_WEIGHT runs after a hidden node exists', () => {
      describe('when mutate() is called', () => {
        it('changes the connection-weight signature', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 438 });
          network.mutate(mutation.ADD_NODE);
          const weightSignatureBeforeMutation = readWeightSignature(network);

          // Act
          network.mutate(mutation.REINIT_WEIGHT);
          const weightSignatureAfterMutation = readWeightSignature(network);

          // Assert
          expect(
            weightSignatureAfterMutation === weightSignatureBeforeMutation,
          ).toBe(false);
        });
      });
    });

    describe('given BATCH_NORM runs after a hidden node exists', () => {
      describe('when mutate() is called', () => {
        it('sets a hidden-node batch-norm tag', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 439 });
          network.mutate(mutation.ADD_NODE);

          // Act
          network.mutate(mutation.BATCH_NORM);
          const hiddenBatchNormWasSet = hasHiddenBatchNormTag(network);

          // Assert
          expect(hiddenBatchNormWasSet).toBe(true);
        });
      });
    });

    describe('given ADD_LSTM_NODE runs with acyclicity disabled', () => {
      describe('when mutate() is called', () => {
        it('increases the node count', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 440,
            enforceAcyclic: false,
          });
          const nodeCountBeforeMutation = network.nodes.length;

          // Act
          network.mutate(mutation.ADD_LSTM_NODE);
          const nodeCountAfterMutation = network.nodes.length;

          // Assert
          expect(nodeCountAfterMutation > nodeCountBeforeMutation).toBe(true);
        });

        it('emits one LSTM temporal descriptor on serialization', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 442,
            enforceAcyclic: false,
          });

          // Act
          network.mutate(mutation.ADD_LSTM_NODE);
          const temporalSummary = summarizeTemporalExtensionBag(network);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 1,
            gatedBlockCount: 1,
            recurrentKinds: ['lstm'],
          });
        });

        it('captures the mutated LSTM block as a strict genome without dropping its internal edges', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 444,
            enforceAcyclic: false,
          });

          // Act / Assert
          network.mutate(mutation.ADD_LSTM_NODE);
          expect(() => createGenomeFromNetwork(network)).not.toThrow();
        });
      });
    });

    describe('given ADD_GRU_NODE runs with acyclicity disabled', () => {
      describe('when mutate() is called', () => {
        it('increases the node count', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 441,
            enforceAcyclic: false,
          });
          const nodeCountBeforeMutation = network.nodes.length;

          // Act
          network.mutate(mutation.ADD_GRU_NODE);
          const nodeCountAfterMutation = network.nodes.length;

          // Assert
          expect(nodeCountAfterMutation > nodeCountBeforeMutation).toBe(true);
        });

        it('emits one GRU temporal descriptor on serialization', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 443,
            enforceAcyclic: false,
          });

          // Act
          network.mutate(mutation.ADD_GRU_NODE);
          const temporalSummary = summarizeTemporalExtensionBag(network);

          // Assert
          expect(temporalSummary).toEqual({
            recurrentModuleCount: 1,
            gatedBlockCount: 1,
            recurrentKinds: ['gru'],
          });
        });

        it('captures the mutated GRU block as a strict genome without dropping its internal edges', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 445,
            enforceAcyclic: false,
          });

          // Act / Assert
          network.mutate(mutation.ADD_GRU_NODE);
          expect(() => createGenomeFromNetwork(network)).not.toThrow();
        });
      });
    });
  });
});
