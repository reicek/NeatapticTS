import Connection from '../connection/connection';
import Activation from '../../methods/activation/activation';
import { Network, methods } from '../../neataptic';
import Node from './node';

describe('Node', () => {
  describe('constructor', () => {
    describe('given no explicit arguments', () => {
      let node: Node;

      beforeEach(() => {
        // Arrange
        node = new Node();
      });

      describe('when reading the default type', () => {
        it('starts as hidden', () => {
          // Arrange
          const expectedType = 'hidden';

          // Act
          const actualType = node.type;

          // Assert
          expect(actualType).toBe(expectedType);
        });
      });

      describe('when reading the default squash function', () => {
        it('starts with logistic activation', () => {
          // Arrange
          const expectedSquash = Activation.logistic;

          // Act
          const actualSquash = node.squash;

          // Assert
          expect(actualSquash).toBe(expectedSquash);
        });
      });

      describe('when reading the initial activation state', () => {
        it('starts at zero', () => {
          // Arrange
          const expectedActivation = 0;

          // Act
          const actualActivation = node.activation;

          // Assert
          expect(actualActivation).toBe(expectedActivation);
        });
      });

      describe('when reading the incoming connection list', () => {
        it('starts empty', () => {
          // Arrange
          const expectedConnections: Connection[] = [];

          // Act
          const actualConnections = node.connections.in;

          // Assert
          expect(actualConnections).toStrictEqual(expectedConnections);
        });
      });
    });

    describe('given an input node type', () => {
      describe('when the node is created', () => {
        it('uses a zero bias', () => {
          // Arrange
          const inputNode = new Node('input');

          // Act
          const actualBias = inputNode.bias;

          // Assert
          expect(actualBias).toBe(0);
        });
      });
    });
  });

  describe('describe()', () => {
    describe('given an output node', () => {
      describe('when applying label and scalar metadata', () => {
        it('stores additive primitive descriptor state', () => {
          // Arrange
          const node = new Node('output');

          // Act
          node.describe({
            label: 'readoutNode',
            metadata: { priority: 2, reusable: true },
          });

          // Assert
          expect({
            label: node.label,
            intent: node.intent,
            metadata: node.metadata,
          }).toStrictEqual({
            label: 'readoutNode',
            intent: 'output',
            metadata: { priority: 2, reusable: true },
          });
        });
      });
    });
  });

  describe('activate()', () => {
    describe('given an identity node with only bias', () => {
      describe('when activating without an explicit input', () => {
        it('returns the bias value', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.identity;
          node.bias = 0.1;

          // Act
          const actualActivation = node.activate();

          // Assert
          expect(actualActivation).toBeCloseTo(0.1, 12);
        });
      });
    });

    describe('given an explicit input on a hidden node', () => {
      describe('when activating the node', () => {
        it('applies the node squash function to that input', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.identity;

          // Act
          const actualActivation = node.activate(0.5);

          // Assert
          expect(actualActivation).toBeCloseTo(0.5, 12);
        });
      });
    });

    describe('given a hidden node with a non-neutral response', () => {
      describe('when activating the node from an explicit input', () => {
        it('applies the response multiplier before squashing', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.identity;
          node.response = 1.5;

          // Act
          const actualActivation = node.activate(0.5);

          // Assert
          expect(actualActivation).toBeCloseTo(0.75, 12);
        });
      });
    });

    describe('given one incoming connection', () => {
      describe('when activating the node', () => {
        it('sums weighted input and bias before squashing', () => {
          // Arrange
          const hiddenNode = new Node('hidden');
          hiddenNode.squash = Activation.identity;
          hiddenNode.bias = 0.1;
          const inputNode = new Node('input');
          inputNode.activation = 1;
          hiddenNode.connections.in.push(
            new Connection(inputNode, hiddenNode, 0.2),
          );

          // Act
          const actualActivation = hiddenNode.activate();

          // Assert
          expect(actualActivation).toBeCloseTo(0.3, 12);
        });
      });
    });

    describe('given a self-connection', () => {
      describe('when activating twice', () => {
        it('feeds the previous state back into the new state', () => {
          // Arrange
          const node = new Node('hidden');
          node.squash = Activation.identity;
          node.bias = 0.1;
          node.connect(node, 0.5);
          node.activate(1);

          // Act
          node.activate();

          // Assert
          expect(node.state).toBeCloseTo(0.6, 12);
        });
      });
    });
  });

  describe('noTraceActivate()', () => {
    describe('given a dropped node', () => {
      describe('when activating without traces', () => {
        it('returns zero', () => {
          // Arrange
          const node = new Node('hidden');
          node.mask = 0;

          // Act
          const actualActivation = node.noTraceActivate();

          // Assert
          expect(actualActivation).toBe(0);
        });
      });
    });
  });

  describe('setActivation()', () => {
    describe('given a custom activation function', () => {
      describe('when activating the node after setting it', () => {
        it('uses the new activation function', () => {
          // Arrange
          const node = new Node('hidden');
          node.setActivation((inputValue, shouldDerivative = false) =>
            shouldDerivative ? -7 : inputValue + 5,
          );

          // Act
          const actualActivation = node.activate(4);

          // Assert
          expect(actualActivation).toBe(9);
        });
      });
    });
  });

  describe('connect()', () => {
    describe('given the node itself as the target', () => {
      describe('when creating the connection', () => {
        it('creates one self-connection', () => {
          // Arrange
          const node = new Node('hidden');

          // Act
          const selfConnections = node.connect(node);

          // Assert
          expect(selfConnections).toHaveLength(1);
        });
      });
    });

    describe('given a group target', () => {
      describe('when connecting to all group nodes', () => {
        it('creates one outgoing connection per target node', () => {
          // Arrange
          const sourceNode = new Node('hidden');
          const targetGroup = {
            nodes: [new Node('hidden'), new Node('hidden')],
          };

          // Act
          const createdConnections = sourceNode.connect(targetGroup);

          // Assert
          expect(createdConnections).toHaveLength(2);
        });
      });
    });
  });

  describe('gate() and ungate()', () => {
    describe('given an outgoing connection', () => {
      describe('when gating and then ungating it', () => {
        it('restores the connection gater to null', () => {
          // Arrange
          const gaterNode = new Node('hidden');
          const sourceNode = new Node('hidden');
          const targetNode = new Node('hidden');
          const connection = sourceNode.connect(targetNode)[0];
          gaterNode.gate(connection);
          gaterNode.ungate(connection);

          // Act
          const actualGater = connection.gater;

          // Assert
          expect(actualGater).toBeNull();
        });
      });
    });
  });

  describe('toJSON() and fromJSON()', () => {
    describe('given a serialized node description', () => {
      describe('when rehydrating the node', () => {
        it('restores the named squash function', () => {
          // Arrange
          const restoredNode = Node.fromJSON({
            bias: 0.25,
            type: 'output',
            squash: 'relu',
            mask: 1,
          });

          // Act
          const actualSquash = restoredNode.squash;

          // Assert
          expect(actualSquash).toBe(Activation.relu);
        });
      });

      describe('when rehydrating the response value', () => {
        it('restores the explicit response parameter', () => {
          // Arrange
          const restoredNode = Node.fromJSON({
            bias: 0.25,
            type: 'output',
            squash: 'relu',
            mask: 1,
            response: 1.5,
          });

          // Act
          const actualResponse = restoredNode.response;

          // Assert
          expect(actualResponse).toBe(1.5);
        });
      });
    });
  });

  describe('property updates inside a network', () => {
    describe('given a simple input-output network', () => {
      let network: Network;
      let initialConnectionCount: number;
      let inputNode: Node;
      let outputNode: Node;

      beforeEach(() => {
        // Arrange
        network = new Network(2, 1, { seed: 412 });
        initialConnectionCount = network.connections.length;

        const candidateInputNode = network.nodes.find(
          (node) => node.type === 'input',
        );
        const candidateOutputNode = network.nodes.find(
          (node) => node.type === 'output',
        );

        if (!candidateInputNode || !candidateOutputNode) {
          throw new Error('Expected input and output nodes to exist');
        }

        inputNode = candidateInputNode;
        outputNode = candidateOutputNode;
      });

      describe('when updating the input bias', () => {
        it('keeps the network connection count unchanged', () => {
          // Act
          inputNode.bias = 0.5;

          // Assert
          expect(network.connections.length).toBe(initialConnectionCount);
        });
      });

      describe('when updating the output squash function', () => {
        it('keeps the network connection count unchanged', () => {
          // Act
          outputNode.squash = methods.Activation.tanh;

          // Assert
          expect(network.connections.length).toBe(initialConnectionCount);
        });
      });

      describe('when updating both node properties', () => {
        it('preserves the existing projection from input to output', () => {
          // Act
          inputNode.bias = 0.5;
          outputNode.squash = methods.Activation.tanh;

          // Assert
          expect(inputNode.isProjectingTo(outputNode)).toBe(true);
        });
      });
    });
  });
});
