import Connection from './connection';
import Node from '../node/node';

interface ConnectionJsonShape {
  from: number | undefined;
  to: number | undefined;
  weight: number;
  gain: number;
  innovation: number;
  enabled: boolean;
  gater?: number;
}

describe('Connection', () => {
  let fromNode: Node;
  let toNode: Node;
  let gaterNode: Node;

  beforeEach(() => {
    // Arrange
    fromNode = new Node();
    toNode = new Node();
    gaterNode = new Node();
    fromNode.index = 0;
    toNode.index = 1;
    gaterNode.index = 2;
  });

  describe('constructor', () => {
    describe('given an explicit weight', () => {
      let connection: Connection;

      beforeEach(() => {
        // Arrange
        connection = new Connection(fromNode, toNode, 0.75);
      });

      describe('when reading the source node', () => {
        it('keeps the provided from node', () => {
          // Arrange
          const expectedNode = fromNode;

          // Act
          const actualNode = connection.from;

          // Assert
          expect(actualNode).toBe(expectedNode);
        });
      });

      describe('when reading the target node', () => {
        it('keeps the provided to node', () => {
          // Arrange
          const expectedNode = toNode;

          // Act
          const actualNode = connection.to;

          // Assert
          expect(actualNode).toBe(expectedNode);
        });
      });

      describe('when reading the stored weight', () => {
        it('uses the provided weight', () => {
          // Arrange
          const expectedWeight = 0.75;

          // Act
          const actualWeight = connection.weight;

          // Assert
          expect(actualWeight).toBe(expectedWeight);
        });
      });

      describe('when reading the default gain', () => {
        it('starts at one', () => {
          // Arrange
          const expectedGain = 1;

          // Act
          const actualGain = connection.gain;

          // Assert
          expect(actualGain).toBe(expectedGain);
        });
      });

      describe('when reading the default eligibility', () => {
        it('starts at zero', () => {
          // Arrange
          const expectedEligibility = 0;

          // Act
          const actualEligibility = connection.eligibility;

          // Assert
          expect(actualEligibility).toBe(expectedEligibility);
        });
      });

      describe('when reading the xtrace node list', () => {
        it('starts empty', () => {
          // Arrange
          const expectedNodes: Node[] = [];

          // Act
          const actualNodes = connection.xtrace.nodes;

          // Assert
          expect(actualNodes).toStrictEqual(expectedNodes);
        });
      });

      describe('when reading the xtrace value list', () => {
        it('starts empty', () => {
          // Arrange
          const expectedValues: number[] = [];

          // Act
          const actualValues = connection.xtrace.values;

          // Assert
          expect(actualValues).toStrictEqual(expectedValues);
        });
      });

      describe('when reading the gater', () => {
        it('starts as null', () => {
          // Arrange
          const expectedGater = null;

          // Act
          const actualGater = connection.gater;

          // Assert
          expect(actualGater).toBe(expectedGater);
        });
      });
    });

    describe('given no explicit weight', () => {
      describe('when the connection is created', () => {
        it('chooses a small default weight inside the initialization range', () => {
          // Arrange
          const connection = new Connection(fromNode, toNode);

          // Act
          const actualWeight = connection.weight;

          // Assert
          expect(actualWeight >= -0.1 && actualWeight <= 0.1).toBe(true);
        });
      });
    });
  });

  describe('gater', () => {
    let connection: Connection;

    beforeEach(() => {
      // Arrange
      connection = new Connection(fromNode, toNode, 0.5);
    });

    describe('given a real node', () => {
      describe('when assigning the gater', () => {
        it('stores the provided node', () => {
          // Arrange
          connection.gater = gaterNode;

          // Act
          const actualGater = connection.gater;

          // Assert
          expect(actualGater).toBe(gaterNode);
        });
      });
    });

    describe('given an existing gater', () => {
      describe('when clearing it back to null', () => {
        it('removes the gater reference', () => {
          // Arrange
          connection.gater = gaterNode;
          connection.gater = null;

          // Act
          const actualGater = connection.gater;

          // Assert
          expect(actualGater).toBeNull();
        });
      });
    });
  });

  describe('toJSON()', () => {
    describe('given a plain connection', () => {
      describe('when serializing the connection', () => {
        it('omits the gater field', () => {
          // Arrange
          const connection = new Connection(fromNode, toNode, -0.3);

          // Act
          const actualJson = connection.toJSON();

          // Assert
          expect(actualJson.gater).toBeUndefined();
        });

        it('preserves the core serialization fields', () => {
          // Arrange
          const connection = new Connection(fromNode, toNode, -0.3);
          const expectedJson: ConnectionJsonShape = {
            from: 0,
            to: 1,
            weight: -0.3,
            gain: 1,
            innovation: connection.innovation,
            enabled: true,
          };

          // Act
          const actualJson = connection.toJSON();

          // Assert
          expect(actualJson).toStrictEqual(expectedJson);
        });
      });
    });

    describe('given a gated connection', () => {
      describe('when serializing the connection', () => {
        it('includes the gater index', () => {
          // Arrange
          const connection = new Connection(fromNode, toNode, 0.6);
          connection.gater = gaterNode;

          // Act
          const actualJson = connection.toJSON();

          // Assert
          expect(actualJson.gater).toBe(gaterNode.index);
        });
      });
    });
  });

  describe('innovationID()', () => {
    describe('given the same ordered pair twice', () => {
      describe('when generating ids', () => {
        it('returns the same deterministic value', () => {
          // Arrange
          const sourceNodeId = 5;
          const targetNodeId = 7;

          // Act
          const generatedIds = [
            Connection.innovationID(sourceNodeId, targetNodeId),
            Connection.innovationID(sourceNodeId, targetNodeId),
          ];

          // Assert
          expect(generatedIds[0]).toBe(generatedIds[1]);
        });
      });
    });

    describe('given a reversed ordered pair', () => {
      describe('when generating ids', () => {
        it('returns a different value', () => {
          // Arrange
          const sourceNodeId = 3;
          const targetNodeId = 8;

          // Act
          const generatedIds = [
            Connection.innovationID(sourceNodeId, targetNodeId),
            Connection.innovationID(targetNodeId, sourceNodeId),
          ];

          // Assert
          expect(generatedIds[0] === generatedIds[1]).toBe(false);
        });
      });
    });
  });
});
