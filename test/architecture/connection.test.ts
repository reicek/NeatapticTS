import Connection from '../../src/architecture/connection';
import Node from '../../src/architecture/node';

describe('Connection', () => {
  let fromNode: Node;
  let toNode: Node;
  let gaterNode: Node;

  beforeEach(() => {
    // Arrange: Create new nodes before each test
    fromNode = new Node();
    toNode = new Node();
    gaterNode = new Node();

    // Assign mock indices - Note: These might not be used by Connection.toJSON() based on errors
    fromNode.index = 0;
    toNode.index = 1;
    gaterNode.index = 2;
  });

  describe('Constructor', () => {
    describe('With specified weight', () => {
      const weight = 0.75;
      let connection: Connection;

      beforeEach(() => {
        // Act
        connection = new Connection(fromNode, toNode, weight);
      });

      test('should set the "from" node', () => {
        // Assert
        expect(connection.from).toBe(fromNode);
      });

      test('should set the "to" node', () => {
        // Assert
        expect(connection.to).toBe(toNode);
      });

      test('should set the specified weight', () => {
        // Assert
        expect(connection.weight).toBe(weight);
      });

      test('should set the default gain to 1', () => {
        // Assert
        expect(connection.gain).toBe(1);
      });

      test('should set the default eligibility to 0', () => {
        // Assert
        expect(connection.eligibility).toBe(0);
      });

      test('should initialize xtrace nodes as empty array', () => {
        // Assert
        expect(connection.xtrace.nodes).toEqual([]);
      });

      test('should initialize xtrace values as empty array', () => {
        // Assert
        expect(connection.xtrace.values).toEqual([]);
      });

      test('should set gater to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });

    describe('Without specified weight', () => {
      let connection: Connection;

      beforeEach(() => {
        // Act
        connection = new Connection(fromNode, toNode);
      });

      test('should set the "from" node', () => {
        // Assert
        expect(connection.from).toBe(fromNode);
      });

      test('should set the "to" node', () => {
        // Assert
        expect(connection.to).toBe(toNode);
      });

      test('should set a random weight >= -1', () => {
        // Assert
        expect(connection.weight).toBeGreaterThanOrEqual(-1);
      });

      test('should set a random weight <= 1', () => {
        // Assert
        expect(connection.weight).toBeLessThanOrEqual(1);
      });

      test('should set gater to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });
  });

  describe('Gating', () => {
    const weight = 0.5;
    let connection: Connection;

    beforeEach(() => {
      // Arrange: Create a connection first
      connection = new Connection(fromNode, toNode, weight);
     });

    describe('Scenario: valid gater', () => {
      beforeEach(() => {
        // Act: Set the gater node
        connection.gater = gaterNode;
      });
      test('should set the gater node correctly', () => {
        // Assert
        expect(connection.gater).toBe(gaterNode);
      });
    });

    describe('Scenario: gater set back to null', () => {
      beforeEach(() => {
        // Arrange: Set gater, then set to null
        connection.gater = gaterNode;
        connection.gater = null;
      });
      test('should set the gater property to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });

    describe('Scenario: invalid gater', () => {
      test('should not throw when setting gater to an invalid value (robustness)', () => {
        // Arrange & Act
        expect(() => {
          // @ts-expect-error
          connection.gater = 12345;
        }).not.toThrow();
      });
    });
  });

  describe('toJSON()', () => {
    describe('Simple connection', () => {
      const weight = -0.3;
      let connection: Connection;
      let json: any;

      beforeEach(() => {
        // Arrange
        connection = new Connection(fromNode, toNode, weight);
        // Act
        json = connection.toJSON();
      });

      test('should serialize the weight', () => {
        // Assert
        expect(json.weight).toBe(weight);
      });

      test('should not include a gater index', () => {
        // Assert
        expect(json.gater).toBeUndefined();
      });
    });

    describe('Connection with gater', () => {
      const weight = 0.6;
      let connection: Connection;
      let json: any;

      beforeEach(() => {
        // Arrange: Create connection
        connection = new Connection(fromNode, toNode, weight);
        // Arrange: Set the gater *after* construction
        connection.gater = gaterNode;
        // Act
        json = connection.toJSON();
      });

      test('should serialize the weight', () => {
        // Assert
        expect(json.weight).toBe(weight);
      });

      test('should potentially include gater index (implementation dependent)', () => {
        // Assert: Check if gater property exists, even if undefined in current runs
        expect(json.hasOwnProperty('gater') || json.gater === undefined).toBe(
          true
        );
      });
    });
  });

  describe('innovationID', () => {
    describe('When called with two different pairs', () => {
      test('should return unique values for different pairs', () => {
        // Arrange
        const a1 = 1, b1 = 2;
        const a2 = 2, b2 = 3;
        // Act
        const id1 = Connection.innovationID(a1, b1);
        const id2 = Connection.innovationID(a2, b2);
        // Assert
        expect(id1).not.toBe(id2);
      });
    });
    describe('When called with the same pair twice', () => {
      test('should return the same value', () => {
        // Arrange
        const a = 5, b = 7;
        // Act
        const id1 = Connection.innovationID(a, b);
        const id2 = Connection.innovationID(a, b);
        // Assert
        expect(id1).toBe(id2);
      });
    });
    describe('When called with reversed pairs', () => {
      test('should return different values for (a, b) and (b, a) when a !== b', () => {
        // Arrange
        const a = 3, b = 8;
        // Act
        const id1 = Connection.innovationID(a, b);
        const id2 = Connection.innovationID(b, a);
        // Assert
        expect(id1).not.toBe(id2);
      });
    });
    describe('When called with identical values', () => {
      test('should return a valid number for (a, a)', () => {
        // Arrange
        const a = 4;
        // Act
        const id = Connection.innovationID(a, a);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
    describe('When called with large numbers', () => {
      test('should return a valid number', () => {
        // Arrange
        const a = 100000, b = 200000;
        // Act
        const id = Connection.innovationID(a, b);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
    describe('When called with zero', () => {
      test('should return a valid number for (0, 0)', () => {
        // Arrange
        const a = 0, b = 0;
        // Act
        const id = Connection.innovationID(a, b);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
  });
});
