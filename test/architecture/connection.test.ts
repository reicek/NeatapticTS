import { suppressConsoleOutput } from '../utils/console-helper';
import Connection from '../../src/architecture/connection';
import Node from '../../src/architecture/node';

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

  describe('Constructor', () => {
    describe('Scenario: with specified weight', () => {
      let connection: Connection;
      const weight = 0.75;

      beforeEach(() => {
        // Arrange & Act
        connection = new Connection(fromNode, toNode, weight);
      });

      it('sets the "from" node', () => {
        // Assert
        expect(connection.from).toBe(fromNode);
      });

      it('sets the "to" node', () => {
        // Assert
        expect(connection.to).toBe(toNode);
      });

      it('sets the specified weight', () => {
        // Assert
        expect(connection.weight).toBe(weight);
      });

      it('sets the default gain to 1', () => {
        // Assert
        expect(connection.gain).toBe(1);
      });

      it('sets the default eligibility to 0', () => {
        // Assert
        expect(connection.eligibility).toBe(0);
      });

      it('initializes xtrace nodes as empty array', () => {
        // Assert
        expect(connection.xtrace.nodes).toEqual([]);
      });

      it('initializes xtrace values as empty array', () => {
        // Assert
        expect(connection.xtrace.values).toEqual([]);
      });

      it('sets gater to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });

    describe('Scenario: without specified weight', () => {
      let connection: Connection;

      beforeEach(() => {
        // Arrange & Act
        connection = new Connection(fromNode, toNode);
      });

      it('sets the "from" node', () => {
        // Assert
        expect(connection.from).toBe(fromNode);
      });

      it('sets the "to" node', () => {
        // Assert
        expect(connection.to).toBe(toNode);
      });

      it('sets a random weight >= -0.1', () => {
        // Assert
        expect(connection.weight).toBeGreaterThanOrEqual(-0.1);
      });

      it('sets a random weight <= 0.1', () => {
        // Assert
        expect(connection.weight).toBeLessThanOrEqual(0.1);
      });

      it('sets gater to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });
  });

  describe('Gating', () => {
    let connection: Connection;
    const weight = 0.5;

    beforeEach(() => {
      // Arrange
      connection = new Connection(fromNode, toNode, weight);
    });

    describe('Scenario: valid gater', () => {
      beforeEach(() => {
        // Act
        connection.gater = gaterNode;
      });
      it('sets the gater node correctly', () => {
        // Assert
        expect(connection.gater).toBe(gaterNode);
      });
    });

    describe('Scenario: gater set back to null', () => {
      beforeEach(() => {
        // Arrange
        connection.gater = gaterNode;
        // Act
        connection.gater = null;
      });
      it('sets the gater property to null', () => {
        // Assert
        expect(connection.gater).toBeNull();
      });
    });

    describe('Scenario: invalid gater', () => {
      beforeEach(() => {
        // Arrange
        suppressConsoleOutput();
      });
      it('does not throw when setting gater to an invalid value', () => {
        // Arrange & Act & Assert
        expect(() => {
          // @ts-expect-error
          connection.gater = 12345;
        }).not.toThrow();
      });
    });
  });

  describe('toJSON()', () => {
    describe('Scenario: simple connection', () => {
      let connection: Connection;
      let json: any;
      const weight = -0.3;

      beforeEach(() => {
        // Arrange
        connection = new Connection(fromNode, toNode, weight);
        // Act
        json = connection.toJSON();
      });

      it('serializes the weight', () => {
        // Assert
        expect(json.weight).toBe(weight);
      });

      it('serializes the from index', () => {
        // Assert
        expect(json.from).toBe(fromNode.index);
      });

      it('serializes the to index', () => {
        // Assert
        expect(json.to).toBe(toNode.index);
      });

      it('serializes the gain', () => {
        // Assert
        expect(json.gain).toBe(connection.gain);
      });

      it('does not include a gater index', () => {
        // Assert
        expect(json.gater).toBeUndefined();
      });
    });

    describe('Scenario: connection with gater', () => {
      let connection: Connection;
      let json: any;
      const weight = 0.6;

      beforeEach(() => {
        // Arrange
        connection = new Connection(fromNode, toNode, weight);
        connection.gater = gaterNode;
        // Act
        json = connection.toJSON();
      });

      it('serializes the weight', () => {
        // Assert
        expect(json.weight).toBe(weight);
      });

      it('includes gater property with correct index', () => {
        // Assert
        expect(json.gater).toBe(gaterNode.index);
      });
    });
  });

  describe('innovationID', () => {
    describe('Scenario: two different pairs', () => {
      it('returns unique values for different pairs', () => {
        // Arrange
        const a1 = 1,
          b1 = 2;
        const a2 = 2,
          b2 = 3;
        // Act
        const id1 = Connection.innovationID(a1, b1);
        const id2 = Connection.innovationID(a2, b2);
        // Assert
        expect(id1).not.toBe(id2);
      });
    });
    describe('Scenario: same pair twice', () => {
      it('returns the same value', () => {
        // Arrange
        const a = 5,
          b = 7;
        // Act
        const id1 = Connection.innovationID(a, b);
        const id2 = Connection.innovationID(a, b);
        // Assert
        expect(id1).toBe(id2);
      });
    });
    describe('Scenario: reversed pairs', () => {
      it('returns different values for (a, b) and (b, a) when a !== b', () => {
        // Arrange
        const a = 3,
          b = 8;
        // Act
        const id1 = Connection.innovationID(a, b);
        const id2 = Connection.innovationID(b, a);
        // Assert
        expect(id1).not.toBe(id2);
      });
    });
    describe('Scenario: identical values', () => {
      it('returns a valid number for (a, a)', () => {
        // Arrange
        const a = 4;
        // Act
        const id = Connection.innovationID(a, a);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
    describe('Scenario: large numbers', () => {
      it('returns a valid number', () => {
        // Arrange
        const a = 100000,
          b = 200000;
        // Act
        const id = Connection.innovationID(a, b);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
    describe('Scenario: zero', () => {
      it('returns a valid number for (0, 0)', () => {
        // Arrange
        const a = 0,
          b = 0;
        // Act
        const id = Connection.innovationID(a, b);
        // Assert
        expect(typeof id).toBe('number');
      });
    });
  });
});
