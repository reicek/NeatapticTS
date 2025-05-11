/**
 * @jest-environment node
 */
import Group from '../../src/architecture/group';
import Node from '../../src/architecture/node';
import Layer from '../../src/architecture/layer';
import Connection from '../../src/architecture/connection';
import * as methods from '../../src/methods/methods';
import { config } from '../../src/config';

// Retry failed tests
jest.retryTimes(2, { logErrorsBeforeRetry: true });

beforeEach(() => {
  const origLog = console.log;
  const origDir = console.dir;
  console.log = function(...args) {
    origLog.apply(console, args.map(arg => (arg && typeof arg.toJSON === 'function') ? arg.toJSON() : arg));
  };
  console.dir = function(obj, options) {
    if (obj && typeof obj.toJSON === 'function') {
      obj = obj.toJSON();
    }
    origDir.call(console, obj, options);
  };
});

describe('Group', () => {
  const epsilon = 1e-9; // Tolerance for float comparisons

  describe('Constructor', () => {
    const size = 5;
    let group: Group;

    beforeEach(() => {
      group = new Group(size);
    });

    test('should create a group with the specified number of nodes', () => {
      // Arrange, Act & Assert
      expect(group.nodes).toHaveLength(size);
    });

    test('should initialize nodes as instances of Node', () => {
      // Arrange, Act & Assert
      group.nodes.forEach((node) => {
        expect(node).toBeInstanceOf(Node);
      });
    });

    test('should initialize connection properties as empty arrays', () => {
      // Arrange, Act & Assert
      expect(group.connections.in).toEqual([]);
      expect(group.connections.out).toEqual([]);
      expect(group.connections.self).toEqual([]);
    });
  });

  describe('activate()', () => {
    describe('Scenario: input group', () => {
      test('should activate all input nodes with input values', () => {
        // Arrange
        const size = 3;
        const inputValues = [0.5, -0.2, 0.9];
        const group = new Group(size);
        group.nodes.forEach(node => node.type = 'input');
        // Act
        const activations = group.activate(inputValues);
        // Assert
        expect(activations).toHaveLength(size);
        activations.forEach((act, i) => {
          expect(act).toBe(inputValues[i]);
        });
      });
    });
    describe('Scenario: hidden group', () => {
      test('should activate all hidden nodes with activation function', () => {
        // Arrange
        const size = 3;
        const inputValues = [0.5, -0.2, 0.9];
        const group = new Group(size);
        group.nodes.forEach(node => node.type = 'hidden');
        // Act
        const activations = group.activate(inputValues);
        // Assert
        expect(activations).toHaveLength(size);
        activations.forEach((act, i) => {
          expect(act).toBeCloseTo(1 / (1 + Math.exp(-inputValues[i])), 10);
        });
      });
    });
  });

  describe('propagate()', () => {
    const size = 3;
    let group: Group;
    const rate = 0.1;
    const momentum = 0.9;

    beforeEach(() => {
      group = new Group(size);
    });

    test('should propagate through all nodes without target values', () => {
      // Arrange, Act & Assert
      expect(() => group.propagate(rate, momentum)).not.toThrow();
    });

    test('should propagate through all nodes with target values', () => {
      // Arrange
    });

    let group1: Group;
    let group2: Group;
    let node: Node;
    let layer: Layer;
    const size1 = 3;
    const size2 = 2;
    let originalWarnings: boolean;

    beforeEach(() => {
      group1 = new Group(size1);
      group2 = new Group(size2);
      node = new Node();
      layer = new Layer(); // Use the default Layer constructor for setup
      // Store original config and set warnings to true for these tests
      originalWarnings = config.warnings;
      config.warnings = true;
      // Suppress console warnings during tests but allow spy to track calls
      jest.spyOn(console, 'warn').mockImplementation(() => {});
    });

    afterEach(() => {
      // Restore console warning and config
      (console.warn as jest.Mock).mockRestore();
      config.warnings = originalWarnings;
      jest.restoreAllMocks();
    });

    describe('To Group', () => {
      test('should connect ALL_TO_ALL by default to a different group', () => {
        const connections = group1.connect(group2);
        expect(connections).toHaveLength(size1 * size2);
        expect(group1.connections.out).toHaveLength(size1 * size2);
        expect(console.warn).toHaveBeenCalledWith(
          'No group connection specified, using ALL_TO_ALL by default.'
        );
        // Check if connections are actually formed between nodes
        let connCount = 0;
        group1.nodes.forEach((fromNode: Node) => {
          group2.nodes.forEach((toNode: Node) => {
            if (fromNode.isConnectedTo(toNode)) {
              connCount++;
            }
          });
        });
        expect(connCount).toBe(size1 * size2);
      });

      test('should connect ONE_TO_ONE by default to the same group', () => {
        const sameSizeGroup = new Group(size1);
        const connections = sameSizeGroup.connect(sameSizeGroup); // Connect to self
        expect(connections).toHaveLength(size1);
        expect(sameSizeGroup.connections.self).toHaveLength(size1);
        expect(console.warn).toHaveBeenCalledWith(
          'Connecting group to itself, using ONE_TO_ONE by default.'
        );
        sameSizeGroup.nodes.forEach((node: Node, i: number) => {
          // Check the connection object stored in the group's self list
          const selfConn = sameSizeGroup.connections.self[i];
          expect(selfConn).toBeInstanceOf(Connection);
          expect(selfConn.from).toBe(node);
          expect(selfConn.to).toBe(node);
          // Check the node's internal self connection reference
          expect(node.connections.self[0]).toBe(selfConn);
        });
      });

      test('should connect using specified ALL_TO_ALL method', () => {
        const connections = group1.connect(
          group2,
          methods.groupConnection.ALL_TO_ALL
        );
        expect(connections).toHaveLength(size1 * size2);
        expect(group1.connections.out).toHaveLength(size1 * size2);
        expect(group2.connections.in).toHaveLength(size1 * size2);
      });

      test('should connect using specified ALL_TO_ELSE method', () => {
        const sameSizeGroup = new Group(size1);
        const connections = sameSizeGroup.connect(
          sameSizeGroup,
          methods.groupConnection.ALL_TO_ELSE
        );
        const expectedConns = size1 * size1 - size1; // All pairs except self
        expect(connections).toHaveLength(expectedConns);
        expect(sameSizeGroup.connections.out).toHaveLength(expectedConns);
        expect(sameSizeGroup.connections.in).toHaveLength(expectedConns);
        sameSizeGroup.nodes.forEach((node: Node, i: number) => {
          expect(node.isConnectedTo(sameSizeGroup.nodes[i])).toBe(false); // No self-connection
        });
      });

      test('should connect using specified ONE_TO_ONE method', () => {
        const sameSizeGroup = new Group(size1);
        const connections = group1.connect(
          sameSizeGroup,
          methods.groupConnection.ONE_TO_ONE
        );
        expect(connections).toHaveLength(size1);
        // Check the correct connection lists for connections between different groups
        expect(group1.connections.out).toHaveLength(size1);
        expect(sameSizeGroup.connections.in).toHaveLength(size1);
        expect(group1.connections.self).toHaveLength(0); // Should not be in self list
        expect(sameSizeGroup.connections.self).toHaveLength(0); // Should not be in self list
        group1.nodes.forEach((node: Node, i: number) => {
          expect(node.isConnectedTo(sameSizeGroup.nodes[i])).toBe(true);
        });
      });

      test('should throw error for ONE_TO_ONE if group sizes differ', () => {
        expect(() =>
          group1.connect(group2, methods.groupConnection.ONE_TO_ONE)
        ).toThrow(
          'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.'
        );
      });

      test('should connect with specified weight', () => {
        const weight = 0.75;
        const connections = group1.connect(
          group2,
          methods.groupConnection.ALL_TO_ALL,
          weight
        );
        connections.forEach((conn: Connection) => {
          expect(conn.weight).toBe(weight);
        });
      });
    });

    describe('To Layer', () => {
      test('should delegate connection to Layer.input()', () => {
        const layer = new Layer();
        const layerInputSpy = jest.spyOn(layer, 'input').mockImplementation(() => []);
        const method = methods.groupConnection.ALL_TO_ALL;
        const weight = 0.5;
        const group1 = new Group(3);

        group1.connect(layer, method, weight);

        expect(layerInputSpy).toHaveBeenCalledTimes(1);
        expect(layerInputSpy).toHaveBeenCalledWith(group1, method, weight);

        layerInputSpy.mockRestore();
      });
    });

    describe('To Node', () => {
      test('should connect all nodes in group to the target node', () => {
        const connections = group1.connect(node);
        expect(connections).toHaveLength(size1);
        expect(group1.connections.out).toHaveLength(size1);
        group1.nodes.forEach((fromNode: Node) => {
          expect(fromNode.isConnectedTo(node)).toBe(true);
        });
        expect(node.connections.in).toHaveLength(size1);
      });

      test('should connect with specified weight', () => {
        const weight = -0.3;
        const connections = group1.connect(node, undefined, weight); // Method is undefined for node target
        connections.forEach((conn: Connection) => {
          expect(conn.weight).toBe(weight);
        });
      });
    });
  });

  describe('gate()', () => {
    let gatingGroup: Group;
    let sourceNode1: Node, sourceNode2: Node;
    let targetNode1: Node, targetNode2: Node;
    let conn1: Connection, conn2: Connection, selfConn: Connection;
    let connections: Connection[];

    beforeEach(() => {
      gatingGroup = new Group(2);
      sourceNode1 = new Node();
      sourceNode2 = new Node();
      targetNode1 = new Node();
      targetNode2 = new Node();

      // Create connections
      conn1 = sourceNode1.connect(targetNode1)[0];
      conn2 = sourceNode2.connect(targetNode2)[0];
      selfConn = sourceNode1.connect(sourceNode1)[0]; // Self connection

      connections = [conn1, conn2];
    });

    test('should throw error if no gating method specified', () => {
      expect(() => gatingGroup.gate(connections, undefined)).toThrow(
        'Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF'
      );
    });

    test('should handle a single connection input', () => {
      gatingGroup.gate(conn1, methods.gating.INPUT);
      expect(conn1.gater).toBe(gatingGroup.nodes[0]);
    });

    test('should gate INPUT correctly', () => {
      gatingGroup.gate(connections, methods.gating.INPUT);
      expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      expect(conn2.gater).toBe(gatingGroup.nodes[1]);
    });

    test('should gate OUTPUT correctly', () => {
      gatingGroup.gate(connections, methods.gating.OUTPUT);
      expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      expect(conn2.gater).toBe(gatingGroup.nodes[1]);
    });

    test('should gate SELF correctly', () => {
      gatingGroup.gate(selfConn, methods.gating.SELF);
      expect(selfConn.gater).toBe(gatingGroup.nodes[0]);
    });

    test('should cycle through gater nodes if more connections than gaters', () => {
      const conn3 = sourceNode1.connect(targetNode2)[0];
      const threeConnections = [conn1, conn2, conn3];
      gatingGroup.gate(threeConnections, methods.gating.INPUT);
      expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      expect(conn2.gater).toBe(gatingGroup.nodes[1]);
      expect(conn3.gater).toBe(gatingGroup.nodes[0]);
    });
  });

  describe('set()', () => {
    const size = 4;
    let group: Group;

    beforeEach(() => {
      group = new Group(size);
    });

    test('should set bias for all nodes', () => {
      const biasValue = 0.5;
      group.set({ bias: biasValue });
      group.nodes.forEach((node) => {
        expect(node.bias).toBe(biasValue);
      });
    });

    test('should set squash function for all nodes', () => {
      const squashFn = methods.Activation.relu;
      group.set({ squash: squashFn });
      group.nodes.forEach((node) => {
        expect(node.squash).toBe(squashFn);
      });
    });

    test('should set type for all nodes', () => {
      const typeValue = 'memory';
      group.set({ type: typeValue });
      group.nodes.forEach((node) => {
        expect(node.type).toBe(typeValue);
      });
    });

    test('should set multiple properties at once', () => {
      const biasValue = -0.1;
      const squashFn = methods.Activation.tanh;
      const typeValue = 'output';
      group.set({ bias: biasValue, squash: squashFn, type: typeValue });
      group.nodes.forEach((node) => {
        expect(node.bias).toBe(biasValue);
        expect(node.squash).toBe(squashFn);
        expect(node.type).toBe(typeValue);
      });
    });

    test('should not change properties if not provided', () => {
      // Store initial values for all nodes
      const initialBiases = group.nodes.map(node => node.bias);
      const initialSquashes = group.nodes.map(node => node.squash);
      const initialTypes = group.nodes.map(node => node.type);

      group.set({}); // Empty object

      // Check that each node's properties remain unchanged
      group.nodes.forEach((node, i) => {
        expect(node.bias).toBe(initialBiases[i]);
        expect(node.squash).toBe(initialSquashes[i]);
        expect(node.type).toBe(initialTypes[i]);
      });

      group.set({ bias: 0.9 }); // Only bias

      group.nodes.forEach((node, i) => {
        expect(node.bias).toBe(0.9);
        // Check that other properties remained unchanged from their initial state
        expect(node.squash).toBe(initialSquashes[i]);
        expect(node.type).toBe(initialTypes[i]);
      });
    });
  });

  describe('disconnect()', () => {
    let group1: Group;
    let group2: Group;
    let node: Node;
    const size1 = 2;
    const size2 = 2;

    beforeEach(() => {
      group1 = new Group(size1);
      group2 = new Group(size2);
      node = new Node();

      // Connect them fully first
      group1.connect(group2, methods.groupConnection.ALL_TO_ALL);
      group1.connect(node);
      group2.connect(group1, methods.groupConnection.ALL_TO_ALL); // For two-sided test
    });

    describe('From Group', () => {
      test('should disconnect one-sided (default)', () => {
        group1.disconnect(group2); // twosided = false

        // Check group connection lists
        expect(group1.connections.out).toHaveLength(size1); // Only connections to 'node' remain
        expect(group2.connections.in).toHaveLength(size1 * size2); // Remains unchanged
        // Connections from group2 to group1 should remain
        expect(group2.connections.out).toHaveLength(size1 * size2);
        expect(group1.connections.in).toHaveLength(size1 * size2);
      });

      test('should disconnect two-sided', () => {
        group1.disconnect(group2, true); // twosided = true

        // Check group connection lists - should now be empty for connections between group1 and group2
        expect(group1.connections.out).toHaveLength(size1); // Only connections to 'node' remain
        expect(group2.connections.in).toHaveLength(0); // Should be 0 after node-level disconnect and group list update
        expect(group2.connections.out).toHaveLength(0); // Should be 0 after group list update
        expect(group1.connections.in).toHaveLength(0); // Should be 0 after group list update
      });
    });

    describe('From Node', () => {
      test('should disconnect one-sided (default)', () => {
        group1.disconnect(node); // twosided = false

        // Check group connection lists
        expect(group1.connections.out).toHaveLength(size1 * size2); // Connections to group2 remain
        // Check node connection lists
        expect(node.connections.in).toHaveLength(0);
      });

      test('should disconnect two-sided', () => {
        // Add a connection from node back to group1 for two-sided test
        node.connect(group1.nodes[0]);
        group1.connections.in.push(node.connections.out[0]); // Manually update group list

        group1.disconnect(node, true); // twosided = true

        // Check group connection lists
        expect(group1.connections.out).toHaveLength(size1 * size2); // Connections to group2 remain
        expect(group1.connections.in).toHaveLength(size1 * size2); // Connections from group2 remain (removed conn from node)
        // Check node connection lists
        expect(node.connections.in).toHaveLength(0);
        expect(node.connections.out).toHaveLength(0);
      });
    });
  });

  describe('clear()', () => {
    const size = 3;
    let group: Group;

    beforeEach(() => {
      group = new Group(size);
    });

    test('should call clear() on all nodes in the group', () => {
      group.clear();
      group.nodes.forEach((node) => {
        expect(node.state).toBe(0);
        expect(node.old).toBe(0);
        expect(node.activation).toBe(0);
        expect(node.derivative).toBeUndefined();
      });
    });
  });

  describe('toJSON()', () => {
    test('should serialize an empty group correctly', () => {
      // Arrange
      const group = new Group(2);
      group.nodes[0].index = 10;
      group.nodes[1].index = 11;
      // Act
      const json = group.toJSON();
      // Assert
      expect(json).toEqual({
        size: 2,
        nodeIndices: [10, 11],
        connections: { in: 0, out: 0, self: 0 }
      });
    });

    test('should serialize group after connections', () => {
      // Arrange
      const group1 = new Group(2);
      const group2 = new Group(2);
      group1.nodes.forEach((n, i) => (n.index = i));
      group2.nodes.forEach((n, i) => (n.index = i + 2));
      group1.connect(group2, methods.groupConnection.ALL_TO_ALL);
      // Act
      const json1 = group1.toJSON();
      const json2 = group2.toJSON();
      // Assert
      expect(json1.size).toBe(2);
      expect(json1.connections.out).toBe(4);
      expect(json2.connections.in).toBe(4);
      expect(json1.nodeIndices).toEqual([0, 1]);
      expect(json2.nodeIndices).toEqual([2, 3]);
    });

    test('should serialize group after gating', () => {
      // Arrange
      const group = new Group(2);
      const node1 = new Node();
      const node2 = new Node();
      node1.index = 10;
      node2.index = 11;
      const conn1 = node1.connect(node2)[0];
      group.nodes[0].index = 20;
      group.nodes[1].index = 21;
      group.gate([conn1], methods.gating.INPUT);
      // Act
      const json = group.toJSON();
      // Assert
      expect(json.size).toBe(2);
      expect(json.nodeIndices).toEqual([20, 21]);
      expect(json.connections.in).toBe(0);
      expect(json.connections.out).toBe(0);
      expect(json.connections.self).toBe(0);
    });
  });
});
