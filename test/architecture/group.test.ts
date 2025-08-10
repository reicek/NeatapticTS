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
  console.log = function (...args) {
    origLog.apply(
      console,
      args.map((arg) =>
        arg && typeof arg.toJSON === 'function' ? arg.toJSON() : arg
      )
    );
  };
  console.dir = function (obj, options) {
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
      // Arrange
      group = new Group(size);
    });

    it('should create a group with the specified number of nodes', () => {
      // Arrange done in beforeEach
      // Act
      const nodeCount = group.nodes.length;
      // Assert
      expect(nodeCount).toBe(size);
    });

    it('should initialize all nodes as instances of Node', () => {
      // Arrange done in beforeEach
      // Act
      const allNodesAreNode = group.nodes.every((node) => node instanceof Node);
      // Assert
      expect(allNodesAreNode).toBe(true);
    });

    it('should initialize connection properties as empty arrays (in)', () => {
      // Arrange done in beforeEach
      // Act
      const inConnections = group.connections.in;
      // Assert
      expect(inConnections).toEqual([]);
    });

    it('should initialize connection properties as empty arrays (out)', () => {
      // Arrange done in beforeEach
      // Act
      const outConnections = group.connections.out;
      // Assert
      expect(outConnections).toEqual([]);
    });

    it('should initialize connection properties as empty arrays (self)', () => {
      // Arrange done in beforeEach
      // Act
      const selfConnections = group.connections.self;
      // Assert
      expect(selfConnections).toEqual([]);
    });
  });

  describe('activate()', () => {
    describe('Scenario: input group', () => {
      const size = 3;
      const inputValues = [0.5, -0.2, 0.9];
      let group: Group;
      beforeEach(() => {
        // Arrange
        group = new Group(size);
        group.nodes.forEach((node) => (node.type = 'input'));
      });

      describe('when activating all input nodes with input values', () => {
        let activations: number[];
        beforeEach(() => {
          // Act
          activations = group.activate(inputValues);
        });
        it('returns an array of length equal to group size', () => {
          // Assert
          expect(activations).toHaveLength(size);
        });
        it('assigns correct activation value to node 0', () => {
          // Assert
          expect(activations[0]).toBe(inputValues[0]);
        });
        it('assigns correct activation value to node 1', () => {
          // Assert
          expect(activations[1]).toBe(inputValues[1]);
        });
        it('assigns correct activation value to node 2', () => {
          // Assert
          expect(activations[2]).toBe(inputValues[2]);
        });
      });

      describe('when input array length does not match group size', () => {
        it('throws an error', () => {
          // Act & Assert
          expect(() => group.activate([1, 2])).toThrow(
            'Array with values should be same as the amount of nodes!'
          );
        });
      });
    });

    describe('Scenario: hidden group', () => {
      const size = 3;
      const inputValues = [0.5, -0.2, 0.9];
      let group: Group;
      beforeEach(() => {
        // Arrange
        group = new Group(size);
        group.nodes.forEach((node) => (node.type = 'hidden'));
      });

      describe('when activating all hidden nodes with input values', () => {
        let activations: number[];
        beforeEach(() => {
          // Act
          activations = group.activate(inputValues);
        });
        it('returns an array of length equal to group size', () => {
          // Assert
          expect(activations).toHaveLength(size);
        });
        it('assigns correct activation value to node 0', () => {
          // Assert
          expect(activations[0]).toBeCloseTo(
            1 / (1 + Math.exp(-inputValues[0])),
            10
          );
        });
        it('assigns correct activation value to node 1', () => {
          // Assert
          expect(activations[1]).toBeCloseTo(
            1 / (1 + Math.exp(-inputValues[1])),
            10
          );
        });
        it('assigns correct activation value to node 2', () => {
          // Assert
          expect(activations[2]).toBeCloseTo(
            1 / (1 + Math.exp(-inputValues[2])),
            10
          );
        });
      });

      describe('when input array length does not match group size', () => {
        it('throws an error', () => {
          // Act & Assert
          expect(() => group.activate([1, 2])).toThrow(
            'Array with values should be same as the amount of nodes!'
          );
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
      // Arrange
      group = new Group(size);
    });

    describe('when propagating without target values', () => {
      it('does not throw', () => {
        // Act & Assert
        expect(() => group.propagate(rate, momentum)).not.toThrow();
      });
    });

    describe('when propagating with target values', () => {
      it('does not throw if target length matches group size', () => {
        // Arrange
        const targets = [0.1, 0.2, 0.3];
        // Act & Assert
        expect(() => group.propagate(rate, momentum, targets)).not.toThrow();
      });
      it('throws if target length does not match group size', () => {
        // Arrange
        const targets = [0.1, 0.2];
        // Act & Assert
        expect(() => group.propagate(rate, momentum, targets)).toThrow(
          'Array with values should be same as the amount of nodes!'
        );
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
      // Arrange
      gatingGroup = new Group(2);
      sourceNode1 = new Node();
      sourceNode2 = new Node();
      targetNode1 = new Node();
      targetNode2 = new Node();
      conn1 = sourceNode1.connect(targetNode1)[0];
      conn2 = sourceNode2.connect(targetNode2)[0];
      selfConn = sourceNode1.connect(sourceNode1)[0];
      connections = [conn1, conn2];
    });

    describe('when no gating method is specified', () => {
      it('throws an error', () => {
        expect(() => gatingGroup.gate(connections, undefined)).toThrow(
          'Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF'
        );
      });
    });

    describe('when gating a single connection (INPUT)', () => {
      beforeEach(() => {
        // Act
        gatingGroup.gate(conn1, methods.gating.INPUT);
      });
      it('assigns the first node as gater', () => {
        expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      });
    });

    describe('when gating multiple connections (INPUT)', () => {
      beforeEach(() => {
        // Act
        gatingGroup.gate(connections, methods.gating.INPUT);
      });
      it('assigns the first node as gater for conn1', () => {
        expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      });
      it('assigns the second node as gater for conn2', () => {
        expect(conn2.gater).toBe(gatingGroup.nodes[1]);
      });
    });

    describe('when gating multiple connections (OUTPUT)', () => {
      beforeEach(() => {
        // Act
        gatingGroup.gate(connections, methods.gating.OUTPUT);
      });
      it('assigns the first node as gater for conn1', () => {
        expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      });
      it('assigns the second node as gater for conn2', () => {
        expect(conn2.gater).toBe(gatingGroup.nodes[1]);
      });
    });

    describe('when gating a self connection (SELF)', () => {
      beforeEach(() => {
        // Act
        gatingGroup.gate(selfConn, methods.gating.SELF);
      });
      it('assigns the first node as gater for selfConn', () => {
        expect(selfConn.gater).toBe(gatingGroup.nodes[0]);
      });
    });

    describe('when more connections than gaters (cycle)', () => {
      let conn3: Connection;
      beforeEach(() => {
        conn3 = sourceNode1.connect(targetNode2)[0];
        const threeConnections = [conn1, conn2, conn3];
        gatingGroup.gate(threeConnections, methods.gating.INPUT);
      });
      it('cycles gater assignment for conn1', () => {
        expect(conn1.gater).toBe(gatingGroup.nodes[0]);
      });
      it('cycles gater assignment for conn2', () => {
        expect(conn2.gater).toBe(gatingGroup.nodes[1]);
      });
      it('cycles gater assignment for conn3', () => {
        expect(conn3.gater).toBe(gatingGroup.nodes[0]);
      });
    });
  });

  describe('set()', () => {
    const size = 4;
    let group: Group;

    beforeEach(() => {
      // Arrange
      group = new Group(size);
    });

    describe('when setting bias for all nodes', () => {
      const biasValue = 0.5;
      beforeEach(() => {
        // Act
        group.set({ bias: biasValue });
      });
      it('sets bias for node 0', () => {
        expect(group.nodes[0].bias).toBe(biasValue);
      });
      it('sets bias for node 1', () => {
        expect(group.nodes[1].bias).toBe(biasValue);
      });
      it('sets bias for node 2', () => {
        expect(group.nodes[2].bias).toBe(biasValue);
      });
      it('sets bias for node 3', () => {
        expect(group.nodes[3].bias).toBe(biasValue);
      });
    });

    describe('when setting squash function for all nodes', () => {
      const squashFn = methods.Activation.relu;
      beforeEach(() => {
        // Act
        group.set({ squash: squashFn });
      });
      it('sets squash for node 0', () => {
        expect(group.nodes[0].squash).toBe(squashFn);
      });
      it('sets squash for node 1', () => {
        expect(group.nodes[1].squash).toBe(squashFn);
      });
      it('sets squash for node 2', () => {
        expect(group.nodes[2].squash).toBe(squashFn);
      });
      it('sets squash for node 3', () => {
        expect(group.nodes[3].squash).toBe(squashFn);
      });
    });

    describe('when setting type for all nodes', () => {
      const typeValue = 'memory';
      beforeEach(() => {
        // Act
        group.set({ type: typeValue });
      });
      it('sets type for node 0', () => {
        expect(group.nodes[0].type).toBe(typeValue);
      });
      it('sets type for node 1', () => {
        expect(group.nodes[1].type).toBe(typeValue);
      });
      it('sets type for node 2', () => {
        expect(group.nodes[2].type).toBe(typeValue);
      });
      it('sets type for node 3', () => {
        expect(group.nodes[3].type).toBe(typeValue);
      });
    });

    describe('when setting multiple properties at once', () => {
      const biasValue = -0.1;
      const squashFn = methods.Activation.tanh;
      const typeValue = 'output';
      beforeEach(() => {
        // Act
        group.set({ bias: biasValue, squash: squashFn, type: typeValue });
      });
      it('sets bias for all nodes', () => {
        group.nodes.forEach((node) => {
          expect(node.bias).toBe(biasValue);
        });
      });
      it('sets squash for all nodes', () => {
        group.nodes.forEach((node) => {
          expect(node.squash).toBe(squashFn);
        });
      });
      it('sets type for all nodes', () => {
        group.nodes.forEach((node) => {
          expect(node.type).toBe(typeValue);
        });
      });
    });

    describe('when not changing properties if not provided', () => {
      let initialBiases: number[];
      let initialSquashes: any[];
      let initialTypes: string[];
      beforeEach(() => {
        // Arrange
        initialBiases = group.nodes.map((node) => node.bias);
        initialSquashes = group.nodes.map((node) => node.squash);
        initialTypes = group.nodes.map((node) => node.type);
        // Act
        group.set({});
      });
      it('does not change bias', () => {
        group.nodes.forEach((node, i) => {
          expect(node.bias).toBe(initialBiases[i]);
        });
      });
      it('does not change squash', () => {
        group.nodes.forEach((node, i) => {
          expect(node.squash).toBe(initialSquashes[i]);
        });
      });
      it('does not change type', () => {
        group.nodes.forEach((node, i) => {
          expect(node.type).toBe(initialTypes[i]);
        });
      });
    });

    describe('when setting only bias', () => {
      let initialSquashes: any[];
      let initialTypes: string[];
      beforeEach(() => {
        // Arrange
        initialSquashes = group.nodes.map((node) => node.squash);
        initialTypes = group.nodes.map((node) => node.type);
        // Act
        group.set({ bias: 0.9 });
      });
      it('sets bias for all nodes', () => {
        group.nodes.forEach((node) => {
          expect(node.bias).toBe(0.9);
        });
      });
      it('does not change squash', () => {
        group.nodes.forEach((node, i) => {
          expect(node.squash).toBe(initialSquashes[i]);
        });
      });
      it('does not change type', () => {
        group.nodes.forEach((node, i) => {
          expect(node.type).toBe(initialTypes[i]);
        });
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
      // Arrange
      group1 = new Group(size1);
      group2 = new Group(size2);
      node = new Node();
      group1.connect(group2, methods.groupConnection.ALL_TO_ALL);
      group1.connect(node);
      group2.connect(group1, methods.groupConnection.ALL_TO_ALL);
    });

    describe('Scenario: From Group', () => {
      describe('when disconnecting one-sided (default)', () => {
        beforeEach(() => {
          // Act
          group1.disconnect(group2);
        });
        it('removes out connections from group1 to group2', () => {
          expect(group1.connections.out).toHaveLength(size1);
        });
        it('does not change group2 in connections', () => {
          expect(group2.connections.in).toHaveLength(size1 * size2);
        });
        it('does not change group2 out connections', () => {
          expect(group2.connections.out).toHaveLength(size1 * size2);
        });
        it('does not change group1 in connections', () => {
          expect(group1.connections.in).toHaveLength(size1 * size2);
        });
      });

      describe('when disconnecting two-sided', () => {
        beforeEach(() => {
          // Act
          group1.disconnect(group2, true);
        });
        it('removes out connections from group1 to group2', () => {
          expect(group1.connections.out).toHaveLength(size1);
        });
        it('removes in connections from group2', () => {
          expect(group2.connections.in).toHaveLength(0);
        });
        it('removes out connections from group2', () => {
          expect(group2.connections.out).toHaveLength(0);
        });
        it('removes in connections from group1', () => {
          expect(group1.connections.in).toHaveLength(0);
        });
      });
    });

    describe('Scenario: From Node', () => {
      describe('when disconnecting one-sided (default)', () => {
        beforeEach(() => {
          // Act
          group1.disconnect(node);
        });
        it('does not change group1 out connections to group2', () => {
          expect(group1.connections.out).toHaveLength(size1 * size2);
        });
        it('removes node in connections', () => {
          expect(node.connections.in).toHaveLength(0);
        });
      });

      describe('when disconnecting two-sided', () => {
        beforeEach(() => {
          // Arrange
          node.connect(group1.nodes[0]);
          group1.connections.in.push(node.connections.out[0]);
          // Act
          group1.disconnect(node, true);
        });
        it('does not change group1 out connections to group2', () => {
          expect(group1.connections.out).toHaveLength(size1 * size2);
        });
        it('removes group1 in connections from node', () => {
          expect(group1.connections.in).toHaveLength(size1 * size2);
        });
        it('removes node in connections', () => {
          expect(node.connections.in).toHaveLength(0);
        });
        it('removes node out connections', () => {
          expect(node.connections.out).toHaveLength(0);
        });
      });
    });
  });

  describe('clear()', () => {
    const size = 3;
    let group: Group;

    beforeEach(() => {
      // Arrange
      group = new Group(size);
      // Set non-default values
      group.nodes.forEach((node) => {
        node.state = 1;
        node.old = 2;
        node.activation = 3;
        node.derivative = 4;
      });
    });

    beforeEach(() => {
      // Act
      group.clear();
    });

    it('resets state for all nodes', () => {
      group.nodes.forEach((node) => {
        expect(node.state).toBe(0);
      });
    });
    it('resets old for all nodes', () => {
      group.nodes.forEach((node) => {
        expect(node.old).toBe(0);
      });
    });
    it('resets activation for all nodes', () => {
      group.nodes.forEach((node) => {
        expect(node.activation).toBe(0);
      });
    });
    it('resets derivative for all nodes', () => {
      group.nodes.forEach((node) => {
        expect(node.derivative).toBe(4);
      });
    });
  });

  describe('toJSON()', () => {
    describe('when serializing an empty group', () => {
      let group: Group;
      let json: any;
      beforeEach(() => {
        // Arrange
        group = new Group(2);
        group.nodes[0].index = 10;
        group.nodes[1].index = 11;
        // Act
        json = group.toJSON();
      });
      it('serializes size', () => {
        expect(json.size).toBe(2);
      });
      it('serializes nodeIndices', () => {
        expect(json.nodeIndices).toEqual([10, 11]);
      });
      it('serializes connections.in', () => {
        expect(json.connections.in).toBe(0);
      });
      it('serializes connections.out', () => {
        expect(json.connections.out).toBe(0);
      });
      it('serializes connections.self', () => {
        expect(json.connections.self).toBe(0);
      });
    });

    describe('when serializing group after connections', () => {
      let group1: Group;
      let group2: Group;
      let json1: any;
      let json2: any;
      beforeEach(() => {
        // Arrange
        group1 = new Group(2);
        group2 = new Group(2);
        group1.nodes.forEach((n, i) => (n.index = i));
        group2.nodes.forEach((n, i) => (n.index = i + 2));
        group1.connect(group2, methods.groupConnection.ALL_TO_ALL);
        // Act
        json1 = group1.toJSON();
        json2 = group2.toJSON();
      });
      it('serializes size for group1', () => {
        expect(json1.size).toBe(2);
      });
      it('serializes out connections for group1', () => {
        expect(json1.connections.out).toBe(4);
      });
      it('serializes in connections for group2', () => {
        expect(json2.connections.in).toBe(4);
      });
      it('serializes nodeIndices for group1', () => {
        expect(json1.nodeIndices).toEqual([0, 1]);
      });
      it('serializes nodeIndices for group2', () => {
        expect(json2.nodeIndices).toEqual([2, 3]);
      });
    });

    describe('when serializing group after gating', () => {
      let group: Group;
      let node1: Node;
      let node2: Node;
      let conn1: Connection;
      let json: any;
      beforeEach(() => {
        // Arrange
        group = new Group(2);
        node1 = new Node();
        node2 = new Node();
        node1.index = 10;
        node2.index = 11;
        conn1 = node1.connect(node2)[0];
        group.nodes[0].index = 20;
        group.nodes[1].index = 21;
        group.gate([conn1], methods.gating.INPUT);
        // Act
        json = group.toJSON();
      });
      it('serializes size', () => {
        expect(json.size).toBe(2);
      });
      it('serializes nodeIndices', () => {
        expect(json.nodeIndices).toEqual([20, 21]);
      });
      it('serializes connections.in', () => {
        expect(json.connections.in).toBe(0);
      });
      it('serializes connections.out', () => {
        expect(json.connections.out).toBe(0);
      });
      it('serializes connections.self', () => {
        expect(json.connections.self).toBe(0);
      });
    });
  });

  describe('connect()', () => {
    let group1: Group;
    let group2: Group;
    let node: Node;
    let layer: Layer;
    const size1 = 3;
    const size2 = 2;
    let originalWarnings: boolean;

    beforeEach(() => {
      // Arrange
      group1 = new Group(size1);
      group2 = new Group(size2);
      node = new Node();
      layer = new Layer();
      originalWarnings = config.warnings;
      config.warnings = true;
      jest.spyOn(console, 'warn').mockImplementation(() => {});
    });

    afterEach(() => {
      (console.warn as jest.Mock).mockRestore();
      config.warnings = originalWarnings;
      jest.restoreAllMocks();
    });

    describe('Scenario: To Group', () => {
      describe('when connecting ALL_TO_ALL by default to a different group', () => {
        let connections: any[];
        beforeEach(() => {
          // Act
          connections = group1.connect(group2);
        });
        it('creates the correct number of connections', () => {
          // Assert
          expect(connections).toHaveLength(size1 * size2);
        });
        it('updates group1.connections.out', () => {
          expect(group1.connections.out).toHaveLength(size1 * size2);
        });
        it('warns about default ALL_TO_ALL', () => {
          expect(console.warn).toHaveBeenCalledWith(
            'No group connection specified, using ALL_TO_ALL by default.'
          );
        });
        it('forms connections between all node pairs', () => {
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
      });

      describe('when connecting ONE_TO_ONE by default to the same group', () => {
        let sameSizeGroup: Group;
        let connections: any[];
        beforeEach(() => {
          // Act
          sameSizeGroup = new Group(size1);
          connections = sameSizeGroup.connect(sameSizeGroup);
        });
        it('creates the correct number of connections', () => {
          expect(connections).toHaveLength(size1);
        });
        it('updates self connections', () => {
          expect(sameSizeGroup.connections.self).toHaveLength(size1);
        });
        it('warns about default ONE_TO_ONE', () => {
          expect(console.warn).toHaveBeenCalledWith(
            'Connecting group to itself, using ONE_TO_ONE by default.'
          );
        });
        it('stores self-connection in group', () => {
          sameSizeGroup.nodes.forEach((node: Node, i: number) => {
            const selfConn = sameSizeGroup.connections.self[i];
            expect(selfConn).toBeInstanceOf(Connection);
            expect(selfConn.from).toBe(node);
            expect(selfConn.to).toBe(node);
            expect(node.connections.self[0]).toBe(selfConn);
          });
        });
      });

      describe('when connecting using specified ALL_TO_ALL method', () => {
        let connections: any[];
        beforeEach(() => {
          // Act
          connections = group1.connect(
            group2,
            methods.groupConnection.ALL_TO_ALL
          );
        });
        it('creates the correct number of connections', () => {
          expect(connections).toHaveLength(size1 * size2);
        });
        it('updates group1.connections.out', () => {
          expect(group1.connections.out).toHaveLength(size1 * size2);
        });
        it('updates group2.connections.in', () => {
          expect(group2.connections.in).toHaveLength(size1 * size2);
        });
      });

      describe('when connecting using specified ALL_TO_ELSE method', () => {
        let sameSizeGroup: Group;
        let connections: any[];
        const expectedConns = size1 * size1 - size1;
        beforeEach(() => {
          // Act
          sameSizeGroup = new Group(size1);
          connections = sameSizeGroup.connect(
            sameSizeGroup,
            methods.groupConnection.ALL_TO_ELSE
          );
        });
        it('creates the correct number of connections', () => {
          expect(connections).toHaveLength(expectedConns);
        });
        it('updates out and in connections', () => {
          expect(sameSizeGroup.connections.out).toHaveLength(expectedConns);
          expect(sameSizeGroup.connections.in).toHaveLength(expectedConns);
        });
        it('does not create self-connections', () => {
          sameSizeGroup.nodes.forEach((node: Node, i: number) => {
            expect(node.isConnectedTo(sameSizeGroup.nodes[i])).toBe(false);
          });
        });
      });

      describe('when connecting using specified ONE_TO_ONE method', () => {
        let sameSizeGroup: Group;
        let connections: any[];
        beforeEach(() => {
          // Act
          sameSizeGroup = new Group(size1);
          connections = group1.connect(
            sameSizeGroup,
            methods.groupConnection.ONE_TO_ONE
          );
        });
        it('creates the correct number of connections', () => {
          expect(connections).toHaveLength(size1);
        });
        it('updates out and in connections', () => {
          expect(group1.connections.out).toHaveLength(size1);
          expect(sameSizeGroup.connections.in).toHaveLength(size1);
        });
        it('does not update self connections for different groups', () => {
          expect(group1.connections.self).toHaveLength(0);
          expect(sameSizeGroup.connections.self).toHaveLength(0);
        });
        it('connects corresponding nodes', () => {
          group1.nodes.forEach((node: Node, i: number) => {
            expect(node.isConnectedTo(sameSizeGroup.nodes[i])).toBe(true);
          });
        });
      });

      describe('when connecting ONE_TO_ONE with different group sizes', () => {
        it('throws an error', () => {
          expect(() =>
            group1.connect(group2, methods.groupConnection.ONE_TO_ONE)
          ).toThrow(
            'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.'
          );
        });
      });

      describe('when connecting with specified weight', () => {
        const weight = 0.75;
        let connections: any[];
        beforeEach(() => {
          // Act
          connections = group1.connect(
            group2,
            methods.groupConnection.ALL_TO_ALL,
            weight
          );
        });
        it('sets the weight for all connections', () => {
          connections.forEach((conn: Connection) => {
            expect(conn.weight).toBe(weight);
          });
        });
      });
    });

    describe('Scenario: To Layer', () => {
      it('delegates connection to Layer.input()', () => {
        const layer = new Layer();
        const layerInputSpy = jest
          .spyOn(layer, 'input')
          .mockImplementation(() => []);
        const method = methods.groupConnection.ALL_TO_ALL;
        const weight = 0.5;
        const group1 = new Group(3);

        group1.connect(layer, method, weight);

        expect(layerInputSpy).toHaveBeenCalledTimes(1);
        expect(layerInputSpy).toHaveBeenCalledWith(group1, method, weight);

        layerInputSpy.mockRestore();
      });
    });

    describe('Scenario: To Node', () => {
      let connections: any[];
      beforeEach(() => {
        // Act
        connections = group1.connect(node);
      });
      it('creates the correct number of connections', () => {
        expect(connections).toHaveLength(size1);
      });
      it('updates group1.connections.out', () => {
        expect(group1.connections.out).toHaveLength(size1);
      });
      it('connects all nodes in group to the target node', () => {
        group1.nodes.forEach((fromNode: Node) => {
          expect(fromNode.isConnectedTo(node)).toBe(true);
        });
      });
      it('updates node.connections.in', () => {
        expect(node.connections.in).toHaveLength(size1);
      });
    });

    describe('when connecting to node with specified weight', () => {
      const weight = -0.3;
      let connections: any[];
      beforeEach(() => {
        // Act
        connections = group1.connect(node, undefined, weight);
      });
      it('sets the weight for all connections', () => {
        connections.forEach((conn: Connection) => {
          expect(conn.weight).toBe(weight);
        });
      });
    });
  });
});
