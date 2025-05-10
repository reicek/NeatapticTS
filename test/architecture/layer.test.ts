import Layer from '../../src/architecture/layer';
import Node from '../../src/architecture/node';
import Group from '../../src/architecture/group';
import Connection from '../../src/architecture/connection';
import * as methods from '../../src/methods/methods';
import { config } from '../../src/config';

// Retry failed tests
jest.retryTimes(2, { logErrorsBeforeRetry: true });

// Helper function to check group connectivity
const isGroupConnectedTo = (
  groupA: Group,
  groupB: Group,
  method?: any
): boolean => {
  if (!groupA || !groupB || !groupA.nodes || !groupB.nodes) return false; // Basic validation

  if (method === methods.groupConnection.ONE_TO_ONE) {
    if (groupA.nodes.length !== groupB.nodes.length) return false;
    return groupA.nodes.every((nodeA, i) => {
      const nodeB = groupB.nodes[i];
      if (!nodeA || !nodeB) return false; // Node validation
      if (nodeA.connections.out.some(conn => conn.to === nodeB)) return true;
      if (nodeA === nodeB && nodeA.connections.self.some((conn: Connection) => conn.to === nodeB)) return true;
      return false;
    });
  } else {
    return groupA.nodes.some((nodeA) =>
      nodeA && groupB.nodes.some((nodeB) => {
        if (!nodeB) return false; // Node validation
        if (nodeA.connections.out.some(conn => conn.to === nodeB)) return true;
        if (nodeA === nodeB && nodeA.connections.self.some((conn: Connection) => conn.to === nodeB)) return true;
        return false;
      })
    );
  }
};

describe('Layer', () => {
  const epsilon = 1e-9; // Tolerance for float comparisons

  describe('Constructor', () => {
    let layer: Layer;

    beforeEach(() => {
      layer = new Layer();
    });

    test('should initialize output to null', () => {
      // Arrange, Act & Assert
      expect(layer.output).toBeNull();
    });

    test('should initialize nodes as an empty array', () => {
      // Arrange, Act & Assert
      expect(layer.nodes).toEqual([]);
    });

    test('should initialize connections.in as an empty array', () => {
      // Arrange, Act & Assert
      expect(layer.connections.in).toEqual([]);
    });

    test('should initialize connections.out as an empty array', () => {
      // Arrange, Act & Assert
      expect(layer.connections.out).toEqual([]);
    });

    test('should initialize connections.self as an empty array', () => {
      // Arrange, Act & Assert
      expect(layer.connections.self).toEqual([]);
    });
  });

  const createTestLayer = (size: number): Layer => {
    const layer = new Layer();
    const group = new Group(size);
    layer.nodes.push(...group.nodes);
    layer.output = group;
    return layer;
  };

  describe('Instance Methods', () => {
    describe('activate()', () => {
      const size = 3;
      let layer: Layer;
      let nodeSpies: jest.SpyInstance[];

      beforeEach(() => {
        layer = createTestLayer(size);
        nodeSpies = layer.nodes.map((node) => jest.spyOn(node, 'activate'));
        layer.nodes.forEach((node, i) => (node.bias = (i + 1) * 0.1));
      });

      afterEach(() => {
        nodeSpies.forEach((spy) => spy.mockRestore());
      });

      test('should call activate on all nodes without input values', () => {
        // Arrange
        // Act
        layer.activate();
        // Assert
        expect(nodeSpies).toHaveLength(size);
        nodeSpies.forEach((spy) => {
          expect(spy).toHaveBeenCalledTimes(1);
          expect(spy).toHaveBeenCalledWith();
        });
      });

      test('should return activation values from nodes without input', () => {
        // Arrange
        // Act
        const activations = layer.activate();
        // Assert
        expect(activations).toHaveLength(size);
        activations.forEach((act, i) => {
          const expected = methods.Activation.logistic((i + 1) * 0.1);
          expect(act).toBeCloseTo(expected, epsilon);
        });
      });

      test('should call activate on all nodes with provided input values', () => {
        // Arrange
        const inputValues = [0.5, -0.2, 1.0];
        // Act
        layer.activate(inputValues);
        // Assert
        expect(nodeSpies).toHaveLength(size);
        nodeSpies.forEach((spy, i) => {
          expect(spy).toHaveBeenCalledTimes(1);
          expect(spy).toHaveBeenCalledWith(inputValues[i]);
        });
      });

      test('should return activation values from nodes with input', () => {
        // Arrange
        const inputValues = [0.5, -0.2, 1.0];
        // Act
        const activations = layer.activate(inputValues);
        // Assert
        expect(activations).toHaveLength(size);
        activations.forEach((act, i) => {
          expect(act).toBe(inputValues[i]);
        });
      });

      test('should throw error if input value array length mismatches layer node count', () => {
        // Arrange
        const invalidInput = [0.1, 0.2];
        // Act & Assert
        expect(() => layer.activate(invalidInput)).toThrow(
          'Array with values should be same as the amount of nodes!'
        );
      });
    });

    describe('propagate()', () => {
      const size = 3;
      let layer: Layer;
      let nodeSpies: jest.SpyInstance[];
      const rate = 0.1;
      const momentum = 0.9;

      beforeEach(() => {
        layer = createTestLayer(size);
        nodeSpies = layer.nodes.map((node) => jest.spyOn(node, 'propagate'));
      });

      afterEach(() => {
        nodeSpies.forEach((spy) => spy.mockRestore());
      });

      test('should call propagate on all nodes without target values', () => {
        // Arrange
        // Act
        layer.propagate(rate, momentum);
        // Assert
        expect(nodeSpies).toHaveLength(size);
        nodeSpies.forEach((spy) => {
          expect(spy).toHaveBeenCalledTimes(1);
          expect(spy).toHaveBeenCalledWith(rate, momentum, true);
        });
      });

      test('should call propagate on nodes in reverse order', () => {
        // Arrange
        const callOrder: number[] = [];
        nodeSpies.forEach((spy, i) => spy.mockImplementation(() => callOrder.push(i)));
        // Act
        layer.propagate(rate, momentum);
        // Assert
        expect(callOrder).toEqual([2, 1, 0]);
      });

      test('should call propagate on all nodes with target values', () => {
        // Arrange
        const targetValues = [0.8, 0.1, 0.5];
        // Act
        layer.propagate(rate, momentum, targetValues);
        // Assert
        expect(nodeSpies).toHaveLength(size);
        nodeSpies.reverse().forEach((spy, i) => {
          const originalIndex = size - 1 - i;
          expect(spy).toHaveBeenCalledTimes(1);
          expect(spy).toHaveBeenCalledWith(
            rate,
            momentum,
            true,
            targetValues[originalIndex]
          );
        });
      });

      test('should throw error if target value array length mismatches layer node count', () => {
        // Arrange
        const invalidTarget = [0.1, 0.2];
        // Act & Assert
        expect(() => layer.propagate(rate, momentum, invalidTarget)).toThrow(
          'Array with values should be same as the amount of nodes!'
        );
      });
    });

    describe('connect()', () => {
      let sourceLayer: Layer;
      let targetLayer: Layer;
      let targetGroup: Group;
      let targetNode: Node;
      let sourceOutputSpy: jest.SpyInstance | undefined;
      let targetInputSpy: jest.SpyInstance;

      beforeEach(() => {
        sourceLayer = createTestLayer(3);
        targetLayer = createTestLayer(3);
        targetGroup = new Group(2);
        targetNode = new Node();

        if (sourceLayer.output) {
          sourceOutputSpy = jest.spyOn(sourceLayer.output, 'connect');
        }
        targetInputSpy = jest.spyOn(targetLayer, 'input');
      });

      afterEach(() => {
        sourceOutputSpy?.mockRestore();
        targetInputSpy?.mockRestore();
      });

      test('should throw error if source layer output is not defined', () => {
        // Arrange
        const layerWithoutOutput = new Layer();
        // Act & Assert
        expect(() => layerWithoutOutput.connect(targetGroup)).toThrow(
          'Layer output is not defined. Cannot connect from this layer.'
        );
      });

      test('should call output.connect when connecting to a Group', () => {
        // Arrange, Act
        const method = methods.groupConnection.ALL_TO_ALL;
        const weight = 0.5;
        sourceLayer.connect(targetGroup, method, weight);
        // Assert
        expect(sourceOutputSpy).toHaveBeenCalledTimes(1);
        expect(sourceOutputSpy).toHaveBeenCalledWith(targetGroup, method, weight);
      });

      test('should call output.connect when connecting to a Node', () => {
        // Arrange, Act
        const weight = 0.6;
        sourceLayer.connect(targetNode, undefined, weight);
        // Assert
        expect(sourceOutputSpy).toHaveBeenCalledTimes(1);
        expect(sourceOutputSpy).toHaveBeenCalledWith(targetNode, undefined, weight);
      });

      test('should call targetLayer.input when connecting to another Layer', () => {
        // Arrange, Act
        const method = methods.groupConnection.ONE_TO_ONE;
        const weight = 0.7;
        sourceLayer.connect(targetLayer, method, weight);
        // Assert
        expect(targetInputSpy).toHaveBeenCalledTimes(1);
        expect(targetInputSpy).toHaveBeenCalledWith(sourceLayer, method, weight);
        expect(sourceOutputSpy).toHaveBeenCalledTimes(1);
        expect(sourceOutputSpy).toHaveBeenCalledWith(targetLayer.output, method, weight);
      });

      test('should return the connections created by output.connect (Group target)', () => {
        // Arrange
        const mockConnections = [new Connection(new Node(), new Node())];
        sourceOutputSpy?.mockReturnValue(mockConnections);
        // Act
        const result = sourceLayer.connect(targetGroup);
        // Assert
        expect(result).toBe(mockConnections);
      });

      test('should return the connections created by targetLayer.input (Layer target)', () => {
        // Arrange
        const mockConnections = [new Connection(new Node(), new Node())];
        targetInputSpy.mockReturnValue(mockConnections);
        // Act
        const result = sourceLayer.connect(targetLayer);
        // Assert
        expect(result).toBe(mockConnections);
      });
    });

    describe('gate()', () => {
      let layer: Layer;
      let connectionsToGate: Connection[];
      let outputGroupSpy: jest.SpyInstance;

      beforeEach(() => {
        layer = createTestLayer(2);
        const node1 = new Node();
        const node2 = new Node();
        connectionsToGate = [new Connection(node1, node2)];

        if (layer.output) {
          outputGroupSpy = jest.spyOn(layer.output, 'gate');
        }
      });

      afterEach(() => {
        outputGroupSpy?.mockRestore();
      });

      test('should throw error if layer output is not defined', () => {
        // Arrange
        const layerWithoutOutput = new Layer();
        // Act & Assert
        expect(() =>
          layerWithoutOutput.gate(connectionsToGate, methods.gating.INPUT)
        ).toThrow('Layer output is not defined. Cannot gate from this layer.');
      });

      test('should call output.gate with the provided connections and method', () => {
        // Arrange, Act
        const method = methods.gating.OUTPUT;
        layer.gate(connectionsToGate, method);
        // Assert
        expect(outputGroupSpy).toHaveBeenCalledTimes(1);
        expect(outputGroupSpy).toHaveBeenCalledWith(connectionsToGate, method);
      });
    });

    describe('set()', () => {
      const size = 3;
      let layer: Layer;

      beforeEach(() => {
        layer = createTestLayer(size);
        layer.nodes.forEach((node, i) => {
          node.bias = i * 0.1;
          node.squash = methods.Activation.logistic;
          node.type = 'hidden';
        });
      });

      test('should set bias for all nodes', () => {
        // Arrange, Act
        const biasValue = 0.7;
        layer.set({ bias: biasValue });
        // Assert
        layer.nodes.forEach((node) => {
          expect(node.bias).toBe(biasValue);
        });
      });

      test('should set squash function for all nodes', () => {
        // Arrange, Act
        const squashFn = methods.Activation.relu;
        layer.set({ squash: squashFn });
        // Assert
        layer.nodes.forEach((node) => {
          expect(node.squash).toBe(squashFn);
        });
      });

      test('should set type for all nodes', () => {
        // Arrange, Act
        const typeValue = 'output';
        layer.set({ type: typeValue });
        // Assert
        layer.nodes.forEach((node) => {
          expect(node.type).toBe(typeValue);
        });
      });

      test('should set multiple properties at once', () => {
        // Arrange, Act
        const biasValue = -0.2;
        const squashFn = methods.Activation.tanh;
        const typeValue = 'input';
        layer.set({ bias: biasValue, squash: squashFn, type: typeValue });
        // Assert
        layer.nodes.forEach((node) => {
          expect(node.bias).toBe(biasValue);
          expect(node.squash).toBe(squashFn);
          expect(node.type).toBe(typeValue);
        });
      });

      test('should not change properties if not provided in values object', () => {
        // Arrange
        const initialBiases = layer.nodes.map((node) => node.bias);
        const initialSquashes = layer.nodes.map((node) => node.squash);
        const initialTypes = layer.nodes.map((node) => node.type);
        // Act
        layer.set({ bias: 0.99 });
        // Assert
        layer.nodes.forEach((node, i) => {
          expect(node.bias).toBe(0.99);
          expect(node.squash).toBe(initialSquashes[i]);
          expect(node.type).toBe(initialTypes[i]);
        });
      });

      test('should call set on Group instances within the layer nodes', () => {
        const memoryLayer = Layer.memory(2, 2);
        const groupSetSpies = memoryLayer.nodes.map((groupNode) =>
          jest.spyOn(groupNode as unknown as Group, 'set')
        );

        const settings = { bias: 0.1, squash: methods.Activation.relu };
        memoryLayer.set(settings);

        groupSetSpies.forEach((spy) => {
          expect(spy).toHaveBeenCalledTimes(1);
          expect(spy).toHaveBeenCalledWith(settings);
        });

        groupSetSpies.forEach((spy) => spy.mockRestore());
      });
    });

    describe('disconnect()', () => {
      let layer: Layer;
      let targetGroup: Group;
      let targetNode: Node;
      let nodeDisconnectSpies: jest.SpyInstance[];

      beforeEach(() => {
        layer = createTestLayer(2);
        targetGroup = new Group(2);
        targetNode = new Node();

        layer.nodes.forEach((node) => {
          targetGroup.nodes.forEach((target) => node.connect(target));
          node.connect(targetNode);
        });
        layer.nodes[0].connections.out.forEach(conn => layer.connections.out.push(conn));
        layer.nodes[1].connections.out.forEach(conn => layer.connections.out.push(conn));

        nodeDisconnectSpies = layer.nodes.map((node) =>
          jest.spyOn(node, 'disconnect')
        );
      });

      afterEach(() => {
        nodeDisconnectSpies.forEach((spy) => spy.mockRestore());
      });

      describe('Disconnecting from Group', () => {
        test('should call disconnect on each layer node for each target group node (one-sided)', () => {
          layer.disconnect(targetGroup, false);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledTimes(targetGroup.nodes.length);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledTimes(targetGroup.nodes.length);
          targetGroup.nodes.forEach((target) => {
            expect(nodeDisconnectSpies[0]).toHaveBeenCalledWith(target, false);
            expect(nodeDisconnectSpies[1]).toHaveBeenCalledWith(target, false);
          });
        });

        test('should call disconnect on each layer node for each target group node (two-sided)', () => {
          layer.disconnect(targetGroup, true);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledTimes(targetGroup.nodes.length);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledTimes(targetGroup.nodes.length);
          targetGroup.nodes.forEach((target) => {
            expect(nodeDisconnectSpies[0]).toHaveBeenCalledWith(target, true);
            expect(nodeDisconnectSpies[1]).toHaveBeenCalledWith(target, true);
          });
        });

        test('should remove outgoing connections from layer.connections.out (one-sided)', () => {
          const initialOutCount = layer.connections.out.length;
          expect(initialOutCount).toBe(6);
          layer.disconnect(targetGroup, false);
          expect(layer.connections.out).toHaveLength(initialOutCount - 4);
          layer.connections.out.forEach(conn => {
            expect(conn.to).toBe(targetNode);
          });
        });

        test('should remove incoming connections from layer.connections.in if two-sided', () => {
          targetGroup.nodes[0].connect(layer.nodes[0]);
          targetGroup.nodes[1].connect(layer.nodes[1]);
          layer.connections.in.push(targetGroup.nodes[0].connections.out[0]);
          layer.connections.in.push(targetGroup.nodes[1].connections.out[0]);
          const initialInCount = layer.connections.in.length;
          expect(initialInCount).toBe(2);

          layer.disconnect(targetGroup, true);

          expect(layer.connections.in).toHaveLength(0);
        });
      });

      describe('Disconnecting from Node', () => {
        test('should call disconnect on each layer node for the target node (one-sided)', () => {
          layer.disconnect(targetNode, false);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledTimes(1);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledTimes(1);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledWith(targetNode, false);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledWith(targetNode, false);
        });

        test('should call disconnect on each layer node for the target node (two-sided)', () => {
          layer.disconnect(targetNode, true);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledTimes(1);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledTimes(1);
          expect(nodeDisconnectSpies[0]).toHaveBeenCalledWith(targetNode, true);
          expect(nodeDisconnectSpies[1]).toHaveBeenCalledWith(targetNode, true);
        });

        test('should remove outgoing connections to the node from layer.connections.out (one-sided)', () => {
          const initialOutCount = layer.connections.out.length;
          layer.disconnect(targetNode, false);
          expect(layer.connections.out).toHaveLength(initialOutCount - 2);
          layer.connections.out.forEach(conn => {
            expect(targetGroup.nodes).toContain(conn.to);
          });
        });

        test('should remove incoming connections from the node in layer.connections.in if two-sided', () => {
          targetNode.connect(layer.nodes[0]);
          layer.connections.in.push(targetNode.connections.out[0]);
          const initialInCount = layer.connections.in.length;
          expect(initialInCount).toBe(1);

          layer.disconnect(targetNode, true);

          expect(layer.connections.in).toHaveLength(0);
        });
      });
    });

    describe('clear()', () => {
      const size = 3;
      let layer: Layer;
      let nodeSpies: jest.SpyInstance[];

      beforeEach(() => {
        layer = createTestLayer(size);
        nodeSpies = layer.nodes.map((node) => jest.spyOn(node, 'clear'));
      });

      afterEach(() => {
        nodeSpies.forEach((spy) => spy.mockRestore());
      });

      test('should call clear on all nodes in the layer', () => {
        layer.clear();
        expect(nodeSpies).toHaveLength(size);
        nodeSpies.forEach((spy) => {
          expect(spy).toHaveBeenCalledTimes(1);
        });
      });

      test('should call clear on Group instances within the layer nodes (e.g., Memory layer)', () => {
        const memoryLayer = Layer.memory(2, 2);
        const groupClearSpies = memoryLayer.nodes.map((groupNode) =>
          jest.spyOn(groupNode as unknown as Group, 'clear')
        );

        memoryLayer.clear();

        groupClearSpies.forEach((spy) => {
          expect(spy).toHaveBeenCalledTimes(1);
        });

        groupClearSpies.forEach((spy) => spy.mockRestore());
      });
    });

    describe('input()', () => {
      let targetLayer: Layer;
      let sourceLayer: Layer;
      let sourceGroup: Group;
      let sourceOutputConnectSpy: jest.SpyInstance | undefined;
      let sourceGroupConnectSpy: jest.SpyInstance;
      let targetOutputConnectSpy: jest.SpyInstance | undefined;

      beforeEach(() => {
        targetLayer = createTestLayer(2);
        sourceLayer = createTestLayer(3);
        sourceGroup = new Group(3);

        if (sourceLayer.output) {
          sourceOutputConnectSpy = jest.spyOn(sourceLayer.output, 'connect');
        }
        sourceGroupConnectSpy = jest.spyOn(sourceGroup, 'connect');

        if (targetLayer.output) {
          targetOutputConnectSpy = jest.spyOn(targetLayer.output, 'connect');
        }
      });

      afterEach(() => {
        sourceOutputConnectSpy?.mockRestore();
        sourceGroupConnectSpy.mockRestore();
        targetOutputConnectSpy?.mockRestore();
      });

      test('should throw error if target layer output (acting as input) is not defined', () => {
        const layerWithoutOutput = new Layer();
        expect(() => layerWithoutOutput.input(sourceGroup)).toThrow(
          'Layer output (acting as input target) is not defined.'
        );
      });

      test('should use source Layer output group when connecting from a Layer', () => {
        targetLayer.input(sourceLayer);
        expect(sourceOutputConnectSpy).toHaveBeenCalledTimes(1);
        expect(sourceOutputConnectSpy).toHaveBeenCalledWith(
          targetLayer.output,
          methods.groupConnection.ALL_TO_ALL,
          undefined
        );
        expect(sourceGroupConnectSpy).not.toHaveBeenCalled();
      });

      test('should use source Group directly when connecting from a Group', () => {
        targetLayer.input(sourceGroup);
        expect(sourceGroupConnectSpy).toHaveBeenCalledTimes(1);
        expect(sourceGroupConnectSpy).toHaveBeenCalledWith(
          targetLayer.output,
          methods.groupConnection.ALL_TO_ALL,
          undefined
        );
        expect(sourceOutputConnectSpy).not.toHaveBeenCalled();
      });

      test('should use default connection method ALL_TO_ALL if none provided', () => {
        targetLayer.input(sourceGroup);
        expect(sourceGroupConnectSpy).toHaveBeenCalledWith(
          expect.anything(),
          methods.groupConnection.ALL_TO_ALL,
          undefined
        );
      });

      test('should use provided connection method', () => {
        const method = methods.groupConnection.ONE_TO_ONE;
        sourceGroup = new Group(2);
        sourceGroupConnectSpy = jest.spyOn(sourceGroup, 'connect');

        targetLayer.input(sourceGroup, method);
        expect(sourceGroupConnectSpy).toHaveBeenCalledWith(
          targetLayer.output,
          method,
          undefined
        );
      });

      test('should use provided weight', () => {
        const weight = 0.88;
        targetLayer.input(sourceGroup, undefined, weight);
        expect(sourceGroupConnectSpy).toHaveBeenCalledWith(
          targetLayer.output,
          methods.groupConnection.ALL_TO_ALL,
          weight
        );
      });

      test('should return the connections created by the source connect call', () => {
        const mockConnections = [new Connection(new Node(), new Node())];
        sourceGroupConnectSpy.mockReturnValue(mockConnections);
        const result = targetLayer.input(sourceGroup);
        expect(result).toBe(mockConnections);
      });
    });
  });

  describe('Static Factory Methods', () => {
    describe('Layer.dense()', () => {
      const size = 5;
      let layer: Layer;

      beforeEach(() => {
        layer = Layer.dense(size);
      });

      test('should create a layer with the specified number of nodes', () => {
        expect(layer.nodes).toHaveLength(size);
        layer.nodes.forEach((node) => expect(node).toBeInstanceOf(Node));
      });

      test('should set the output to a Group containing all nodes', () => {
        expect(layer.output).toBeInstanceOf(Group);
        expect(layer.output?.nodes).toHaveLength(size);
        expect(layer.output?.nodes).toEqual(layer.nodes);
      });

      test('should have a custom input method', () => {
        expect(typeof layer.input).toBe('function');
        expect(layer.input).not.toBe(Layer.prototype.input);
      });

      describe('Dense Layer input() method', () => {
        let sourceLayer: Layer;
        let sourceGroup: Group;
        let sourceOutputConnectSpy: jest.SpyInstance | undefined;
        let sourceGroupConnectSpy: jest.SpyInstance;

        beforeEach(() => {
          sourceLayer = Layer.dense(3);
          sourceGroup = new Group(3);

          if (sourceLayer.output) {
            sourceOutputConnectSpy = jest.spyOn(sourceLayer.output, 'connect');
          }
          sourceGroupConnectSpy = jest.spyOn(sourceGroup, 'connect');
        });

        afterEach(() => {
          sourceOutputConnectSpy?.mockRestore();
          sourceGroupConnectSpy.mockRestore();
        });

        test('should connect source Layer output to the dense layers internal block', () => {
          const method = methods.groupConnection.ALL_TO_ALL;
          const weight = 0.1;
          layer.input(sourceLayer, method, weight);

          expect(sourceOutputConnectSpy).toHaveBeenCalledTimes(1);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(layer.output, method, weight);
          expect(sourceGroupConnectSpy).not.toHaveBeenCalled();
        });

        test('should connect source Group to the dense layers internal block', () => {
          const method = methods.groupConnection.ONE_TO_ONE;
          sourceGroup = new Group(size);
          sourceGroupConnectSpy = jest.spyOn(sourceGroup, 'connect');
          const weight = 0.2;

          layer.input(sourceGroup, method, weight);

          expect(sourceGroupConnectSpy).toHaveBeenCalledTimes(1);
          expect(sourceGroupConnectSpy).toHaveBeenCalledWith(layer.output, method, weight);
          expect(sourceOutputConnectSpy).not.toHaveBeenCalled();
        });

        test('should use ALL_TO_ALL by default if method not provided', () => {
          layer.input(sourceGroup);
          expect(sourceGroupConnectSpy).toHaveBeenCalledWith(expect.anything(), methods.groupConnection.ALL_TO_ALL, undefined);
        });
      });
    });

    describe('Layer.lstm()', () => {
      const size = 2;
      let layer: Layer;
      let inputGate: Group, forgetGate: Group, memoryCell: Group, outputGate: Group, outputBlock: Group;

      beforeEach(() => {
        layer = Layer.lstm(size);
        inputGate = new Group(size);
        inputGate.nodes = layer.nodes.slice(0, size);

        forgetGate = new Group(size);
        forgetGate.nodes = layer.nodes.slice(size, size * 2);

        memoryCell = new Group(size);
        memoryCell.nodes = layer.nodes.slice(size * 2, size * 3);

        outputGate = new Group(size);
        outputGate.nodes = layer.nodes.slice(size * 3, size * 4);

        outputBlock = new Group(size);
        outputBlock.nodes = layer.nodes.slice(size * 4, size * 5);

        layer.output = outputBlock;
      });

      test('should create a layer with 5 * size nodes', () => {
        expect(layer.nodes).toHaveLength(5 * size);
      });

      test('should set the output to the outputBlock group', () => {
        expect(layer.output).toBe(outputBlock);
      });

      test('should set initial bias for gates', () => {
        inputGate.nodes.forEach(node => expect(node.bias).toBe(1));
        forgetGate.nodes.forEach(node => expect(node.bias).toBe(1));
        outputGate.nodes.forEach(node => expect(node.bias).toBe(1));
        memoryCell.nodes.forEach(node => expect(node.bias).toBe(0));
        outputBlock.nodes.forEach(node => expect(node.bias).toBe(0));
      });

      test('should establish internal connections (memoryCell to gates)', () => {
        expect(isGroupConnectedTo(memoryCell, inputGate)).toBe(true);
        expect(isGroupConnectedTo(memoryCell, forgetGate)).toBe(true);
        expect(isGroupConnectedTo(memoryCell, outputGate)).toBe(true);
      });

      test('should establish internal connections (memoryCell self-connection)', () => {
        expect(isGroupConnectedTo(memoryCell, memoryCell, methods.groupConnection.ONE_TO_ONE)).toBe(true);
      });

      test('should establish internal connections (memoryCell to outputBlock)', () => {
        expect(isGroupConnectedTo(memoryCell, outputBlock)).toBe(true);
      });

      test('should gate the memoryCell self-connection with the forgetGate', () => {
        memoryCell.nodes.forEach((node, i) => {
          const selfConnection = node.connections.self.find((conn: Connection) => conn.to === node);
          expect(selfConnection).toBeDefined();
          expect(selfConnection?.gater).toBe(forgetGate.nodes[i]);
        });
      });

      test('should gate the memoryCell to outputBlock connection with the outputGate', () => {
        memoryCell.nodes.forEach((node, i) => {
          const outputConnection = node.connections.out.find(conn => conn.to === outputBlock.nodes[0]);
          expect(outputConnection).toBeDefined();
          expect(outputConnection?.gater).toBe(outputGate.nodes[i]);

          if (size > 1) {
            const outputConnection1 = node.connections.out.find(conn => conn.to === outputBlock.nodes[1]);
            expect(outputConnection1).toBeDefined();
            expect(outputConnection1?.gater).toBe(outputGate.nodes[i]);
          }
        });
      });

      test('should have a custom input method', () => {
        expect(typeof layer.input).toBe('function');
        expect(layer.input).not.toBe(Layer.prototype.input);
      });

      describe('LSTM Layer input() method', () => {
        let sourceLayer: Layer;
        let sourceOutputConnectSpy: jest.SpyInstance | undefined;

        beforeEach(() => {
          sourceLayer = Layer.dense(3);
          if (sourceLayer.output) {
            sourceOutputConnectSpy = jest.spyOn(sourceLayer.output, 'connect');
          }
        });

        afterEach(() => {
          sourceOutputConnectSpy?.mockRestore();
        });

        test('should connect source to inputGate, forgetGate, memoryCell, and outputGate', () => {
          layer.input(sourceLayer);

          expect(sourceOutputConnectSpy).toHaveBeenCalledTimes(4);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: inputGate.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: forgetGate.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: memoryCell.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: outputGate.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
        });

        test('should gate the source-to-memoryCell connection with the inputGate', () => {
          const connections = layer.input(sourceLayer);
          const inputToMemoryConnection = connections.find(conn =>
            sourceLayer.output!.nodes.includes(conn.from) && memoryCell.nodes.includes(conn.to)
          );
          expect(inputToMemoryConnection).toBeDefined();
          const targetNodeIndex = memoryCell.nodes.indexOf(inputToMemoryConnection.to);
          expect(inputToMemoryConnection.gater).toBe(inputGate.nodes[targetNodeIndex]);
        });
      });
    });

    describe('Layer.gru()', () => {
      const size = 2;
      let layer: Layer;
      let updateGate: Group, inverseUpdateGate: Group, resetGate: Group, memoryCell: Group, output: Group, previousOutput: Group;

      beforeEach(() => {
        layer = Layer.gru(size);
        updateGate = new Group(size);
        updateGate.nodes = layer.nodes.slice(0, size);

        inverseUpdateGate = new Group(size);
        inverseUpdateGate.nodes = layer.nodes.slice(size, size * 2);

        resetGate = new Group(size);
        resetGate.nodes = layer.nodes.slice(size * 2, size * 3);

        memoryCell = new Group(size);
        memoryCell.nodes = layer.nodes.slice(size * 3, size * 4);

        output = new Group(size);
        output.nodes = layer.nodes.slice(size * 4, size * 5);

        previousOutput = new Group(size);
        previousOutput.nodes = layer.nodes.slice(size * 5, size * 6);

        layer.output = output;
      });

      test('should create a layer with 6 * size nodes', () => {
        expect(layer.nodes).toHaveLength(6 * size);
      });

      test('should set the output to the output group', () => {
        expect(layer.output).toBe(output);
      });

      test('should set specific node properties', () => {
        previousOutput.nodes.forEach(node => {
          expect(node.bias).toBe(0);
          expect(node.squash).toBe(methods.Activation.identity);
          expect(node.type).toBe('variant');
        });
        memoryCell.nodes.forEach(node => expect(node.squash).toBe(methods.Activation.tanh));
        inverseUpdateGate.nodes.forEach(node => {
          expect(node.bias).toBe(0);
          expect(node.squash).toBe(methods.Activation.inverse);
          expect(node.type).toBe('variant');
        });
        updateGate.nodes.forEach(node => expect(node.bias).toBe(1));
        resetGate.nodes.forEach(node => expect(node.bias).toBe(0));
      });

      test('should establish internal connections (previousOutput to gates)', () => {
        expect(isGroupConnectedTo(previousOutput, updateGate)).toBe(true);
        expect(isGroupConnectedTo(previousOutput, resetGate)).toBe(true);
      });

      test('should establish internal connections (updateGate to inverseUpdateGate)', () => {
        expect(isGroupConnectedTo(updateGate, inverseUpdateGate, methods.groupConnection.ONE_TO_ONE)).toBe(true);
        updateGate.nodes.forEach((node, i) => {
          const conn = node.connections.out.find(c => c.to === inverseUpdateGate.nodes[i]);
          expect(conn).toBeDefined();
          expect(conn?.weight).toBe(1);
        });
      });

      test('should establish internal connections (previousOutput to memoryCell)', () => {
        expect(isGroupConnectedTo(previousOutput, memoryCell)).toBe(true);
      });

      test('should establish internal connections (previousOutput and memoryCell to output)', () => {
        expect(isGroupConnectedTo(previousOutput, output)).toBe(true);
        expect(isGroupConnectedTo(memoryCell, output)).toBe(true);
      });

      test('should establish internal connections (output to previousOutput)', () => {
        expect(isGroupConnectedTo(output, previousOutput, methods.groupConnection.ONE_TO_ONE)).toBe(true);
        output.nodes.forEach((node, i) => {
          const conn = node.connections.out.find(c => c.to === previousOutput.nodes[i]);
          expect(conn).toBeDefined();
          expect(conn?.weight).toBe(1);
        });
      });

      test('should gate previousOutput->memoryCell connection with resetGate', () => {
        previousOutput.nodes.forEach((node, i) => {
          const conn = node.connections.out.find(c => c.to === memoryCell.nodes[0]);
          expect(conn).toBeDefined();
          expect(conn?.gater).toBe(resetGate.nodes[i]);
        });
      });

      test('should gate previousOutput->output connection with updateGate', () => {
        previousOutput.nodes.forEach((node, i) => {
          const conn = node.connections.out.find(c => c.to === output.nodes[0]);
          expect(conn).toBeDefined();
          expect(conn?.gater).toBe(updateGate.nodes[i]);
        });
      });

      test('should gate memoryCell->output connection with inverseUpdateGate', () => {
        memoryCell.nodes.forEach((node, i) => {
          const conn = node.connections.out.find(c => c.to === output.nodes[0]);
          expect(conn).toBeDefined();
          expect(conn?.gater).toBe(inverseUpdateGate.nodes[i]);
        });
      });

      test('should have a custom input method', () => {
        expect(typeof layer.input).toBe('function');
        expect(layer.input).not.toBe(Layer.prototype.input);
      });

      describe('GRU Layer input() method', () => {
        let sourceLayer: Layer;
        let sourceOutputConnectSpy: jest.SpyInstance | undefined;

        beforeEach(() => {
          sourceLayer = Layer.dense(3);
          if (sourceLayer.output) {
            sourceOutputConnectSpy = jest.spyOn(sourceLayer.output, 'connect');
          }
        });

        afterEach(() => {
          sourceOutputConnectSpy?.mockRestore();
        });

        test('should connect source to updateGate, resetGate, and memoryCell', () => {
          layer.input(sourceLayer);

          expect(sourceOutputConnectSpy).toHaveBeenCalledTimes(3);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: updateGate.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: resetGate.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: memoryCell.nodes }), methods.groupConnection.ALL_TO_ALL, undefined);
        });
      });
    });

    describe('Layer.memory()', () => {
      const size = 3;
      const memoryDepth = 2;
      let layer: Layer;

      beforeEach(() => {
        layer = Layer.memory(size, memoryDepth);
      });

      test('should create a layer with memoryDepth groups in nodes array', () => {
        expect(layer.nodes).toHaveLength(memoryDepth);
        layer.nodes.forEach(nodeOrGroup => {
          expect((layer as any).isGroup(nodeOrGroup)).toBe(true);
          expect((nodeOrGroup as unknown as Group).nodes).toHaveLength(size);
        });
      });

      test('should set specific properties for nodes within memory blocks', () => {
        layer.nodes.forEach(group => {
          (group as unknown as Group).nodes.forEach(node => {
            expect(node.squash).toBe(methods.Activation.identity);
            expect(node.bias).toBe(0);
            expect(node.type).toBe('variant');
          });
        });
      });

      test('should connect previous memory block to current one (ONE_TO_ONE, weight 1)', () => {
        // After reversal in factory: nodes[0] is newest, nodes[1] is second newest, etc.
        // Connection is made from older (previous) to newer (block).
        // So, connection should exist from nodes[1] (older) to nodes[0] (newer).
        const block1 = layer.nodes[0] as unknown as Group; // Newest block
        const block2 = layer.nodes[1] as unknown as Group; // Second newest (older) block

        // Check connection from the older block (block2) to the newer block (block1)
        expect(isGroupConnectedTo(block2, block1, methods.groupConnection.ONE_TO_ONE)).toBe(true);

        // Check weight on the node level (connection from block2 node to block1 node)
        block2.nodes.forEach((node, i) => {
            const conn = node.connections.out.find(c => c.to === block1.nodes[i]);
            expect(conn).toBeDefined();
            expect(conn?.weight).toBe(1);
        });
      });

      test('should create a concatenated output group', () => {
        expect(layer.output).toBeInstanceOf(Group);
        expect(layer.output?.nodes).toHaveLength(size * memoryDepth);
        const block1Nodes = (layer.nodes[0] as unknown as Group).nodes;
        const block2Nodes = (layer.nodes[1] as unknown as Group).nodes;
        expect(layer.output?.nodes).toEqual([...block1Nodes, ...block2Nodes]);
      });

      test('should have a custom input method', () => {
        expect(typeof layer.input).toBe('function');
        expect(layer.input).not.toBe(Layer.prototype.input);
      });

      describe('Memory Layer input() method', () => {
        let sourceLayer: Layer;
        let sourceGroup: Group;
        let sourceOutputConnectSpy: jest.SpyInstance | undefined;
        let sourceGroupConnectSpy: jest.SpyInstance;
        let lastBlock: Group;

        beforeEach(() => {
          sourceLayer = Layer.dense(size);
          sourceGroup = new Group(size);
          lastBlock = layer.nodes[memoryDepth - 1] as unknown as Group;

          if (sourceLayer.output) {
            sourceOutputConnectSpy = jest.spyOn(sourceLayer.output, 'connect');
          }
          sourceGroupConnectSpy = jest.spyOn(sourceGroup, 'connect');
        });

        afterEach(() => {
          sourceOutputConnectSpy?.mockRestore();
          sourceGroupConnectSpy.mockRestore();
        });

        test('should connect source Layer output to the last memory block (ONE_TO_ONE, weight 1)', () => {
          layer.input(sourceLayer);

          expect(sourceOutputConnectSpy).toHaveBeenCalledTimes(1);
          expect(sourceOutputConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: lastBlock.nodes }), methods.groupConnection.ONE_TO_ONE, 1);
          expect(sourceGroupConnectSpy).not.toHaveBeenCalled();
        });

        test('should connect source Group to the last memory block (ONE_TO_ONE, weight 1)', () => {
          layer.input(sourceGroup);

          expect(sourceGroupConnectSpy).toHaveBeenCalledTimes(1);
          expect(sourceGroupConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: lastBlock.nodes }), methods.groupConnection.ONE_TO_ONE, 1);
          expect(sourceOutputConnectSpy).not.toHaveBeenCalled();
        });

        test('should ignore provided method and weight, forcing ONE_TO_ONE and weight 1', () => {
          layer.input(sourceGroup, methods.groupConnection.ALL_TO_ALL, 0.5);

          expect(sourceGroupConnectSpy).toHaveBeenCalledTimes(1);
          expect(sourceGroupConnectSpy).toHaveBeenCalledWith(expect.objectContaining({ nodes: lastBlock.nodes }), methods.groupConnection.ONE_TO_ONE, 1);
        });

        test('should throw error if source size does not match memory block size', () => {
          const wrongSizeSource = new Group(size + 1);
          expect(() => layer.input(wrongSizeSource)).toThrow(
            `Previous layer size (${wrongSizeSource.nodes.length}) must be same as memory size (${size})`
          );
        });

        test('should throw error if the target input block is not a Group (edge case)', () => {
          layer.nodes[memoryDepth - 1] = new Node();
          expect(() => layer.input(sourceGroup)).toThrow('Memory layer input block is not a Group.');
        });
      });
    });
  });

  describe('isGroup (private helper)', () => {
    let layer: Layer;

    beforeEach(() => {
      layer = new Layer();
    });

    test('should return true for a Group instance', () => {
      const group = new Group(1);
      expect((layer as any).isGroup(group)).toBe(true);
    });

    test('should return false for a Node instance', () => {
      const node = new Node();
      expect((layer as any).isGroup(node)).toBe(false);
    });
 
    test('should return false for a plain object', () => {
      const obj = { nodes: [], set: () => {} };
      expect((layer as any).isGroup(obj)).toBe(true);

      const objMissingNodes = { set: () => {} };
      expect((layer as any).isGroup(objMissingNodes)).toBe(false);

      const objMissingSet = { nodes: [] };
      expect((layer as any).isGroup(objMissingSet)).toBe(false);
    });

    test('should return false for null', () => {
      expect((layer as any).isGroup(null)).toBe(false);
    });

    test('should return false for undefined', () => {
      expect((layer as any).isGroup(undefined)).toBe(false);
    });

    test('should return false for primitive types', () => {
      expect((layer as any).isGroup(123)).toBe(false);
      expect((layer as any).isGroup("string")).toBe(false);
      expect((layer as any).isGroup(true)).toBe(false);
    });
  });
});
