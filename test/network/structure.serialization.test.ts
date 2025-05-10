import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';

let globalWarnSpy: jest.SpyInstance;
beforeAll(() => {
  globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
});
afterAll(() => {
  if (globalWarnSpy && typeof globalWarnSpy.mockRestore === 'function') {
    globalWarnSpy.mockRestore();
  }
});

describe('Structure & Serialization', () => {
  describe('Feed-forward Property', () => {
    test('maintains feed-forward connections after mutations and crossover', () => {
      // Arrange
      jest.setTimeout(30000);
      const network1 = new Network(2, 2);
      const network2 = new Network(2, 2);
      let i;
      for (i = 0; i < 100; i++) {
        network1.mutate(methods.mutation.ADD_NODE);
        network2.mutate(methods.mutation.ADD_NODE);
      }
      for (i = 0; i < 400; i++) {
        network1.mutate(methods.mutation.ADD_CONN);
        network2.mutate(methods.mutation.ADD_NODE);
      }
      // Act
      const network = Network.crossOver(network1, network2);
      const allFeedForward = network.connections.every((conn) => {
        const fromNode = conn.from;
        const toNode = conn.to;
        if (
          network.nodes.includes(fromNode) &&
          network.nodes.includes(toNode)
        ) {
          const fromIndex = network.nodes.indexOf(fromNode);
          const toIndex = network.nodes.indexOf(toNode);
          return fromIndex < toIndex;
        } else {
          console.error(
            `Connection node not found in network nodes array: from=${fromNode?.index}, to=${toNode?.index}`
          );
          return false;
        }
      });
      // Assert
      expect(allFeedForward).toBe(true);
    });
  });

  describe('fromJSON / toJSON Equivalency', () => {
    jest.setTimeout(15000);
    const runEquivalencyTests = (
      architectureName: string,
      createNetwork: () => Network
    ) => {
      describe(`${architectureName}`, () => {
        let original: Network;
        let copy: Network;
        let input: number[];
        let originalOutput: number[];
        let copyOutput: number[];
        beforeAll(() => {
          // Arrange
          original = createNetwork();
          const json: any = original.toJSON();
          copy = Network.fromJSON(json);
          input = Array.from({ length: original.input }, () => Math.random());
          originalOutput = original.activate(input);
          copyOutput = copy.activate(input);
        });
        test('produces the same output length as the original', () => {
          // Assert
          expect(copyOutput.length).toEqual(originalOutput.length);
        });
        test('produces numerically close outputs to the original', () => {
          // Assert
          const outputsAreEqual = copyOutput.every(
            (val, i) =>
              typeof originalOutput[i] === 'number' &&
              Math.abs(val - originalOutput[i]) < 1e-9
          );
          expect(outputsAreEqual).toBe(true);
        });
        describe('Scenario: fromJSON throws on corrupted data', () => {
          test('throws error if nodes field is missing', () => {
            // Arrange
            const json: any = original.toJSON();
            delete json.nodes;
            // Act
            const act = () => Network.fromJSON(json);
            // Assert
            expect(act).toThrow();
          });
          test('throws error if connections field is missing', () => {
            // Arrange
            const json: any = original.toJSON();
            delete json.connections;
            // Act
            const act = () => Network.fromJSON(json);
            // Assert
            expect(act).toThrow();
          });
        });
      });
    };
    runEquivalencyTests('Perceptron', () =>
      Architect.perceptron(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      )
    );
    runEquivalencyTests(
      'Basic Network',
      () =>
        new Network(
          Math.floor(Math.random() * 5 + 1),
          Math.floor(Math.random() * 5 + 1)
        )
    );
    runEquivalencyTests('LSTM', () =>
      Architect.lstm(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      )
    );
    runEquivalencyTests('GRU', () =>
      Architect.gru(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      )
    );
    runEquivalencyTests('Random', () =>
      Architect.random(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 10 + 1),
        Math.floor(Math.random() * 5 + 1)
      )
    );
    runEquivalencyTests('NARX', () =>
      Architect.narx(
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1),
        Math.floor(Math.random() * 5 + 1)
      )
    );
    runEquivalencyTests('Hopfield', () =>
      Architect.hopfield(Math.floor(Math.random() * 5 + 1))
    );
  });

  describe('Serialize/Deserialize Equivalency', () => {
    describe('Scenario: standard perceptron', () => {
      test('should produce the same output length after serialize/deserialize', () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const input = [Math.random(), Math.random()];
        const originalOutput = net.activate(input);
        // Act
        const arr = net.serialize();
        const deserialized = Network.deserialize(arr, net.input, net.output);
        const deserializedOutput = deserialized.activate(input);
        // Assert
        expect(deserializedOutput.length).toBe(originalOutput.length);
      });
      test('should produce numerically close outputs after serialize/deserialize', () => {
        // Arrange
        const net = Architect.perceptron(2, 4, 1);
        const input = [Math.random(), Math.random()];
        const originalOutput = net.activate(input);
        // Act
        const arr = net.serialize();
        const deserialized = Network.deserialize(arr, net.input, net.output);
        const deserializedOutput = deserialized.activate(input);
        // Assert
        const epsilon = 0.05;
        const diffs = deserializedOutput.map((val, i) => Math.abs(val - originalOutput[i]));
        const allClose = deserializedOutput.length === originalOutput.length &&
          diffs.every(diff => diff < epsilon);
        expect(allClose).toBe(true);
      });
    });
    describe('Scenario: network with no connections', () => {
      test('should serialize and deserialize without error', () => {
        // Arrange
        const net = new Network(2, 1);
        net.connections = [];
        // Act
        const arr = net.serialize();
        const deserialized = Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(deserialized).toBeDefined();
      });
    });
  });

  describe('Deserialization/Serialization Scenarios', () => {
    describe('Scenario: invalid connection indices', () => {
      test('should skip invalid connection indices', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[3][0].from = 999;
        arr[3][0].to = 999;
        // Act
        const deserialized = Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(deserialized.connections.length).toBeLessThanOrEqual(net.connections.length);
      });
      test('should warn for invalid connection indices', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[3][0].from = 999;
        arr[3][0].to = 999;
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid connection indices'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: invalid gater index', () => {
      test('should skip invalid gater index', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[3][0].gater = 999;
        // Act
        const deserialized = Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(deserialized.gates.length).toBeLessThanOrEqual(net.gates.length);
      });
      test('should warn for invalid gater index', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[3][0].gater = 999;
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid gater index'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: unknown squash function', () => {
      test('should fall back to identity for unknown squash', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[2][0] = 'notARealSquashFn';
        // Act
        const deserialized = Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(deserialized.nodes[0].squash).toBe(methods.Activation.identity);
      });
      test('should warn for unknown squash function', () => {
        // Arrange
        const net = new Network(2, 1);
        const arr = net.serialize();
        arr[2][0] = 'notARealSquashFn';
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.deserialize(arr, net.input, net.output);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Unknown squash function'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: fromJSON with unknown squash', () => {
      test('should fall back to identity for unknown squash in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.nodes[0].squash = 'notARealSquashFn';
        // Act
        const deserialized = Network.fromJSON(json);
        // Assert
        expect(deserialized.nodes[0].squash).toBe(methods.Activation.identity);
      });
      test('should warn for unknown squash in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.nodes[0].squash = 'notARealSquashFn';
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.fromJSON(json);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Unknown squash function'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: fromJSON with invalid connection indices', () => {
      test('should skip invalid connection indices in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.connections[0].from = 999;
        json.connections[0].to = 999;
        // Act
        const deserialized = Network.fromJSON(json);
        // Assert
        expect(deserialized.connections.length).toBeLessThanOrEqual(net.connections.length);
      });
      test('should warn for invalid connection indices in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.connections[0].from = 999;
        json.connections[0].to = 999;
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.fromJSON(json);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid connection indices'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: fromJSON with invalid gater index', () => {
      test('should skip invalid gater index in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.connections[0].gater = 999;
        // Act
        const deserialized = Network.fromJSON(json);
        // Assert
        expect(deserialized.gates.length).toBeLessThanOrEqual(net.gates.length);
      });
      test('should warn for invalid gater index in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        json.connections[0].gater = 999;
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.fromJSON(json);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid gater index'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
    describe('Scenario: fromJSON with failed connection creation', () => {
      test('should warn if connection creation fails in fromJSON', () => {
        // Arrange
        const net = new Network(2, 1);
        const json = net.toJSON() as any;
        // Make from and to indices valid but break the connection logic by removing nodes
        json.nodes = json.nodes.slice(1);
        // Spy
        globalWarnSpy.mockRestore();
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        Network.fromJSON(json);
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Invalid connection indices'));
        warnSpy.mockRestore();
        globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
    });
  });
});
