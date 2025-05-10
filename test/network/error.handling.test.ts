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

describe('Network Error Handling & Scenarios', () => {
  describe('activate()', () => {
    describe('Scenario: too few input values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: too many input values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1, 2, 3]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: correct input size', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1, 2]);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('noTraceActivate()', () => {
    describe('Scenario: too few input values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: too many input values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1, 2, 3]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: correct input size', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1, 2]);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('propagate()', () => {
    describe('Scenario: too many target values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, [1, 2]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: too few target values', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, []);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: correct target size', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, [1]);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('train()', () => {
    beforeAll(() => {
      const { config } = require('../../src/config');
      config.warnings = true;
    });
    describe('Scenario: input/output size mismatch', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1], output: [1] }], { iterations: 1 });
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: output size mismatch', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1, 2] }], { iterations: 1 });
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: batch size too large', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], { batchSize: 2, iterations: 1 });
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: missing iterations and error', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], {});
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: valid training options', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], { iterations: 1 });
        // Assert
        expect(act).not.toThrow();
      });
    });
    describe('Scenario: missing rate option', () => {
      test('warns about missing rate', () => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        net.train([{ input: [1, 2], output: [1] }], { iterations: 1 });
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `rate` option'));
        warnSpy.mockRestore();
      });
    });
    describe('Scenario: missing iterations and error option', () => {
      test('warns about missing iterations and error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        try { net.train([{ input: [1, 2], output: [1] }], {}); } catch {};
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `iterations` or `error` option'));
        warnSpy.mockRestore();
      });
    });
    describe('Scenario: missing iterations option', () => {
      test('warns about missing iterations', () => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        net.train([{ input: [1, 2], output: [1] }], { error: 0.1 });
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `iterations` option'));
        warnSpy.mockRestore();
      });
    });
  });

  describe('test()', () => {
    describe('Scenario: input size mismatch', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1], output: [1] }]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: output size mismatch', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1, 2], output: [1, 2] }]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: empty test set', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([]);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: valid test set', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1, 2], output: [1] }]);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('merge()', () => {
    describe('Scenario: output/input sizes do not match', () => {
      test('throws error', () => {
        // Arrange
        const net1 = new Network(2, 3);
        const net2 = new Network(2, 1);
        // Act
        const act = () => Network.merge(net1, net2);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: output/input sizes match', () => {
      test('does not throw', () => {
        // Arrange
        const net1 = new Network(2, 2);
        const net2 = new Network(2, 2);
        // Act
        const act = () => Network.merge(net1, net2);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('ungate()', () => {
    describe('Scenario: connection not in gates list', () => {
      test('throws error', () => {
        // Arrange
        const net = new Network(2, 1);
        const [conn] = net.nodes[0].connect(net.nodes[1]);
        // Act
        const act = () => net.ungate(conn);
        // Assert
        expect(act).toThrow();
      });
    });
    describe('Scenario: connection in gates list', () => {
      test('does not throw', () => {
        // Arrange
        const net = new Network(2, 1);
        const [conn] = net.connect(net.nodes[0], net.nodes[1]);
        net.gate(net.nodes[1], conn);
        // Act
        const act = () => net.ungate(conn);
        // Assert
        expect(act).not.toThrow();
      });
    });
  });

  describe('deserialize()', () => {
    describe('Scenario: no input/output info', () => {
      test('creates network with generic nodes', () => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0.5, 0.5]);
        const arr = net.serialize();
        // Act
        const deserialized = Network.deserialize(arr);
        // Assert
        expect(deserialized.nodes.length).toBe(net.nodes.length);
      });
    });
    describe('Scenario: deserialization returns nodes array', () => {
      test('returns nodes array', () => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0.5, 0.5]);
        const arr = net.serialize();
        // Act
        const deserialized = Network.deserialize(arr);
        // Assert
        expect(Array.isArray(deserialized.nodes)).toBe(true);
      });
    });
  });
});
