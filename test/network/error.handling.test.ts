import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';
// Define our own fail function instead of importing from Jest
const fail = (message: string): void => {
  throw new Error(message);
};

let globalWarnSpy: jest.SpyInstance;
beforeAll(() => {
  globalWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
});
afterAll(() => {
  globalWarnSpy.mockRestore();
});

// Helper to wrap a test with a timeout and retry logic
function testWithTimeoutAndRetry(
  name: string,
  fn: (done: jest.DoneCallback) => void,
  timeoutMs: number = 5000,
  retries: number = 3
) {
  test(name, (done: jest.DoneCallback) => {
    let attempts = 0;
    let lastError: any;
    function runAttempt() {
      let finished = false;
      const timer = setTimeout(() => {
        if (!finished) {
          finished = true;
          if (++attempts < retries) {
            runAttempt();
          } else {
            done.fail(new Error(`Test timed out after ${retries} attempts (${timeoutMs}ms each)`));
          }
        }
      }, timeoutMs);
      try {
        fn(done);
      } catch (err: any) {
        lastError = err;
        if (!finished) {
          finished = true;
          clearTimeout(timer);
          if (++attempts < retries) {
            runAttempt();
          } else {
            done.fail(lastError instanceof Error ? lastError : String(lastError));
          }
        }
      }
    }
    runAttempt();
  });
}

xdescribe('Network Error Handling & Scenarios', () => {
  describe('activate()', () => {
    describe('Scenario: too few input values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: too many input values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1, 2, 3]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: correct input size', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.activate([1, 2]);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('noTraceActivate()', () => {
    describe('Scenario: too few input values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: too many input values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1, 2, 3]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: correct input size', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.noTraceActivate([1, 2]);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('propagate()', () => {
    describe('Scenario: too many target values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, [1, 2]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: too few target values', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, []);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: correct target size', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0, 1]);
        // Act
        const act = () => net.propagate(0.1, 0, true, [1]);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('train()', () => {
    beforeAll(() => {
      const { config } = require('../../src/config');
      config.warnings = true;
    });
    describe('Scenario: input/output size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1], output: [1] }], { iterations: 1 });
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: output size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1, 2] }], { iterations: 1 });
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: batch size too large', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], { batchSize: 2, iterations: 1 });
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: missing iterations and error', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], {});
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: valid training options', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.train([{ input: [1, 2], output: [1] }], { iterations: 1 });
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
    describe('Scenario: missing rate option', () => {
      testWithTimeoutAndRetry('warns about missing rate', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        net.train([{ input: [1, 2], output: [1] }], { iterations: 1 });
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `rate` option'));
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: missing iterations and error option', () => {
      testWithTimeoutAndRetry('warns about missing iterations and error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        try { net.train([{ input: [1, 2], output: [1] }], {}); } catch {};
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `iterations` or `error` option'));
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: missing iterations option', () => {
      testWithTimeoutAndRetry('warns about missing iterations', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Spy
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act
        net.train([{ input: [1, 2], output: [1] }], { error: 0.1 });
        // Assert
        expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Missing `iterations` option'));
        warnSpy.mockRestore();
        done();
      });
    });
  });

  describe('test()', () => {
    describe('Scenario: input size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1], output: [1] }]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: output size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1, 2], output: [1, 2] }]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: empty test set', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([]);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: valid test set', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        const act = () => net.test([{ input: [1, 2], output: [1] }]);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('merge()', () => {
    describe('Scenario: output/input sizes do not match', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net1 = new Network(2, 3);
        const net2 = new Network(2, 1);
        // Act
        const act = () => Network.merge(net1, net2);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: output/input sizes match', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net1 = new Network(2, 2);
        const net2 = new Network(2, 2);
        // Act
        const act = () => Network.merge(net1, net2);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('ungate()', () => {
    describe('Scenario: connection not in gates list', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const [conn] = net.nodes[0].connect(net.nodes[1]);
        // Act
        const act = () => net.ungate(conn);
        // Assert
        expect(act).toThrow();
        done();
      });
    });
    describe('Scenario: connection in gates list', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const [conn] = net.connect(net.nodes[0], net.nodes[1]);
        net.gate(net.nodes[1], conn);
        // Act
        const act = () => net.ungate(conn);
        // Assert
        expect(act).not.toThrow();
        done();
      });
    });
  });

  describe('deserialize()', () => {
    describe('Scenario: no input/output info', () => {
      testWithTimeoutAndRetry('creates network with generic nodes', (done) => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0.5, 0.5]);
        const arr = net.serialize();
        // Act
        const deserialized = Network.deserialize(arr);
        // Assert
        expect(deserialized.nodes.length).toBe(net.nodes.length);
        done();
      });
    });
    describe('Scenario: deserialization returns nodes array', () => {
      testWithTimeoutAndRetry('returns nodes array', (done) => {
        // Arrange
        const net = new Network(2, 1);
        net.activate([0.5, 0.5]);
        const arr = net.serialize();
        // Act
        const deserialized = Network.deserialize(arr);
        // Assert
        expect(Array.isArray(deserialized.nodes)).toBe(true);
        done();
      });
    });
  });

  describe('Advanced Error Handling Scenarios', () => {
    testWithTimeoutAndRetry('should throw if cost function is invalid', (done) => {
      // Arrange
      const net = new Network(2, 1);
      // Act & Assert
      expect(() => net.train([{ input: [1, 2], output: [1] }], { iterations: 1, cost: 'notARealCostFn' })).toThrow();
      done();
    });

    testWithTimeoutAndRetry('should warn if activation function is invalid', (done) => {
      // Arrange
      const net = new Network(2, 1);
      // Set invalid squash on a hidden node (not input)
      net.mutate(methods.mutation.ADD_NODE);
      const hidden = net.nodes.find(n => n.type === 'hidden');
      hidden!.squash = (null as any);
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      // Act
      try { net.activate([1, 2]); } catch {}
      // Assert
      expect(warnSpy).toHaveBeenCalled();
      warnSpy.mockRestore();
      done();
    });
  });
  
  describe('Training robustness', () => {
    testWithTimeoutAndRetry('should recover from error in a single iteration and continue', (done) => {
      // Arrange
      const net = new Network(2, 1);
      const goodData = [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] }
      ];
      
      // Instead of using fail(), test our implementation differently
      let errorThrown = false;
      const originalActivate = net.activate;
      
      // Create a mock activate that throws once
      net.activate = jest.fn(function(input, training) {
        if (!errorThrown && input[0] === 0 && input[1] === 1) {
          errorThrown = true;
          throw new Error("Test error");
        }
        return originalActivate.apply(net, [input, training]);
      });
      
      // Act - Train the network despite the error in one iteration
      try {
        // This training will encounter our error but should continue
        net.train(goodData, { 
          iterations: 10, 
          error: 0.01,
          rate: 0.3
        });
        
        // If we got here without an exception, count it as success
        expect(errorThrown).toBe(true);
      } catch (e) {
        // This should not throw, but if it does, make it a regular test failure
        expect(e).toBe(undefined);
      } finally {
        // Restore original method
        net.activate = originalActivate;
      }
      done();
    });
  });
});
