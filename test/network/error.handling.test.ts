import { Architect, Network, methods } from '../../src/neataptic';
import Node from '../../src/architecture/node';
import { config } from '../../src/config';
// Define our own fail function instead of importing from Jest
const fail = (message: string): void => {
  throw new Error(message);
};

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
            clearTimeout(timer);
            runAttempt();
          } else {
            clearTimeout(timer);
            done(new Error(`Test timed out after ${retries} attempts (${timeoutMs}ms each)`));
          }
        }
      }, timeoutMs);
      try {
        fn(Object.assign(function (...args: any[]) {
          if (!finished) {
            finished = true;
            clearTimeout(timer);
            done(...args);
          }
        }, { fail: done.fail }));
      } catch (err: any) {
        lastError = err;
        if (!finished) {
          finished = true;
          clearTimeout(timer);
          if (++attempts < retries) {
            runAttempt();
          } else {
            done(lastError instanceof Error ? lastError : new Error(String(lastError)));
          }
        }
      }
    }
    runAttempt();
  });
}

describe('Network Error Handling & Scenarios', () => {
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
      config.warnings = true;
    });
    describe('Scenario: input/output size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act & Assert
        let threw = false;
        try {
          net.train([{ input: [1], output: [1] }], { iterations: 1 });
        } catch {
          threw = true;
        }
        expect(threw).toBe(true);
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: output size mismatch', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act & Assert
        let threw = false;
        try {
          net.train([{ input: [1, 2], output: [1, 2] }], { iterations: 1 });
        } catch {
          threw = true;
        }
        expect(threw).toBe(true);
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: batch size too large', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act & Assert
        let threw = false;
        try {
          net.train([{ input: [1, 2], output: [1] }], { batchSize: 2, iterations: 1 });
        } catch {
          threw = true;
        }
        expect(threw).toBe(true);
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: missing iterations and error', () => {
      testWithTimeoutAndRetry('throws error', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act & Assert
        let threw = false;
        try {
          net.train([{ input: [1, 2], output: [1] }], {});
        } catch {
          threw = true;
        }
        expect(threw).toBe(true);
        warnSpy.mockRestore();
        done();
      });
    });
    describe('Scenario: valid training options', () => {
      testWithTimeoutAndRetry('does not throw', (done) => {
        // Arrange
        const net = new Network(2, 1);
        // Act
        net.train([{ input: [1, 2], output: [1] }], { iterations: 1 });
        // Assert: should not throw
        done();
      });
    });
    describe('Scenario: missing rate option', () => {
      testWithTimeoutAndRetry('warns about missing rate or throws', (done) => {
        // Arrange
        const net = new Network(2, 1);
        const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
        // Act & Assert
        let threw = false;
        try {
          net.train([{ input: [1, 2], output: [1] }], { iterations: 1 }); // No rate provided
        } catch {
          threw = true;
        }
        if (threw) {
          expect(threw).toBe(true);
        } else {
          const calls = warnSpy.mock.calls;
          const found = calls.some(call => call[0] && call[0].includes('Missing `rate` option'));
          expect(found).toBe(true);
        }
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
        // Assert (synchronously)
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
        try {
          net.train([{ input: [1, 2], output: [1] }], { error: 0.1 });
        } catch {}
        // Assert (synchronously)
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
      describe('when called with a connection not in gates', () => {
        testWithTimeoutAndRetry('warns and does not throw', (done) => {
          // Arrange
          const net = new Network(2, 1);
          const [conn] = net.nodes[0].connect(net.nodes[1]);
          // Spy
          const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
          // Act
          const act = () => net.ungate(conn);
          // Assert
          expect(act).not.toThrow();
          expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('Attempted to ungate a connection not in the gates list.'));
          warnSpy.mockRestore();
          done();
        });
      });
      describe('when called with an invalid connection', () => {
        testWithTimeoutAndRetry('warns and does not throw', (done) => {
          // Arrange
          const net = new Network(2, 1);
          // Spy
          const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
          // Act
          const act = () => net.ungate(undefined as any);
          // Assert
          expect(act).not.toThrow();
          expect(warnSpy).toHaveBeenCalled();
          warnSpy.mockRestore();
          done();
        });
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
    describe('Scenario: invalid cost function', () => {
      it('should throw if cost function is invalid', () => {
        // Arrange
        const net = new Network(2, 1);
        // Act & Assert
        expect(() => net.train([{ input: [1, 2], output: [1] }], { iterations: 1, cost: 'notARealCostFn' })).toThrow();
      });
    });
    describe('Scenario: invalid activation function', () => {
      let warnSpy: jest.SpyInstance;
      beforeEach(() => {
        // Spy
        warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
      });
      afterEach(() => {
        warnSpy.mockRestore();
      });
      it('should warn if activation function is invalid', () => {
        // Arrange
        const net = new Network(2, 1);
        net.mutate(methods.mutation.ADD_NODE);
        const hidden = net.nodes.find(n => n.type === 'hidden');
        hidden!.squash = (null as any);
        // Act
        try { net.activate([1, 2]); } catch {}
        // Assert
        expect(warnSpy).toHaveBeenCalled();
      });
    });
  });
  
  describe('Training robustness', () => {
    describe('Scenario: error in a single iteration', () => {
      let net: Network;
      let originalActivate: any;
      beforeEach(() => {
        // Arrange
        net = new Network(2, 1);
        originalActivate = net.activate;
      });
      afterEach(() => {
        // Restore original method
        net.activate = originalActivate;
      });
      it('should set errorThrown to true if error occurs', () => {
        // Arrange
        const goodData = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] }
        ];
        let errorThrown = false;
        // Spy
        net.activate = jest.fn(function(input, training) {
          if (!errorThrown && input[0] === 0 && input[1] === 1) {
            errorThrown = true;
            throw new Error("Test error");
          }
          return originalActivate.apply(net, [input, training]);
        });
        // Act
        try {
          net.train(goodData, { iterations: 10, error: 0.01, rate: 0.3 });
        } catch {}
        // Assert
        expect(errorThrown).toBe(true);
      });
      it('should not throw from train even if error occurs in one iteration', () => {
        // Arrange
        const goodData = [
          { input: [0, 0], output: [0] },
          { input: [0, 1], output: [1] },
          { input: [1, 0], output: [1] },
          { input: [1, 1], output: [0] }
        ];
        let errorThrown = false;
        // Spy
        net.activate = jest.fn(function(input, training) {
          if (!errorThrown && input[0] === 0 && input[1] === 1) {
            errorThrown = true;
            throw new Error("Test error");
          }
          return originalActivate.apply(net, [input, training]);
        });
        // Act & Assert
        expect(() => {
          net.train(goodData, { iterations: 10, error: 0.01, rate: 0.3 });
        }).not.toThrow();
      });
    });
  });
});
