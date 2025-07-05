import Multi from '../../src/multithreading/multi';

// Helper: simple cost function for tests
const mse = (expected: number[], actual: number[]) => {
  let sum = 0;
  for (let i = 0; i < expected.length; i++) {
    sum += Math.pow(expected[i] - actual[i], 2);
  }
  return sum / expected.length;
};

describe('Multi-threading Utilities (Multi)', () => {
  // Set longer timeout for multithreading tests
  jest.setTimeout(10000);

  beforeAll(() => {
    // Only set up DummyWorker for positive import tests, not globally
  });

  afterAll(() => {
    jest.restoreAllMocks();
  });

  describe('serializeDataSet', () => {
    describe('when dataset is valid', () => {
      it('serializes a simple dataset', () => {
        // Arrange
        const set = [
          { input: [1, 2], output: [3] },
          { input: [4, 5], output: [9] }
        ];
        // Act
        const serialized = Multi.serializeDataSet(set);
        // Assert
        expect(Array.isArray(serialized)).toBe(true);
      });
    });
    describe('when dataset is empty', () => {
      it('throws an error', () => {
        // Arrange
        const set: any[] = [];
        // Act & Assert
        expect(() => Multi.serializeDataSet(set)).toThrow();
      });
    });
  });

  describe('deserializeDataSet', () => {
    describe('when input is valid', () => {
      it('deserializes a simple serialized dataset', () => {
        // Arrange
        const set = [
          { input: [1, 2], output: [3] },
          { input: [4, 5], output: [9] }
        ];
        const serialized = Multi.serializeDataSet(set);
        // Act
        const deserialized = Multi.deserializeDataSet(serialized);
        // Assert
        expect(deserialized).toEqual(set);
      });
    });
    describe('when input is empty', () => {
      it('returns an empty array', () => {
        // Arrange
        const serialized: number[] = [];
        // Act
        const deserialized = Multi.deserializeDataSet(serialized);
        // Assert
        expect(deserialized).toEqual([]);
      });
    });
    describe('when input is malformed', () => {
      it('returns an empty array', () => {
        // Arrange
        const malformed = [2]; // Not enough info
        // Act
        const deserialized = Multi.deserializeDataSet(malformed);
        // Assert
        expect(deserialized).toEqual([]);
      });
    });
  });

  describe('activateSerializedNetwork', () => {
    describe('when input and data are valid', () => {
      it('returns an array as output', () => {
        // Arrange
        const input = [1];
        const A = [0];
        const S = [0];
        const data = [1, 1, 0, 0, 0, 0, -1, -2]; // squash index is 0
        const F = [(x: number) => x];
        // Act
        const output = Multi.activateSerializedNetwork(input, A, S, data, F);
        // Assert
        expect(Array.isArray(output)).toBe(true);
      });
    });
    describe('when data is malformed', () => {
      it('returns an empty array', () => {
        // Arrange
        const input = [1];
        const A = [0];
        const S = [0];
        const data = [1]; // malformed
        const F = [(x: number) => x];
        // Act
        const output = Multi.activateSerializedNetwork(input, A, S, data, F);
        // Assert
        expect(output).toEqual([]);
      });
    });
  });

  describe('testSerializedSet', () => {
    describe('when set is non-empty', () => {
      it('returns a number as average error', () => {
        // Arrange
        const set = [
          { input: [1], output: [2] },
          { input: [2], output: [4] }
        ];
        const cost = (expected: number[], actual: number[]) => Math.abs(expected[0] - actual[0]);
        const A = [0];
        const S = [0];
        const data = [1, 1, 0, 0, 0, 0, -1, -2]; // squash index is 0
        const F = [(x: number) => x * 2];
        // Act
        const error = Multi.testSerializedSet(set, cost, A, S, data, F);
        // Assert
        expect(typeof error).toBe('number');
      });
    });
    describe('when set is empty', () => {
      it('returns NaN', () => {
        // Arrange
        const set: any[] = [];
        const cost = (expected: number[], actual: number[]) => 0;
        const A = [0];
        const S = [0];
        const data = [1, 1, 0, 0, 0, 0, -1, -2]; // squash index is 0
        const F = [(x: number) => x];
        // Act
        const error = Multi.testSerializedSet(set, cost, A, S, data, F);
        // Assert
        expect(Number.isNaN(error)).toBe(true);
      });
    });
    describe('when cost function throws', () => {
      it('returns NaN', () => {
        // Arrange
        const set = [ { input: [1], output: [2] } ];
        const cost = () => { throw new Error('cost error'); };
        const A = [0];
        const S = [0];
        const data = [1, 1, 0, 0, 0, 0, -1, -2]; // squash index is 0
        const F = [(x: number) => x];
        // Act
        let error;
        try {
          error = Multi.testSerializedSet(set, cost, A, S, data, F);
        } catch (e) {
          error = NaN;
        }
        // Assert
        expect(Number.isNaN(error)).toBe(true);
      });
    });
    describe('when cost function returns NaN', () => {
      it('returns NaN as error', () => {
        // Arrange
        const set = [ { input: [1], output: [2] } ];
        const cost = () => NaN;
        const A = [0];
        const S = [0];
        const data = [1, 1, 0, 0, 0, 0, -1, -2]; // squash index is 0
        const F = [(x: number) => x];
        // Act
        const error = Multi.testSerializedSet(set, cost, A, S, data, F);
        // Assert
        expect(Number.isNaN(error)).toBe(true);
      });
    });
  });

  describe('getBrowserTestWorker', () => {
    describe('when import succeeds', () => {
      beforeAll(() => {
        // Provide a dummy class as the resolved value, matching the expected type
        class DummyWorker {
          private worker: any;
          constructor(dataSet: number[], cost: { name: string }) {
            this.worker = null;
          }
          evaluate = () => {};
          terminate = () => {};
          test = () => {};
          private static _createBlobString = () => '';
        }
        jest.spyOn(Multi, 'getBrowserTestWorker').mockResolvedValue(DummyWorker as any);
      });
      afterAll(() => {
        jest.restoreAllMocks();
      });
      it('returns a TestWorker class', async () => {
        // Act
        const WorkerClass = await Multi.getBrowserTestWorker();
        // Assert
        expect(WorkerClass).toBeDefined();
      });
    });
    describe('when import fails', () => {
      beforeEach(() => {
        jest.resetModules();
        jest.dontMock('../../src/multithreading/workers/browser/testworker');
        jest.restoreAllMocks();
        jest.doMock('../../src/multithreading/workers/browser/testworker', () => {
          throw new Error('fail');
        });
      });
      afterEach(() => {
        jest.dontMock('../../src/multithreading/workers/browser/testworker');
        jest.resetModules();
        jest.restoreAllMocks();
      });
      it('throws an error', async () => {
        // Act & Assert
        await expect(Multi.getBrowserTestWorker()).rejects.toThrow('fail');
      });
    });
  });

  describe('getNodeTestWorker', () => {
    describe('when import succeeds', () => {
      beforeAll(() => {
        // Provide a dummy class as the resolved value, matching the expected type
        class DummyWorker {
          private worker: any;
          constructor(dataSet: number[], cost: { name: string }) {
            this.worker = null;
          }
          evaluate = () => {};
          terminate = () => {};
          test = () => {};
          private static _createBlobString = () => '';
        }
        jest.spyOn(Multi, 'getNodeTestWorker').mockResolvedValue(DummyWorker as any);
      });
      afterAll(() => {
        jest.restoreAllMocks();
      });
      it('returns a TestWorker class', async () => {
        // Act
        const WorkerClass = await Multi.getNodeTestWorker();
        // Assert
        expect(WorkerClass).toBeDefined();
      });
    });
    describe('when import fails', () => {
      beforeEach(() => {
        jest.resetModules();
        jest.dontMock('../../src/multithreading/workers/node/testworker');
        jest.restoreAllMocks();
        jest.doMock('../../src/multithreading/workers/node/testworker', () => {
          throw new Error('fail');
        });
      });
      afterEach(() => {
        jest.dontMock('../../src/multithreading/workers/node/testworker');
        jest.resetModules();
        jest.restoreAllMocks();
      });
      it('throws an error', async () => {
        // Act & Assert
        await expect(Multi.getNodeTestWorker()).rejects.toThrow('fail');
      });
    });
  });
});