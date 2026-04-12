import { TestWorker } from './testworker';

type BrowserHarness = {
  blobMock: jest.Mock;
  createObjectURL: jest.Mock;
  revokeObjectURL: jest.Mock;
  workerHarness: {
    onmessage: ((event: MessageEvent) => void) | null;
    postMessage: jest.Mock;
    terminate: jest.Mock;
  };
  workerMock: jest.Mock;
};

const globalScope = globalThis as typeof globalThis & {
  Blob?: unknown;
  Worker?: unknown;
  window?: unknown;
};
const originalBlob = globalScope.Blob;
const originalWorker = globalScope.Worker;
const originalWindow = globalScope.window;

describe('browser worker wrapper chapter', () => {
  afterEach(() => {
    restoreGlobal('Blob', originalBlob);
    restoreGlobal('Worker', originalWorker);
    restoreGlobal('window', originalWindow);
  });

  describe('constructor', () => {
    describe('given the wrapper is initialized with a serialized dataset', () => {
      it('creates the blob worker and transfers the dataset buffer once', () => {
        // Arrange
        const browserHarness = installBrowserHarness();

        // Act
        const testWorker = new TestWorker([1, 2], mseCost);
        testWorker.terminate();
        const constructorPayload = browserHarness.workerHarness.postMessage.mock
          .calls[0]?.[0] as { set: ArrayBuffer };
        const transferBuffer = browserHarness.workerHarness.postMessage.mock
          .calls[0]?.[1]?.[0] as ArrayBuffer;
        const blobRecord = browserHarness.createObjectURL.mock.calls[0]?.[0] as {
          parts: unknown[];
        };

        // Assert
        expect({
          datasetValues: Array.from(new Float64Array(constructorPayload.set)),
          transferByteLength: transferBuffer.byteLength,
          workerUrl: browserHarness.workerMock.mock.calls[0]?.[0],
          blobContainsDeserializer: String(blobRecord.parts[0]).includes(
            'deserializeDataSet',
          ),
        }).toEqual({
          datasetValues: [1, 2],
          transferByteLength: 16,
          workerUrl: 'blob:test-worker',
          blobContainsDeserializer: true,
        });
      });
    });
  });

  describe('evaluate', () => {
    describe('given the worker responds with a scalar error buffer', () => {
      it('transfers the serialized network buffers and resolves the error', async () => {
        // Arrange
        const browserHarness = installBrowserHarness();
        const testWorker = new TestWorker([1, 2], mseCost);
        const candidateNetwork = {
          serialize(): [number[], number[], number[]] {
            return [[1, 2], [3], [4, 5]];
          },
        };

        // Act
        const evaluationPromise = testWorker.evaluate(candidateNetwork);
        const evaluationPayload = browserHarness.workerHarness.postMessage.mock
          .calls[1]?.[0] as {
          activations: ArrayBuffer;
          conns: ArrayBuffer;
          states: ArrayBuffer;
        };
        browserHarness.workerHarness.onmessage?.({
          data: { buffer: new Float64Array([0.25]).buffer },
        } as MessageEvent);
        const evaluationResult = await evaluationPromise;
        testWorker.terminate();

        // Assert
        expect({
          activations: Array.from(new Float64Array(evaluationPayload.activations)),
          conns: Array.from(new Float64Array(evaluationPayload.conns)),
          evaluationResult,
          states: Array.from(new Float64Array(evaluationPayload.states)),
          transferCount: browserHarness.workerHarness.postMessage.mock.calls[1]?.[1]
            ?.length,
        }).toEqual({
          activations: [1, 2],
          conns: [4, 5],
          evaluationResult: 0.25,
          states: [3],
          transferCount: 3,
        });
      });
    });
  });

  describe('terminate', () => {
    describe('given the wrapper is no longer needed', () => {
      it('terminates the worker and revokes the object URL', () => {
        // Arrange
        const browserHarness = installBrowserHarness();
        const testWorker = new TestWorker([1], mseCost);

        // Act
        testWorker.terminate();

        // Assert
        expect({
          revokedUrl: browserHarness.revokeObjectURL.mock.calls[0]?.[0],
          terminateCalls: browserHarness.workerHarness.terminate.mock.calls.length,
        }).toEqual({ revokedUrl: 'blob:test-worker', terminateCalls: 1 });
      });
    });
  });

  describe('blob program creation', () => {
    describe('given a named cost function is embedded into the worker script', () => {
      it('includes the cost source and the required multithreading helpers', () => {
        // Arrange
        const blobString = (
          TestWorker as unknown as {
            _createBlobString(cost: typeof mseCost): string;
          }
        )._createBlobString(mseCost);

        // Assert
        expect({
          hasActivationShelf: blobString.includes('const F = ['),
          hasCostSource: blobString.includes('function mse'),
          hasDeserializer: blobString.includes('deserializeDataSet'),
          hasSerializedSetHelper: blobString.includes('testSerializedSet'),
        }).toEqual({
          hasActivationShelf: true,
          hasCostSource: true,
          hasDeserializer: true,
          hasSerializedSetHelper: true,
        });
      });
    });
  });
});

function installBrowserHarness(): BrowserHarness {
  const workerHarness = {
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: jest.fn(),
    terminate: jest.fn(),
  };
  const workerMock = jest.fn(() => workerHarness);
  const blobMock = jest.fn((parts: unknown[]) => ({ parts }));
  const createObjectURL = jest.fn(() => 'blob:test-worker');
  const revokeObjectURL = jest.fn();

  Object.defineProperty(globalScope, 'Blob', {
    configurable: true,
    value: blobMock,
    writable: true,
  });
  Object.defineProperty(globalScope, 'Worker', {
    configurable: true,
    value: workerMock,
    writable: true,
  });
  Object.defineProperty(globalScope, 'window', {
    configurable: true,
    value: {
      URL: {
        createObjectURL,
        revokeObjectURL,
      },
    },
    writable: true,
  });

  return {
    blobMock,
    createObjectURL,
    revokeObjectURL,
    workerHarness,
    workerMock,
  };
}

function restoreGlobal(name: 'Blob' | 'Worker' | 'window', value: unknown): void {
  if (value === undefined) {
    Reflect.deleteProperty(globalScope, name);
    return;
  }

  Object.defineProperty(globalScope, name, {
    configurable: true,
    value,
    writable: true,
  });
}

function mse(expected: number[], actual: number[]): number {
  return Math.abs(expected[0] - actual[0]);
}

const mseCost = mse;