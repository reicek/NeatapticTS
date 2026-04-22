import { EventEmitter } from 'events';
import { type ChildProcess, fork } from 'child_process';
import TestWorker from './testworker';

jest.mock('child_process', () => ({
  fork: jest.fn(),
}));

type EvaluationPayload = {
  activations: number[];
  states: number[];
  conns: number[];
};

type ChildProcessHarness = EventEmitter & {
  send: jest.MockedFunction<(message: unknown) => boolean>;
  kill: jest.MockedFunction<() => boolean>;
};

const mockedFork = jest.mocked(fork);

function createChildProcessHarness(input?: {
  onEvaluationPayload?: (childProcessHarness: ChildProcessHarness) => void;
}): ChildProcessHarness {
  const childProcessHarness = new EventEmitter() as ChildProcessHarness;

  childProcessHarness.send = jest.fn((message: unknown) => {
    if (isEvaluationPayload(message)) {
      queueMicrotask(() => {
        input?.onEvaluationPayload?.(childProcessHarness) ??
          childProcessHarness.emit('message', 0.25);
      });
    }

    return true;
  });

  childProcessHarness.kill = jest.fn(() => true);

  return childProcessHarness;
}

function isEvaluationPayload(message: unknown): message is EvaluationPayload {
  if (typeof message !== 'object' || message === null) {
    return false;
  }

  return (
    'activations' in message &&
    'states' in message &&
    'conns' in message &&
    Array.isArray(message.activations) &&
    Array.isArray(message.states) &&
    Array.isArray(message.conns)
  );
}

function createCandidateNetwork() {
  return {
    serialize(): [number[], number[], number[]] {
      return [[1], [2], [3]];
    },
  };
}

async function captureRejectionMessage(
  evaluationPromise: Promise<number>,
): Promise<string> {
  try {
    await evaluationPromise;
    return 'resolved';
  } catch (error: unknown) {
    return error instanceof Error ? error.message : String(error);
  }
}

describe('node worker wrapper chapter', () => {
  beforeEach(() => {
    mockedFork.mockReset();
  });

  describe('constructor', () => {
    describe('given the wrapper starts a child process for one serialized dataset', () => {
      it('forks the worker entrypoint and sends the initialization payload', () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness();
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );

        // Act
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        testWorker.terminate();

        // Assert
        expect({
          forkPath: mockedFork.mock.calls[0]?.[0],
          initializationPayload: childProcessHarness.send.mock.calls[0]?.[0],
        }).toEqual({
          forkPath: expect.stringContaining('worker'),
          initializationPayload: { set: [1, 2, 3], cost: 'mse' },
        });
      });
    });
  });

  describe('evaluate', () => {
    describe('given the child process responds with a numeric score', () => {
      it('forwards the serialized network and resolves that score', async () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness();
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        const candidateNetwork = createCandidateNetwork();

        // Act
        const evaluationResult = await testWorker.evaluate(candidateNetwork);

        // Assert
        expect({
          evaluationResult,
          evaluationPayload: childProcessHarness.send.mock.calls[1]?.[0],
          listenerCounts: {
            message: childProcessHarness.listenerCount('message'),
            error: childProcessHarness.listenerCount('error'),
            exit: childProcessHarness.listenerCount('exit'),
          },
        }).toEqual({
          evaluationResult: 0.25,
          evaluationPayload: {
            activations: [1],
            states: [2],
            conns: [3],
          },
          listenerCounts: {
            message: 0,
            error: 0,
            exit: 0,
          },
        });
      });
    });

    describe('given the child process emits an error during evaluation', () => {
      it('rejects with that error and removes evaluation listeners', async () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness({
          onEvaluationPayload: (workerHarness) => {
            workerHarness.emit('error', new Error('worker error'));
          },
        });
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        const candidateNetwork = createCandidateNetwork();

        // Act
        const rejectionMessage = await captureRejectionMessage(
          testWorker.evaluate(candidateNetwork),
        );

        // Assert
        expect({
          rejectionMessage,
          listenerCounts: {
            message: childProcessHarness.listenerCount('message'),
            error: childProcessHarness.listenerCount('error'),
            exit: childProcessHarness.listenerCount('exit'),
          },
        }).toEqual({
          rejectionMessage: 'worker error',
          listenerCounts: {
            message: 0,
            error: 0,
            exit: 0,
          },
        });
      });
    });

    describe('given the child process exits with a numeric exit code', () => {
      it('rejects with the exit-code message and removes evaluation listeners', async () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness({
          onEvaluationPayload: (workerHarness) => {
            workerHarness.emit('exit', 9, null);
          },
        });
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        const candidateNetwork = createCandidateNetwork();

        // Act
        const rejectionMessage = await captureRejectionMessage(
          testWorker.evaluate(candidateNetwork),
        );

        // Assert
        expect({
          rejectionMessage,
          listenerCounts: {
            message: childProcessHarness.listenerCount('message'),
            error: childProcessHarness.listenerCount('error'),
            exit: childProcessHarness.listenerCount('exit'),
          },
        }).toEqual({
          rejectionMessage: 'worker exited with code 9',
          listenerCounts: {
            message: 0,
            error: 0,
            exit: 0,
          },
        });
      });
    });

    describe('given the child process exits because of a signal', () => {
      it('rejects with the signal message and removes evaluation listeners', async () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness({
          onEvaluationPayload: (workerHarness) => {
            workerHarness.emit('exit', null, 'SIGTERM');
          },
        });
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        const candidateNetwork = createCandidateNetwork();

        // Act
        const rejectionMessage = await captureRejectionMessage(
          testWorker.evaluate(candidateNetwork),
        );

        // Assert
        expect({
          rejectionMessage,
          listenerCounts: {
            message: childProcessHarness.listenerCount('message'),
            error: childProcessHarness.listenerCount('error'),
            exit: childProcessHarness.listenerCount('exit'),
          },
        }).toEqual({
          rejectionMessage: 'worker exited with signal SIGTERM',
          listenerCounts: {
            message: 0,
            error: 0,
            exit: 0,
          },
        });
      });
    });

    describe('given the child process exits without a code or signal', () => {
      it('rejects with the generic exit message and removes evaluation listeners', async () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness({
          onEvaluationPayload: (workerHarness) => {
            workerHarness.emit('exit', null, null);
          },
        });
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });
        const candidateNetwork = createCandidateNetwork();

        // Act
        const rejectionMessage = await captureRejectionMessage(
          testWorker.evaluate(candidateNetwork),
        );

        // Assert
        expect({
          rejectionMessage,
          listenerCounts: {
            message: childProcessHarness.listenerCount('message'),
            error: childProcessHarness.listenerCount('error'),
            exit: childProcessHarness.listenerCount('exit'),
          },
        }).toEqual({
          rejectionMessage: 'worker exited',
          listenerCounts: {
            message: 0,
            error: 0,
            exit: 0,
          },
        });
      });
    });
  });

  describe('terminate', () => {
    describe('given the wrapper is no longer needed', () => {
      it('kills the child process', () => {
        // Arrange
        const childProcessHarness = createChildProcessHarness();
        mockedFork.mockReturnValue(
          childProcessHarness as unknown as ChildProcess,
        );
        const testWorker = new TestWorker([1, 2, 3], { name: 'mse' });

        // Act
        testWorker.terminate();

        // Assert
        expect(childProcessHarness.kill).toHaveBeenCalled();
      });
    });
  });
});
