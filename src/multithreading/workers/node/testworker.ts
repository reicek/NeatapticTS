/**
 * Node-side evaluation worker wrapper for serialized networks.
 *
 * This chapter is the server-runtime twin of the browser worker wrapper. It
 * keeps the same public evaluation contract - serialized dataset in,
 * serialized network in, scalar score out - but implements it with a forked
 * helper process instead of a browser `Worker` instance.
 *
 * That split matters because Node and the browser expose different isolation
 * primitives even when the computation is the same. This file owns process
 * startup, one-time dataset handoff, repeated `evaluate()` calls, and cleanup.
 * The neighboring `worker.ts` file owns the child-process message handler that
 * turns those payloads back into a local evaluation run.
 *
 * Read the folder as two layers: `testworker.ts` is the host-side facade and
 * `worker.ts` is the child-process entrypoint. Together they keep evaluation
 * parallelism explicit without leaking process lifecycle details into the rest
 * of the library.
 *
 * ```mermaid
 * flowchart LR
 *   Host[Node host] --> Fork[Fork helper process]
 *   Fork --> Init[Send dataset and cost]
 *   Init --> Evaluate[Send serialized network]
 *   Evaluate --> Score[Receive scalar score]
 * ```
 *
 * For background on the Node primitive used here, see Node.js Documentation,
 * [child_process.fork](https://nodejs.org/api/child_process.html#child_processforkmodulepath-args-options).
 *
 * Example: keep one worker process around while scoring several candidate
 * networks.
 *
 * ```ts
 * const worker = new TestWorker(serializedSet, { name: 'mse' });
 * const score = await worker.evaluate(network);
 * worker.terminate();
 * ```
 */
import { fork, ChildProcess } from 'child_process';
import * as path from 'path';

/**
 * Interface for serializable network used in worker evaluation.
 */
interface SerializableNetwork {
  serialize(): [number[], number[], number[]];
}

/**
 * Interface for cost function used in worker evaluation.
 */
interface CostFunction {
  name: string;
}

/**
 * TestWorker class for handling network evaluations in a Node.js environment
 * through a forked helper process.
 *
 * This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
 * neural networks in parallel environments. The use of a forked process allows for offloading
 * computationally expensive tasks, such as network evaluation, to separate threads.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#4-constraints Instinct Algorithm - Section 4 Constraints}
 *
 * This class provides methods to evaluate neural networks and manage the worker process.
 *
 * @example
 * // Typical usage in an async context
 * (async () => {
 *   // example serialized dataset numbers placeholder
 *   const dataSet = [0, 1, 2];
 *   const cost = { name: 'mse' };
 *   const worker = new TestWorker(dataSet, cost);
 *   try {
 *     const mockNetwork = { serialize: () => [[0], [0], [0]] };
 *     const score = await worker.evaluate(mockNetwork);
 *     console.log('score', score);
 *   } finally {
 *     worker.terminate();
 *   }
 * })();
 */
export class TestWorker {
  private worker: ChildProcess;

  /**
   * Creates a new TestWorker instance.
   *
   * This initializes a new worker process and sends the dataset and cost function
   * to the worker for further processing.
   *
   * @param {number[]} dataSet - The serialized dataset to be used by the worker.
   * @param {CostFunction} cost - The cost function to evaluate the network.
   */
  constructor(dataSet: number[], cost: CostFunction) {
    // Use path module to join paths correctly for worker
    const workerPath = path.join(__dirname, '/worker');
    this.worker = fork(workerPath);
    this.worker.send({ set: dataSet, cost: cost.name });
  }

  /**
   * Evaluates a neural network using the worker process.
   *
   * The network is serialized and sent to the worker for evaluation. The worker
   * sends back the evaluation result, which is returned as a promise.
   *
   * @param {SerializableNetwork} network - The neural network to evaluate. It must implement a `serialize` method.
   * @returns {Promise<number>} A promise that resolves to the evaluation result.
   *
   * @example
   * // Example: evaluate a mock network (assumes `worker` is an instance of TestWorker)
   * // Note: `evaluate` returns a Promise — use `await` inside an async function.
   * const mockNetwork = { serialize: () => [[0], [0], [0]] };
   * const score = await worker.evaluate(mockNetwork);
   * console.log('score', score);
   */
  async evaluate(network: SerializableNetwork): Promise<number> {
    const serialized = network.serialize();

    const data = {
      activations: serialized[0],
      states: serialized[1],
      conns: serialized[2],
    };

    return new Promise<number>((resolve, reject) => {
      /**
       * Handler for the worker 'message' event.
       * Resolves the outer promise with the numeric evaluation result sent by the worker.
       * @param {number} e - The numeric result returned by the worker process.
       */
      const onMessage = (e: number) => {
        cleanup();
        resolve(e);
      };

      /**
       * Handler for the worker 'error' event.
       * Cleans up listeners and rejects the promise with the received Error.
       * @param {Error} err - The error emitted by the worker process.
       */
      const onError = (err: Error) => {
        cleanup();
        reject(err);
      };

      /**
       * Handler for the worker 'exit' event.
       * Called when the worker terminates unexpectedly; rejects the promise with a descriptive Error.
       * @param {number|null} code - Exit code if available.
       * @param {string|undefined} signal - Kill signal if the process was terminated by a signal.
       */
      const onExit = (code: number | null, signal?: string) => {
        cleanup();
        reject(
          new Error(
            `worker exited${
              code != null
                ? ` with code ${code}`
                : signal
                  ? ` with signal ${signal}`
                  : ''
            }`,
          ),
        );
      };

      /**
       * Remove all registered event listeners for this evaluation cycle.
       * Keeps the worker EventEmitter clean and prevents memory leaks when multiple
       * evaluations are run sequentially or concurrently.
       */
      const cleanup = () => {
        // use off which is available on EventEmitter in modern Node.js
        this.worker.off('message', onMessage);
        this.worker.off('error', onError);
        this.worker.off(
          'exit',
          onExit as (
            code: number | null,
            signal: NodeJS.Signals | null,
          ) => void,
        );
      };

      this.worker.once('message', onMessage);
      this.worker.once('error', onError);
      this.worker.once(
        'exit',
        onExit as (code: number | null, signal: NodeJS.Signals | null) => void,
      );

      this.worker.send(data);
    });
  }

  /**
   * Terminates the worker process.
   *
   * This method ensures that the worker process is properly terminated to free up system resources.
   *
   * @example
   * // Create and terminate a worker when it's no longer needed
   * const worker = new TestWorker([0, 1, 2], { name: 'mse' });
   * // ...use worker.evaluate(...) as needed
   * worker.terminate();
   */
  terminate(): void {
    this.worker.kill();
  }
}

// Add default export to match the original JavaScript implementation.
export default TestWorker;
