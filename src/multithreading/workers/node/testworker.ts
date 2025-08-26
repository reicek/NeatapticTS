import { fork, ChildProcess } from 'child_process';

/**
 * TestWorker class for handling network evaluations in a Node.js environment using Worker Threads.
 *
 * This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
 * neural networks in parallel environments. The use of Worker Threads allows for offloading
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
   * @param {{ name: string }} cost - The cost function to evaluate the network.
   */
  constructor(dataSet: number[], cost: { name: string }) {
    // Lazily require 'path' at runtime to avoid bundlers resolving Node builtins
    let pathModule: any = null;
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      pathModule = require('path');
    } catch {}
    const workerPath = pathModule
      ? pathModule.join(__dirname, '/worker')
      : './worker';
    this.worker = fork(workerPath);
    this.worker.send({ set: dataSet, cost: cost.name });
  }

  /**
   * Evaluates a neural network using the worker process.
   *
   * The network is serialized and sent to the worker for evaluation. The worker
   * sends back the evaluation result, which is returned as a promise.
   *
   * @param {any} network - The neural network to evaluate. It must implement a `serialize` method.
   * @returns {Promise<number>} A promise that resolves to the evaluation result.
   *
   * @example
   * // Example: evaluate a mock network (assumes `worker` is an instance of TestWorker)
   * // Note: `evaluate` returns a Promise — use `await` inside an async function.
   * const mockNetwork = { serialize: () => [[0], [0], [0]] };
   * const score = await worker.evaluate(mockNetwork);
   * console.log('score', score);
   */
  async evaluate(network: any): Promise<number> {
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
            }`
          )
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
        this.worker.off('exit', onExit as any);
      };

      this.worker.once('message', onMessage);
      this.worker.once('error', onError);
      this.worker.once('exit', onExit as any);

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
