import { fork, ChildProcess } from 'child_process';
import path from 'path';

/**
 * TestWorker class for handling network evaluations in a separate Node.js process.
 *
 * This class provides methods to evaluate neural networks and manage the worker process.
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
    this.worker = fork(path.join(__dirname, '/worker'));
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
   */
  evaluate(network: any): Promise<number> {
    return new Promise((resolve) => {
      const serialized = network.serialize();

      const data = {
        activations: serialized[0],
        states: serialized[1],
        conns: serialized[2],
      };

      const _that = this.worker;
      this.worker.on('message', function callback(e: number) {
        _that.removeListener('message', callback);
        resolve(e);
      });

      this.worker.send(data);
    });
  }

  /**
   * Terminates the worker process.
   *
   * This method ensures that the worker process is properly terminated to free up system resources.
   */
  terminate(): void {
    this.worker.kill();
  }
}

// Add default export to match the original JavaScript implementation.
export default TestWorker;
