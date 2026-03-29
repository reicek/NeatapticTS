/**
 * Browser-side evaluation worker wrapper for serialized networks.
 *
 * This chapter is the Web Worker half of the multithreading boundary. It does
 * not teach activation math again. Its job is to take the flat dataset and
 * network contract from `multithreading/`, package it into browser-friendly
 * transfer buffers, and return one scalar error without blocking the main
 * thread.
 *
 * The lifecycle is intentionally coarse grained. The constructor transfers the
 * dataset once and creates a blob-backed worker program. Each `evaluate()` call
 * then sends only the activation, state, and connection buffers for one
 * candidate network. That keeps the wrapper focused on whole-network scoring
 * jobs rather than chatty per-layer messaging.
 *
 * Read the file in runtime order: constructor first, `evaluate()` second,
 * `_createBlobString()` last. That matches the actual browser story: bootstrap
 * a worker, run repeated evaluations, then inspect how the inline worker code
 * is assembled.
 *
 * ```mermaid
 * flowchart LR
 *   Host[Browser host] --> Dataset[Transfer dataset once]
 *   Dataset --> Blob[Create blob worker]
 *   Blob --> Evaluate[Transfer network buffers]
 *   Evaluate --> Error[Return scalar error]
 * ```
 *
 * For background on the browser primitive behind this wrapper, see MDN Web
 * Docs,
 * [Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).
 *
 * Example: create one browser worker and reuse it across repeated evaluations.
 *
 * ```ts
 * const worker = new TestWorker(serializedSet, { name: 'mse' });
 * const score = await worker.evaluate(network);
 * worker.terminate();
 * ```
 */
import Multi from '../../multi';

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
 * TestWorker class for handling network evaluations in a browser environment using Web Workers.
 *
 * This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
 * neural networks in parallel environments. The use of Web Workers allows for offloading
 * computationally expensive tasks, such as network evaluation, to separate threads.
 *
 * @see Instinct Algorithm - Section 4 Constraints
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
 */
export class TestWorker {
  private worker: Worker;
  private url: string;

  /**
   * Creates a new TestWorker instance.
   * @param {number[]} dataSet - The serialized dataset to be used by the worker.
   * @param {CostFunction} cost - The cost function to evaluate the network.
   */
  constructor(dataSet: number[], cost: CostFunction) {
    const blob = new Blob([TestWorker._createBlobString(cost)]);
    this.url = window.URL.createObjectURL(blob);
    this.worker = new Worker(this.url);

    const data = { set: new Float64Array(dataSet).buffer };
    this.worker.postMessage(data, [data.set]);
  }

  /**
   * Evaluates a network using the worker process.
   * @param {SerializableNetwork} network - The network to evaluate.
   * @returns {Promise<number>} A promise that resolves to the evaluation result.
   */
  evaluate(network: SerializableNetwork): Promise<number> {
    return new Promise((resolve) => {
      const serialized = network.serialize();

      const data = {
        activations: new Float64Array(serialized[0]).buffer,
        states: new Float64Array(serialized[1]).buffer,
        conns: new Float64Array(serialized[2]).buffer,
      };

      this.worker.onmessage = (e: MessageEvent) => {
        const error = new Float64Array(e.data.buffer)[0];
        resolve(error);
      };

      this.worker.postMessage(data, [
        data.activations,
        data.states,
        data.conns,
      ]);
    });
  }

  /**
   * Terminates the worker process and revokes the object URL.
   */
  terminate(): void {
    this.worker.terminate();
    window.URL.revokeObjectURL(this.url);
  }

  /**
   * Creates a string representation of the worker's blob.
   * @param {CostFunction} cost - The cost function to be used by the worker.
   * @returns {string} The blob string.
   */
  private static _createBlobString(cost: CostFunction): string {
    return `
      const F = [${Multi.activations.toString()}];
      const cost = ${cost.toString()};
      const multi = {
        deserializeDataSet: ${Multi.deserializeDataSet.toString()},
        testSerializedSet: ${Multi.testSerializedSet.toString()},
        activateSerializedNetwork: ${Multi.activateSerializedNetwork.toString()}
      };

      let set;

      this.onmessage = function (e) {
        if (typeof e.data.set === 'undefined') {
          const A = new Float64Array(e.data.activations);
          const S = new Float64Array(e.data.states);
          const data = new Float64Array(e.data.conns);

          const error = multi.testSerializedSet(set, cost, A, S, data, F);

          const answer = { buffer: new Float64Array([error]).buffer };
          postMessage(answer, [answer.buffer]);
        } else {
          set = multi.deserializeDataSet(new Float64Array(e.data.set));
        }
      };`;
  }
}
