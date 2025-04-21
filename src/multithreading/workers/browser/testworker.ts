import Multi from '../../multi';

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
   * @param {any} cost - The cost function to evaluate the network.
   */
  constructor(dataSet: number[], cost: { name: string }) {
    const blob = new Blob([TestWorker._createBlobString(cost)]);
    this.url = window.URL.createObjectURL(blob);
    this.worker = new Worker(this.url);

    const data = { set: new Float64Array(dataSet).buffer };
    this.worker.postMessage(data, [data.set]);
  }

  /**
   * Evaluates a network using the worker process.
   * @param {any} network - The network to evaluate.
   * @returns {Promise<number>} A promise that resolves to the evaluation result.
   */
  evaluate(network: any): Promise<number> {
    return new Promise((resolve, reject) => {
      const serialized = network.serialize();

      const data = {
        activations: new Float64Array(serialized[0]).buffer,
        states: new Float64Array(serialized[1]).buffer,
        conns: new Float64Array(serialized[2]).buffer,
      };

      this.worker.onmessage = function (e: MessageEvent) {
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
   * @param {any} cost - The cost function to be used by the worker.
   * @returns {string} The blob string.
   */
  private static _createBlobString(cost: any): string {
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
