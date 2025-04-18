import Multi from '../../../multithreading/multi';
import * as methods from '../../../methods/methods';

/**
 * The dataset to be used by the worker.
 * This is an array of objects where each object contains:
 * - `input`: An array of input values for the network.
 * - `output`: An array of expected output values for the network.
 */
let set: Array<{ input: number[]; output: number[] }> = [];

/**
 * The cost function to evaluate the network's performance.
 * It takes the expected and actual output arrays and returns a numerical cost value.
 */
let cost: (expected: number[], actual: number[]) => number;

/**
 * The activation functions to be used by the worker.
 * These are imported from the `Multi` module.
 */
const F = Multi.activations;

/**
 * Handles messages sent to the worker process.
 *
 * This function listens for messages sent to the worker process and performs one of two actions:
 * 1. If the message contains serialized activations, states, and connections, it evaluates the network using the dataset.
 * 2. If the message contains a dataset and cost function, it initializes the worker with the provided data.
 *
 * @param e - The message object sent to the worker process. It can contain:
 *   - `set`: Serialized dataset to initialize the worker. This is an array of objects with `input` and `output` properties.
 *   - `cost`: The name of the cost function to use. This should match a key in the `methods.Cost` object.
 *   - `activations`: Serialized activation values for the network.
 *   - `states`: Serialized state values for the network.
 *   - `conns`: Serialized connection data for the network.
 */
process.on(
  'message',
  (e: {
    set?: any;
    cost?: string;
    activations?: any;
    states?: any;
    conns?: any;
  }) => {
    if (typeof e.set === 'undefined') {
      // Deserialize the activations, states, and connections from the message
      const { activations: A, states: S, conns: data } = e;

      // Evaluate the network using the serialized dataset and send the result back
      const result = Multi.testSerializedSet(set, cost, A, S, data, F);

      // Send the evaluation result back to the parent process
      if (process.send) {
        process.send(result);
      }
    } else {
      // Initialize the cost function using the provided name
      // The cost function is retrieved from the `methods.Cost` object
      cost = methods.Cost[e.cost as keyof typeof methods.Cost] as (
        expected: number[],
        actual: number[]
      ) => number;

      // Deserialize the dataset from the message and store it in the `set` variable
      set = Multi.deserializeDataSet(e.set);
    }
  }
);
