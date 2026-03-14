# multithreading/workers/node

Handles messages sent to the worker process.

This function listens for messages sent to the worker process and performs one of two actions:
1. If the message contains serialized activations, states, and connections, it evaluates the network using the dataset.
2. If the message contains a dataset and cost function, it initializes the worker with the provided data.

## multithreading/workers/node/worker.ts

### WorkerMessage

Interface for messages sent to the worker process.

## multithreading/workers/node/testworker.ts

### CostFunction

Interface for cost function used in worker evaluation.

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### TestWorker

TestWorker class for handling network evaluations in a Node.js environment using Worker Threads.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of Worker Threads allows for offloading
computationally expensive tasks, such as network evaluation, to separate threads.

Example:

// Typical usage in an async context
(async () => {
  // example serialized dataset numbers placeholder
  const dataSet = [0, 1, 2];
  const cost = { name: 'mse' };
  const worker = new TestWorker(dataSet, cost);
  try {
    const mockNetwork = { serialize: () => [[0], [0], [0]] };
    const score = await worker.evaluate(mockNetwork);
    console.log('score', score);
  } finally {
    worker.terminate();
  }
})();

#### evaluate

```ts
evaluate(
  network: SerializableNetwork,
): Promise<number>
```

Evaluates a neural network using the worker process.

The network is serialized and sent to the worker for evaluation. The worker
sends back the evaluation result, which is returned as a promise.

Returns: A promise that resolves to the evaluation result.

Example:

// Example: evaluate a mock network (assumes `worker` is an instance of TestWorker)
// Note: `evaluate` returns a Promise — use `await` inside an async function.
const mockNetwork = { serialize: () => [[0], [0], [0]] };
const score = await worker.evaluate(mockNetwork);
console.log('score', score);

#### terminate

```ts
terminate(): void
```

Terminates the worker process.

This method ensures that the worker process is properly terminated to free up system resources.

Example:

// Create and terminate a worker when it's no longer needed
const worker = new TestWorker([0, 1, 2], { name: 'mse' });
// ...use worker.evaluate(...) as needed
worker.terminate();
