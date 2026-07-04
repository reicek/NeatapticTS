# multithreading/workers/node

Node-side evaluation worker wrapper for serialized networks.

This chapter is the server-runtime twin of the browser worker wrapper. It
keeps the same public evaluation contract - serialized dataset in,
serialized network in, scalar score out - but implements it with a forked
helper process instead of a browser `Worker` instance.

That split matters because Node and the browser expose different isolation
primitives even when the computation is the same. This file owns process
startup, one-time dataset handoff, repeated `evaluate()` calls, and cleanup.
The neighboring `worker.ts` file owns the child-process message handler that
turns those payloads back into a local evaluation run.

Read the folder as two layers: `testworker.ts` is the host-side facade and
`worker.ts` is the child-process entrypoint. Together they keep evaluation
parallelism explicit without leaking process lifecycle details into the rest
of the library.

```mermaid
flowchart LR
  Host[Node host] --> Fork[Fork helper process]
  Fork --> Init[Send dataset and cost]
  Init --> Evaluate[Send serialized network]
  Evaluate --> Score[Receive scalar score]
```

For background on the Node primitive used here, see Node.js Documentation,
[child_process.fork](https://nodejs.org/api/child_process.html#child_processforkmodulepath-args-options).

Example: keep one worker process around while scoring several candidate
networks.

```ts
const worker = new TestWorker(serializedSet, { name: 'mse' });
const score = await worker.evaluate(network);
worker.terminate();
```

## multithreading/workers/node/testworker.ts

### CostFunction

Interface for cost function used in worker evaluation.

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### TestWorker

TestWorker class for handling network evaluations in a Node.js environment
through a forked helper process.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of a forked process allows for offloading
computationally expensive tasks, such as network evaluation, to a separate process.

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

## multithreading/workers/node/worker.ts

Handles messages sent to the worker process.

This function listens for messages sent to the worker process and performs one of two actions:
1. If the message contains serialized activations, states, and connections, it evaluates the network using the dataset.
2. If the message contains a dataset and cost function, it initializes the worker with the provided data.

### WorkerMessage

Interface for messages sent to the worker process.
