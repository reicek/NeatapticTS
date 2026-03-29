# multithreading/workers/browser

Browser-side evaluation worker wrapper for serialized networks.

This chapter is the Web Worker half of the multithreading boundary. It does
not teach activation math again. Its job is to take the flat dataset and
network contract from `multithreading/`, package it into browser-friendly
transfer buffers, and return one scalar error without blocking the main
thread.

The lifecycle is intentionally coarse grained. The constructor transfers the
dataset once and creates a blob-backed worker program. Each `evaluate()` call
then sends only the activation, state, and connection buffers for one
candidate network. That keeps the wrapper focused on whole-network scoring
jobs rather than chatty per-layer messaging.

Read the file in runtime order: constructor first, `evaluate()` second,
`_createBlobString()` last. That matches the actual browser story: bootstrap
a worker, run repeated evaluations, then inspect how the inline worker code
is assembled.

```mermaid
flowchart LR
  Host[Browser host] --> Dataset[Transfer dataset once]
  Dataset --> Blob[Create blob worker]
  Blob --> Evaluate[Transfer network buffers]
  Evaluate --> Error[Return scalar error]
```

For background on the browser primitive behind this wrapper, see MDN Web
Docs,
[Using Web Workers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers).

Example: create one browser worker and reuse it across repeated evaluations.

```ts
const worker = new TestWorker(serializedSet, { name: 'mse' });
const score = await worker.evaluate(network);
worker.terminate();
```

## multithreading/workers/browser/testworker.ts

### CostFunction

Interface for cost function used in worker evaluation.

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### TestWorker

TestWorker class for handling network evaluations in a browser environment using Web Workers.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of Web Workers allows for offloading
computationally expensive tasks, such as network evaluation, to separate threads.

#### _createBlobString

```ts
_createBlobString(
  cost: CostFunction,
): string
```

Creates a string representation of the worker's blob.

Returns: The blob string.

#### evaluate

```ts
evaluate(
  network: SerializableNetwork,
): Promise<number>
```

Evaluates a network using the worker process.

Returns: A promise that resolves to the evaluation result.

#### terminate

```ts
terminate(): void
```

Terminates the worker process and revokes the object URL.
