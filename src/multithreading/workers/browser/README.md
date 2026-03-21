# multithreading/workers/browser

## multithreading/workers/browser/testworker.ts

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

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### CostFunction

Interface for cost function used in worker evaluation.
